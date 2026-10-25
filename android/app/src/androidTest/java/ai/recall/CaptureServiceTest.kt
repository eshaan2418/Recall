package ai.recall

import ai.recall.core.OperatingPoint
import ai.recall.service.CaptureService
import ai.recall.service.DebugHooks
import ai.recall.service.RecallRuntime
import android.Manifest
import android.app.NotificationManager
import android.content.Intent
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Foreground service: recording indicator, pause/stop, operating-point extras, counters.json, file-replay input. */
@RunWith(AndroidJUnit4::class)
class CaptureServiceTest {
    @get:Rule val perms: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA, Manifest.permission.POST_NOTIFICATIONS)
    private val ctx = Synthetic.target

    @Before fun setUp() {
        RecallRuntime.resetPipeline()
        RecallRuntime.pipelineFactory = { c, op -> Synthetic.pipeline(RecallRuntime.db(c), op) }
        RecallRuntime.db(ctx).wipe()
        File(ctx.filesDir, CaptureService.COUNTERS_FILE).delete()
    }

    @After fun tearDown() {
        ctx.startService(Intent(ctx, CaptureService::class.java).setAction(CaptureService.ACTION_STOP))
        waitFor(10_000) { !RecallRuntime.recording.value }
        RecallRuntime.pipelineFactory = null
        RecallRuntime.resetPipeline()
        RecallRuntime.db(ctx).wipe()
    }

    private fun waitFor(ms: Long, cond: () -> Boolean): Boolean {
        val end = SystemClock.elapsedRealtime() + ms
        while (SystemClock.elapsedRealtime() < end) { if (cond()) return true; Thread.sleep(100) }
        return cond()
    }

    private fun start(op: OperatingPoint, realtime: Boolean) = ctx.startForegroundService(
        CaptureService.startIntent(ctx, op).putExtra(DebugHooks.EXTRA_REPLAY_FILE, Synthetic.clip().absolutePath)
            .putExtra(DebugHooks.EXTRA_REPLAY_REALTIME, realtime))

    /** Final counters (written by the pipeline thread after recording stops). */
    private fun counters(): JSONObject {
        val f = File(ctx.filesDir, CaptureService.COUNTERS_FILE)
        assertTrue("final counters.json written", waitFor(10_000) { f.exists() && !JSONObject(f.readText()).getBoolean("recording") })
        return JSONObject(f.readText())
    }

    @Test fun replayRunsPipelineAndShowsIndicator() {
        start(OperatingPoint(OperatingPoint.Policy.ALL), realtime = true)
        assertTrue("service starts recording", waitFor(10_000) { RecallRuntime.recording.value })
        val nm = ctx.getSystemService(NotificationManager::class.java)
        assertTrue("ongoing recording notification", waitFor(5_000) { nm.activeNotifications.any { it.id == CaptureService.NOTIF_ID && it.isOngoing } })
        assertTrue("replay finishes and service stops", waitFor(120_000) { !RecallRuntime.recording.value })
        assertTrue("events stored", waitFor(5_000) { RecallRuntime.db(ctx).count() > 0 })
        assertTrue("indicator removed after stop", waitFor(5_000) { nm.activeNotifications.none { it.id == CaptureService.NOTIF_ID } })
        val c = counters()
        for (k in listOf("steps", "gated", "heavy", "events", "db_bytes", "thermal_status", "backoff_level")) assertTrue("counters.json has $k", c.has(k))
        assertEquals("policy=all: every step heavy", c.getLong("steps"), c.getLong("heavy"))
        assertEquals(0L, c.getLong("gated"))
        assertTrue(c.getLong("db_bytes") > 0)
        assertEquals("all", c.getString("policy"))
    }

    @Test fun uniformPolicyRunsEveryKthStep() {
        start(OperatingPoint(OperatingPoint.Policy.UNIFORM, k = 4, backoff = false), realtime = false)
        assertTrue(waitFor(10_000) { RecallRuntime.recording.value })
        assertTrue(waitFor(120_000) { !RecallRuntime.recording.value })
        val c = counters()
        val steps = c.getLong("steps")
        assertTrue("decoded the clip ($steps steps)", steps >= 30)
        assertEquals((steps + 3) / 4, c.getLong("heavy"))
        assertEquals(false, c.getBoolean("backoff"))
    }

    @Test fun learnedPolicyUsesGate() {
        start(OperatingPoint(OperatingPoint.Policy.LEARNED), realtime = false)
        assertTrue(waitFor(10_000) { RecallRuntime.recording.value })
        assertTrue(waitFor(120_000) { !RecallRuntime.recording.value })
        val c = counters()
        assertEquals("gate runs every step at nominal thermal", c.getLong("steps"), c.getLong("gated"))
    }

    /** Live CameraX path (emulator: emulated back camera) through the foreground service; no replay extra. */
    @Test fun liveCameraFeedsPipeline() {
        ctx.startForegroundService(CaptureService.startIntent(ctx, OperatingPoint(OperatingPoint.Policy.UNIFORM, k = 2)))
        assertTrue(waitFor(10_000) { RecallRuntime.recording.value })
        assertTrue("camera frames arrive at ~2 fps (${RecallRuntime.status.value})",
            waitFor(20_000) { (RecallRuntime.pipelineOrNull()?.stats?.steps ?: 0) >= 6 })
        val st = RecallRuntime.pipelineOrNull()!!.stats
        assertTrue("uniform k=2 → about half heavy", st.heavy in (st.steps / 2 - 1)..(st.steps / 2 + 1))
        ctx.startService(Intent(ctx, CaptureService::class.java).setAction(CaptureService.ACTION_PAUSE))
        assertTrue(waitFor(10_000) { !RecallRuntime.recording.value })
    }

    @Test fun stopActionPausesImmediately() {
        start(OperatingPoint(), realtime = true)
        assertTrue(waitFor(10_000) { RecallRuntime.recording.value })
        Thread.sleep(1500)
        ctx.startForegroundService(Intent(ctx, CaptureService::class.java).setAction(CaptureService.ACTION_STOP))
        assertTrue("stopped", waitFor(10_000) { !RecallRuntime.recording.value })
        assertEquals("0", RecallRuntime.db(ctx).setting(RecallRuntime.KEY_WANT_RECORDING))
        assertTrue(RecallRuntime.status.value.startsWith("Paused"))
    }
}
