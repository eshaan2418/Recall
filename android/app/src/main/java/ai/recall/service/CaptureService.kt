package ai.recall.service

import ai.recall.R
import ai.recall.core.OperatingPoint
import ai.recall.core.Scheduler
import ai.recall.ui.MainActivity
import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.BatteryManager
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Foreground capture service. A persistent notification (recording indicator) with a one-tap Pause action is
 * shown whenever the camera may be in use. Processing happens on a single background thread.
 *
 * Intents:
 *  - ACTION_START (`ai.recall.START`) with optional extras `policy` (all|uniform|learned), `k` (int), `backoff` (bool)
 *  - ACTION_STOP (`ai.recall.STOP`) / ACTION_PAUSE (notification button): stop capture and the service.
 *  - null intent: START_STICKY restart after process death → resumes only if the user had recording on.
 *
 * While running, files/counters.json is rewritten every [COUNTER_PERIOD_S] s for the energy harness.
 */
class CaptureService : LifecycleService() {
    private val exec = Executors.newSingleThreadExecutor { Thread(it, "recall-pipeline") }
    private val analyzerExec = Executors.newSingleThreadExecutor { Thread(it, "recall-camera") }
    private var timer: ScheduledExecutorService? = null
    private var source: FrameSource? = null
    private var op = OperatingPoint()
    private val busy = AtomicBoolean(false)
    @Volatile private var thermal = 0
    @Volatile private var battery = 100 to false
    private var batteryReadAt = 0L

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        val db = RecallRuntime.db(this)
        when (intent?.action) {
            ACTION_STOP, ACTION_PAUSE -> {
                // Started via startForegroundService(STOP) → must still enter the foreground before stopping.
                tryStartForeground()
                db.setSetting(RecallRuntime.KEY_WANT_RECORDING, "0")
                shutdown("Paused")
                return START_NOT_STICKY
            }
            null -> { // sticky restart after process death
                if (db.setting(RecallRuntime.KEY_WANT_RECORDING) != "1") { tryStartForeground(); shutdown(""); return START_NOT_STICKY }
                op = db.setting(RecallRuntime.KEY_OPERATING_POINT)?.let { decodeOp(it) } ?: OperatingPoint()
            }
            else -> {
                op = try {
                    OperatingPoint.parse(intent.getStringExtra(EXTRA_POLICY),
                        if (intent.hasExtra(EXTRA_K)) intent.getIntExtra(EXTRA_K, 5) else null,
                        if (intent.hasExtra(EXTRA_BACKOFF)) intent.getBooleanExtra(EXTRA_BACKOFF, true) else null)
                } catch (e: IllegalArgumentException) {
                    tryStartForeground(); shutdown("Bad start intent: ${e.message}"); return START_NOT_STICKY
                }
            }
        }
        val replay = DebugHooks.replaySource(intent)
        if (replay == null && ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            tryStartForeground()
            shutdown("Camera permission not granted — recording stopped") // e.g. revoked in Settings
            return START_NOT_STICKY
        }
        if (!tryStartForeground()) {
            shutdown("Could not start the recording indicator (open Recall to resume)")
            return START_NOT_STICKY
        }
        val pipeline = RecallRuntime.pipeline(this, op)
        if (pipeline == null) {
            shutdown("Models not installed or invalid: ${RecallRuntime.loadError}")
            return START_NOT_STICKY
        }
        db.setSetting(RecallRuntime.KEY_WANT_RECORDING, "1")
        db.setSetting(RecallRuntime.KEY_OPERATING_POINT, encodeOp(op))
        if (source != null && replay == null) return START_STICKY // already capturing; operating point updated
        source?.stop()
        val src = replay ?: CameraXSource(this, this, analyzerExec)
        source = src
        watchThermal()
        startTimer()
        post { RecallRuntime.applyRetention(this) }
        src.start({ bm, t ->
            val work = Runnable {
                try {
                    val (lvl, charging) = batteryState()
                    pipeline.onFrame(bm, t, FloatArray(7), Scheduler.plan(thermal, lvl, charging))
                    if (pipeline.stats.events != lastEvents) { lastEvents = pipeline.stats.events; RecallRuntime.memoryChanged() }
                } catch (e: Exception) {
                    Log.e(TAG, "pipeline step failed", e)
                    RecallRuntime.status.value = "Pipeline error: ${e.message}"
                } finally {
                    busy.set(false)
                }
            }
            try {
                if (src.blocking) { busy.set(true); exec.submit(work).get() }
                else if (busy.compareAndSet(false, true)) exec.execute(work) // drop frames while busy (keep-latest)
            } catch (_: RejectedExecutionException) { // service already destroyed
                busy.set(false); src.stop()
            }
        }, { err ->
            post { pipeline.flush(); writeCounters(); RecallRuntime.memoryChanged() }
            if (src === source) {
                ContextCompat.getMainExecutor(this).execute {
                    if (src !== source) return@execute
                    if (src === replay) db.setSetting(RecallRuntime.KEY_WANT_RECORDING, "0")
                    shutdown(err ?: "Replay finished")
                }
            }
        })
        RecallRuntime.recording.value = true
        RecallRuntime.status.value = "Remembering (${op.policy.name.lowercase()}${if (op.policy == OperatingPoint.Policy.UNIFORM) " k=${op.k}" else ""}, backoff ${if (op.backoff) "on" else "off"})"
        return START_STICKY
    }

    @Volatile private var lastEvents = 0L

    /** Runs [r] on the pipeline thread unless the service is already shutting down. */
    private fun post(r: () -> Unit) {
        try { exec.execute(r) } catch (_: RejectedExecutionException) {}
    }

    private fun batteryState(): Pair<Int, Boolean> {
        val now = SystemClock.elapsedRealtime()
        if (now - batteryReadAt > 30_000 || batteryReadAt == 0L) {
            batteryReadAt = now
            val bat = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val level = bat?.getIntExtra(BatteryManager.EXTRA_LEVEL, 100) ?: 100
            val scale = bat?.getIntExtra(BatteryManager.EXTRA_SCALE, 100)?.takeIf { it > 0 } ?: 100
            val st = bat?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
            battery = (level * 100 / scale) to (st == BatteryManager.BATTERY_STATUS_CHARGING || st == BatteryManager.BATTERY_STATUS_FULL)
        }
        return battery
    }

    private val thermalListener = PowerManager.OnThermalStatusChangedListener { thermal = it }

    private var thermalWatched = false

    private fun watchThermal() {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        thermal = pm.currentThermalStatus
        if (!thermalWatched) {
            pm.addThermalStatusListener(ContextCompat.getMainExecutor(this), thermalListener)
            thermalWatched = true
        }
    }

    private fun unwatchThermal() {
        if (!thermalWatched) return
        (getSystemService(Context.POWER_SERVICE) as PowerManager).removeThermalStatusListener(thermalListener)
        thermalWatched = false
    }

    private fun startTimer() {
        if (timer != null) return
        timer = Executors.newSingleThreadScheduledExecutor().also { t ->
            t.scheduleWithFixedDelay({ writeCounters() }, 0, COUNTER_PERIOD_S, TimeUnit.SECONDS)
            t.scheduleWithFixedDelay({ post { RecallRuntime.applyRetention(this) } }, 1, 1, TimeUnit.HOURS)
        }
    }

    /** Atomic rewrite of files/counters.json (energy harness: `adb shell run-as ai.recall cat files/counters.json`). */
    private fun writeCounters() {
        try {
            val p = RecallRuntime.pipelineOrNull() ?: return
            val json = p.stats.toJson(mapOf(
                "db_bytes" to RecallRuntime.db(this).bytes(), "thermal_status" to thermal,
                "policy" to op.policy.name.lowercase(), "k" to op.k, "backoff" to op.backoff,
                "recording" to RecallRuntime.recording.value, "t" to System.currentTimeMillis() / 1000.0,
            ))
            val f = File(filesDir, COUNTERS_FILE)
            val tmp = File(filesDir, "$COUNTERS_FILE.tmp")
            tmp.writeText(json.toString())
            tmp.renameTo(f)
        } catch (e: Exception) {
            Log.w(TAG, "counters write failed", e)
        }
    }

    private fun tryStartForeground(): Boolean = try {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, getString(R.string.recording_channel), NotificationManager.IMPORTANCE_LOW))
        val pause = PendingIntent.getService(this, 0, Intent(this, CaptureService::class.java).setAction(ACTION_PAUSE),
            PendingIntent.FLAG_IMMUTABLE)
        val open = PendingIntent.getActivity(this, 1, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val n: Notification = NotificationCompat.Builder(this, CHANNEL)
            .setContentTitle(getString(R.string.recording_title))
            .setContentText(getString(R.string.recording_text))
            .setSmallIcon(android.R.drawable.presence_video_online)
            .setOngoing(true)
            .setContentIntent(open)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .addAction(android.R.drawable.ic_media_pause, getString(R.string.pause), pause)
            .build()
        val hasCam = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        // Camera FGS type requires the camera permission (API 34+); without it we can only enter the foreground
        // briefly to satisfy startForegroundService() and stop.
        val type = if (hasCam) ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA else 0
        ServiceCompat.startForeground(this, NOTIF_ID, n, type)
        true
    } catch (e: Exception) { // ForegroundServiceStartNotAllowedException, SecurityException …
        Log.w(TAG, "startForeground failed", e)
        false
    }

    private fun shutdown(msg: String) {
        stopCapture()
        if (msg.isNotEmpty()) RecallRuntime.status.value = msg
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun stopCapture() {
        source?.stop()
        source = null
        post { RecallRuntime.pipelineOrNull()?.flush(); writeCounters(); RecallRuntime.memoryChanged() }
        timer?.shutdown(); timer = null
        unwatchThermal()
        RecallRuntime.recording.value = false
    }

    override fun onDestroy() {
        stopCapture()
        exec.shutdown()
        analyzerExec.shutdown()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "RecallCapture"
        const val CHANNEL = "recall-capture"
        const val NOTIF_ID = 42
        const val ACTION_START = "ai.recall.START"
        const val ACTION_STOP = "ai.recall.STOP"
        const val ACTION_PAUSE = "ai.recall.PAUSE"
        const val EXTRA_POLICY = "policy"
        const val EXTRA_K = "k"
        const val EXTRA_BACKOFF = "backoff"
        const val COUNTERS_FILE = "counters.json"
        const val COUNTER_PERIOD_S = 10L

        fun encodeOp(o: OperatingPoint) = "${o.policy.name.lowercase()},${o.k},${o.backoff}"
        fun decodeOp(s: String): OperatingPoint? = try {
            val p = s.split(","); OperatingPoint.parse(p[0], p[1].toInt(), p[2].toBooleanStrict())
        } catch (_: Exception) { null }

        fun startIntent(ctx: Context, op: OperatingPoint = OperatingPoint()): Intent =
            Intent(ctx, CaptureService::class.java).setAction(ACTION_START)
                .putExtra(EXTRA_POLICY, op.policy.name.lowercase()).putExtra(EXTRA_K, op.k).putExtra(EXTRA_BACKOFF, op.backoff)
    }
}
