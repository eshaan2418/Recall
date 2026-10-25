package ai.recall

import ai.recall.core.OperatingPoint
import ai.recall.core.QueryEngine
import ai.recall.core.Scheduler
import ai.recall.data.KeyManager
import ai.recall.data.MemoryDb
import ai.recall.ml.LiteRtEmbedder
import ai.recall.ml.LiteRtGate
import ai.recall.privacy.Redactor
import ai.recall.service.Pipeline
import ai.recall.service.RecallRuntime
import ai.recall.service.ReplaySource
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Scripted end-to-end demo (scripts/demo_e2e.sh): the synthetic clip (a hand puts the keys on the table) is injected
 * as camera input, the pipeline runs, then we ask "Where are my keys?" and assert the answer.
 *
 *  - plumbing: synthetic gate/embedder/heads (no model files needed) — verifies the whole device path.
 *  - realModels: the installed LiteRT embedder + exported heads/calibrator; the HOI student is replaced by the
 *    colour-key detector because the clip's hand is drawn, not real (instrumentation arg gate=model uses gate.tflite).
 */
@RunWith(AndroidJUnit4::class)
class E2eDemoTest {
    private val ctx = Synthetic.target
    private val dbName = "e2e-memory.db"
    private lateinit var db: MemoryDb
    private val args = InstrumentationRegistry.getArguments()

    @Before fun open() { ctx.deleteDatabase(dbName); db = MemoryDb(ctx, dbName) }
    @After fun close() { db.close(); ctx.deleteDatabase(dbName); KeyManager(ctx, "recall-key-$dbName").destroy() }

    private fun replay(p: Pipeline, t0: Double) {
        val src = ReplaySource(Synthetic.clip(), startEpochS = t0)
        var err: String? = "not finished"
        src.start({ bm, t -> p.onFrame(bm, t, FloatArray(7), Scheduler.NOMINAL) }, { err = it })
        src.join()
        assertEquals(null, err)
        p.flush()
        Log.i(TAG, "replayed ${src.frames} frames; stats=${p.stats.toJson()}")
        assertTrue("decoded the clip", src.frames >= 30)
    }

    private fun report(mode: String, q: String, text: String, loc: String?, conf: Double) {
        val line = "E2E[$mode] Q: \"$q\" → A: \"$text\" (location=$loc, confidence=${"%.2f".format(conf)})"
        Log.i(TAG, line)
        InstrumentationRegistry.getInstrumentation().sendStatus(0, android.os.Bundle().apply { putString("stream", "$line\n") })
    }

    @Test fun plumbing_whereAreMyKeys() {
        val t0 = 1_700_000_000.0
        val p = Synthetic.pipeline(db, OperatingPoint(OperatingPoint.Policy.ALL))
        replay(p, t0)
        val events = db.events().map { it.event }
        assertTrue("events extracted: ${events.map { it.kind }}", events.any { it.kind == "put" })
        val r = QueryEngine(Synthetic.bundle()).ask("Where are my keys?", events, t0 + 20)
        report("plumbing", "Where are my keys?", r.rendered.text, r.locName, r.confidence)
        assertFalse(r.rendered.text, r.rendered.abstained)
        assertEquals("table", r.locName)
        assertTrue("put-down time ≈ 10–12.5 s into the clip, got ${r.answer.t!! - t0}", r.answer.t!! - t0 in 10.0..12.5)
        assertNotNull("redacted thumbnail stored", db.thumb(r.answer.event!!.id))
        val never = QueryEngine(Synthetic.bundle()).ask("Where is my phone?", events, t0 + 20)
        report("plumbing", "Where is my phone?", never.rendered.text, never.locName, never.confidence)
        assertTrue("never-seen object → abstain", never.rendered.abstained)
    }

    @Test fun realModels_whereAreMyKeys() {
        val missing = RecallRuntime.missingAssets(ctx)
        if (missing.isNotEmpty()) { // AGP reports JUnit assumptions as failures, so skip explicitly and say so
            report("real-models", "Where are my keys?", "SKIPPED: real models not installed (missing ${missing.joinToString()})", null, 0.0)
            return
        }
        val b = assertNotNullBundle()
        val emb = LiteRtEmbedder(ctx, mean = b.preprocess.embedMean, std = b.preprocess.embedStd)
        assertEquals(b.dim, emb.dim)
        val gate = if (args.getString("gate") == "model") LiteRtGate(ctx) else Synthetic.AlwaysOnGate()
        val p = Pipeline(gate, Synthetic.ColourKeyHandObject(), emb, b, db, Redactor(RecallRuntime.redactionDetectors(ctx, b.preprocess.faceThresh)),
            OperatingPoint(if (args.getString("gate") == "model") OperatingPoint.Policy.LEARNED else OperatingPoint.Policy.ALL))
        val t0 = 1_700_000_000.0
        replay(p, t0)
        val events = db.events().map { it.event }
        val r = QueryEngine(b).ask("Where are my keys?", events, t0 + 20)
        report("real-models", "Where are my keys?", r.rendered.text, r.locName, r.confidence)
        assertTrue("put events extracted", events.any { it.kind == "put" })
        assertFalse("answered: ${r.rendered.text}", r.answer.abstain)
        val expect = args.getString("expect_loc") ?: "table"
        assertEquals(expect, r.locName)
    }

    private fun assertNotNullBundle() = requireNotNull(RecallRuntime.bundle(ctx)) { "bundle invalid: ${RecallRuntime.loadError}" }

    companion object { const val TAG = "RecallE2E" }
}
