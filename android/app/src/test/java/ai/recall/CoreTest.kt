package ai.recall

import ai.recall.core.Answer
import ai.recall.core.BlendHead
import ai.recall.core.Bundle
import ai.recall.core.Event
import ai.recall.core.Fp16
import ai.recall.core.QueryEngine
import ai.recall.core.unit
import ai.recall.service.Pipeline
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

class CoreTest {
    @Test fun fp16RoundTrip() {
        val r = Random(0)
        val xs = FloatArray(2000) { (r.nextFloat() - 0.5f) * 4f } + floatArrayOf(0f, -0f, 1f, -1f, 65504f, 1e-5f, 6e-8f, 1e9f)
        val back = Fp16.decode(Fp16.encode(xs))
        for (i in xs.indices) {
            val x = xs[i]
            when {
                x == 1e9f -> assertEquals(Float.POSITIVE_INFINITY, back[i])
                kotlin.math.abs(x) < 6.2e-5f -> assertEquals(x, back[i], 6e-8f) // subnormal range
                else -> assertEquals("x=$x", x, back[i], kotlin.math.abs(x) * 1e-3f)
            }
        }
        // exact values
        assertEquals(0x3c00.toShort(), Fp16.toHalf(1f))
        assertEquals(0xc000.toShort(), Fp16.toHalf(-2f))
        assertEquals(0x7bff.toShort(), Fp16.toHalf(65504f))
        assertEquals(0.333251953125f, Fp16.toFloat(Fp16.toHalf(1f / 3f)), 0f)
        assertTrue(Fp16.toFloat(Fp16.toHalf(Float.NaN)).isNaN())
    }

    @Test fun cropRectMatchesPythonExpandBox() {
        // recall/perception/extract.py: expand_box((0.4,0.4,0.5,0.6), 1.15) on 456x256 → crop [x0:x1, y0:y1]
        val r = Pipeline.cropRect(456, 256, floatArrayOf(0.4f, 0.4f, 0.5f, 0.6f), 1.15f)
        // l=0.3925 t=0.385 r=0.5075 b=0.615 → x0=int(178.98)=178, y0=int(98.56)=98, x1=ceil(231.42)=232, y1=ceil(157.44)=158
        assertArrayEquals(intArrayOf(178, 98, 232, 158), r)
        val edge = Pipeline.cropRect(456, 256, floatArrayOf(0.95f, 0.95f, 1f, 1f), 3f)
        assertTrue(edge[2] <= 456 && edge[3] <= 256 && edge[2] - edge[0] >= 2 && edge[3] - edge[1] >= 2)
        val tiny = Pipeline.cropRect(456, 256, floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f), 1.15f)
        assertTrue("min size 0.06", tiny[2] - tiny[0] >= 27)
    }

    private fun head(c: Int, d: Int, r: Random, probe: Boolean): BlendHead {
        val bank = Array(c) { unit(FloatArray(d) { r.nextFloat() - 0.5f }) }
        if (!probe) return BlendHead(bank, 50f, 1f)
        return BlendHead(bank, 50f, 0.5f, Array(2) { FloatArray(d) { r.nextFloat() } }, floatArrayOf(0.1f, -0.2f), intArrayOf(1, 3))
    }

    @Test fun headsBinRoundTrip() {
        val r = Random(1)
        for (h in listOf(head(6, 8, r, false), head(6, 8, r, true))) {
            val b = Bundle.readHead(ByteBuffer.wrap(Bundle.writeHead(h)).order(ByteOrder.LITTLE_ENDIAN))
            assertEquals(h.scale, b.scale); assertEquals(h.alpha, b.alpha)
            for (i in h.textBank.indices) assertArrayEquals(h.textBank[i], b.textBank[i], 0f)
            assertEquals(h.probeClasses?.toList(), b.probeClasses?.toList())
            val e = unit(FloatArray(8) { r.nextFloat() })
            assertArrayEquals(h.logProbs(e), b.logProbs(e), 1e-12)
        }
    }

    private fun bundleJson(nouns: List<String>, locs: List<String>, extraCfg: JSONObject.() -> Unit = {}): Array<String> {
        val vocab = JSONObject().put("nouns", JSONArray(nouns)).put("loc_ids", JSONArray(locs.indices.toList())).put("loc_names", JSONArray(locs))
            .put("synonyms", JSONObject().apply { nouns.forEachIndexed { i, n -> put(n, i) }; put("car keys", 0) })
        val calib = JSONObject().put("features", JSONArray(listOf("score", "kind=where"))).put("coef", JSONArray(listOf(4.0, 0.5)))
            .put("intercept", -1.0).put("iso_x", JSONArray()).put("iso_y", JSONArray()).put("threshold", 0.2)
        val cfg = JSONObject().put("configs", JSONObject().apply {
            for (k in listOf("where", "temporal", "state")) put(k, JSONObject().put("mode", "latest").put("tau", 0.05).put("recency_s", 600.0))
        }).put("gate_threshold", 0.42).put("state_prior", JSONObject().put("2", 0.9)).apply(extraCfg)
        return arrayOf(vocab.toString(), calib.toString(), cfg.toString())
    }

    @Test fun bundleLoadAndQueryEndToEnd() {
        val d = 8
        val nouns = listOf("keys", "phone", "tap")
        val locs = listOf("table", "drawer")
        // one-hot text banks: dim 0..2 objects, dim 4..5 locations
        val obj = BlendHead(Array(3) { c -> FloatArray(d).also { it[c] = 1f } }, 30f, 1f)
        val loc = BlendHead(Array(2) { c -> FloatArray(d).also { it[4 + c] = 1f } }, 30f, 1f)
        val (v, c, cfg) = bundleJson(nouns, locs)
        val b = Bundle.load(v, Bundle.writeHead(obj) + Bundle.writeHead(loc), c, cfg)
        assertEquals(0.42f, b.gateThreshold); assertEquals(d, b.dim)
        assertEquals(0.9, b.pOn(Event(0.0, "touch", 0, FloatArray(d), FloatArray(d), FloatArray(d)), 2), 0.0)
        val q = QueryEngine(b)
        assertEquals(0, q.resolve("Keys")); assertEquals(0, q.resolve("car keys")); assertEquals(1, q.resolve("phones"))
        assertNull(q.resolve("giraffe"))
        fun e(t: Double, kind: String, o: Int, l: Int) = Event(t, kind, 1, FloatArray(d).also { it[o] = 1f }, FloatArray(d).also { it[4 + l] = 1f }, FloatArray(d))
        val evs = listOf(e(100.0, "take", 0, 0), e(110.0, "put", 0, 1), e(200.0, "put", 1, 0), e(300.0, "take", 0, 0), e(305.0, "put", 0, 0))
        val r = q.ask("Where are my keys?", evs, 1000.0)
        assertFalse(r.rendered.abstained)
        assertEquals("table", r.locName)
        assertEquals(305.0, r.answer.t!!, 0.0)
        assertTrue(r.rendered.text, r.rendered.text.startsWith("Probably the table"))
        val past = q.ask("where did I leave my keys", evs, 200.0) // before the second put
        assertEquals("drawer", past.locName)
        val unknown = q.ask("where is the giraffe", evs, 1000.0)
        assertTrue(unknown.rendered.abstained)
        val never = q.ask("where is my phone", evs.take(2), 1000.0)
        assertTrue("phone never seen → abstain", never.rendered.abstained)
        val temporal = q.ask("When did I last pick up the keys?", evs, 1000.0)
        assertEquals(300.0, temporal.answer.t!!, 0.0)
    }

    @Test fun bundleRejectsMismatchedVocab() {
        val d = 4
        val obj = BlendHead(Array(3) { FloatArray(d) }, 30f, 1f)
        val loc = BlendHead(Array(2) { FloatArray(d) }, 30f, 1f)
        val (v, c, cfg) = bundleJson(listOf("a", "b"), listOf("x", "y"))
        try {
            Bundle.load(v, Bundle.writeHead(obj) + Bundle.writeHead(loc), c, cfg); throw AssertionError("expected failure")
        } catch (e: IllegalArgumentException) { assertTrue(e.message!!.contains("classes")) }
    }

    @Test fun bundleOptionalBlocks() {
        val d = 4
        val obj = BlendHead(Array(2) { FloatArray(d) }, 30f, 1f)
        val loc = BlendHead(Array(2) { FloatArray(d) }, 30f, 1f)
        val (v, c, cfg) = bundleJson(listOf("a", "b"), listOf("x", "y")) {
            put("state_probe", JSONObject().put("coef", JSONArray(DoubleArray(2 * d) { 1.0 }.toList())).put("intercept", 0.0))
            put("models", JSONObject().put("embedder", JSONObject().put("size", 224).put("mean", JSONArray(listOf(0.5, 0.5, 0.5)))))
        }
        val b = Bundle.load(v, Bundle.writeHead(obj) + Bundle.writeHead(loc), c, cfg)
        assertEquals(224, b.preprocess.embedSize)
        assertEquals(0.5f, b.preprocess.embedMean[0])
        assertEquals(1f, b.preprocess.embedStd[0])
        val ev = Event(0.0, "touch", 0, FloatArray(d), FloatArray(d) { 1f }, FloatArray(d) { 1f })
        assertEquals(1 / (1 + Math.exp(-8.0)), b.pOn(ev, 0), 1e-12)
    }

    @Test fun calibratorAbstainsAtZero() {
        val (v, c, cfg) = bundleJson(listOf("a"), listOf("x"))
        val b = Bundle.load(v, Bundle.writeHead(BlendHead(arrayOf(FloatArray(2)), 1f, 1f)) + Bundle.writeHead(BlendHead(arrayOf(FloatArray(2)), 1f, 1f)), c, cfg)
        assertEquals(0.0, b.calibrator.confidence(Answer("where", true)), 0.0)
        assertEquals(1 / (1 + Math.exp(-(4 * 0.9 + 0.5 - 1))), b.calibrator.confidence(Answer("where", false, feats = mapOf("score" to 0.9))), 1e-12)
    }
}
