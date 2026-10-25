package ai.recall.core

import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/** Logistic on-probe for appliance state (recall/qa/state.py::StateHead): x = [loc_emb, scene_emb]. */
class StateProbe(val coef: DoubleArray, val intercept: Double) {
    fun pOn(e: Event): Double {
        val d = e.locEmb.size
        require(coef.size == d + e.sceneEmb.size) { "state probe expects ${coef.size} features, got ${d + e.sceneEmb.size}" }
        var z = intercept
        for (i in 0 until d) z += coef[i] * e.locEmb[i]
        for (i in e.sceneEmb.indices) z += coef[d + i] * e.sceneEmb[i]
        return 1.0 / (1.0 + exp(-z))
    }
}

/** Optional per-model preprocessing overrides (answer_config.json → "models"). Defaults match the export contract. */
data class Preprocess(
    val embedSize: Int = 256,
    val embedMean: FloatArray = floatArrayOf(0f, 0f, 0f),
    val embedStd: FloatArray = floatArrayOf(1f, 1f, 1f),
    val hoiW: Int = 256,
    val hoiH: Int = 144,
    val hoiPresentThresh: Float = 0.5f,
    val faceThresh: Float = 0.4f,
)

/** Vocabulary + heads + calibrator + answer configs, exported by the Python side (see android/README.md). */
class Bundle(
    val nounNames: List<String>,
    val synonyms: Map<String, Int>,
    val locIds: List<Int>, // location head index -> noun class id
    val locNames: List<String>,
    val obj: BlendHead,
    val loc: BlendHead,
    val calibrator: Calibrator,
    val configs: Map<String, AnswerConfig>,
    val gateThreshold: Float,
    val stateOnPrior: Map<Int, Double>,
    val globalStatePrior: Double = 0.5,
    val stateProbe: StateProbe? = null,
    val preprocess: Preprocess = Preprocess(),
) {
    val dim: Int get() = obj.textBank.firstOrNull()?.size ?: 0

    fun pOn(e: Event, cls: Int): Double = stateProbe?.pOn(e) ?: stateOnPrior[cls] ?: globalStatePrior

    companion object {
        private fun JSONObject.floats(k: String, def: FloatArray): FloatArray =
            optJSONArray(k)?.let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } } ?: def

        /** heads.bin layout: little-endian; see android/README.md "Asset contract". */
        fun load(vocabJson: String, headsBin: ByteArray, calibJson: String, configJson: String): Bundle {
            val v = JSONObject(vocabJson)
            val nouns = v.getJSONArray("nouns").let { a -> List(a.length()) { a.getString(it) } }
            val syn = v.getJSONObject("synonyms").let { o -> o.keys().asSequence().associateWith { o.getInt(it) } }
            val locIds = v.getJSONArray("loc_ids").let { a -> List(a.length()) { a.getInt(it) } }
            val locNames = v.getJSONArray("loc_names").let { a -> List(a.length()) { a.getString(it) } }
            val b = ByteBuffer.wrap(headsBin).order(ByteOrder.LITTLE_ENDIAN)
            val obj = readHead(b)
            val loc = readHead(b)
            require(obj.nClasses == nouns.size) { "heads.bin object head has ${obj.nClasses} classes, vocab.json ${nouns.size}" }
            require(loc.nClasses == locNames.size) { "heads.bin location head has ${loc.nClasses} classes, vocab.json ${locNames.size}" }
            require(obj.textBank[0].size == loc.textBank[0].size) { "object/location heads disagree on embedding dim" }
            val c = JSONObject(calibJson)
            val cal = Calibrator(
                c.getJSONArray("features").let { a -> List(a.length()) { a.getString(it) } },
                c.getJSONArray("coef").let { a -> DoubleArray(a.length()) { a.getDouble(it) } },
                c.getDouble("intercept"),
                c.getJSONArray("iso_x").let { a -> DoubleArray(a.length()) { a.getDouble(it) } },
                c.getJSONArray("iso_y").let { a -> DoubleArray(a.length()) { a.getDouble(it) } },
                c.getDouble("threshold"),
            )
            val cj = JSONObject(configJson)
            val cfgs = listOf("where", "temporal", "state").associateWith { k ->
                val o = cj.getJSONObject("configs").getJSONObject(k)
                AnswerConfig(o.getString("mode"), o.getDouble("tau"), o.optDouble("recency_s", 600.0), o.optString("score", "prob"),
                    o.optBoolean("fallback_take", true), o.optString("loc_mode", "event"))
            }
            val prior = cj.optJSONObject("state_prior")?.let { o -> o.keys().asSequence().associate { it.toInt() to o.getDouble(it) } } ?: emptyMap()
            val probe = cj.optJSONObject("state_probe")?.let { o ->
                StateProbe(o.getJSONArray("coef").let { a -> DoubleArray(a.length()) { a.getDouble(it) } }, o.getDouble("intercept"))
            }
            val pre = cj.optJSONObject("models")?.let { m ->
                val d = Preprocess()
                val e = m.optJSONObject("embedder") ?: JSONObject()
                val h = m.optJSONObject("hoi") ?: JSONObject()
                val f = m.optJSONObject("face") ?: JSONObject()
                Preprocess(e.optInt("size", d.embedSize), e.floats("mean", d.embedMean), e.floats("std", d.embedStd),
                    h.optInt("w", d.hoiW), h.optInt("h", d.hoiH), h.optDouble("present_thresh", d.hoiPresentThresh.toDouble()).toFloat(),
                    f.optDouble("thresh", d.faceThresh.toDouble()).toFloat())
            } ?: Preprocess()
            return Bundle(nouns, syn, locIds, locNames, obj, loc, cal, cfgs, cj.optDouble("gate_threshold", 0.5).toFloat(), prior,
                cj.optDouble("global_state_prior", 0.5), probe, pre)
        }

        fun readHead(b: ByteBuffer): BlendHead {
            val c = b.int; val d = b.int
            val scale = b.float; val alpha = b.float
            val bank = Array(c) { FloatArray(d) { b.float } }
            val k = b.int
            if (k == 0) return BlendHead(bank, scale, alpha)
            val cls = IntArray(k) { b.int }
            val w = Array(k) { FloatArray(d) { b.float } }
            val bias = FloatArray(k) { b.float }
            return BlendHead(bank, scale, alpha, w, bias, cls)
        }

        /** Inverse of [readHead]; used by tests and the synthetic (no-model) bundle. */
        fun writeHead(h: BlendHead): ByteArray {
            val c = h.textBank.size; val d = h.textBank[0].size; val k = h.probeW?.size ?: 0
            val b = ByteBuffer.allocate(4 * (5 + c * d + k * (1 + d + 1))).order(ByteOrder.LITTLE_ENDIAN)
            b.putInt(c); b.putInt(d); b.putFloat(h.scale); b.putFloat(h.alpha)
            for (r in h.textBank) for (x in r) b.putFloat(x)
            b.putInt(k)
            if (k > 0) {
                for (x in h.probeClasses!!) b.putInt(x)
                for (r in h.probeW!!) for (x in r) b.putFloat(x)
                for (x in h.probeB!!) b.putFloat(x)
            }
            return b.array()
        }
    }
}

data class QueryResult(val parsed: ParsedQuery, val cls: Int?, val answer: Answer, val confidence: Double, val rendered: Rendered,
                       val locName: String?)

class QueryEngine(private val b: Bundle) {
    val bundle: Bundle get() = b

    /** Synonym lookup; mirrors recall/qa/resolve.py (t, t.rstrip("s"), t[:-2] if t ends with "es"). */
    fun resolve(text: String): Int? {
        val t = text.lowercase().trim()
        for (c in listOf(t, t.trimEnd('s'), if (t.endsWith("es")) t.dropLast(2) else t)) b.synonyms[c]?.let { return it }
        return null
    }

    fun topObject(e: Event): Int = b.obj.logProbs(e.objEmb).let { lp -> lp.indices.maxByOrNull { lp[it] }!! }

    fun ask(question: String, events: List<Event>, nowT: Double, zoneOffsetS: Int = 0): QueryResult {
        val pq = QueryParser.parse(question)
        val cls = resolve(pq.obj)
        if (cls == null) {
            val a = Answer(pq.kind, true)
            return QueryResult(pq, null, a, 0.0, Renderer.render(a, pq.obj, null, 0.0, b.calibrator.abstainBelow), null)
        }
        val view = MemoryView(events, b.obj, b.loc)
        val cfg = b.configs[pq.kind] ?: AnswerConfig()
        val a = when (pq.kind) {
            "where" -> Answerer.where(view, cls, nowT, cfg)
            "temporal" -> Answerer.temporal(view, cls, pq.verb, nowT, cfg)
            else -> Answerer.state(view, cls, nowT, cfg, b::pOn)
        }
        val conf = b.calibrator.confidence(a)
        val locName = a.locIdx?.let { b.locNames[it] }
        return QueryResult(pq, cls, a, conf, Renderer.render(a, pq.obj, locName, conf, b.calibrator.abstainBelow, zoneOffsetS), locName)
    }
}
