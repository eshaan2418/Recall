package ai.recall.core

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * Blended zero-shot + linear-probe classifier (recall/perception/heads.py::BlendHead).
 * textBank: [C][D]; probeW: [K][D], probeB: [K] for classes probeClasses (subset of C).
 */
class BlendHead(
    val textBank: Array<FloatArray>,
    val scale: Float,
    val alpha: Float,
    val probeW: Array<FloatArray>? = null,
    val probeB: FloatArray? = null,
    val probeClasses: IntArray? = null,
) {
    val nClasses get() = textBank.size

    fun logProbs(emb: FloatArray): DoubleArray {
        val C = textBank.size
        // Python: lz = log(softmax(scale * <emb, bank>) + 1e-9)
        val lz = DoubleArray(C) { scale.toDouble() * dotD(emb, textBank[it]) }
        logSoftmaxInPlace(lz)
        for (i in lz.indices) lz[i] = ln(exp(lz[i]) + 1e-9)
        if (probeW == null || alpha >= 1f) return lz
        val lp = DoubleArray(C) { ln(1e-4) }
        val logits = DoubleArray(probeW.size) { k -> dotD(emb, probeW[k]) + probeB!![k] }
        logSoftmaxInPlace(logits)
        for (k in logits.indices) lp[probeClasses!![k]] = ln(exp(logits[k]) + 1e-9)
        val z = DoubleArray(C) { alpha * lz[it] + (1 - alpha) * lp[it] }
        logSoftmaxInPlace(z)
        return z
    }

    companion object {
        fun dotD(a: FloatArray, b: FloatArray): Double {
            var s = 0.0
            for (i in a.indices) s += a[i].toDouble() * b[i]
            return s
        }

        fun logSoftmaxInPlace(x: DoubleArray) {
            val m = x.max()
            var s = 0.0
            for (v in x) s += exp(v - m)
            val l = m + ln(s)
            for (i in x.indices) x[i] -= l
        }
    }
}

data class AnswerConfig(
    val mode: String = "latest", // latest | recency | best
    val tau: Double = 0.05,
    val recencyS: Double = 600.0,
    val score: String = "prob", // prob | ratio
    val fallbackTake: Boolean = true,
)

data class Answer(
    val kind: String,
    val abstain: Boolean,
    val event: Event? = null,
    val t: Double? = null,
    val locIdx: Int? = null,
    val state: String? = null,
    val feats: Map<String, Double> = emptyMap(),
)

/** Column view over events with class log-probs (recall/qa/answer.py::MemoryView). */
class MemoryView(val events: List<Event>, objHead: BlendHead, locHead: BlendHead) {
    val objLogp: List<DoubleArray> = events.map { objHead.logProbs(it.objEmb) }
    val locLogp: List<DoubleArray> = events.map { locHead.logProbs(it.locEmb) }
}

object Answerer {
    private val kindForVerb = mapOf(
        "take" to setOf("take"), "put" to setOf("put"), "open" to setOf("touch", "take"),
        "close" to setOf("touch", "take"), "turn-on" to setOf("touch"), "turn-off" to setOf("touch"),
        "wash" to setOf("take", "touch"),
    )

    private fun scores(v: MemoryView, cls: Int, mask: BooleanArray, cfg: AnswerConfig): DoubleArray =
        DoubleArray(v.events.size) { i ->
            if (!mask[i]) 0.0
            else {
                var lp = v.objLogp[i][cls]
                if (cfg.score == "ratio") lp -= v.objLogp[i].max()
                exp(lp)
            }
        }

    private fun pick(s: DoubleArray, v: MemoryView, tq: Double, cfg: AnswerConfig): Pair<Int?, List<Int>> {
        val cand = s.indices.filter { s[it] >= cfg.tau }
        if (cand.isEmpty()) return null to cand
        // Tie-breaking mirrors numpy: argmax -> first maximum; lexsort(...)[-1] -> last maximum.
        val j = when (cfg.mode) {
            "latest" -> cand.maxByOrNull { v.events[it].t }!!
            "best" -> {
                var b = cand[0]
                for (c in cand) {
                    if (s[c] > s[b] || (s[c] == s[b] && v.events[c].t >= v.events[b].t)) b = c
                }
                b
            }
            else -> cand.maxByOrNull { s[it] * exp(-(tq - v.events[it].t) / cfg.recencyS) }!!
        }
        return j to cand
    }

    private fun features(v: MemoryView, j: Int, cls: Int, s: DoubleArray, cand: List<Int>, tq: Double, tau: Double): Map<String, Double> {
        val lp = v.objLogp[j]
        val sc = s[j]
        var otherMax = 0.0
        for (c in lp.indices) if (c != cls) otherMax = max(otherMax, exp(lp[c]))
        val locp = v.locLogp[j].map { exp(it) }.sortedDescending()
        val later = cand.count { v.events[it].t > v.events[j].t - 1e-6 && it != j }
        val rest = cand.filter { it != j }.map { s[it] }
        val argmax = lp.indices.maxByOrNull { lp[it] } == cls
        return mapOf(
            "score" to sc, "is_argmax" to if (argmax) 1.0 else 0.0, "score_margin" to sc - otherMax,
            "loc_p" to locp[0], "loc_margin" to locp[0] - (locp.getOrElse(1) { 0.0 }),
            "log_n_obs" to ln(1.0 + v.events[j].nObs), "log_age_s" to ln(1.0 + max(0.0, tq - v.events[j].t)),
            "n_conflicts" to (later + rest.count { it >= tau }).toDouble(), "second_score" to (rest.maxOrNull() ?: 0.0),
            "release_obs" to if (v.events[j].releaseObserved) 1.0 else 0.0, "hand_score" to v.events[j].handScore.toDouble(),
            "n_candidates" to cand.size.toDouble(),
        )
    }

    fun where(v: MemoryView, cls: Int, tq: Double, cfg: AnswerConfig): Answer {
        var mask = BooleanArray(v.events.size) { v.events[it].kind == "put" && v.events[it].t < tq }
        var s = scores(v, cls, mask, cfg)
        var (j, cand) = pick(s, v, tq, cfg)
        if (j == null && cfg.fallbackTake) {
            mask = BooleanArray(v.events.size) { v.events[it].kind in setOf("take", "touch") && v.events[it].t < tq }
            s = scores(v, cls, mask, cfg)
            val p = pick(s, v, tq, cfg); j = p.first; cand = p.second
        }
        if (j == null) return Answer("where", true)
        val f = features(v, j, cls, s, cand, tq, cfg.tau) + ("fallback" to if (v.events[j].kind != "put") 1.0 else 0.0)
        val loc = v.locLogp[j].indices.maxByOrNull { v.locLogp[j][it] }
        return Answer("where", false, v.events[j], v.events[j].t, loc, feats = f)
    }

    fun temporal(v: MemoryView, cls: Int, verb: String?, tq: Double, cfg: AnswerConfig): Answer {
        val kinds = kindForVerb[verb ?: "take"] ?: setOf("take")
        val mask = BooleanArray(v.events.size) { v.events[it].kind in kinds && v.events[it].t < tq }
        val s = scores(v, cls, mask, cfg)
        val (j, cand) = pick(s, v, tq, cfg)
        if (j == null) return Answer("temporal", true)
        return Answer("temporal", false, v.events[j], v.events[j].t, feats = features(v, j, cls, s, cand, tq, cfg.tau))
    }

    fun state(v: MemoryView, cls: Int, tq: Double, cfg: AnswerConfig, pOn: (Event, Int) -> Double): Answer {
        val mask = BooleanArray(v.events.size) { v.events[it].kind in setOf("touch", "take") && v.events[it].t < tq }
        val s = scores(v, cls, mask, cfg)
        val (j, cand) = pick(s, v, tq, cfg)
        if (j == null) return Answer("state", true)
        val p = pOn(v.events[j], cls)
        val f = features(v, j, cls, s, cand, tq, cfg.tau) + ("p_on" to p)
        return Answer("state", false, v.events[j], v.events[j].t, state = if (p >= 0.5) "on" else "off", feats = f)
    }
}
