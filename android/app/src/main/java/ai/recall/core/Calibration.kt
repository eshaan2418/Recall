package ai.recall.core

import kotlin.math.exp

/**
 * Logistic-on-features calibrator followed by an isotonic map (recall/qa/calibrate.py), exported from Python as
 * JSON: {"features": [...], "coef": [...], "intercept": x, "iso_x": [...], "iso_y": [...], "threshold": t}.
 */
class Calibrator(
    private val featureNames: List<String>,
    private val coef: DoubleArray,
    private val intercept: Double,
    private val isoX: DoubleArray,
    private val isoY: DoubleArray,
    val abstainBelow: Double,
) {
    fun confidence(a: Answer): Double {
        if (a.abstain) return 0.0
        val x = DoubleArray(featureNames.size) { i ->
            val n = featureNames[i]
            when {
                n.startsWith("kind=") -> if (a.kind == n.removePrefix("kind=")) 1.0 else 0.0
                n == "p_on_margin" -> kotlin.math.abs((a.feats["p_on"] ?: 0.5) - 0.5)
                n == "p_on" -> a.feats["p_on"] ?: 0.5
                else -> a.feats[n] ?: 0.0
            }
        }
        var z = intercept
        for (i in x.indices) z += coef[i] * x[i]
        val p = 1.0 / (1.0 + exp(-z))
        return iso(p)
    }

    private fun iso(p: Double): Double {
        if (isoX.isEmpty()) return p
        if (p <= isoX.first()) return isoY.first()
        if (p >= isoX.last()) return isoY.last()
        var k = isoX.indexOfFirst { it >= p }
        if (isoX[k] == p) return isoY[k]
        val x0 = isoX[k - 1]; val x1 = isoX[k]
        return isoY[k - 1] + (isoY[k] - isoY[k - 1]) * (p - x0) / (x1 - x0)
    }
}
