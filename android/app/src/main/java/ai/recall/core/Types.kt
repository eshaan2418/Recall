package ai.recall.core

const val PORTABLE = 3
const val STATIONARY = 4

/** One hand observation from the hand–object detector (normalised box). */
data class HandObs(val present: Boolean, val score: Float, val state: Int, val objBox: FloatArray?)

/** Everything heavy perception produced for one processed step, for one hand side. */
data class SideObs(val hand: HandObs, val objEmb: FloatArray?, val relCtxEmb: FloatArray?)

data class Event(
    val t: Double,
    val kind: String, // put | take | touch
    val side: Int,
    val objEmb: FloatArray,
    val locEmb: FloatArray,
    val sceneEmb: FloatArray,
    val nObs: Int = 1,
    val handScore: Float = 0f,
    val releaseObserved: Boolean = false,
    val box: FloatArray = floatArrayOf(0f, 0f, 1f, 1f),
    var id: Long = -1,
    var thumb: ByteArray? = null,
)

fun unit(x: FloatArray): FloatArray {
    var n = 0.0
    for (v in x) n += v * v
    val s = (1.0 / maxOf(Math.sqrt(n), 1e-8)).toFloat()
    return FloatArray(x.size) { x[it] * s }
}

fun dot(a: FloatArray, b: FloatArray): Float {
    var s = 0f
    for (i in a.indices) s += a[i] * b[i]
    return s
}

fun mean(xs: List<FloatArray>): FloatArray {
    val d = xs[0].size
    val out = FloatArray(d)
    for (x in xs) for (i in 0 until d) out[i] += x[i]
    for (i in 0 until d) out[i] /= xs.size
    return out
}

/** IEEE-754 half-precision conversion (pure Kotlin so it is JVM-testable; android.util.Half is a stub there). */
object Fp16 {
    fun toHalf(f: Float): Short {
        val bits = java.lang.Float.floatToRawIntBits(f)
        val sign = (bits ushr 16) and 0x8000
        var exp = ((bits ushr 23) and 0xff) - 127 + 15
        var mant = bits and 0x7fffff
        if (((bits ushr 23) and 0xff) == 0xff) return (sign or 0x7c00 or (if (mant != 0) 0x200 else 0)).toShort() // inf/nan
        if (exp >= 0x1f) return (sign or 0x7c00).toShort() // overflow -> inf
        if (exp <= 0) {
            if (exp < -10) return sign.toShort() // underflow -> signed zero
            mant = mant or 0x800000
            val shift = 14 - exp
            var h = mant ushr shift
            val rem = mant and ((1 shl shift) - 1)
            val half = 1 shl (shift - 1)
            if (rem > half || (rem == half && (h and 1) == 1)) h++
            return (sign or h).toShort()
        }
        var h = (exp shl 10) or (mant ushr 13)
        val rem = mant and 0x1fff
        if (rem > 0x1000 || (rem == 0x1000 && (h and 1) == 1)) h++ // round half to even (may carry into exponent)
        return (sign or h).toShort()
    }

    fun toFloat(h: Short): Float {
        val x = h.toInt() and 0xffff
        val sign = (x and 0x8000) shl 16
        val exp = (x ushr 10) and 0x1f
        val mant = x and 0x3ff
        val bits = when {
            exp == 0 && mant == 0 -> sign
            exp == 0 -> { // subnormal
                var e = -1; var m = mant
                while (m and 0x400 == 0) { m = m shl 1; e++ }
                sign or ((127 - 15 - e) shl 23) or ((m and 0x3ff) shl 13)
            }
            exp == 0x1f -> sign or 0x7f800000 or (mant shl 13)
            else -> sign or ((exp - 15 + 127) shl 23) or (mant shl 13)
        }
        return java.lang.Float.intBitsToFloat(bits)
    }

    fun encode(x: FloatArray): ByteArray {
        val b = java.nio.ByteBuffer.allocate(2 * x.size).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (v in x) b.putShort(toHalf(v))
        return b.array()
    }

    fun decode(b: ByteArray): FloatArray {
        val s = java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        return FloatArray(b.size / 2) { toFloat(s.getShort()) }
    }
}
