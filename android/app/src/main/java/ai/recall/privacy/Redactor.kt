package ai.recall.privacy

import android.graphics.Bitmap
import java.io.ByteArrayOutputStream

/** Integer pixel rectangle [l, r) × [t, b) (pure Kotlin; android.graphics.Rect is a stub in JVM unit tests). */
data class Box(val l: Int, val t: Int, val r: Int, val b: Int) {
    fun clip(w: Int, h: Int) = Box(l.coerceIn(0, w), t.coerceIn(0, h), r.coerceIn(0, w), b.coerceIn(0, h))
    val isEmpty get() = r <= l || b <= t
}

/** ARGB pixel buffer view; [px] is row-major, length w*h. */
class Pixels(val px: IntArray, val w: Int, val h: Int) {
    companion object {
        fun of(bm: Bitmap): Pixels = Pixels(IntArray(bm.width * bm.height).also { bm.getPixels(it, 0, bm.width, 0, 0, bm.width, bm.height) },
            bm.width, bm.height)
    }

    fun toBitmap(): Bitmap = Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
}

interface SensitiveRegionDetector {
    val name: String

    /** Boxes in pixel coordinates of [img]. */
    fun detect(img: Pixels): List<Box>
}

object Pixelate {
    /** In-place mosaic of [r] with [block]-px cells (each cell takes the colour of its centre pixel). */
    fun apply(p: Pixels, r: Box, block: Int = 12) {
        val rr = r.clip(p.w, p.h)
        if (rr.isEmpty) return
        var y = rr.t
        while (y < rr.b) {
            var x = rr.l
            while (x < rr.r) {
                val c = p.px[minOf(y + block / 2, rr.b - 1) * p.w + minOf(x + block / 2, rr.r - 1)]
                for (yy in y until minOf(y + block, rr.b)) for (xx in x until minOf(x + block, rr.r)) p.px[yy * p.w + xx] = c
                x += block
            }
            y += block
        }
    }
}

/**
 * Privacy filter applied before anything is stored: every detected face / screen / document region is
 * pixelated in the thumbnail. Fail closed: if the face detector is unavailable (no model) or any detector throws,
 * the thumbnail is NOT stored at all.
 */
class Redactor(private val detectors: List<SensitiveRegionDetector>, private val thumbW: Int = 160, private val thumbH: Int = 90,
               private val requireFaceDetector: Boolean = true) {
    val available get() = detectors.isNotEmpty() && (!requireFaceDetector || detectors.any { it.name == "face" })

    /** Redacts in place on a copy and returns the redacted pixels, or null if redaction is unavailable/failed. */
    fun redact(img: Pixels): Pixels? {
        if (!available) return null
        val p = Pixels(img.px.copyOf(), img.w, img.h)
        return try {
            for (d in detectors) for (r in d.detect(p)) Pixelate.apply(p, r, blockFor(r))
            p
        } catch (_: Exception) {
            null
        }
    }

    private fun blockFor(r: Box) = maxOf(6, minOf(r.r - r.l, r.b - r.t) / 6)

    /** Returns a redacted JPEG thumbnail of [crop], or null if redaction is unavailable. */
    fun thumbnail(crop: Bitmap): ByteArray? {
        val red = redact(Pixels.of(crop)) ?: return null
        val t = Bitmap.createScaledBitmap(red.toBitmap(), thumbW, thumbH, true)
        return ByteArrayOutputStream().use { t.compress(Bitmap.CompressFormat.JPEG, 70, it); it.toByteArray() }
    }
}
