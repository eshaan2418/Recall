package ai.recall.privacy

import ai.recall.ml.LiteRtModel
import android.content.Context
import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.exp

/**
 * BlazeFace short-range (MediaPipe face_detection_short_range.tflite, Apache-2.0): input [1,128,128,3] float32 in
 * [-1,1] (NHWC, RGB); outputs regressors [1,896,16] and scores [1,896,1] (identified by shape, not index).
 * Anchors per the MediaPipe SSD config (strides 8,16,16,16; 2 anchors per stride entry; fixed size 1).
 * Low threshold + box padding: redaction prefers over-blurring to missing a face.
 */
class BlazeFaceDetector(ctx: Context, private val thresh: Float = 0.4f, private val pad: Float = 0.25f) :
    LiteRtModel(ctx, "face_detector.tflite"), SensitiveRegionDetector {
    override val name = "face"
    private val regIdx: Int
    private val scoreIdx: Int

    init {
        val shapes = (0 until interp.outputTensorCount).map { interp.getOutputTensor(it).shape().toList() }
        regIdx = shapes.indexOfFirst { it.last() == 16 }
        scoreIdx = shapes.indexOfFirst { it.last() == 1 && it.size == 3 }
        require(regIdx >= 0 && scoreIdx >= 0 && shapes[regIdx][1] == 896) {
            "face_detector.tflite: expected outputs [1,896,16] and [1,896,1], got $shapes"
        }
        val inShape = interp.getInputTensor(0).shape().toList()
        require(inShape == listOf(1, 128, 128, 3)) { "face_detector.tflite: expected input [1,128,128,3], got $inShape" }
    }

    override fun detect(img: Pixels): List<Box> {
        val s = Bitmap.createScaledBitmap(img.toBitmap(), 128, 128, true)
        val px = IntArray(128 * 128).also { s.getPixels(it, 0, 128, 0, 0, 128, 128) }
        val buf = ByteBuffer.allocateDirect(4 * 3 * 128 * 128).order(ByteOrder.nativeOrder())
        for (p in px) for (sh in intArrayOf(16, 8, 0)) buf.putFloat(((p shr sh) and 0xff) / 127.5f - 1f)
        buf.rewind()
        val reg = Array(1) { Array(896) { FloatArray(16) } }
        val sc = Array(1) { Array(896) { FloatArray(1) } }
        interp.runForMultipleInputsOutputs(arrayOf(buf), mapOf(regIdx to reg, scoreIdx to sc))
        return BlazeFaceDecoder.decode(reg[0], sc[0], img.w, img.h, thresh, pad)
    }
}

object BlazeFaceDecoder {
    val anchors: List<FloatArray> = buildList {
        val strides = intArrayOf(8, 16, 16, 16)
        var i = 0
        while (i < strides.size) {
            var j = i; var n = 0
            while (j < strides.size && strides[j] == strides[i]) { n += 2; j++ }
            val fm = 128 / strides[i]
            for (y in 0 until fm) for (x in 0 until fm) repeat(n) { add(floatArrayOf((x + 0.5f) / fm, (y + 0.5f) / fm)) }
            i = j
        }
    }

    fun decode(reg: Array<FloatArray>, sc: Array<FloatArray>, w: Int, h: Int, thresh: Float, pad: Float): List<Box> {
        val out = mutableListOf<Box>()
        for (k in reg.indices) {
            val p = 1f / (1f + exp(-sc[k][0].coerceIn(-100f, 100f)))
            if (p < thresh) continue
            val a = anchors[k]
            val cx = a[0] + reg[k][0] / 128f; val cy = a[1] + reg[k][1] / 128f
            val bw = reg[k][2] / 128f * (1 + pad); val bh = reg[k][3] / 128f * (1 + pad)
            out += Box(((cx - bw / 2) * w).toInt(), ((cy - bh / 2) * h).toInt(), ((cx + bw / 2) * w).toInt() + 1, ((cy + bh / 2) * h).toInt() + 1)
        }
        return out
    }
}

/**
 * Screen / document heuristic: grid cells that are bright with dense high-contrast strokes (text) are flagged.
 * Deliberately conservative (over-redacts bright printed packaging).
 */
class ScreenDetector(private val gridW: Int = 16, private val gridH: Int = 9, private val minMean: Float = 150f,
                     private val minEdge: Float = 0.10f) : SensitiveRegionDetector {
    override val name = "screen"

    override fun detect(img: Pixels): List<Box> {
        val w = img.w; val h = img.h; val px = img.px
        fun lum(i: Int): Float { val p = px[i]; return 0.299f * ((p shr 16) and 0xff) + 0.587f * ((p shr 8) and 0xff) + 0.114f * (p and 0xff) }
        val cw = w / gridW; val ch = h / gridH
        val out = mutableListOf<Box>()
        if (cw < 3 || ch < 3) return out
        for (gy in 0 until gridH) for (gx in 0 until gridW) {
            var sum = 0f; var edges = 0; var n = 0
            for (y in gy * ch until (gy + 1) * ch - 1) for (x in gx * cw until (gx + 1) * cw - 1) {
                val l = lum(y * w + x)
                sum += l; n++
                if (abs(l - lum(y * w + x + 1)) > 40f || abs(l - lum((y + 1) * w + x)) > 40f) edges++
            }
            if (sum / n >= minMean && edges.toFloat() / n >= minEdge) out += Box(gx * cw, gy * ch, (gx + 1) * cw, (gy + 1) * ch)
        }
        return out
    }
}
