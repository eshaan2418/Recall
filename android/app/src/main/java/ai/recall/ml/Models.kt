package ai.recall.ml

import ai.recall.core.HandObs
import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/*
 * Asset contract (see android/README.md for the full table). All models: float32 input/output tensors
 * (fp32 or dynamic-range-int8 weights; fully-int8 I/O models are rejected at load with a clear message).
 *
 *  gate.tflite        in0 x   [1,8,9,36,64]  (frame-0.5, 4·(f-f[t-1]), 4·(f-f[t-2]); RGB 0..1; CTX=8 oldest→newest)
 *                     in1 imu [1,8,7]        (optional input; zeros + flag 0 when no IMU)   out [1] or [1,1] logit
 *  embedder.tflite    in  [1,3,S,S] NCHW (or [1,S,S,3] NHWC) RGB, (px/255 - mean)/std, default S=256 mean 0 std 1;
 *                     shortest-side resize + centre crop.   out [1,D] (re-L2-normalised on device)
 *  hoi_student.tflite in  [1,3,H,W] NCHW (or NHWC) RGB 0..1, default H=144 W=256 (full frame, squashed)
 *                     out [1,2,14] per side (0 left, 1 right): present_logit, 5 state logits
 *                     (none, self, other, portable, stationary), hand box l,t,r,b, object box l,t,r,b (0..1)
 *  face_detector.tflite  MediaPipe BlazeFace short-range (see privacy/Detectors.kt)
 */

/** Always-on gate: last CTX low-res frames (+IMU) → P(interaction now). */
interface Gate {
    fun score(frame36x64: Bitmap, imu: FloatArray): Float
}

/** Image → L2-normalised embedding (MobileCLIP2-S2 image tower export). */
interface Embedder {
    val dim: Int
    fun embed(img: Bitmap): FloatArray
}

/** Hand–object student: frame → per side (left, right) hand presence, contact state, object box. */
interface HandObjectDetector {
    fun detect(frame: Bitmap): Array<HandObs>
}

enum class Accel { CPU, GPU }

class ModelContractException(msg: String) : IllegalStateException(msg)

open class LiteRtModel(ctx: Context, val asset: String, accel: Accel = Accel.CPU, threads: Int = 2) : AutoCloseable {
    private var gpu: GpuDelegate? = null
    protected val interp: Interpreter

    init {
        val buf = loadAsset(ctx, asset)
        var ip: Interpreter? = null
        if (accel == Accel.GPU) {
            ip = try {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    val g = GpuDelegate().also { gpu = it }
                    Interpreter(buf, Interpreter.Options().setNumThreads(threads).addDelegate(g))
                } else null
            } catch (_: Throwable) { // GPU delegate unavailable (e.g. emulator) or op unsupported → CPU
                gpu?.close(); gpu = null; null
            }
        }
        interp = ip ?: Interpreter(buf, Interpreter.Options().setNumThreads(threads).setUseXNNPACK(!isEmulator))
        for (i in 0 until interp.inputTensorCount) requireFloat("input $i", interp.getInputTensor(i).dataType())
        for (i in 0 until interp.outputTensorCount) requireFloat("output $i", interp.getOutputTensor(i).dataType())
    }

    val usingGpu get() = gpu != null

    private fun requireFloat(what: String, t: DataType) {
        if (t != DataType.FLOAT32) throw ModelContractException("$asset: $what is $t; the app expects float32 I/O (ship the fp32 or int8dyn export)")
    }

    protected fun inShape(i: Int = 0): IntArray = interp.getInputTensor(i).shape()

    override fun close() {
        interp.close()
        gpu?.close()
    }

    companion object {
        /**
         * XNNPACK's CPU-feature probe dies with SIGILL on the arm64 emulator under Apple-Silicon hypervisors, so the
         * plain CPU kernels are used there. Real phones keep XNNPACK (the fast path).
         */
        val isEmulator: Boolean by lazy {
            android.os.Build.HARDWARE in setOf("ranchu", "goldfish") || android.os.Build.FINGERPRINT.contains("generic") ||
                android.os.Build.PRODUCT.startsWith("sdk_gphone")
        }

        fun loadAsset(ctx: Context, name: String): MappedByteBuffer {
            val fd = ctx.assets.openFd(name)
            FileInputStream(fd.fileDescriptor).use { s ->
                return s.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }

        /** True if the asset exists and is non-empty (works for compressed and uncompressed assets). */
        fun hasAsset(ctx: Context, name: String): Boolean = try {
            ctx.assets.open(name).use { it.read() != -1 }
        } catch (_: Exception) { false }

        /** Image layout of a 4-D image input: NCHW if dim 1 == 3, NHWC if dim 3 == 3. Returns (nchw, h, w). */
        fun imageLayout(shape: IntArray, asset: String): Triple<Boolean, Int, Int> = when {
            shape.size == 4 && shape[0] == 1 && shape[1] == 3 -> Triple(true, shape[2], shape[3])
            shape.size == 4 && shape[0] == 1 && shape[3] == 3 -> Triple(false, shape[1], shape[2])
            else -> throw ModelContractException("$asset: expected image input [1,3,H,W] or [1,H,W,3], got ${shape.toList()}")
        }

        /** Float image buffer, values (pixel/255 - mean)/std per channel, NCHW or NHWC. */
        fun imageBuffer(bm: Bitmap, w: Int, h: Int, mean: FloatArray, std: FloatArray, nchw: Boolean = true): ByteBuffer {
            val s = if (bm.width == w && bm.height == h) bm else Bitmap.createScaledBitmap(bm, w, h, true)
            val px = IntArray(w * h)
            s.getPixels(px, 0, w, 0, 0, w, h)
            return packPixels(px, mean, std, nchw)
        }

        fun packPixels(px: IntArray, mean: FloatArray, std: FloatArray, nchw: Boolean): ByteBuffer {
            val buf = ByteBuffer.allocateDirect(4 * 3 * px.size).order(ByteOrder.nativeOrder())
            fun ch(p: Int, c: Int) = when (c) { 0 -> (p shr 16) and 0xff; 1 -> (p shr 8) and 0xff; else -> p and 0xff }
            if (nchw) for (c in 0 until 3) for (p in px) buf.putFloat((ch(p, c) / 255f - mean[c]) / std[c])
            else for (p in px) for (c in 0 until 3) buf.putFloat((ch(p, c) / 255f - mean[c]) / std[c])
            buf.rewind()
            return buf
        }

        /** Resize shortest side to [size] and centre-crop (open_clip eval transform). */
        fun centerCrop(bm: Bitmap, size: Int): Bitmap {
            val sc = size.toFloat() / minOf(bm.width, bm.height)
            val r = Bitmap.createScaledBitmap(bm, maxOf(size, Math.round(bm.width * sc)), maxOf(size, Math.round(bm.height * sc)), true)
            return Bitmap.createBitmap(r, (r.width - size) / 2, (r.height - size) / 2, size, size)
        }
    }
}

class LiteRtEmbedder(ctx: Context, asset: String = "embedder.tflite", accel: Accel = Accel.GPU,
                     private val mean: FloatArray = floatArrayOf(0f, 0f, 0f),
                     private val std: FloatArray = floatArrayOf(1f, 1f, 1f)) : LiteRtModel(ctx, asset, accel), Embedder {
    private val layout = imageLayout(inShape(), asset)
    private val size = layout.second
    override val dim: Int = interp.getOutputTensor(0).shape().last()

    override fun embed(img: Bitmap): FloatArray {
        val out = Array(1) { FloatArray(dim) }
        interp.run(imageBuffer(centerCrop(img, size), size, size, mean, std, layout.first), out)
        return out[0]
    }
}

/**
 * Gate input = 9 channels (frame-0.5, 4·diff(t-1), 4·diff(t-2)) over the last CTX steps; this class keeps the
 * ring buffer of past frames itself (mirrors recall/trigger/gate.py::frame_inputs + score_video left padding).
 */
class LiteRtGate(ctx: Context, asset: String = "gate.tflite") : LiteRtModel(ctx, asset, Accel.CPU, 1), Gate {
    private val xIdx: Int
    private val imuIdx: Int
    private val ctxLen: Int
    private val imuDim: Int
    private val frames = ArrayDeque<FloatArray>()
    private val inputs = ArrayDeque<FloatArray>()
    private val imus = ArrayDeque<FloatArray>()

    init {
        val shapes = (0 until interp.inputTensorCount).map { inShape(it) }
        xIdx = shapes.indexOfFirst { it.size == 5 }
        imuIdx = shapes.indexOfFirst { it.size == 3 }
        if (xIdx < 0 || shapes[xIdx].toList().drop(2) != listOf(9, 36, 64)) {
            throw ModelContractException("$asset: expected x input [1,CTX,9,36,64], got ${shapes.map { it.toList() }}")
        }
        ctxLen = shapes[xIdx][1]
        imuDim = if (imuIdx >= 0) shapes[imuIdx][2] else 0
        val outN = interp.getOutputTensor(0).numElements()
        if (outN != 1) throw ModelContractException("$asset: expected a single logit output, got ${interp.getOutputTensor(0).shape().toList()}")
    }

    override fun score(frame36x64: Bitmap, imu: FloatArray): Float {
        val f = toFloat(frame36x64)
        val p1 = frames.lastOrNull() ?: f
        val p2 = if (frames.size >= 2) frames[frames.size - 2] else p1
        val hw = 36 * 64
        val x = FloatArray(9 * hw)
        for (c in 0 until 3) for (i in 0 until hw) {
            val v = f[c * hw + i]
            x[c * hw + i] = v - 0.5f
            x[(3 + c) * hw + i] = (v - p1[c * hw + i]) * 4f
            x[(6 + c) * hw + i] = (v - p2[c * hw + i]) * 4f
        }
        frames.addLast(f); if (frames.size > 2) frames.removeFirst()
        inputs.addLast(x); imus.addLast(FloatArray(imuDim) { imu.getOrElse(it) { 0f } })
        while (inputs.size < ctxLen) { inputs.addFirst(inputs.first()); imus.addFirst(imus.first()) }
        while (inputs.size > ctxLen) { inputs.removeFirst(); imus.removeFirst() }
        val xb = ByteBuffer.allocateDirect(4 * ctxLen * x.size).order(ByteOrder.nativeOrder())
        for (a in inputs) for (v in a) xb.putFloat(v)
        xb.rewind()
        val ins = arrayOfNulls<Any>(interp.inputTensorCount)
        ins[xIdx] = xb
        if (imuIdx >= 0) {
            val ib = ByteBuffer.allocateDirect(4 * ctxLen * imuDim).order(ByteOrder.nativeOrder())
            for (a in imus) for (v in a) ib.putFloat(v)
            ib.rewind()
            ins[imuIdx] = ib
        }
        val out = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
        interp.runForMultipleInputsOutputs(ins, mapOf(0 to out))
        out.rewind()
        return (1.0 / (1.0 + Math.exp(-out.float.toDouble()))).toFloat()
    }

    private fun toFloat(bm: Bitmap): FloatArray {
        val s = if (bm.width == 64 && bm.height == 36) bm else Bitmap.createScaledBitmap(bm, 64, 36, true)
        val px = IntArray(64 * 36)
        s.getPixels(px, 0, 64, 0, 0, 64, 36)
        val hw = 64 * 36
        val out = FloatArray(3 * hw)
        for (i in 0 until hw) {
            out[i] = ((px[i] shr 16) and 0xff) / 255f
            out[hw + i] = ((px[i] shr 8) and 0xff) / 255f
            out[2 * hw + i] = (px[i] and 0xff) / 255f
        }
        return out
    }
}

/** Student HOI detector export; see the contract at the top of this file. */
class LiteRtHandObject(ctx: Context, asset: String = "hoi_student.tflite", private val presentThresh: Float = 0.5f) :
    LiteRtModel(ctx, asset, Accel.GPU), HandObjectDetector {
    private val layout = imageLayout(inShape(), asset)

    init {
        val o = interp.getOutputTensor(0).shape().toList()
        if (o != listOf(1, 2, 14)) throw ModelContractException("$asset: expected output [1,2,14], got $o")
    }

    override fun detect(frame: Bitmap): Array<HandObs> {
        val out = Array(1) { Array(2) { FloatArray(14) } }
        interp.run(imageBuffer(frame, layout.third, layout.second, floatArrayOf(0f, 0f, 0f), floatArrayOf(1f, 1f, 1f), layout.first), out)
        return Array(2) { s -> decodeSide(out[0][s], presentThresh) }
    }

    companion object {
        fun decodeSide(o: FloatArray, presentThresh: Float): HandObs {
            val p = (1.0 / (1.0 + Math.exp(-o[0].toDouble()))).toFloat()
            val state = (1..5).maxByOrNull { o[it] }!! - 1
            return HandObs(p >= presentThresh, p, state, floatArrayOf(o[10], o[11], o[12], o[13]))
        }
    }
}
