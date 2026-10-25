package ai.recall

import ai.recall.privacy.Box
import ai.recall.privacy.Pixels
import ai.recall.privacy.Redactor
import ai.recall.privacy.ScreenDetector
import ai.recall.privacy.SensitiveRegionDetector
import ai.recall.service.RecallRuntime
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class RedactorInstrumentedTest {
    private class FaceAt(val b: Box) : SensitiveRegionDetector {
        override val name = "face"
        override fun detect(img: Pixels) = listOf(b)
    }

    private fun checker(): Bitmap = Bitmap.createBitmap(320, 180, Bitmap.Config.ARGB_8888).apply {
        for (y in 0 until height) for (x in 0 until width) setPixel(x, y, if ((x / 2 + y / 2) % 2 == 0) Color.BLACK else Color.WHITE)
    }

    @Test fun thumbnailIsRedactedJpeg() {
        val jpeg = Redactor(listOf(FaceAt(Box(0, 0, 160, 180))), 320, 180).thumbnail(checker())
        assertNotNull(jpeg)
        val bm = BitmapFactory.decodeByteArray(jpeg, 0, jpeg!!.size)
        assertEquals(320, bm.width)
        fun contrast(x0: Int, x1: Int): Int {
            var s = 0
            for (y in 20 until 160) for (x in x0 until x1 - 1) s += kotlin.math.abs(Color.red(bm.getPixel(x, y)) - Color.red(bm.getPixel(x + 1, y)))
            return s
        }
        assertTrue("redacted half has much less fine detail", contrast(10, 150) * 4 < contrast(170, 310))
    }

    @Test fun noFaceModelNoThumbnail() {
        assertNull(Redactor(emptyList()).thumbnail(checker()))
        assertNull(Redactor(listOf(ScreenDetector())).thumbnail(checker()))
    }

    @Test fun faceModelLoadsIfInstalled() {
        val ctx = Synthetic.target
        val d = RecallRuntime.redactionDetectors(ctx, 0.4f)
        if (ai.recall.ml.LiteRtModel.hasAsset(ctx, RecallRuntime.FACE_ASSET)) {
            assertTrue("BlazeFace must load from assets", d.any { it.name == "face" })
            val blank = Bitmap.createBitmap(256, 144, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.GRAY) }
            assertTrue("no faces in a flat grey image", d.first { it.name == "face" }.detect(Pixels.of(blank)).isEmpty())
        } else {
            assertTrue(d.isEmpty())
        }
    }
}
