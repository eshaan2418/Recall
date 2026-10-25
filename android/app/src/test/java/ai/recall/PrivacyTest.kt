package ai.recall

import ai.recall.privacy.BlazeFaceDecoder
import ai.recall.privacy.Box
import ai.recall.privacy.Pixelate
import ai.recall.privacy.Pixels
import ai.recall.privacy.Redactor
import ai.recall.privacy.ScreenDetector
import ai.recall.privacy.SensitiveRegionDetector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PrivacyTest {
    private fun noise(w: Int, h: Int, seed: Int = 0): Pixels {
        val r = java.util.Random(seed.toLong())
        return Pixels(IntArray(w * h) { (0xff shl 24) or r.nextInt(0xffffff) }, w, h)
    }

    @Test fun pixelateDestroysDetailInsideBoxOnly() {
        val p = noise(64, 48)
        val orig = p.px.copyOf()
        val box = Box(10, 10, 40, 34)
        Pixelate.apply(p, box, 8)
        // inside: each 8x8 cell is a single colour
        for (y in 10 until 34) for (x in 10 until 40) {
            val cx = 10 + (x - 10) / 8 * 8; val cy = 10 + (y - 10) / 8 * 8
            assertEquals(p.px[cy * 64 + cx], p.px[y * 64 + x])
        }
        // outside untouched
        for (y in 0 until 48) for (x in 0 until 64) if (x !in 10 until 40 || y !in 10 until 34) assertEquals(orig[y * 64 + x], p.px[y * 64 + x])
        val distinct = (10 until 34).flatMap { y -> (10 until 40).map { x -> p.px[y * 64 + x] } }.toSet().size
        assertTrue("mosaic leaves few distinct colours: $distinct", distinct <= 16)
    }

    @Test fun pixelateClipsOutOfBounds() {
        val p = noise(20, 20)
        Pixelate.apply(p, Box(-5, -5, 100, 100), 4) // must not throw
        Pixelate.apply(p, Box(30, 30, 40, 40), 4)   // fully outside: no-op
    }

    private class Fixed(override val name: String, val boxes: List<Box>) : SensitiveRegionDetector {
        override fun detect(img: Pixels) = boxes
    }

    private class Throwing : SensitiveRegionDetector {
        override val name = "face"
        override fun detect(img: Pixels): List<Box> = throw IllegalStateException("boom")
    }

    @Test fun redactorFailsClosed() {
        val img = noise(32, 32)
        assertFalse(Redactor(emptyList()).available)
        assertNull("no detectors → no thumbnail", Redactor(emptyList()).redact(img))
        assertNull("screen heuristic alone is not enough", Redactor(listOf(ScreenDetector())).redact(img))
        assertNull("detector failure → no thumbnail", Redactor(listOf(Throwing())).redact(img))
        val out = Redactor(listOf(Fixed("face", listOf(Box(0, 0, 16, 16))))).redact(img)
        assertNotNull(out)
        assertEquals("input not modified", noise(32, 32).px.toList(), img.px.toList())
        assertTrue("face region changed", (0 until 16).any { out!!.px[it] != img.px[it] })
    }

    @Test fun screenHeuristicFlagsBrightTextAndNotFlatOrDark() {
        val w = 160; val h = 90
        // left half: white page with dense black strokes; right half: flat grey
        val px = IntArray(w * h) { i ->
            val x = i % w; val y = i / w
            if (x < 80) (if ((x / 2 + y / 3) % 3 == 0) 0xff000000.toInt() else 0xffffffff.toInt()) else 0xff808080.toInt()
        }
        val boxes = ScreenDetector().detect(Pixels(px, w, h))
        assertTrue(boxes.isNotEmpty())
        assertTrue("only text cells flagged", boxes.all { it.r <= 80 })
        assertTrue(ScreenDetector().detect(Pixels(IntArray(w * h) { 0xff101010.toInt() }, w, h)).isEmpty())
    }

    @Test fun blazeFaceAnchorsAndDecode() {
        assertEquals(896, BlazeFaceDecoder.anchors.size)
        assertEquals(0.5f / 16, BlazeFaceDecoder.anchors[0][0], 1e-6f)
        assertEquals(0.5f / 8, BlazeFaceDecoder.anchors[512][0], 1e-6f)
        val reg = Array(896) { FloatArray(16) }
        val sc = Array(896) { floatArrayOf(-10f) }
        val k = 512 + 6 * (4 * 8 + 4) // anchor at cell (4,4) of the 8x8 map: centre (0.5625, 0.5625)
        sc[k][0] = 5f
        reg[k][2] = 32f; reg[k][3] = 32f // 0.25 of the image
        val boxes = BlazeFaceDecoder.decode(reg, sc, 400, 200, 0.4f, 0f)
        assertEquals(1, boxes.size)
        val b = boxes[0]
        assertEquals(225 - 50, b.l); assertTrue(b.t in 87..88)
        assertTrue(b.r in 275..276 && b.b in 137..138)
    }
}
