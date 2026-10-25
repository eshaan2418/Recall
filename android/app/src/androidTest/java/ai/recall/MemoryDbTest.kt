package ai.recall

import ai.recall.core.Event
import ai.recall.data.KeyManager
import ai.recall.data.MemoryDb
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MemoryDbTest {
    private val ctx = Synthetic.target
    private val name = "test-memory.db"
    private lateinit var db: MemoryDb

    @Before fun open() { ctx.deleteDatabase(name); db = MemoryDb(ctx, name) }
    @After fun close() { db.close(); ctx.deleteDatabase(name); KeyManager(ctx, "recall-key-$name").destroy() }

    private fun ev(t: Double, cls: Int = 0) = Event(t, "put", 1, FloatArray(8) { it * 0.1f }, FloatArray(8) { 1f - it * 0.1f }, FloatArray(8) { 0.5f },
        3, 0.9f, true, floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f), thumb = byteArrayOf(1, 2, 3))

    @Test fun fileIsEncryptedAndKeyIsKeystoreBound() {
        db.insert(ev(1.0), listOf(0 to 0.9f), listOf(1 to 0.8f), 0)
        db.close()
        val head = db.file.readBytes().copyOf(16)
        assertFalse("plaintext SQLite header must not appear", String(head, Charsets.US_ASCII).startsWith("SQLite format 3"))
        assertFalse("no plaintext rows", String(db.file.readBytes(), Charsets.ISO_8859_1).contains("put"))
        db = MemoryDb(ctx, name) // reopen with the Keystore-unwrapped key
        assertEquals(1, db.count())
        try {
            MemoryDb(ctx, name, ByteArray(32)).count(); throw AssertionError("wrong key must not open the DB")
        } catch (_: Exception) {}
    }

    @Test fun roundTripForgetRetentionAndPolicies() {
        val e = ev(1000.0)
        val id = db.insert(e, listOf(5 to 0.7f, 2 to 0.2f), listOf(1 to 0.8f), 3)
        db.insert(ev(10.0), listOf(7 to 0.9f), listOf(0 to 0.5f), 4)
        val back = db.events().last()
        assertEquals(1000.0, back.event.t, 0.0)
        assertArrayEquals(e.objEmb, back.event.objEmb, 1e-3f)
        assertEquals(listOf(5 to 0.7f, 2 to 0.2f), back.objTop)
        assertArrayEquals(byteArrayOf(1, 2, 3), db.thumb(id))
        assertEquals("beforeT filter (regression: varargs binding)", listOf(10.0), db.events(beforeT = 500.0).map { it.event.t })
        assertNull(db.thumb(id + 100))
        assertEquals(1, db.applyRetention(nowS = 999.0 + 86400, days = 1.0)) // cutoff t=999: drops t=10 only
        assertEquals(1, db.count())
        assertEquals(1, db.blockObject(5))
        assertTrue(5 in db.blockedObjects())
        assertEquals(0, db.count())
        db.upsertPlace(4, FloatArray(8) { 1f }, 1)
        db.insert(ev(20.0), listOf(1 to 0.9f), listOf(0 to 0.5f), 4)
        assertEquals(1, db.setPlaceBlocked(4, true))
        assertTrue(db.places().single().blocked)
        db.setSetting("k", "v"); assertEquals("v", db.setting("k")); assertNull(db.setting("missing"))
        db.insert(ev(30.0), listOf(1 to 0.9f), listOf(0 to 0.5f), 1)
        assertEquals(1, db.forget(db.events().single().event.id))
        db.wipe(); assertEquals(0, db.count()); assertTrue(db.bytes() > 0)
    }
}
