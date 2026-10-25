package ai.recall.data

import ai.recall.core.Event
import ai.recall.core.Fp16
import android.content.ContentValues
import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import net.zetetic.database.sqlcipher.SQLiteDatabase
import java.io.File
import java.security.KeyStore
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Database passphrase: 32 random bytes, stored only encrypted (AES-256-GCM) by a non-exportable
 * Android Keystore key. Deleting the Keystore entry cryptographically erases the memory.
 */
class KeyManager(private val ctx: Context, private val alias: String = "recall-db-key") {
    private val wrapped = File(ctx.noBackupFilesDir, "$alias.enc")

    private fun keystoreKey(): SecretKey {
        val ks = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (ks.getKey(alias, null) as? SecretKey)?.let { return it }
        val kg = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        kg.init(KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setKeySize(256).build())
        return kg.generateKey()
    }

    @Synchronized fun passphrase(): ByteArray {
        val key = keystoreKey()
        if (wrapped.exists()) {
            val b = wrapped.readBytes()
            val c = Cipher.getInstance("AES/GCM/NoPadding")
            c.init(Cipher.DECRYPT_MODE, key, GCMParameterSpec(128, b.copyOfRange(0, 12)))
            return c.doFinal(b.copyOfRange(12, b.size))
        }
        val pass = ByteArray(32).also { SecureRandom().nextBytes(it) }
        val c = Cipher.getInstance("AES/GCM/NoPadding")
        c.init(Cipher.ENCRYPT_MODE, key)
        wrapped.writeBytes(c.iv + c.doFinal(pass))
        return pass
    }

    fun destroy() {
        KeyStore.getInstance("AndroidKeyStore").apply { load(null) }.deleteEntry(alias)
        wrapped.delete()
    }
}

data class StoredEvent(val event: Event, val objTop: List<Pair<Int, Float>>, val locTop: List<Pair<Int, Float>>, val placeId: Int)

data class Place(val id: Int, val centroid: FloatArray, val n: Int, val name: String?, val blocked: Boolean)

/**
 * Encrypted (SQLCipher) episodic memory.
 * NB: SQLCipher's rawQuery(String, Object...) is varargs — pass bind values directly, never as an Array<Any>
 * (Kotlin would bind the array itself as a single argument and the WHERE clause silently misbehaves). Thread-safe: SQLiteDatabase serialises access internally. */
class MemoryDb(ctx: Context, val name: String = "memory.db", passphrase: ByteArray? = null) : AutoCloseable {
    private val db: SQLiteDatabase
    val file: File = ctx.getDatabasePath(name)

    init {
        System.loadLibrary("sqlcipher")
        file.parentFile?.mkdirs()
        val pass = passphrase ?: KeyManager(ctx, "recall-key-$name").passphrase()
        db = SQLiteDatabase.openOrCreateDatabase(file, pass, null, null, null)
        db.execSQL("""CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, t REAL, kind TEXT, side INTEGER,
            place_id INTEGER, n_obs INTEGER, hand_score REAL, release_obs INTEGER, obj_cls INTEGER, obj_top TEXT, loc_top TEXT,
            box TEXT, obj_emb BLOB, loc_emb BLOB, scene_emb BLOB, thumb BLOB)""")
        db.execSQL("CREATE INDEX IF NOT EXISTS ev_t ON events(t)")
        db.execSQL("CREATE TABLE IF NOT EXISTS places (id INTEGER PRIMARY KEY, centroid BLOB, n INTEGER, name TEXT, blocked INTEGER DEFAULT 0)")
        db.execSQL("CREATE TABLE IF NOT EXISTS blocked_objects (cls INTEGER PRIMARY KEY)")
        db.execSQL("CREATE TABLE IF NOT EXISTS settings (k TEXT PRIMARY KEY, v TEXT)")
    }

    fun insert(e: Event, objTop: List<Pair<Int, Float>>, locTop: List<Pair<Int, Float>>, placeId: Int): Long {
        val v = ContentValues().apply {
            put("t", e.t); put("kind", e.kind); put("side", e.side); put("place_id", placeId); put("n_obs", e.nObs)
            put("hand_score", e.handScore); put("release_obs", if (e.releaseObserved) 1 else 0)
            put("obj_cls", objTop.firstOrNull()?.first ?: -1)
            put("obj_top", objTop.joinToString(";") { "${it.first}:${it.second}" })
            put("loc_top", locTop.joinToString(";") { "${it.first}:${it.second}" })
            put("box", e.box.joinToString(",")); put("obj_emb", Fp16.encode(e.objEmb)); put("loc_emb", Fp16.encode(e.locEmb))
            put("scene_emb", Fp16.encode(e.sceneEmb)); e.thumb?.let { put("thumb", it) }
        }
        return db.insert("events", null, v).also { e.id = it }
    }

    fun events(beforeT: Double = Double.MAX_VALUE): List<StoredEvent> {
        val out = mutableListOf<StoredEvent>()
        db.rawQuery("SELECT id,t,kind,side,place_id,n_obs,hand_score,release_obs,obj_top,loc_top,box,obj_emb,loc_emb,scene_emb FROM events WHERE t<? ORDER BY t",
            beforeT).use { c ->
            while (c.moveToNext()) {
                val e = Event(c.getDouble(1), c.getString(2), c.getInt(3), Fp16.decode(c.getBlob(11)), Fp16.decode(c.getBlob(12)),
                    Fp16.decode(c.getBlob(13)), c.getInt(5), c.getFloat(6), c.getInt(7) == 1,
                    c.getString(10).split(",").map { it.toFloat() }.toFloatArray(), c.getLong(0))
                out += StoredEvent(e, top(c.getString(8)), top(c.getString(9)), c.getInt(4))
            }
        }
        return out
    }

    fun thumb(id: Long): ByteArray? = db.rawQuery("SELECT thumb FROM events WHERE id=?", id).use {
        if (it.moveToFirst() && !it.isNull(0)) it.getBlob(0) else null
    }

    /** "Forget this": hard delete. */
    fun forget(id: Long): Int = db.delete("events", "id=?", arrayOf<Any>(id))

    /** Retention policy: delete events older than [days] relative to [nowS]. */
    fun applyRetention(nowS: Double, days: Double): Int = db.delete("events", "t<?", arrayOf<Any>(nowS - days * 86400))

    /** "Never track this object": future events with this top class are dropped, existing ones deleted. */
    fun blockObject(cls: Int): Int {
        db.execSQL("INSERT OR IGNORE INTO blocked_objects VALUES (?)", arrayOf<Any>(cls))
        return db.delete("events", "obj_cls=?", arrayOf<Any>(cls))
    }

    fun unblockObject(cls: Int) = db.delete("blocked_objects", "cls=?", arrayOf<Any>(cls))

    fun blockedObjects(): Set<Int> = db.rawQuery("SELECT cls FROM blocked_objects", null).use { c ->
        buildSet { while (c.moveToNext()) add(c.getInt(0)) }
    }

    /** "Never record here": blocks the place and deletes what was already stored there. */
    fun setPlaceBlocked(id: Int, blocked: Boolean): Int {
        db.execSQL("UPDATE places SET blocked=? WHERE id=?", arrayOf<Any>(if (blocked) 1 else 0, id))
        return if (blocked) db.delete("events", "place_id=?", arrayOf<Any>(id)) else 0
    }

    fun renamePlace(id: Int, name: String) = db.execSQL("UPDATE places SET name=? WHERE id=?", arrayOf<Any>(name, id))

    fun upsertPlace(id: Int, centroid: FloatArray, n: Int) = db.execSQL(
        "INSERT INTO places(id,centroid,n) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET centroid=excluded.centroid, n=excluded.n",
        arrayOf<Any>(id, Fp16.encode(centroid), n))

    fun places(): List<Place> = db.rawQuery("SELECT id, centroid, n, name, blocked FROM places ORDER BY id", null).use { c ->
        buildList {
            while (c.moveToNext()) add(Place(c.getInt(0), Fp16.decode(c.getBlob(1)), c.getInt(2), if (c.isNull(3)) null else c.getString(3), c.getInt(4) == 1))
        }
    }

    fun count(): Int = db.rawQuery("SELECT COUNT(*) FROM events", null).use { it.moveToFirst(); it.getInt(0) }

    fun setting(k: String): String? = db.rawQuery("SELECT v FROM settings WHERE k=?", k).use {
        if (it.moveToFirst()) it.getString(0) else null
    }

    fun setSetting(k: String, v: String) = db.execSQL("INSERT OR REPLACE INTO settings(k,v) VALUES(?,?)", arrayOf<Any>(k, v))

    /** Bytes on disk of the encrypted DB (+ WAL/journal if present). */
    fun bytes(): Long = listOf(file, File(file.path + "-wal"), File(file.path + "-journal")).sumOf { if (it.exists()) it.length() else 0L }

    fun wipe() {
        db.delete("events", null, null); db.delete("places", null, null)
        db.execSQL("VACUUM")
    }

    override fun close() = db.close()

    companion object {
        private fun top(s: String): List<Pair<Int, Float>> =
            if (s.isEmpty()) emptyList() else s.split(";").map { it.split(":").let { p -> p[0].toInt() to p[1].toFloat() } }
    }
}
