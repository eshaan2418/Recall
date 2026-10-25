package ai.recall

import ai.recall.core.AnswerConfig
import ai.recall.core.BlendHead
import org.json.JSONArray
import org.json.JSONObject

/** Loads the JSON fixtures exported by android/tools/make_parity_fixtures.py from the Python reference. */
object Fixtures {
    fun text(name: String): String =
        requireNotNull(javaClass.classLoader!!.getResourceAsStream("parity/$name.json")) { "missing fixture $name (run scripts/android_fixtures.sh)" }
            .bufferedReader().readText()

    fun obj(name: String) = JSONObject(text(name))
    fun arr(name: String) = JSONArray(text(name))
}

fun JSONArray.floats() = FloatArray(length()) { getDouble(it).toFloat() }
fun JSONArray.doubles() = DoubleArray(length()) { getDouble(it) }
fun JSONArray.ints() = IntArray(length()) { getInt(it) }
fun JSONArray.matrix() = Array(length()) { getJSONArray(it).floats() }
fun JSONArray.objects() = List(length()) { getJSONObject(it) }
fun JSONObject.optDoubleOrNull(k: String): Double? = if (isNull(k)) null else getDouble(k)
fun JSONObject.optStringOrNull(k: String): String? = if (isNull(k)) null else getString(k)

fun headFrom(o: JSONObject): BlendHead {
    val bank = o.getJSONArray("bank").matrix()
    val p = o.optJSONObject("probe") ?: return BlendHead(bank, o.getDouble("scale").toFloat(), o.getDouble("alpha").toFloat())
    return BlendHead(bank, o.getDouble("scale").toFloat(), o.getDouble("alpha").toFloat(), p.getJSONArray("w").matrix(),
        p.getJSONArray("b").floats(), p.getJSONArray("classes").ints())
}

fun cfgFrom(o: JSONObject) = AnswerConfig(o.getString("mode"), o.getDouble("tau"), o.getDouble("recency_s"), o.getString("score"),
    o.getBoolean("fallback_take"), o.optString("loc_mode", "event"))
