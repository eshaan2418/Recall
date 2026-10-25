package ai.recall.core

/** Grounded answer text (recall/qa/generate.py): every slot value comes from the stored event. */
data class Rendered(val text: String, val abstained: Boolean, val evidenceT: Double?, val confidence: Double?)

object Renderer {
    fun clock(tEpochS: Double, zoneOffsetS: Int = 0): String {
        val s = (((tEpochS + zoneOffsetS).toLong()) % 86400 + 86400) % 86400 // Python: int(start + t) % 86400
        val h = (s / 3600).toInt()
        val m = ((s % 3600) / 60).toInt()
        val ampm = if (h < 12) "am" else "pm"
        val h12 = if (h % 12 == 0) 12 else h % 12
        return "%d:%02d %s".format(h12, m, ampm)
    }

    fun render(a: Answer, obj: String, locName: String?, conf: Double?, abstainBelow: Double, zoneOffsetS: Int = 0): Rendered {
        if (a.abstain || conf == null || conf < abstainBelow) {
            return Rendered("I don't know — I haven't seen the $obj in my memory.", true, null, conf)
        }
        val pct = Math.rint(100 * conf).toInt() // Python round(): half to even
        val tm = clock(a.t ?: 0.0, zoneOffsetS)
        val txt = when (a.kind) {
            "where" -> if (conf < 0.5) "I'm not sure. My best guess is the $locName, last seen at $tm — but only $pct% confident."
            else "Probably the $locName — you put the $obj there at $tm. $pct% confident."
            "temporal" -> "Around $tm. $pct% confident."
            else -> "The last time I saw you at the $obj was $tm; it looked ${a.state ?: "unknown"} after that. $pct% confident."
        }
        return Rendered(txt, false, a.t, conf)
    }
}
