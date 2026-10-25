package ai.recall.core

/** Rule-based query parser; mirrors recall/qa/parse.py (parity-tested). */
data class ParsedQuery(val kind: String, val obj: String, val verb: String? = null)

object QueryParser {
    private val verbAliases = linkedMapOf(
        "take" to "take", "took" to "take", "pick up" to "take", "picked up" to "take", "grab" to "take",
        "grabbed" to "take", "use" to "take", "used" to "take",
        "put" to "put", "put down" to "put", "place" to "put", "placed" to "put", "leave" to "put", "left" to "put",
        "open" to "open", "opened" to "open", "close" to "close", "closed" to "close", "shut" to "close",
        "turn on" to "turn-on", "turned on" to "turn-on", "switch on" to "turn-on", "switched on" to "turn-on",
        "turn off" to "turn-off", "turned off" to "turn-off", "switch off" to "turn-off", "switched off" to "turn-off",
        "wash" to "wash", "washed" to "wash", "rinse" to "wash", "rinsed" to "wash",
    )
    private const val ART = "(?:the |my |a |an |some )?"

    private fun clean(s: String) = s.trim().trim('?', '.', '!', ' ').lowercase().replace(Regex("\\s+"), " ")

    fun parse(question: String): ParsedQuery {
        val q = clean(question)
        (Regex("^(?:did|have) i (?:turn|switch)(?:ed)? off $ART(.+)$").find(q)
            ?: Regex("^is $ART(.+?) (?:still )?(?:on|off)$").find(q))?.let { return ParsedQuery("state", it.groupValues[1]) }
        Regex("^when did i (?:last )?(turn on|turn off|switch on|switch off|pick up|put down|\\w+) $ART(.+?)(?: last)?$")
            .find(q)?.let { m ->
                val v = m.groupValues[1].trim()
                val verb = verbAliases[v] ?: verbAliases.entries.sortedByDescending { it.key.length }
                    .firstOrNull { v.startsWith(it.key) }?.value
                return ParsedQuery("temporal", m.groupValues[2], verb ?: "take")
            }
        val patterns = listOf(
            "^where (?:is|are) $ART(.+)$",
            "^where did i (?:leave|put|place|last see|see|park) $ART(.+)$",
            "^where did i (park)$",
            "^where'?s $ART(.+)$",
            "^have you seen $ART(.+)$",
            "^find $ART(.+)$",
        )
        for (p in patterns) Regex(p).find(q)?.let { return ParsedQuery("where", it.groupValues[1]) }
        return ParsedQuery("where", q.replace(Regex("^$ART"), ""))
    }
}
