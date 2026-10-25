package ai.recall.core

/**
 * Streaming port of recall/perception/events.py::extract_events. Feed processed steps in order; emitted
 * events are returned from [onStep] / [finish]. Parity with the Python implementation is checked by
 * EventParityTest on fixtures exported from Python.
 */
class EventExtractor(
    private val maxGap: Int = 6,
    private val switchSim: Float = 0.40f,
    private val releaseFrames: Int = 2,
) {
    private class Held(
        var first: Int, var firstT: Double, var last: Int, var lastT: Double, var lastScene: FloatArray,
        var box: FloatArray,
    ) {
        val embs = mutableListOf<FloatArray>()
        val ctx = mutableListOf<FloatArray>()
        val scores = mutableListOf<Float>()
        var miss = 0
        var relStep: Int? = null
        var relT = 0.0
        var relCtx: FloatArray? = null
        var relScene: FloatArray? = null
    }

    private class Touch(var first: Int, var last: Int, var lastT: Double, var lastScene: FloatArray, val box: FloatArray) {
        val embs = mutableListOf<FloatArray>()
        val ctx = mutableListOf<FloatArray>()
        val scores = mutableListOf<Float>()
    }

    private val held = arrayOfNulls<Held>(2)
    private val touch = arrayOfNulls<Touch>(2)
    private var lastStep = Int.MIN_VALUE / 2

    private fun closeHeld(side: Int, useRel: Boolean, out: MutableList<Event>) {
        val h = held[side] ?: return
        val o = unit(mean(h.embs))
        val rel = useRel && h.relStep != null
        val relObs = rel && h.relCtx != null
        val loc = if (relObs) h.relCtx!! else h.ctx.last()
        val t = if (rel) h.relT else h.lastT
        val scene = if (rel) h.relScene!! else h.lastScene
        out += Event(t, "put", side, o, unit(loc), unit(scene), h.embs.size, h.scores.average().toFloat(), relObs, h.box)
        held[side] = null
    }

    private fun closeTouch(side: Int, out: MutableList<Event>) {
        val c = touch[side] ?: return
        out += Event(c.lastT, "touch", side, unit(mean(c.embs)), unit(mean(c.ctx)), unit(c.lastScene), c.embs.size,
            c.scores.average().toFloat(), false, c.box)
        touch[side] = null
    }

    /** One processed step. [sides] has 2 entries (left, right). */
    fun onStep(step: Int, t: Double, sides: Array<SideObs>, scene: FloatArray): List<Event> {
        val out = mutableListOf<Event>()
        val gap = step - lastStep
        for (side in 0..1) {
            val so = sides[side]
            held[side]?.let { if (gap > maxGap) closeHeld(side, it.miss > 0, out) }
            if (touch[side] != null && gap > maxGap) closeTouch(side, out)
            val hasObj = so.hand.present && so.objEmb != null
            val portable = hasObj && so.hand.state == PORTABLE
            val stationary = hasObj && so.hand.state == STATIONARY
            if (portable) {
                val e = so.objEmb!!
                val box = so.hand.objBox ?: floatArrayOf(0f, 0f, 1f, 1f)
                held[side]?.let { h ->
                    val sim = dot(unit(mean(h.embs)), unit(e))
                    if (sim < switchSim) {
                        if (h.miss > 0) closeHeld(side, true, out)
                        else {
                            if (gap == 1) { h.relStep = step; h.relT = t; h.relCtx = so.relCtxEmb; h.relScene = scene }
                            closeHeld(side, gap == 1, out)
                        }
                    } else {
                        h.miss = 0; h.relStep = null; h.relCtx = null
                    }
                }
                if (held[side] == null) {
                    held[side] = Held(step, t, step, t, scene, box)
                    out += Event(t, "take", side, unit(e), unit(scene), unit(scene), 1, so.hand.score, false, box)
                }
                val h = held[side]!!
                h.embs += e; h.ctx += scene; h.scores += so.hand.score
                h.last = step; h.lastT = t; h.lastScene = scene; h.box = box
            } else {
                held[side]?.let { h ->
                    h.miss += 1
                    if (h.miss == 1) {
                        if (gap == 1) { h.relStep = step; h.relT = t; h.relCtx = so.relCtxEmb; h.relScene = scene }
                        else h.relStep = null
                    }
                    if (h.miss >= releaseFrames) closeHeld(side, true, out)
                }
            }
            if (stationary) {
                val c = touch[side] ?: Touch(step, step, t, scene, so.hand.objBox ?: floatArrayOf(0f, 0f, 1f, 1f))
                    .also { touch[side] = it }
                c.embs += so.objEmb!!; c.ctx += scene; c.scores += so.hand.score
                c.last = step; c.lastT = t; c.lastScene = scene
            } else if (touch[side] != null) {
                closeTouch(side, out)
            }
        }
        lastStep = step
        return out.sortedWith(compareBy({ it.t }, { it.side }))
    }

    fun finish(): List<Event> {
        val out = mutableListOf<Event>()
        for (side in 0..1) {
            held[side]?.let { closeHeld(side, it.miss > 0, out) }
            closeTouch(side, out)
        }
        return out
    }
}
