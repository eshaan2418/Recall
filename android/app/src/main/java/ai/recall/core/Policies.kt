package ai.recall.core

/**
 * Online memory-trigger controller (μ): the always-on gate fires, heavy perception runs for a burst of
 * [burstLength] steps; firings inside a burst extend it. Mirrors recall/trigger/policies.py::burst.
 */
class BurstController(var burstLength: Int, var threshold: Float, var refractory: Int = 0) {
    private var until = -1L
    private var cool = -1L
    private var step = -1L

    /** Call once per gate step with the gate score; returns true if heavy perception must run this step. */
    fun onStep(score: Float): Boolean {
        step += 1
        if (score >= threshold && step > cool) until = maxOf(until, step + burstLength - 1)
        val run = step <= until
        if (run && step == until) cool = step + refractory
        return run
    }
}

fun burst(fire: BooleanArray, length: Int, refractory: Int = 0): BooleanArray {
    val c = BurstController(length, 0.5f, refractory)
    return BooleanArray(fire.size) { c.onStep(if (fire[it]) 1f else 0f) }
}
