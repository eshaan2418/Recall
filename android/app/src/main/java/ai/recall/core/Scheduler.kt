package ai.recall.core

/**
 * Battery- and thermal-aware scheduling of the memory pipeline.
 *
 * Inputs are the platform thermal status (PowerManager.THERMAL_STATUS_*: 0 none … 6 shutdown) and battery
 * level/charging. Output: step stride (run the trigger every k-th camera step), burst length, and a gate threshold
 * offset (fire less readily when hot / low battery). Pure function → unit-tested; the service applies it.
 *
 * [level]: 0 nominal, 1 light, 2 moderate or low battery, 3 severe, 4 paused (thermal critical / battery critical).
 */
data class SchedulePlan(
    val gateStride: Int,
    val burstLength: Int,
    val thresholdOffset: Float,
    val paused: Boolean,
    val reason: String,
    val level: Int = 0,
)

object Scheduler {
    val NOMINAL = SchedulePlan(1, 4, 0f, false, "nominal", 0)

    fun plan(thermalStatus: Int, batteryPct: Int, charging: Boolean, lowBatteryPct: Int = 20, criticalBatteryPct: Int = 8): SchedulePlan {
        if (thermalStatus >= 4) return SchedulePlan(4, 2, 0.3f, true, "thermal critical: paused", 4)
        if (!charging && batteryPct <= criticalBatteryPct) return SchedulePlan(4, 2, 0.3f, true, "battery critical: paused", 4)
        var stride = 1
        var burst = 4
        var off = 0f
        var level = 0
        val why = mutableListOf<String>()
        when (thermalStatus) {
            3 -> { stride = 3; burst = 2; off += 0.2f; level = 3; why += "thermal severe" }
            2 -> { stride = 2; burst = 3; off += 0.1f; level = 2; why += "thermal moderate" }
            1 -> { off += 0.05f; level = 1; why += "thermal light" }
        }
        if (!charging && batteryPct <= lowBatteryPct) {
            stride = maxOf(stride, 2); burst = minOf(burst, 3); off += 0.1f; level = maxOf(level, 2); why += "low battery"
        }
        return SchedulePlan(stride, burst, off, false, if (why.isEmpty()) "nominal" else why.joinToString(", "), level)
    }
}

/**
 * Operating point of the memory trigger (energy harness: recall/energy/harness.py).
 *  - ALL: heavy perception on every step (no gate);
 *  - UNIFORM: heavy perception on every k-th step (no gate);
 *  - LEARNED: tiny gate + burst controller (the Recall default).
 * [backoff]=false ignores battery/thermal state entirely (for controlled measurements).
 */
data class OperatingPoint(val policy: Policy = Policy.LEARNED, val k: Int = 5, val backoff: Boolean = true) {
    enum class Policy { ALL, UNIFORM, LEARNED }

    companion object {
        fun parse(policy: String?, k: Int?, backoff: Boolean?): OperatingPoint {
            val p = when (policy?.lowercase()) {
                "all", "always", "always-heavy" -> Policy.ALL
                "uniform" -> Policy.UNIFORM
                null, "", "learned" -> Policy.LEARNED
                else -> throw IllegalArgumentException("unknown policy '$policy' (all|uniform|learned)")
            }
            return OperatingPoint(p, (k ?: 5).coerceAtLeast(1), backoff ?: true)
        }
    }
}

/** Per-step decision: does the gate run, does heavy perception run. Gate score is requested lazily. */
class StepPolicy(var op: OperatingPoint, gateThreshold: Float) {
    private val controller = BurstController(4, gateThreshold)
    private val baseThreshold = gateThreshold
    var plan: SchedulePlan = Scheduler.NOMINAL
        private set

    data class Decision(val skipped: Boolean, val gated: Boolean, val heavy: Boolean)

    fun applyPlan(p: SchedulePlan) {
        plan = if (op.backoff) p else Scheduler.NOMINAL
        controller.burstLength = plan.burstLength
        controller.threshold = baseThreshold + plan.thresholdOffset
    }

    /** [step] counts camera steps from 0. [gateScore] is only invoked if the policy uses the gate this step. */
    fun decide(step: Long, gateScore: () -> Float): Decision {
        if (plan.paused || step % plan.gateStride != 0L) return Decision(true, false, false)
        val s = step / plan.gateStride
        return when (op.policy) {
            OperatingPoint.Policy.ALL -> Decision(false, false, true)
            OperatingPoint.Policy.UNIFORM -> Decision(false, false, s % op.k == 0L)
            OperatingPoint.Policy.LEARNED -> Decision(false, true, controller.onStep(gateScore()))
        }
    }
}
