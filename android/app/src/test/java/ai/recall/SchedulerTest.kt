package ai.recall

import ai.recall.core.OperatingPoint
import ai.recall.core.Scheduler
import ai.recall.core.StepPolicy
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SchedulerTest {
    @Test fun nominalWhenCoolAndCharged() {
        val p = Scheduler.plan(0, 80, false)
        assertEquals(Scheduler.NOMINAL, p)
    }

    @Test fun thermalBackoffIsMonotone() {
        val plans = (0..3).map { Scheduler.plan(it, 80, false) }
        for (i in 1 until plans.size) {
            assertTrue(plans[i].gateStride >= plans[i - 1].gateStride)
            assertTrue(plans[i].burstLength <= plans[i - 1].burstLength)
            assertTrue(plans[i].thresholdOffset > plans[i - 1].thresholdOffset)
            assertTrue(plans[i].level >= plans[i - 1].level)
        }
        assertEquals(3, Scheduler.plan(3, 80, false).gateStride)
    }

    @Test fun criticalThermalOrBatteryPauses() {
        for (s in 4..6) assertTrue(Scheduler.plan(s, 80, true).paused)
        assertTrue(Scheduler.plan(0, 5, false).paused)
        assertEquals(4, Scheduler.plan(0, 5, false).level)
        assertFalse("charging overrides low battery", Scheduler.plan(0, 5, true).paused)
    }

    @Test fun lowBatteryBacksOff() {
        val p = Scheduler.plan(0, 15, false)
        assertEquals(2, p.gateStride); assertEquals(3, p.burstLength); assertEquals(2, p.level)
        assertEquals("low battery", p.reason)
        val both = Scheduler.plan(3, 15, false)
        assertEquals(3, both.gateStride); assertEquals(2, both.burstLength)
        assertEquals(0.3f, both.thresholdOffset, 1e-6f)
    }

    private fun run(pol: StepPolicy, n: Int, score: (Long) -> Float): Triple<Int, Int, Int> {
        var gated = 0; var heavy = 0; var skipped = 0
        for (s in 0 until n.toLong()) {
            val d = pol.decide(s) { gated++; score(s) }
            if (d.heavy) heavy++
            if (d.skipped) skipped++
        }
        return Triple(gated, heavy, skipped)
    }

    @Test fun operatingPoints() {
        val all = StepPolicy(OperatingPoint(OperatingPoint.Policy.ALL), 0.5f)
        assertEquals(Triple(0, 100, 0), run(all, 100) { 0f })
        val uni = StepPolicy(OperatingPoint(OperatingPoint.Policy.UNIFORM, k = 5), 0.5f)
        assertEquals(Triple(0, 20, 0), run(uni, 100) { 0f })
        val learned = StepPolicy(OperatingPoint(OperatingPoint.Policy.LEARNED), 0.5f)
        // fires once at step 10 → burst of 4 heavy steps
        assertEquals(Triple(100, 4, 0), run(learned, 100) { if (it == 10L) 0.9f else 0.1f })
    }

    @Test fun backoffAppliesOnlyWhenEnabled() {
        val hot = Scheduler.plan(3, 80, false)
        val on = StepPolicy(OperatingPoint(OperatingPoint.Policy.ALL, backoff = true), 0.5f).apply { applyPlan(hot) }
        val off = StepPolicy(OperatingPoint(OperatingPoint.Policy.ALL, backoff = false), 0.5f).apply { applyPlan(hot) }
        assertEquals(34, run(on, 100) { 0f }.second) // every 3rd step
        assertEquals(100, run(off, 100) { 0f }.second)
        assertEquals(0, off.plan.level)
        val paused = StepPolicy(OperatingPoint(), 0.5f).apply { applyPlan(Scheduler.plan(5, 80, false)) }
        assertEquals(Triple(0, 0, 50), run(paused, 50) { 1f })
    }

    @Test fun thresholdOffsetRaisesGateThreshold() {
        val p = StepPolicy(OperatingPoint(), 0.5f).apply { applyPlan(Scheduler.plan(1, 80, false)) } // +0.05
        assertEquals(0, run(p, 20) { 0.52f }.second)
        val q = StepPolicy(OperatingPoint(), 0.5f)
        assertEquals(20, run(q, 20) { 0.52f }.second)
    }

    @Test fun parseOperatingPoint() {
        assertEquals(OperatingPoint(OperatingPoint.Policy.UNIFORM, 5, false), OperatingPoint.parse("uniform", 5, false))
        assertEquals(OperatingPoint(), OperatingPoint.parse(null, null, null))
        assertEquals(OperatingPoint.Policy.ALL, OperatingPoint.parse("all", null, null).policy)
        assertEquals(1, OperatingPoint.parse("uniform", 0, null).k)
        try { OperatingPoint.parse("bogus", null, null); throw AssertionError("expected failure") } catch (_: IllegalArgumentException) {}
    }
}
