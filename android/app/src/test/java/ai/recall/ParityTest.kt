package ai.recall

import ai.recall.core.Answer
import ai.recall.core.Answerer
import ai.recall.core.Calibrator
import ai.recall.core.Event
import ai.recall.core.EventExtractor
import ai.recall.core.HandObs
import ai.recall.core.MemoryView
import ai.recall.core.QueryParser
import ai.recall.core.Renderer
import ai.recall.core.SideObs
import ai.recall.core.StateProbe
import ai.recall.core.burst
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/** Kotlin ports vs the Python reference (fixtures exported by scripts/android_fixtures.sh). */
class ParityTest {
    @Test fun parse() {
        val rows = Fixtures.arr("parse").objects()
        assertTrue(rows.size > 30)
        for (r in rows) {
            val p = QueryParser.parse(r.getString("q"))
            assertEquals(r.getString("q"), r.getString("kind"), p.kind)
            assertEquals(r.getString("q"), r.getString("obj"), p.obj)
            assertEquals(r.getString("q"), r.optStringOrNull("verb"), p.verb)
        }
    }

    @Test fun burstPolicy() {
        for (c in Fixtures.arr("burst").objects()) {
            val fire = c.getJSONArray("fire").let { a -> BooleanArray(a.length()) { a.getBoolean(it) } }
            val exp = c.getJSONArray("out").let { a -> BooleanArray(a.length()) { a.getBoolean(it) } }
            assertArrayEquals(exp, burst(fire, c.getInt("length"), c.getInt("refractory")))
        }
    }

    @Test fun blendHeads() {
        for (c in Fixtures.arr("heads").objects()) {
            val h = headFrom(c.getJSONObject("head"))
            val emb = c.getJSONArray("emb").matrix()
            val exp = c.getJSONArray("logp")
            for (i in emb.indices) {
                val got = h.logProbs(emb[i]); val e = exp.getJSONArray(i).doubles()
                for (k in got.indices) assertEquals("alpha=${h.alpha} row $i class $k", e[k], got[k], 2e-4)
            }
        }
    }

    @Test fun answers() {
        val f = Fixtures.obj("answers")
        val obj = headFrom(f.getJSONObject("obj")); val loc = headFrom(f.getJSONObject("loc"))
        val sp = f.getJSONObject("state_probe").let { StateProbe(it.getJSONArray("coef").doubles(), it.getDouble("intercept")) }
        val events = f.getJSONArray("events").objects().map {
            Event(it.getDouble("t"), it.getString("kind"), it.getInt("side"), it.getJSONArray("obj").floats(), it.getJSONArray("loc").floats(),
                it.getJSONArray("scene").floats(), it.getInt("n_obs"), it.getDouble("hand_score").toFloat(), it.getBoolean("release"))
        }
        val view = MemoryView(events, obj, loc)
        var answered = 0
        val qs = f.getJSONArray("queries").objects()
        for (q in qs) {
            val cfg = cfgFrom(q.getJSONObject("cfg")); val cls = q.getInt("cls"); val tq = q.getDouble("tq")
            val a: Answer = when (q.getString("kind")) {
                "where" -> Answerer.where(view, cls, tq, cfg)
                "temporal" -> Answerer.temporal(view, cls, q.optStringOrNull("verb"), tq, cfg)
                else -> Answerer.state(view, cls, tq, cfg) { e, _ -> sp.pOn(e) }
            }
            val what = "$q"
            assertEquals(what, q.getBoolean("abstain"), a.abstain)
            if (a.abstain) continue
            answered++
            assertEquals(what, q.getInt("event_idx"), events.indexOfFirst { it === a.event })
            assertEquals(what, q.getDouble("t"), a.t!!, 1e-9)
            if (q.isNull("loc_idx")) assertNull(a.locIdx) else assertEquals(what, q.getInt("loc_idx"), a.locIdx)
            assertEquals(what, q.optStringOrNull("state"), a.state)
            val ef = q.getJSONObject("feats")
            assertEquals(what, ef.keys().asSequence().toSet(), a.feats.keys)
            for (k in ef.keys()) assertEquals("$what feat $k", ef.getDouble(k), a.feats[k]!!, 2e-4 * maxOf(1.0, abs(ef.getDouble(k))))
        }
        assertTrue("fixture should exercise answered queries ($answered/${qs.size})", answered > qs.size / 4)
    }

    @Test fun calibration() {
        for (c in Fixtures.arr("calibration").objects()) {
            val j = c.getJSONObject("calibrator")
            val cal = Calibrator(j.getJSONArray("features").let { a -> List(a.length()) { a.getString(it) } }, j.getJSONArray("coef").doubles(),
                j.getDouble("intercept"), j.getJSONArray("iso_x").doubles(), j.getJSONArray("iso_y").doubles(), j.getDouble("threshold"))
            val exp = c.getJSONArray("p").doubles()
            c.getJSONArray("recs").objects().forEachIndexed { i, r ->
                val fe = r.getJSONObject("feats")
                val a = Answer(r.getString("kind"), !r.getBoolean("answered"), feats = fe.keys().asSequence().associateWith { k -> fe.getDouble(k) })
                assertEquals("rec $i", exp[i], cal.confidence(a), 1e-6)
            }
        }
    }

    @Test fun events() {
        for ((ci, c) in Fixtures.arr("events").objects().withIndex()) {
            val t = c.getJSONArray("t").doubles()
            val hands = c.getJSONArray("hands")
            val femb = c.getJSONArray("frame_emb").matrix()
            val processed = c.getJSONArray("processed").let { a -> BooleanArray(a.length()) { a.getBoolean(it) } }
            val reg = HashMap<Triple<Int, String, Int>, JSONObjectView>()
            for (r in c.getJSONArray("regions").objects()) {
                reg[Triple(r.getInt("t"), r.getString("kind"), r.getInt("side"))] = JSONObjectView(r.getJSONArray("emb").floats(), r.getJSONArray("box").floats())
            }
            val ex = EventExtractor()
            val got = mutableListOf<Event>()
            for (i in t.indices) {
                if (!processed[i]) continue
                val sides = Array(2) { s ->
                    val h = hands.getJSONArray(i).getJSONArray(s).doubles()
                    val o = reg[Triple(i, "obj", s)]
                    SideObs(HandObs(h[0] > 0, h[1].toFloat(), h[2].toInt(), o?.box), o?.emb, reg[Triple(i, "pctx", s)]?.emb)
                }
                got += ex.onStep(i, t[i], sides, femb[i])
            }
            got += ex.finish()
            val sorted = got.sortedWith(compareBy({ it.t }, { it.side })) // stable, like Python's list.sort
            val exp = c.getJSONArray("events").objects()
            assertEquals("case $ci event count", exp.size, sorted.size)
            for ((k, e) in exp.withIndex()) {
                val g = sorted[k]; val w = "case $ci event $k"
                assertEquals(w, e.getString("kind"), g.kind)
                assertEquals(w, e.getInt("side"), g.side)
                assertEquals(w, e.getDouble("t"), g.t, 1e-6)
                assertEquals(w, e.getInt("n_obs"), g.nObs)
                assertEquals(w, e.getBoolean("release"), g.releaseObserved)
                assertEquals(w, e.getDouble("hand_score"), g.handScore.toDouble(), 1e-5)
                assertArrayEquals(w, e.getJSONArray("obj").floats(), g.objEmb, 1e-4f)
                assertArrayEquals(w, e.getJSONArray("loc").floats(), g.locEmb, 1e-4f)
                assertArrayEquals(w, e.getJSONArray("scene").floats(), g.sceneEmb, 1e-4f)
                assertArrayEquals(w, e.getJSONArray("box").floats(), g.box, 1e-6f)
            }
        }
    }

    @Test fun render() {
        val f = Fixtures.obj("render")
        for (r in f.getJSONArray("rows").objects()) {
            val a = Answer(r.getString("kind"), r.getBoolean("abstain"), t = r.optDoubleOrNull("t"), state = r.optStringOrNull("state"))
            val out = Renderer.render(a, r.getString("obj"), r.optStringOrNull("loc"), r.optDoubleOrNull("conf"), r.getDouble("abstain_below"))
            assertEquals(r.getString("text"), out.text)
            assertEquals(r.getBoolean("abstained"), out.abstained)
        }
        for (c in f.getJSONArray("clocks").objects()) assertEquals(c.getString("clock"), Renderer.clock(c.getDouble("t")))
    }

    private class JSONObjectView(val emb: FloatArray, val box: FloatArray)
}
