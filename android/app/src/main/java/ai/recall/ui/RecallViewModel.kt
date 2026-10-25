package ai.recall.ui

import ai.recall.core.QueryEngine
import ai.recall.core.Renderer
import ai.recall.service.RecallRuntime
import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.TimeZone

data class AnswerUi(
    val question: String,
    val text: String,
    val abstained: Boolean,
    val confidence: Double,
    val eventId: Long? = null,
    val eventT: Double? = null,
    val thumb: ByteArray? = null,
    val objCls: Int? = null,
    val objName: String = "",
)

data class TimelineItem(val id: Long, val t: Double, val clock: String, val kind: String, val label: String, val thumb: ByteArray?)

data class PlaceUi(val id: Int, val name: String, val n: Int, val blocked: Boolean)

data class UiState(
    val recording: Boolean = false,
    val status: String = "",
    /** Non-null → the app runs in the clearly labelled "models not installed" fallback. */
    val fallbackReason: String? = null,
    val tab: Int = 0,
    val query: String = "",
    val answer: AnswerUi? = null,
    val busy: Boolean = false,
    val listening: Boolean = false,
    val speechNote: String? = null,
    val timeline: List<TimelineItem> = emptyList(),
    val focusT: Double? = null,
    val eventCount: Int = 0,
    val retentionDays: Int = RecallRuntime.DEFAULT_RETENTION_DAYS,
    val places: List<PlaceUi> = emptyList(),
    val currentPlace: Int = -1,
    val blockedObjects: List<Pair<Int, String>> = emptyList(),
    val message: String? = null,
)

class RecallViewModel(app: Application, private val saved: SavedStateHandle) : AndroidViewModel(app) {
    private val ctx get() = getApplication<Application>()
    private val _state = MutableStateFlow(UiState(query = saved["query"] ?: "", tab = saved["tab"] ?: 0))
    val state: StateFlow<UiState> = _state.asStateFlow()
    private var engine: QueryEngine? = null

    init {
        viewModelScope.launch {
            val (eng, reason) = withContext(Dispatchers.IO) {
                val e = RecallRuntime.engine(ctx)
                val missing = RecallRuntime.missingAssets(ctx)
                e to when {
                    e == null -> "Models not installed (${RecallRuntime.loadError ?: missing.joinToString()}). " +
                        "Build with models in ../models/litert (see android/README.md)."
                    missing.isNotEmpty() -> "Capture disabled: missing ${missing.joinToString()}."
                    else -> null
                }
            }
            engine = eng
            _state.update { it.copy(fallbackReason = reason) }
            withContext(Dispatchers.IO) { RecallRuntime.applyRetention(ctx) }
            refresh()
        }
        viewModelScope.launch { RecallRuntime.recording.collect { r -> _state.update { it.copy(recording = r) } } }
        viewModelScope.launch { RecallRuntime.status.collect { s -> _state.update { it.copy(status = s) } } }
        viewModelScope.launch { RecallRuntime.memoryVersion.collect { refresh() } }
    }

    private fun zoneOffset(): Int = TimeZone.getDefault().getOffset(System.currentTimeMillis()) / 1000

    fun setTab(i: Int) { saved["tab"] = i; _state.update { it.copy(tab = i) } }
    fun setQuery(q: String) { saved["query"] = q; _state.update { it.copy(query = q) } }
    fun setListening(b: Boolean, note: String? = null) = _state.update { it.copy(listening = b, speechNote = note) }
    fun dismissMessage() = _state.update { it.copy(message = null) }

    fun ask() {
        val q = _state.value.query.trim()
        val eng = engine ?: return
        if (q.isEmpty()) return
        _state.update { it.copy(busy = true) }
        viewModelScope.launch {
            val ans = withContext(Dispatchers.IO) {
                val db = RecallRuntime.db(ctx)
                val now = System.currentTimeMillis() / 1000.0
                val r = eng.ask(q, db.events(now).map { it.event }, now, zoneOffset())
                val ev = r.answer.event
                AnswerUi(q, r.rendered.text, r.rendered.abstained, r.confidence, ev?.id, ev?.t,
                    ev?.let { e -> if (!r.rendered.abstained) db.thumb(e.id) else null }, r.cls, r.parsed.obj)
            }
            _state.update { it.copy(answer = ans, busy = false) }
        }
    }

    fun showMoment(t: Double) { _state.update { it.copy(focusT = t) }; setTab(1) }

    fun forget(id: Long) = io({ RecallRuntime.db(ctx).forget(id) }) { n ->
        _state.update { s -> s.copy(answer = s.answer?.takeIf { it.eventId != id }, message = if (n > 0) "Forgotten." else null) }
    }

    fun neverTrack(cls: Int) = io({
        val n = RecallRuntime.db(ctx).blockObject(cls); RecallRuntime.pipelineOrNull()?.reloadPolicies(); n
    }) { n -> _state.update { it.copy(answer = null, message = "Won't track this object any more ($n memories deleted).") } }

    fun neverTrackByName(name: String) {
        val cls = engine?.resolve(name)
        if (cls == null) { _state.update { it.copy(message = "Unknown object \"$name\".") }; return }
        neverTrack(cls)
    }

    fun unblockObject(cls: Int) = io({ RecallRuntime.db(ctx).unblockObject(cls); RecallRuntime.pipelineOrNull()?.reloadPolicies() }) {}

    fun setPlaceBlocked(id: Int, blocked: Boolean) = io({
        val n = RecallRuntime.db(ctx).setPlaceBlocked(id, blocked); RecallRuntime.pipelineOrNull()?.reloadPolicies(); n
    }) { n -> if (blocked) _state.update { it.copy(message = "Won't record in this place ($n memories deleted).") } }

    fun blockCurrentPlace() {
        val p = RecallRuntime.pipelineOrNull()?.currentPlace ?: -1
        if (p < 0) { _state.update { it.copy(message = "No current place yet — start recording first.") }; return }
        setPlaceBlocked(p, true)
    }

    fun setRetention(days: Int) = io({
        RecallRuntime.db(ctx).setSetting(RecallRuntime.KEY_RETENTION_DAYS, days.toString()); RecallRuntime.applyRetention(ctx)
    }) {}

    fun wipe() = io({ RecallRuntime.db(ctx).wipe() }) { _state.update { it.copy(answer = null, message = "All memory deleted.") } }

    private fun <T> io(block: () -> T, after: (T) -> Unit) {
        viewModelScope.launch {
            val r = withContext(Dispatchers.IO) { block() }
            after(r)
            RecallRuntime.memoryChanged()
        }
    }

    fun refresh() {
        viewModelScope.launch {
            val snap = withContext(Dispatchers.IO) {
                val db = RecallRuntime.db(ctx)
                val b = engine?.bundle
                val evs = db.events().asReversed().take(300)
                val tl = evs.map { se ->
                    val e = se.event
                    val obj = se.objTop.firstOrNull()?.first?.let { c -> b?.nounNames?.getOrNull(c) } ?: "object"
                    val loc = se.locTop.firstOrNull()?.first?.let { c -> b?.locNames?.getOrNull(c) }
                    val label = when (e.kind) { "put" -> "put $obj${loc?.let { " → $it" } ?: ""}"; "take" -> "took $obj"; else -> "touched $obj" }
                    TimelineItem(e.id, e.t, Renderer.clock(e.t, zoneOffset()), e.kind, label, db.thumb(e.id))
                }
                val places = db.places().map { PlaceUi(it.id, it.name ?: "Place ${it.id + 1}", it.n, it.blocked) }
                val blocked = db.blockedObjects().map { it to (b?.nounNames?.getOrNull(it) ?: "class $it") }
                Snap(tl, db.count(), RecallRuntime.retentionDays(ctx), places, blocked, RecallRuntime.pipelineOrNull()?.currentPlace ?: -1)
            }
            _state.update {
                it.copy(timeline = snap.tl, eventCount = snap.count, retentionDays = snap.retention, places = snap.places,
                    blockedObjects = snap.blocked, currentPlace = snap.current)
            }
        }
    }

    private data class Snap(val tl: List<TimelineItem>, val count: Int, val retention: Int, val places: List<PlaceUi>,
                            val blocked: List<Pair<Int, String>>, val current: Int)
}
