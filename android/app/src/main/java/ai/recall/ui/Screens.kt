package ai.recall.ui

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlin.math.abs

/** All UI callbacks; the Activity binds them to the ViewModel, tests to fakes. */
interface UiActions {
    fun toggleRecording() {}
    fun setTab(i: Int) {}
    fun setQuery(q: String) {}
    fun ask() {}
    fun voice() {}
    fun showMoment(t: Double) {}
    fun forget(id: Long) {}
    fun neverTrack(cls: Int) {}
    fun neverTrackByName(name: String) {}
    fun unblockObject(cls: Int) {}
    fun setRetention(days: Int) {}
    fun setPlaceBlocked(id: Int, blocked: Boolean) {}
    fun blockCurrentPlace() {}
    fun wipe() {}
    fun dismissMessage() {}
}

object Tags {
    const val REC_STATUS = "rec_status"
    const val REC_TOGGLE = "rec_toggle"
    const val FALLBACK = "fallback_banner"
    const val QUERY = "query"
    const val ASK = "ask"
    const val VOICE = "voice"
    const val ANSWER_CARD = "answer_card"
    const val ANSWER_TEXT = "answer_text"
    const val CONFIDENCE = "confidence"
    const val EVIDENCE = "evidence"
    const val SHOW_MOMENT = "show_moment"
    const val FORGET = "forget"
    const val NEVER_TRACK = "never_track"
    const val TIMELINE = "timeline"
    const val FOCUSED = "timeline_focused"
    const val WIPE = "wipe"
    const val BLOCK_PLACE = "block_place"
    const val MESSAGE = "message"
}

@Composable
private fun thumbBitmap(bytes: ByteArray?): ImageBitmap? = remember(bytes) {
    bytes?.let { BitmapFactory.decodeByteArray(it, 0, it.size)?.asImageBitmap() }
}

@Composable
fun RecallApp(s: UiState, a: UiActions) {
    Column(Modifier.fillMaxSize()) {
        RecordingBar(s, a)
        s.fallbackReason?.let {
            Surface(color = Color(0xFFFFF3CD), modifier = Modifier.fillMaxWidth().testTag(Tags.FALLBACK)) {
                Text("Fallback mode — $it", Modifier.padding(12.dp), style = MaterialTheme.typography.bodySmall)
            }
        }
        s.message?.let {
            Row(Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.secondaryContainer).padding(horizontal = 12.dp),
                verticalAlignment = Alignment.CenterVertically) {
                Text(it, Modifier.weight(1f).testTag(Tags.MESSAGE))
                TextButton({ a.dismissMessage() }) { Text("OK") }
            }
        }
        PrimaryTabRow(selectedTabIndex = s.tab) {
            listOf("Ask", "Timeline", "Privacy").forEachIndexed { i, t ->
                Tab(s.tab == i, { a.setTab(i) }, text = { Text(t) }, modifier = Modifier.testTag("tab_$i"))
            }
        }
        when (s.tab) {
            0 -> AskScreen(s, a)
            1 -> TimelineScreen(s, a)
            else -> PrivacyScreen(s, a)
        }
    }
}

@Composable
fun RecordingBar(s: UiState, a: UiActions) {
    Row(Modifier.fillMaxWidth().padding(12.dp), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.weight(1f)) {
            Spacer(Modifier.size(12.dp).clip(CircleShape).background(if (s.recording) Color(0xFFD32F2F) else Color.Gray))
            Spacer(Modifier.width(8.dp))
            Column {
                Text(if (s.recording) "Remembering" else "Paused", fontWeight = FontWeight.Bold, modifier = Modifier.testTag(Tags.REC_STATUS))
                if (s.status.isNotEmpty()) Text(s.status, style = MaterialTheme.typography.bodySmall)
            }
        }
        val canStart = s.recording || s.fallbackReason == null
        Button(onClick = { a.toggleRecording() }, enabled = canStart, modifier = Modifier.testTag(Tags.REC_TOGGLE)) {
            Text(if (s.recording) "Pause" else "Start")
        }
    }
}

@Composable
fun AskScreen(s: UiState, a: UiActions) {
    Column(Modifier.padding(16.dp).verticalScroll(rememberScrollState())) {
        OutlinedTextField(s.query, { a.setQuery(it) }, label = { Text("Ask, e.g. \"Where are my keys?\"") }, singleLine = true,
            modifier = Modifier.fillMaxWidth().testTag(Tags.QUERY))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 8.dp)) {
            Button(onClick = { a.ask() }, enabled = s.fallbackReason?.startsWith("Models not installed") != true && !s.busy,
                modifier = Modifier.testTag(Tags.ASK)) { Text("Ask") }
            OutlinedButton(onClick = { a.voice() }, modifier = Modifier.testTag(Tags.VOICE)) { Text(if (s.listening) "Listening…" else "Voice") }
        }
        s.speechNote?.let { Text(it, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 4.dp)) }
        if (s.busy) LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 8.dp))
        s.answer?.let { AnswerCard(it, a) }
    }
}

@Composable
fun AnswerCard(r: AnswerUi, a: UiActions) {
    Card(Modifier.fillMaxWidth().padding(top = 16.dp).testTag(Tags.ANSWER_CARD)) {
        Column(Modifier.padding(16.dp)) {
            Text(r.text, style = MaterialTheme.typography.titleMedium, modifier = Modifier.testTag(Tags.ANSWER_TEXT))
            if (!r.abstained) {
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(progress = { r.confidence.toFloat() }, modifier = Modifier.fillMaxWidth())
                Text("Confidence ${Math.rint(r.confidence * 100).toInt()}%", Modifier.testTag(Tags.CONFIDENCE))
                val bm = thumbBitmap(r.thumb)
                if (bm != null) {
                    Image(bm, "evidence thumbnail (faces and screens redacted)",
                        Modifier.fillMaxWidth().height(160.dp).padding(top = 8.dp).testTag(Tags.EVIDENCE))
                } else if (r.eventId != null) {
                    Text("No thumbnail stored (redaction unavailable — privacy fail-closed).", style = MaterialTheme.typography.bodySmall)
                }
                if (r.eventId != null) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 8.dp)) {
                        OutlinedButton({ r.eventT?.let { a.showMoment(it) } }, Modifier.testTag(Tags.SHOW_MOMENT)) { Text("Show me the moment") }
                        OutlinedButton({ a.forget(r.eventId) }, Modifier.testTag(Tags.FORGET)) { Text("Forget this") }
                    }
                }
                r.objCls?.let { c ->
                    TextButton({ a.neverTrack(c) }, Modifier.testTag(Tags.NEVER_TRACK)) { Text("Never track \"${r.objName}\"") }
                }
            }
        }
    }
}

@Composable
fun TimelineScreen(s: UiState, a: UiActions) {
    if (s.timeline.isEmpty()) {
        Text("No memories yet.", Modifier.padding(16.dp).testTag(Tags.TIMELINE))
        return
    }
    val list = rememberLazyListState()
    val focusIdx = s.focusT?.let { f -> s.timeline.indexOfFirst { abs(it.t - f) < 1.0 } } ?: -1
    LaunchedEffect(focusIdx) { if (focusIdx >= 0) list.scrollToItem(focusIdx) }
    LazyColumn(Modifier.padding(horizontal = 12.dp).testTag(Tags.TIMELINE), state = list) {
        items(s.timeline, key = { it.id }) { e ->
            val focused = s.focusT != null && abs(e.t - s.focusT) < 1.0
            Card(Modifier.fillMaxWidth().padding(vertical = 4.dp).let { if (focused) it.testTag(Tags.FOCUSED) else it },
                colors = if (focused) CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer) else CardDefaults.cardColors()) {
                Row(Modifier.padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    thumbBitmap(e.thumb)?.let { Image(it, "thumbnail", Modifier.size(96.dp, 54.dp)) } ?: Spacer(Modifier.size(96.dp, 54.dp).background(Color.LightGray))
                    Spacer(Modifier.width(12.dp))
                    Column(Modifier.weight(1f)) {
                        Text(e.clock, fontWeight = FontWeight.Bold)
                        Text(e.label)
                    }
                    TextButton({ a.forget(e.id) }, Modifier.semantics { contentDescription = "forget ${e.id}" }) { Text("Forget") }
                }
            }
        }
    }
}

@Composable
fun PrivacyScreen(s: UiState, a: UiActions) {
    var name by rememberSaveable { mutableStateOf("") }
    var confirmWipe by remember { mutableStateOf(false) }
    Column(Modifier.padding(16.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Everything stays on this phone: the app has no network permission, the memory database is encrypted with a " +
            "hardware-backed key, faces and screens are pixelated before a thumbnail is stored, and no video is kept.")
        Text("Stored memories: ${s.eventCount}")
        HorizontalDivider()
        Text("Auto-delete memories older than", fontWeight = FontWeight.Bold)
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            for ((d, l) in listOf(1 to "1 day", 7 to "7 days", 30 to "30 days", 0 to "Never")) {
                FilterChip(s.retentionDays == d, { a.setRetention(d) }, label = { Text(l) }, modifier = Modifier.testTag("retention_$d"))
            }
        }
        HorizontalDivider()
        Text("Never record here", fontWeight = FontWeight.Bold)
        OutlinedButton({ a.blockCurrentPlace() }, Modifier.testTag(Tags.BLOCK_PLACE)) { Text("Never record in the current place") }
        for (p in s.places) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("${p.name}${if (p.id == s.currentPlace) " (current)" else ""}", Modifier.weight(1f))
                Text(if (p.blocked) "blocked" else "recording", style = MaterialTheme.typography.bodySmall)
                Switch(p.blocked, { a.setPlaceBlocked(p.id, it) }, Modifier.testTag("place_${p.id}"))
            }
        }
        HorizontalDivider()
        Text("Never track these objects", fontWeight = FontWeight.Bold)
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(name, { name = it }, label = { Text("object, e.g. pills") }, singleLine = true,
                modifier = Modifier.weight(1f).testTag("never_track_name"))
            TextButton({ a.neverTrackByName(name); name = "" }, Modifier.testTag("never_track_add")) { Text("Add") }
        }
        for ((c, n) in s.blockedObjects) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(n, Modifier.weight(1f)); TextButton({ a.unblockObject(c) }) { Text("Allow") }
            }
        }
        HorizontalDivider()
        if (!confirmWipe) OutlinedButton({ confirmWipe = true }, Modifier.testTag(Tags.WIPE)) { Text("Delete all memory") }
        else Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button({ a.wipe(); confirmWipe = false }, Modifier.testTag("wipe_confirm")) { Text("Yes, delete everything") }
            TextButton({ confirmWipe = false }) { Text("Cancel") }
        }
    }
}
