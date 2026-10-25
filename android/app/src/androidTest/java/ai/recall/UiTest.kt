package ai.recall

import ai.recall.ui.AnswerUi
import ai.recall.ui.RecallApp
import ai.recall.ui.Tags
import ai.recall.ui.TimelineItem
import ai.recall.ui.UiActions
import ai.recall.ui.UiState
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.hasAnyDescendant
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/** Compose UI tests of the stateless screens with a fake action sink. */
@RunWith(AndroidJUnit4::class)
class UiTest {
    @get:Rule val rule = createComposeRule()

    private class Recorder : UiActions {
        val calls = mutableListOf<String>()
        override fun ask() { calls += "ask" }
        override fun setQuery(q: String) { calls += "query:$q" }
        override fun forget(id: Long) { calls += "forget:$id" }
        override fun showMoment(t: Double) { calls += "moment:$t" }
        override fun neverTrack(cls: Int) { calls += "never:$cls" }
        override fun toggleRecording() { calls += "toggle" }
        override fun wipe() { calls += "wipe" }
        override fun setRetention(days: Int) { calls += "retention:$days" }
        override fun blockCurrentPlace() { calls += "block_place" }
    }

    private val answer = AnswerUi("where are my keys", "Probably the table — you put the keys there at 9:41 am. 87% confident.", false, 0.87,
        eventId = 12, eventT = 1000.0, objCls = 0, objName = "keys")

    @Test fun answerCardShowsConfidenceAndActions() {
        val a = Recorder()
        rule.setContent { MaterialTheme { RecallApp(UiState(answer = answer), a) } }
        rule.onNodeWithTag(Tags.ANSWER_TEXT).assertTextContains("Probably the table", substring = true)
        rule.onNodeWithTag(Tags.CONFIDENCE).assertTextContains("Confidence 87%")
        rule.onNodeWithText("No thumbnail stored", substring = true).assertIsDisplayed() // fail-closed message
        rule.onNodeWithTag(Tags.SHOW_MOMENT).performClick()
        rule.onNodeWithTag(Tags.FORGET).performClick()
        rule.onNodeWithTag(Tags.NEVER_TRACK).performClick()
        assertEquals(listOf("moment:1000.0", "forget:12", "never:0"), a.calls)
    }

    @Test fun abstentionHasNoConfidenceOrActions() {
        rule.setContent { MaterialTheme { RecallApp(UiState(answer = answer.copy(text = "I don't know — I haven't seen the keys in my memory.", abstained = true)), Recorder()) } }
        rule.onNodeWithTag(Tags.ANSWER_TEXT).assertTextContains("I don't know", substring = true)
        rule.onNodeWithTag(Tags.CONFIDENCE).assertDoesNotExist()
        rule.onNodeWithTag(Tags.FORGET).assertDoesNotExist()
    }

    @Test fun fallbackBannerDisablesStartAndAsk() {
        rule.setContent { MaterialTheme { RecallApp(UiState(fallbackReason = "Models not installed (missing gate.tflite)."), Recorder()) } }
        rule.onNodeWithText("Fallback mode", substring = true).assertIsDisplayed()
        rule.onNodeWithTag(Tags.REC_TOGGLE).assertIsNotEnabled()
        rule.onNodeWithTag(Tags.ASK).assertIsNotEnabled()
    }

    @Test fun recordingIndicatorAndToggle() {
        val a = Recorder()
        var s by mutableStateOf(UiState(recording = false))
        rule.setContent { MaterialTheme { RecallApp(s, a) } }
        rule.onNodeWithTag(Tags.REC_STATUS).assertTextContains("Paused")
        s = s.copy(recording = true, status = "Remembering (learned, backoff on)")
        rule.onNodeWithTag(Tags.REC_STATUS).assertTextContains("Remembering")
        rule.onNodeWithTag(Tags.REC_TOGGLE).assertTextContains("Pause").performClick()
        assertEquals(listOf("toggle"), a.calls)
    }

    @Test fun queryInputAndAsk() {
        val a = Recorder()
        rule.setContent { MaterialTheme { RecallApp(UiState(), a) } }
        rule.onNodeWithTag(Tags.QUERY).performTextInput("where are my keys")
        rule.onNodeWithTag(Tags.ASK).performClick()
        assertEquals(listOf("query:where are my keys", "ask"), a.calls)
    }

    @Test fun timelineHighlightsTheMoment() {
        val items = (0 until 5).map { TimelineItem(it.toLong(), 1000.0 + it * 60, "9:4$it am", "put", "put keys → table", null) }
        rule.setContent { MaterialTheme { RecallApp(UiState(tab = 1, timeline = items, focusT = 1120.0), Recorder()) } }
        rule.onNode(hasTestTag(Tags.FOCUSED) and hasAnyDescendant(hasText("9:42 am"))).assertIsDisplayed()
    }

    @Test fun privacyControls() {
        val a = Recorder()
        rule.setContent { MaterialTheme { RecallApp(UiState(tab = 2, eventCount = 3), a) } }
        rule.onNodeWithText("no network permission", substring = true).assertIsDisplayed()
        rule.onNodeWithTag("retention_30").performClick()
        rule.onNodeWithTag(Tags.BLOCK_PLACE).performClick()
        rule.onNodeWithTag(Tags.WIPE).performClick() // asks for confirmation first
        rule.onNodeWithTag("wipe_confirm").performClick()
        assertEquals(listOf("retention:30", "block_place", "wipe"), a.calls)
    }
}
