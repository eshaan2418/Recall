package ai.recall

import ai.recall.service.RecallRuntime
import ai.recall.ui.MainActivity
import ai.recall.ui.Tags
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/** Whole-app smoke tests: launch, fallback labelling, tabs, state across activity recreation (rotation). */
@RunWith(AndroidJUnit4::class)
class MainActivityTest {
    @get:Rule val rule = createAndroidComposeRule<MainActivity>()

    @Test fun launchesAndLabelsFallbackWhenModelsMissing() {
        rule.onNodeWithTag(Tags.REC_STATUS).assertIsDisplayed()
        if (RecallRuntime.missingAssets(rule.activity).isNotEmpty()) {
            rule.waitUntil(5_000) { rule.onAllNodes(androidx.compose.ui.test.hasTestTag(Tags.FALLBACK)).fetchSemanticsNodes().isNotEmpty() }
            rule.onNodeWithText("Fallback mode", substring = true).assertIsDisplayed()
        }
        rule.onNodeWithTag("tab_1").performClick()
        rule.onNodeWithTag(Tags.TIMELINE).assertIsDisplayed()
        rule.onNodeWithTag("tab_2").performClick()
        rule.onNodeWithTag(Tags.WIPE).assertIsDisplayed()
    }

    @Test fun queryAndTabSurviveRecreation() {
        rule.onNodeWithTag(Tags.QUERY).performTextInput("where is my phone")
        rule.activityRule.scenario.recreate() // same path as a rotation / config change
        rule.onNodeWithTag(Tags.QUERY).assertTextContains("where is my phone")
        rule.onNodeWithTag("tab_2").performClick()
        rule.activityRule.scenario.recreate()
        rule.onNodeWithTag(Tags.WIPE).assertIsDisplayed()
    }
}
