package io.github.luilan.stepstone

import android.graphics.Bitmap
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import io.github.luilan.stepstone.data.*
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import java.io.File
import kotlin.random.Random

/**
 * Drives the real app against the published catalog: download a pack, fail a concept gate, retry, pass,
 * answer every question type through to "lesson complete", then check the saved progress.
 * Screenshots go to app/build/ui-shots/. Skipped when offline.
 */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [35], qualifiers = "w411dp-h891dp-xxhdpi")
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class LessonFlowTest {
    @get:Rule val rule = createAndroidComposeRule<MainActivity>()
    private val ctx get() = ApplicationProvider.getApplicationContext<android.content.Context>()
    private val shots = File("build/ui-shots").apply { mkdirs() }
    private var n = 0

    private fun shot(name: String) {
        rule.waitForIdle()
        var bmp: Bitmap? = null
        rule.runOnUiThread {                                         // draw the window straight into a bitmap
            val v = rule.activity.window.decorView
            bmp = Bitmap.createBitmap(v.width, v.height, Bitmap.Config.ARGB_8888).also { v.draw(android.graphics.Canvas(it)) }
        }
        File(shots, "%02d_%s.png".format(++n, name)).outputStream().use { bmp!!.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }

    private fun waitText(text: String, substring: Boolean = true, timeout: Long = 120_000) {
        try {
            rule.waitUntil(timeout) { rule.onAllNodesWithText(text, substring = substring).fetchSemanticsNodes().isNotEmpty() }
        } catch (e: Throwable) {
            runCatching { shot("FAILED_waiting") }
            throw AssertionError("'$text' never appeared. Screen:\n" + rule.onRoot(useUnmergedTree = true).printToString(), e)
        }
    }

    private fun click(text: String, substring: Boolean = true, index: Int = 0) {
        waitText(text, substring)
        val node = rule.onAllNodesWithText(text, substring = substring)[index]
        runCatching { node.performScrollTo() }                     // fixed footers have no scroll parent
        node.performClick()
        rule.waitForIdle()
    }

    private fun clickNoScroll(text: String, substring: Boolean = true, index: Int = 0) {
        waitText(text, substring)
        rule.onAllNodesWithText(text, substring = substring)[index].performClick()
        rule.waitForIdle()
    }

    private fun plain(html: String) = html.replace(Regex("<[^>]+>"), "")
        .replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'").replace("&amp;", "&")

    private fun fmtNum(v: Double) = if (v == Math.floor(v) && Math.abs(v) < 1e15) "%.0f".format(v) else v.toString()

    /** Answer one question on screen; right = false gives a wrong answer where the type allows. */
    private fun answer(e: Exercise, right: Boolean) {
        when (val k = e.key) {
            is Key.Choice -> {
                val i = if (right) k.index else (k.index + 1) % e.options.size
                click(plain(e.options[i]), substring = false)
                click("Check", substring = false)
            }
            is Key.TrueFalse -> {
                click(if (k.value == right) "True" else "False", substring = false)
                click("Check", substring = false)
            }
            is Key.Numbers -> {
                val fields = rule.onAllNodes(hasSetTextAction())
                k.parts.forEachIndexed { i, p -> fields[i].also { runCatching { it.performScrollTo() } }.performTextInput(fmtNum(if (right) p.value else p.value * 3 + 7)) }
                click("Check", substring = false)
            }
            is Key.Order -> {
                // same shuffle as the app, then bubble each item up with its ▲ arrow
                val r = Random(e.id.hashCode())
                var cur = k.items.shuffled(r)
                var guard = 0
                while (cur == k.items && k.items.size > 1 && guard++ < 10) cur = k.items.shuffled(r)
                val list = cur.toMutableList()
                if (right) for (target in k.items.indices) {
                    var at = list.indexOf(k.items[target])
                    while (at > target) {
                        rule.onAllNodesWithText("▲")[at].also { runCatching { it.performScrollTo() } }.performClick(); rule.waitForIdle()
                        list.add(at - 1, list.removeAt(at)); at--
                    }
                }
                click("Check", substring = false)
            }
            null -> {
                click("Show the model answer")
                click(if (right) "✓ Got it" else "✗ Missed")
                return                                                // self-graded questions move on by themselves
            }
        }
        val next = rule.onAllNodesWithText("Next question").fetchSemanticsNodes().isNotEmpty()
        click(if (next) "Next question" else "Finish", substring = false)
    }

    private fun lesson(pack: String, id: String): Lesson =
        Parse.lesson(File(ctx.filesDir, "packs/$pack/lessons/$id.json").readText())

    @Test fun downloadStudyGatePassAndFinish() {
        assumeTrue("no network", runCatching { Http.get(BuildConfig.CATALOG_URL) }.isSuccess)
        waitText("How LLMs Work: Foundations")
        shot("library")
        // the second "Download" button belongs to How LLMs Work (catalog order: foundations, how-llms-work, ...)
        clickNoScroll("Download", substring = false, index = 1)
        waitText("0 of 14 lessons", timeout = 180_000)
        shot("library_installed")

        click("How LLMs Work", substring = false)
        rule.onNode(hasScrollToNodeAction()).performScrollToNode(hasText("Attention I: Tokens Talking to Each Other"))
        shot("series")
        click("Attention I: Tokens Talking to Each Other")
        shot("lesson_start")
        click("Start lesson")
        val l = lesson("how-llms-work", "v05")
        shot("concept1")

        // concept 1: fail the gate on purpose (1 of 3 right), see "Not yet", retry only the missed, pass
        click("Check your understanding")
        l.concepts[0].exercises.forEachIndexed { i, e -> answer(e, right = i == l.concepts[0].exercises.lastIndex) }
        waitText("Not yet")
        shot("gate_not_yet")
        click("Retry missed")
        val missed = l.concepts[0].exercises.dropLast(1)
        missed.forEach { answer(it, right = true) }
        waitText("Stone laid")
        shot("stone_laid")

        // remaining concepts: everything right, screenshots of the question types along the way
        for (ci in 1 until l.concepts.size) {
            click("Continue · concept ${ci + 1}")
            if (ci == 1) shot("concept2_reading")
            click("Check your understanding")
            l.concepts[ci].exercises.forEachIndexed { ei, e ->
                if (ci == 1 && ei == 0) {                                  // show a graded number answer
                    rule.onAllNodes(hasSetTextAction())[0].performTextInput("0.28")
                    click("Check", substring = false); shot("number_right"); click("Next question", substring = false)
                } else if (e.key is Key.Order) {
                    shot("order_question"); answer(e, right = true)
                } else answer(e, right = true)
            }
            if (ci < l.concepts.lastIndex) waitText("Stone laid")
        }
        waitText("COMPLETE")
        shot("lesson_complete")

        // saved progress: one finished attempt, every concept passed, the concept-1 misses queued for review
        val p = JSONObject(File(ctx.filesDir, "progress.json").readText())
        val attempts = p.getJSONObject("lessons").getJSONObject("how-llms-work/v05").getJSONArray("attempts")
        assertEquals(1, attempts.length())
        val a = attempts.getJSONObject(0)
        assertFalse(a.isNull("finished"))
        assertEquals(l.concepts.size, a.getJSONArray("passed").length())
        assertTrue(p.getJSONObject("review").keys().asSequence().any { it.startsWith("how-llms-work/v05/c1e") })
        val first = a.getJSONObject("first")
        val total = l.concepts.sumOf { it.exercises.size }
        assertEquals(total, first.length())
        assertEquals((total - 2).toDouble(), first.keys().asSequence().sumOf { first.getDouble(it) }, 1e-9)

        // progress and review tabs
        rule.runOnUiThread { rule.activity.onBackPressedDispatcher.onBackPressed() }
        rule.runOnUiThread { rule.activity.onBackPressedDispatcher.onBackPressed() }
        rule.runOnUiThread { rule.activity.onBackPressedDispatcher.onBackPressed() }
        rule.waitForIdle()
        click("Progress", substring = false)
        waitText("lessons completed")
        shot("progress")
        click("Review", substring = false)
        waitText("Nothing due today")
        shot("review")
    }
}
