package io.github.luilan.stepstone.ui

import android.content.Context
import androidx.compose.runtime.*
import io.github.luilan.stepstone.BuildConfig
import io.github.luilan.stepstone.data.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import java.io.File

enum class Tab { LIBRARY, REVIEW, PROGRESS }

data class QItem(val pack: String, val lesson: String, val ex: Exercise)

sealed interface Screen {
    data class Home(val tab: Tab) : Screen
    data class Series(val pack: String) : Screen
    data class LessonStart(val pack: String, val lesson: String) : Screen
    data class ConceptView(val pack: String, val lesson: String, val idx: Int) : Screen
    /** concept >= 0: the gate quiz of that concept (counts in the attempt); -1: practice/review (feeds review only). */
    data class Quiz(val items: List<QItem>, val concept: Int, val title: String) : Screen
    data class Gate(val pack: String, val lesson: String, val idx: Int) : Screen
    data class Passed(val pack: String, val lesson: String, val idx: Int) : Screen
    data class LessonDone(val pack: String, val lesson: String) : Screen
    data class Player(val pack: String, val lesson: String, val start: Double) : Screen
}

/** App-wide state: catalog, installed packs, downloads, progress, navigation. */
class AppState(context: Context, val scope: CoroutineScope) {
    val repo = Repo(context)
    val progress = ProgressStore(File(context.filesDir, "progress.json"))

    var catalog by mutableStateOf(repo.cachedCatalog())
    var installed by mutableStateOf(repo.installed())
    var checking by mutableStateOf(false)
    var lastCheck by mutableStateOf<Long?>(null)
    var message by mutableStateOf<String?>(null)
    val downloads = mutableStateMapOf<String, Float>()          // pack id or "video:<pack>/<lesson>" → 0..1
    var tick by mutableIntStateOf(0)                            // bumped when progress changes
    val stack = mutableStateListOf<Screen>(Screen.Home(Tab.LIBRARY))

    val screen get() = stack.last()
    fun go(s: Screen) { stack.add(s) }
    fun back(): Boolean = if (stack.size > 1) { stack.removeAt(stack.lastIndex); true } else false
    fun replace(s: Screen) { stack[stack.lastIndex] = s }
    fun home(tab: Tab) { stack.clear(); stack.add(Screen.Home(tab)) }
    /** Pop back to the given lesson's start screen (or push it). */
    fun toLesson(pack: String, lesson: String) {
        val i = stack.indexOfLast { it is Screen.LessonStart && it.pack == pack && it.lesson == lesson }
        if (i >= 0) while (stack.lastIndex > i) stack.removeAt(stack.lastIndex) else go(Screen.LessonStart(pack, lesson))
    }
    fun changed() { tick++ }

    fun manifest(pack: String) = installed.firstOrNull { it.id == pack }
    fun ref(pack: String, lesson: String) = manifest(pack)?.lessons?.firstOrNull { it.id == lesson }
    fun lesson(pack: String, lesson: String): Lesson? = ref(pack, lesson)?.let { runCatching { repo.lesson(pack, it) }.getOrNull() }
    fun packDir(pack: String) = repo.file(pack, "")

    val appUpdate get() = catalog?.app?.takeIf { it.versionCode > BuildConfig.VERSION_CODE }

    fun refresh() {
        if (checking) return
        checking = true
        scope.launch {
            runCatching { repo.fetchCatalog() }
                .onSuccess { catalog = it; lastCheck = System.currentTimeMillis() }
                .onFailure { message = "Couldn't reach the catalog (${it.message ?: "offline"}). Your downloaded lessons still work." }
            checking = false
        }
    }

    fun install(p: CatalogPack) {
        if (p.id in downloads) return
        downloads[p.id] = 0f
        scope.launch {
            runCatching { repo.install(p) { downloads[p.id] = it } }
                .onSuccess { installed = repo.installed(); message = "${p.title} is ready." }
                .onFailure { message = "Download failed: ${it.message}" }
            downloads.remove(p.id)
        }
    }

    fun remove(pack: String) { repo.remove(pack); installed = repo.installed(); progress.offlineSeries.remove(pack); progress.save() }

    // -------------------------------------------------------------------------------------- offline videos
    fun videoKey(pack: String, lesson: String) = "video:$pack/$lesson"

    fun downloadVideos(pack: String, lessons: List<LessonRef>) {
        scope.launch {
            for (ref in lessons) {
                if (ref.video.offline == null || repo.hasVideo(pack, ref.id)) continue
                if (pack !in progress.offlineSeries && lessons.size > 1) break     // switched off meanwhile
                val k = videoKey(pack, ref.id)
                downloads[k] = 0f
                val r = runCatching { repo.downloadVideo(pack, ref) { downloads[k] = it } }
                downloads.remove(k)
                if (r.isFailure) { message = "Video download stopped: ${r.exceptionOrNull()?.message}"; break }
                changed()
            }
        }
    }

    fun setOffline(pack: String, on: Boolean) {
        val m = manifest(pack) ?: return
        if (on) { progress.offlineSeries.add(pack); progress.save(); downloadVideos(pack, m.lessons) }
        else { progress.offlineSeries.remove(pack); progress.save(); repo.deleteVideos(pack) }
        changed()
    }

    // -------------------------------------------------------------------------------------- lesson logic
    fun lp(pack: String, lesson: String) = progress.lesson(pack, lesson)

    /** Stone states for a lesson: within the current attempt, passed / first unpassed / locked. */
    fun stones(pack: String, lesson: Lesson): List<StoneState> {
        val p = lp(pack, lesson.id)
        val a = p.current ?: return lesson.concepts.map { if (p.completed) StoneState.DONE else StoneState.LOCKED }
            .let { if (!p.completed && it.isNotEmpty()) listOf(StoneState.NOW) + it.drop(1) else it }
        var nowGiven = false
        return lesson.concepts.map { c ->
            when {
                c.id in a.passed -> StoneState.DONE
                !nowGiven -> { nowGiven = true; StoneState.NOW }
                else -> StoneState.LOCKED
            }
        }
    }

    fun firstOpen(pack: String, lesson: Lesson): Int =
        stones(pack, lesson).indexOf(StoneState.NOW).let { if (it < 0) 0 else it }

    /** Questions still to answer for a concept's gate in the current attempt (everything not yet right). */
    fun gateItems(pack: String, lesson: Lesson, idx: Int): List<QItem> {
        val a = lp(pack, lesson.id).current
        return lesson.concepts[idx].exercises.filter { (a?.best?.get(it.id) ?: 0.0) < 1.0 }.map { QItem(pack, lesson.id, it) }
    }

    fun conceptPoints(pack: String, lesson: Lesson, idx: Int): Pair<Double, Int> {
        val a = lp(pack, lesson.id).current
        val ex = lesson.concepts[idx].exercises
        return ex.sumOf { a?.best?.get(it.id) ?: 0.0 } to ex.size
    }

    /** After a gate quiz: lay the stone if the concept now passes, then go to the right screen. */
    fun afterGate(pack: String, lesson: Lesson, idx: Int) {
        // the concept screens under this quiz are done with: back from what follows returns to the lesson page
        val top = stack.last()
        stack.removeAll { it is Screen.ConceptView && it.pack == pack && it.lesson == lesson.id }
        if (stack.last() != top) stack.add(top)
        val (pts, n) = conceptPoints(pack, lesson, idx)
        if (Grading.passed(listOf(pts), n)) {
            progress.passConcept(pack, lesson.id, lesson.concepts[idx].id)
            if (idx == lesson.concepts.lastIndex) {
                progress.finishAttempt(pack, lesson.id)
                replace(Screen.LessonDone(pack, lesson.id))
            } else replace(Screen.Passed(pack, lesson.id, idx))
        } else replace(Screen.Gate(pack, lesson.id, idx))
        changed()
    }

    /** First-try score of a finished attempt: (points, exercises). */
    fun attemptScore(lesson: Lesson, a: Attempt): Pair<Double, Int> {
        val ex = lesson.concepts.flatMap { it.exercises }
        return ex.sumOf { a.first[it.id] ?: 0.0 } to ex.size
    }

    fun bestScore(pack: String, lesson: Lesson): Double? =
        lp(pack, lesson.id).attempts.filter { it.finished != null }
            .maxOfOrNull { attemptScore(lesson, it).let { (p, n) -> if (n == 0) 1.0 else p / n } }

    fun dueItems(): List<QItem> = progress.dueReview().mapNotNull { key ->
        val (pack, lesson, ex) = key.split("/").takeIf { it.size == 3 } ?: return@mapNotNull null
        lesson(pack, lesson)?.concepts?.flatMap { it.exercises }?.firstOrNull { it.id == ex }?.let { QItem(pack, lesson, it) }
    }
}
