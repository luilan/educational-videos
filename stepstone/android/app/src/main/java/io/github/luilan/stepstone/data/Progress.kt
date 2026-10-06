package io.github.luilan.stepstone.data

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.time.LocalDate

/** One pass through a lesson. Scores are per exercise id: first try in this attempt, and best so far (with retries). */
class Attempt(
    val started: Long,
    var finished: Long? = null,
    val first: MutableMap<String, Double> = mutableMapOf(),
    val best: MutableMap<String, Double> = mutableMapOf(),
    val passed: MutableSet<String> = mutableSetOf(),            // concept ids whose stone is laid
    val answers: MutableMap<String, String> = mutableMapOf(),   // last text typed for self-graded exercises
)

class LessonProgress(val attempts: MutableList<Attempt> = mutableListOf()) {
    val current get() = attempts.lastOrNull()?.takeIf { it.finished == null }
    val completed get() = attempts.any { it.finished != null }
}

/** Spaced review: box 1 → due in 1 day, box 2 → 3 days, box 3 → 7 days; right in box 3 retires the question. */
data class ReviewItem(val box: Int, val due: Long)

private val INTERVALS = listOf(1L, 3L, 7L)

/** Everything the learner has done, saved as one JSON file. Keys: "<pack>/<lesson>" and "<pack>/<lesson>/<exercise>". */
class ProgressStore(private val file: File) {
    val lessons = mutableMapOf<String, LessonProgress>()
    val review = mutableMapOf<String, ReviewItem>()
    val studyDays = sortedSetOf<Long>()
    val offlineSeries = mutableSetOf<String>()

    init { load() }

    fun lesson(pack: String, lesson: String) = lessons.getOrPut("$pack/$lesson") { LessonProgress() }

    fun today() = LocalDate.now().toEpochDay()

    fun startAttempt(pack: String, lesson: String): Attempt =
        Attempt(System.currentTimeMillis()).also { lesson(pack, lesson).attempts.add(it); save() }

    /** Record one answer inside the current attempt. Misses enter the review queue. */
    fun record(pack: String, lessonId: String, exerciseId: String, score: Double, text: String? = null) {
        val a = lesson(pack, lessonId).current ?: startAttempt(pack, lessonId)
        if (exerciseId !in a.first) a.first[exerciseId] = score
        a.best[exerciseId] = maxOf(score, a.best[exerciseId] ?: 0.0)
        if (text != null) a.answers[exerciseId] = text
        if (score < 1.0) schedule("$pack/$lessonId/$exerciseId", right = false)
        studyDays.add(today())
        save()
    }

    /** Answer given in a review or practice session (outside an attempt). */
    fun recordReview(pack: String, lessonId: String, exerciseId: String, score: Double) {
        schedule("$pack/$lessonId/$exerciseId", right = score >= 1.0)
        studyDays.add(today())
        save()
    }

    private fun schedule(key: String, right: Boolean) {
        val cur = review[key]
        if (!right) {
            review[key] = ReviewItem(1, today() + INTERVALS[0])
        } else if (cur != null) {
            val next = cur.box + 1
            if (next > INTERVALS.size) review.remove(key) else review[key] = ReviewItem(next, today() + INTERVALS[next - 1])
        }
    }

    fun passConcept(pack: String, lessonId: String, conceptId: String) {
        lesson(pack, lessonId).current?.passed?.add(conceptId); save()
    }

    fun finishAttempt(pack: String, lessonId: String) {
        lesson(pack, lessonId).current?.finished = System.currentTimeMillis(); save()
    }

    fun dueReview(): List<String> = review.filter { it.value.due <= today() }.keys.sorted()

    /** Days in a row with study, ending today (or yesterday, if nothing yet today). */
    fun streak(): Int {
        var d = if (today() in studyDays) today() else today() - 1
        var n = 0
        while (d in studyDays) { n++; d-- }
        return n
    }

    // ------------------------------------------------------------------------------------------ persistence
    private fun load() {
        if (!file.exists()) return
        runCatching {
            val o = JSONObject(file.readText())
            o.optJSONObject("lessons")?.let { ls ->
                ls.keys().forEach { k ->
                    val arr = ls.getJSONObject(k).getJSONArray("attempts")
                    lessons[k] = LessonProgress(MutableList(arr.length()) { i -> attempt(arr.getJSONObject(i)) })
                }
            }
            o.optJSONObject("review")?.let { r ->
                r.keys().forEach { k -> r.getJSONObject(k).let { review[k] = ReviewItem(it.getInt("box"), it.getLong("due")) } }
            }
            o.optJSONArray("study_days")?.let { a -> for (i in 0 until a.length()) studyDays.add(a.getLong(i)) }
            o.optJSONArray("offline_series")?.let { a -> for (i in 0 until a.length()) offlineSeries.add(a.getString(i)) }
        }
    }

    private fun scores(o: JSONObject?) = mutableMapOf<String, Double>().apply { o?.keys()?.forEach { put(it, o.getDouble(it)) } }

    private fun attempt(o: JSONObject) = Attempt(
        o.getLong("started"), if (o.isNull("finished")) null else o.optLong("finished"),
        scores(o.optJSONObject("first")), scores(o.optJSONObject("best")),
        o.optJSONArray("passed")?.let { a -> MutableList(a.length()) { a.getString(it) }.toMutableSet() } ?: mutableSetOf(),
        mutableMapOf<String, String>().apply { o.optJSONObject("answers")?.let { a -> a.keys().forEach { put(it, a.getString(it)) } } },
    )

    fun toJson(): JSONObject = JSONObject().apply {
        put("format", 1)
        put("lessons", JSONObject().apply {
            lessons.forEach { (k, lp) ->
                put(k, JSONObject().put("attempts", JSONArray().apply {
                    lp.attempts.forEach { a ->
                        put(JSONObject().apply {
                            put("started", a.started); put("finished", a.finished ?: JSONObject.NULL)
                            put("first", JSONObject(a.first as Map<*, *>)); put("best", JSONObject(a.best as Map<*, *>))
                            put("passed", JSONArray(a.passed.toList())); put("answers", JSONObject(a.answers as Map<*, *>))
                        })
                    }
                }))
            }
        })
        put("review", JSONObject().apply { review.forEach { (k, v) -> put(k, JSONObject().put("box", v.box).put("due", v.due)) } })
        put("study_days", JSONArray(studyDays.toList()))
        put("offline_series", JSONArray(offlineSeries.toList()))
    }

    fun save() {
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(toJson().toString())
        tmp.renameTo(file)
    }

    /** Replace everything with an exported backup (validated by parsing it first). */
    fun importFrom(text: String) {
        JSONObject(text)                                         // throws if not JSON
        file.writeText(text)
        lessons.clear(); review.clear(); studyDays.clear(); offlineSeries.clear()
        load()
    }
}
