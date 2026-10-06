package io.github.luilan.stepstone.data

import org.json.JSONArray
import org.json.JSONObject

/** Content format version this app understands (see stepstone/FORMAT.md). */
const val SUPPORTED_FORMAT = 1

data class AppRelease(val versionCode: Int, val versionName: String, val url: String)

data class CatalogPack(
    val id: String, val title: String, val description: String, val subject: String, val order: Int,
    val requires: List<String>, val version: Int, val file: String, val url: String, val size: Long,
    val sha256: String, val lessons: Int, val contentHash: String,
)

data class Catalog(val format: Int, val packs: List<CatalogPack>, val app: AppRelease?)

data class OfflineVideo(val url: String, val size: Long, val sha256: String)
data class Video(val youtube: String?, val offline: OfflineVideo?)

data class LessonRef(
    val id: String, val number: Int, val title: String, val tagline: String, val duration: String,
    val file: String, val concepts: Int, val exercises: Int, val video: Video,
)

data class Manifest(
    val format: Int, val id: String, val title: String, val description: String, val subject: String,
    val order: Int, val requires: List<String>, val cover: String?, val contentHash: String, val lessons: List<LessonRef>,
)

data class Figure(val src: String, val caption: String, val t: Double)

sealed interface Block {
    data class Text(val html: String, val center: Boolean) : Block
    data class Callout(val style: String, val title: String?, val html: String) : Block
    data class Figures(val items: List<Figure>) : Block
    data class Table(val header: List<String>?, val rows: List<List<String>>) : Block
    data class Code(val language: String, val text: String) : Block
    data class Html(val html: String) : Block
    data class Unknown(val type: String) : Block
}

data class NumberPart(val label: String?, val value: Double, val tol: Double, val unit: String?)

sealed interface Key {
    data class Choice(val index: Int) : Key
    data class TrueFalse(val value: Boolean) : Key
    data class Order(val items: List<String>) : Key
    data class Numbers(val parts: List<NumberPart>) : Key
}

data class Exercise(
    val id: String, val kind: String, val prompt: List<Block>, val options: List<String>, val code: String?,
    val key: Key?, val answer: List<Block>, val why: List<Block>,
) {
    /** Self-graded when there is no usable key (short answers, code, keys marked self). */
    val selfGraded get() = key == null
}

data class Concept(
    val id: String, val title: String, val segment: Pair<Double, Double>, val blocks: List<Block>,
    val exercises: List<Exercise>,
)

data class Lesson(
    val id: String, val number: Int, val label: String, val title: String, val tagline: String, val duration: String,
    val intro: List<Block>, val prereq: List<Block>, val video: Video, val concepts: List<Concept>,
    val studyPdf: String?, val code: String?,
)

// ------------------------------------------------------------------------------------------------- parsing
private fun JSONObject.str(k: String) = if (isNull(k)) null else optString(k, null)
private fun JSONArray.strings() = List(length()) { getString(it) }
private fun JSONArray.objects() = List(length()) { getJSONObject(it) }

object Parse {
    fun catalog(text: String): Catalog {
        val o = JSONObject(text)
        val app = o.optJSONObject("app")?.let { AppRelease(it.getInt("version_code"), it.getString("version_name"), it.getString("url")) }
        val packs = o.getJSONArray("packs").objects().map {
            CatalogPack(
                it.getString("id"), it.getString("title"), it.optString("description"), it.optString("subject"),
                it.optInt("order"), it.optJSONArray("requires")?.strings() ?: emptyList(), it.getInt("version"),
                it.getString("file"), it.getString("url"), it.getLong("size"), it.getString("sha256"), it.optInt("lessons"),
                it.optString("content_hash"),
            )
        }
        return Catalog(o.getInt("format"), packs, app)
    }

    private fun video(o: JSONObject?): Video {
        if (o == null) return Video(null, null)
        val off = o.optJSONObject("offline")?.let { OfflineVideo(it.getString("url"), it.getLong("size"), it.getString("sha256")) }
        return Video(o.str("youtube"), off)
    }

    fun manifest(text: String): Manifest {
        val o = JSONObject(text)
        return Manifest(
            o.getInt("format"), o.getString("id"), o.getString("title"), o.optString("description"), o.optString("subject"),
            o.optInt("order"), o.optJSONArray("requires")?.strings() ?: emptyList(), o.str("cover"), o.optString("content_hash"),
            o.getJSONArray("lessons").objects().map {
                LessonRef(it.getString("id"), it.getInt("number"), it.getString("title"), it.optString("tagline"),
                    it.optString("duration"), it.getString("file"), it.optInt("concepts"), it.optInt("exercises"),
                    video(it.optJSONObject("video")))
            },
        )
    }

    fun blocks(a: JSONArray?): List<Block> = a?.objects()?.map { b ->
        when (b.optString("type")) {
            "text" -> Block.Text(b.getString("html"), b.optString("align") == "center")
            "callout" -> Block.Callout(b.optString("style", "note"), b.str("title"), b.getString("html"))
            "figures" -> Block.Figures(b.getJSONArray("items").objects().map {
                Figure(it.getString("src"), it.optString("caption"), it.optDouble("t", 0.0))
            })
            "table" -> Block.Table(b.optJSONArray("header")?.strings(), b.getJSONArray("rows").let { r ->
                List(r.length()) { r.getJSONArray(it).strings() }
            })
            "code" -> Block.Code(b.optString("language"), b.getString("text"))
            "html" -> Block.Html(b.getString("html"))
            else -> Block.Unknown(b.optString("type"))
        }
    } ?: emptyList()

    fun key(kind: String, grading: String, o: JSONObject?): Key? {
        if (grading != "auto" || o == null) return null
        return when (kind) {
            "mc" -> Key.Choice(o.getInt("choice"))
            "tf" -> Key.TrueFalse(o.getBoolean("value"))
            "order" -> Key.Order(o.getJSONArray("items").strings())
            "number" -> Key.Numbers(o.getJSONArray("parts").objects().map {
                NumberPart(it.str("label"), it.getDouble("value"), it.getDouble("tol"), it.str("unit"))
            })
            else -> null
        }
    }

    fun lesson(text: String): Lesson {
        val o = JSONObject(text)
        val links = o.optJSONObject("links")
        return Lesson(
            o.getString("id"), o.getInt("number"), o.optString("label"), o.getString("title"), o.optString("tagline"),
            o.optString("duration"), blocks(o.optJSONArray("intro")), blocks(o.optJSONArray("prereq")),
            video(o.optJSONObject("video")),
            o.getJSONArray("concepts").objects().map { c ->
                val seg = c.optJSONArray("segment")
                Concept(
                    c.getString("id"), c.getString("title"),
                    if (seg != null && seg.length() == 2) seg.getDouble(0) to seg.getDouble(1) else 0.0 to 0.0,
                    blocks(c.optJSONArray("blocks")),
                    c.getJSONArray("exercises").objects().map { e ->
                        val kind = e.getString("kind")
                        Exercise(
                            e.getString("id"), kind, blocks(e.optJSONArray("prompt")),
                            e.optJSONArray("options")?.strings() ?: emptyList(), e.str("code"),
                            runCatching { key(kind, e.optString("grading"), e.optJSONObject("key")) }.getOrNull(),
                            blocks(e.optJSONArray("answer")), blocks(e.optJSONArray("why")),
                        )
                    },
                )
            },
            links?.str("study_pdf"), links?.str("code"),
        )
    }
}
