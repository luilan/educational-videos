package io.github.luilan.stepstone.data

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.zip.ZipInputStream

/** Where the lesson catalog lives. Debug builds can point elsewhere (see app/build.gradle.kts). */
val CATALOG_URL: String = io.github.luilan.stepstone.BuildConfig.CATALOG_URL

/** Packs, lessons and videos on disk, and the network calls that fetch them. */
class Repo(context: Context) {
    private val root = context.filesDir
    private val packsDir = File(root, "packs").apply { mkdirs() }
    private val videosDir = File(root, "videos").apply { mkdirs() }
    private val catalogCache = File(root, "catalog.json")
    private val lessonCache = mutableMapOf<String, Lesson>()

    fun cachedCatalog(): Catalog? = runCatching { Parse.catalog(catalogCache.readText()) }.getOrNull()

    suspend fun fetchCatalog(): Catalog = withContext(Dispatchers.IO) {
        val text = Http.get("$CATALOG_URL?t=${System.currentTimeMillis() / 60000}").decodeToString()
        val c = Parse.catalog(text)                            // validate before caching
        catalogCache.writeText(text)
        c
    }

    fun installed(): List<Manifest> = packsDir.listFiles { f -> f.isDirectory && !f.name.endsWith(".tmp") }
        ?.mapNotNull { d -> runCatching { Parse.manifest(File(d, "manifest.json").readText()) }.getOrNull() }
        ?.sortedBy { it.order } ?: emptyList()

    fun file(pack: String, path: String) = File(File(packsDir, pack), path)

    fun lesson(pack: String, ref: LessonRef): Lesson =
        lessonCache.getOrPut("$pack/${ref.id}") { Parse.lesson(file(pack, ref.file).readText()) }

    /** Download, verify and unpack a pack; the old version is replaced only after the new one is complete. */
    suspend fun install(p: CatalogPack, progress: (Float) -> Unit) = withContext(Dispatchers.IO) {
        val zip = File(root, "${p.id}.download")
        Http.download(p.url, zip, p.size, progress)
        check(Http.sha256(zip) == p.sha256) { "download corrupted (checksum mismatch)" }
        val tmp = File(packsDir, "${p.id}.tmp").apply { deleteRecursively(); mkdirs() }
        ZipInputStream(zip.inputStream().buffered()).use { z ->
            generateSequence { z.nextEntry }.forEach { e ->
                val out = File(tmp, e.name).canonicalFile
                check(out.path.startsWith(tmp.canonicalPath + File.separator)) { "bad path in pack: ${e.name}" }
                if (e.isDirectory) out.mkdirs() else { out.parentFile?.mkdirs(); out.outputStream().use { z.copyTo(it) } }
            }
        }
        val m = Parse.manifest(File(tmp, "manifest.json").readText())
        check(m.format <= SUPPORTED_FORMAT) { "this pack needs a newer StepStone" }
        val dest = File(packsDir, p.id)
        val old = File(packsDir, "${p.id}.old")
        old.deleteRecursively()
        if (dest.exists()) dest.renameTo(old)
        tmp.renameTo(dest)
        old.deleteRecursively()
        zip.delete()
        lessonCache.keys.removeAll { it.startsWith("${p.id}/") }
    }

    fun remove(pack: String) {
        File(packsDir, pack).deleteRecursively(); File(videosDir, pack).deleteRecursively()
        lessonCache.keys.removeAll { it.startsWith("$pack/") }
    }

    // ------------------------------------------------------------------------------------------ offline videos
    fun videoFile(pack: String, lesson: String) = File(File(videosDir, pack), "$lesson.mp4")
    fun hasVideo(pack: String, lesson: String) = videoFile(pack, lesson).exists()
    fun videoBytes(pack: String) = File(videosDir, pack).listFiles()?.sumOf { it.length() } ?: 0L

    suspend fun downloadVideo(pack: String, ref: LessonRef, progress: (Float) -> Unit) = withContext(Dispatchers.IO) {
        val v = ref.video.offline ?: throw IOException("no offline video for ${ref.id}")
        val dest = videoFile(pack, ref.id).apply { parentFile?.mkdirs() }
        val part = File(dest.path + ".part")
        Http.download(v.url, part, v.size, progress)
        check(Http.sha256(part) == v.sha256) { "video corrupted (checksum mismatch)" }
        part.renameTo(dest)
    }

    fun deleteVideos(pack: String) { File(videosDir, pack).deleteRecursively() }
}

/** Plain HTTP helpers (no Android dependencies, so they can be tested on the JVM). */
object Http {
    fun open(url: String): HttpURLConnection {
        var u = URL(url)
        repeat(5) {                                              // follow redirects (GitHub → its file CDN)
            val c = (u.openConnection() as HttpURLConnection).apply {
                connectTimeout = 15000; readTimeout = 30000; instanceFollowRedirects = false
                setRequestProperty("User-Agent", "StepStone")
            }
            when (c.responseCode) {
                in 300..399 -> { u = URL(u, c.getHeaderField("Location")); c.disconnect() }
                in 200..299 -> return c
                else -> throw IOException("HTTP ${c.responseCode} for $u")
            }
        }
        throw IOException("too many redirects")
    }

    fun get(url: String): ByteArray = open(url).let { c -> c.inputStream.use { it.readBytes() }.also { c.disconnect() } }

    fun download(url: String, dest: File, size: Long, progress: (Float) -> Unit) {
        val c = open(url)
        try {
            var done = 0L
            var last = -1
            c.inputStream.use { inp ->
                dest.outputStream().use { out ->
                    val buf = ByteArray(1 shl 16)
                    while (true) {
                        val n = inp.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        done += n
                        val pct = if (size > 0) (100 * done / size).toInt() else 0
                        if (pct != last) { last = pct; progress(pct / 100f) }
                    }
                }
            }
        } finally { c.disconnect() }
    }

    fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { inp -> val b = ByteArray(1 shl 16); while (true) { val n = inp.read(b); if (n < 0) break; md.update(b, 0, n) } }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
