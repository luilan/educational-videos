package io.github.luilan.stepstone

import io.github.luilan.stepstone.data.Http
import io.github.luilan.stepstone.data.Parse
import io.github.luilan.stepstone.data.SUPPORTED_FORMAT
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.util.zip.ZipFile

/** End-to-end against the published catalog (skipped when offline): redirects, checksums, parsing every lesson. */
class LiveCatalogTest {
    private val url = "https://luilan.github.io/educational-videos/stepstone/catalog.json"

    private fun online() = runCatching { Http.get(url) }.isSuccess

    @Test fun catalogPackAndLessonsParse() {
        assumeTrue("no network", online())
        val cat = Parse.catalog(Http.get(url).decodeToString())
        assertTrue(cat.packs.isNotEmpty())
        val p = cat.packs.minBy { it.size }                       // smallest pack keeps the test quick
        val f = File.createTempFile("pack", ".zip")
        var last = 0f
        Http.download(p.url, f, p.size) { last = it }             // GitHub release → CDN redirect
        assertEquals(p.size, f.length())
        assertEquals(p.sha256, Http.sha256(f))
        assertEquals(1f, last, 0.001f)
        ZipFile(f).use { z ->
            val m = Parse.manifest(z.getInputStream(z.getEntry("manifest.json")).readBytes().decodeToString())
            assertTrue(m.format <= SUPPORTED_FORMAT)
            assertEquals(p.lessons, m.lessons.size)
            var exercises = 0
            for (ref in m.lessons) {
                val l = Parse.lesson(z.getInputStream(z.getEntry(ref.file)).readBytes().decodeToString())
                assertEquals(ref.concepts, l.concepts.size)
                l.concepts.forEach { c ->
                    c.blocks.filterIsInstance<io.github.luilan.stepstone.data.Block.Figures>().flatMap { it.items }
                        .forEach { assertNotNull("missing ${it.src}", z.getEntry(it.src)) }
                    c.exercises.forEach { e -> if (e.kind in setOf("mc", "tf", "order")) assertNotNull("no key ${l.id}/${e.id}", e.key) }
                }
                exercises += l.concepts.sumOf { it.exercises.size }
            }
            assertTrue(exercises > 0)
        }
        f.delete()
    }

    @Test fun offlineVideoLinkResolves() {
        assumeTrue("no network", online())
        val cat = Parse.catalog(Http.get(url).decodeToString())
        val p = cat.packs.minBy { it.size }
        val f = File.createTempFile("pack", ".zip")
        Http.download(p.url, f, p.size) {}
        val v = ZipFile(f).use { z -> Parse.manifest(z.getInputStream(z.getEntry("manifest.json")).readBytes().decodeToString()) }
            .lessons.first().video.offline
        assertNotNull(v)
        val c = Http.open(v!!.url)                                // only the headers: the redirect must land on a 200
        assertEquals(200, c.responseCode)
        assertEquals(v.size, c.contentLengthLong)
        c.disconnect()
        f.delete()
    }
}
