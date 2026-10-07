package io.github.luilan.stepstone.ui

import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.luilan.stepstone.BuildConfig
import io.github.luilan.stepstone.data.CATALOG_URL
import io.github.luilan.stepstone.data.CatalogPack
import io.github.luilan.stepstone.data.Manifest
import java.io.File

@Composable
fun HomeScreen(s: AppState, tab: Tab) {
    Column(Modifier.fillMaxSize()) {
        Box(Modifier.weight(1f)) {
            when (tab) {
                Tab.LIBRARY -> LibraryTab(s)
                Tab.REVIEW -> ReviewTab(s)
                Tab.PROGRESS -> ProgressTab(s)
            }
        }
        NavigationBar(containerColor = C.surface, tonalElevation = 0.dp) {
            listOf(Tab.LIBRARY to ("◧" to "Library"), Tab.REVIEW to ("↻" to "Review"), Tab.PROGRESS to ("▤" to "Progress")).forEach { (t, l) ->
                NavigationBarItem(selected = t == tab, onClick = { s.replace(Screen.Home(t)) },
                    icon = { Text(l.first, fontSize = 18.sp) }, label = { Text(l.second) },
                    colors = NavigationBarItemDefaults.colors(selectedIconColor = C.accent, selectedTextColor = C.accent,
                        indicatorColor = C.accent.copy(alpha = 0.14f), unselectedIconColor = C.muted, unselectedTextColor = C.muted))
            }
        }
    }
}

@Composable
private fun Header(title: @Composable () -> Unit, action: (@Composable () -> Unit)? = null) {
    Row(Modifier.fillMaxWidth().padding(start = 16.dp, end = 8.dp, top = 14.dp, bottom = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.weight(1f)) { title() }
        action?.invoke()
    }
}

const val TAGLINE = "Learn one concept at a time"

@Composable
fun Brand() = Text(androidx.compose.ui.text.buildAnnotatedString {
    append("Step"); pushStyle(androidx.compose.ui.text.SpanStyle(color = C.accent)); append("Stone"); pop()
}, fontFamily = Serif, fontWeight = FontWeight.SemiBold, fontSize = 23.sp, color = C.ink)

@Composable
fun Cover(file: File?, size: Int = 62) {
    val bmp = remember(file) { file?.takeIf { it.exists() }?.let { runCatching { BitmapFactory.decodeFile(it.path)?.asImageBitmap() }.getOrNull() } }
    if (bmp != null) Image(bmp, null, Modifier.size(size.dp).clip(RoundedCornerShape(12.dp)), contentScale = ContentScale.Crop)
    else Box(Modifier.size(size.dp).clip(RoundedCornerShape(12.dp)).background(C.surface2), contentAlignment = Alignment.Center) {
        Text("🪨", fontSize = 24.sp)
    }
}

// ------------------------------------------------------------------------------------------------ library
@Composable
private fun LibraryTab(s: AppState) {
    val ctx = LocalContext.current
    s.tick
    val cat = s.catalog
    val installed = s.installed
    val available = cat?.packs?.filter { p -> installed.none { it.id == p.id } } ?: emptyList()
    Column(Modifier.fillMaxSize()) {
        Header({
            Column {
                Brand()
                Text(TAGLINE, color = C.muted, fontSize = 12.5.sp)
            }
        }) {
            if (s.checking) CircularProgressIndicator(Modifier.size(22.dp).padding(2.dp), strokeWidth = 2.dp, color = C.accent)
            else TextButton(onClick = { s.refresh() }) { Text("⟳ Check", color = C.ink2) }
        }
        val listState = rememberLazyListState()
        LaunchedEffect(installed.size) { listState.animateScrollToItem(0) }   // a new install appears at the top
        LazyColumn(Modifier.fillMaxSize().padding(horizontal = 16.dp), state = listState, verticalArrangement = Arrangement.spacedBy(10.dp)) {
            s.appUpdate?.let { u ->
                item {
                    CardBox(border = C.gold.copy(alpha = 0.5f), onClick = { ctx.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(u.url))) }) {
                        Text("StepStone ${u.versionName} is available", fontWeight = FontWeight.SemiBold, color = C.gold)
                        Text("Tap to download the new version (you have ${BuildConfig.VERSION_NAME}).", color = C.ink2, fontSize = 13.sp)
                    }
                }
            }
            if (installed.isEmpty() && available.isEmpty()) item {
                CardBox {
                    Text("Welcome to StepStone", fontWeight = FontWeight.SemiBold)
                    Text("$TAGLINE: each concept unlocks when you've mastered the one before it.", color = C.ink2, fontSize = 14.sp)
                    Text(if (s.checking) "Looking for lessons…" else "Tap ⟳ Check to load the lesson catalog.", color = C.ink2, fontSize = 14.sp)
                }
            }
            items(installed, key = { it.id }) { m -> InstalledCard(s, m, cat?.packs?.firstOrNull { it.id == m.id }) }
            items(available, key = { "new-" + it.id }) { p -> AvailableCard(s, p) }
            item {
                Text(s.lastCheck?.let { "Catalog checked ${ago(it)}" } ?: if (cat != null) "Showing the saved catalog" else "",
                    color = C.muted, fontSize = 12.sp, modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp),
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center)
            }
        }
    }
}

private fun ago(t: Long): String {
    val m = (System.currentTimeMillis() - t) / 60000
    return if (m < 1) "just now" else if (m < 60) "$m min ago" else "${m / 60} h ago"
}

@Composable
private fun InstalledCard(s: AppState, m: Manifest, remote: CatalogPack?) {
    val done = m.lessons.count { s.lp(m.id, it.id).completed }
    val inProgress = m.lessons.firstOrNull { s.lp(m.id, it.id).current != null }
    val update = remote != null && remote.contentHash.isNotEmpty() && remote.contentHash != m.contentHash
    val dl = s.downloads[m.id]
    CardBox(border = if (update) C.gold.copy(alpha = 0.5f) else C.line, onClick = { s.go(Screen.Series(m.id)) }) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Cover(m.cover?.let { s.repo.file(m.id, it) })
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                if (update) Badge("Updated")
                Text(m.title, fontWeight = FontWeight.SemiBold, fontSize = 15.sp)
                Text("${m.lessons.size} lessons · ${m.subject}", color = C.muted, fontSize = 12.sp)
                Meter(if (m.lessons.isEmpty()) 0f else done / m.lessons.size.toFloat(), Modifier.padding(vertical = 6.dp))
                Text("$done of ${m.lessons.size} lessons" + (inProgress?.let { " · on lesson ${it.number}" } ?: ""),
                    color = C.muted, fontSize = 12.sp)
            }
        }
        if (update && remote != null) {
            Spacer(Modifier.height(8.dp))
            if (dl != null) LinearProgressIndicator(progress = { dl }, Modifier.fillMaxWidth(), color = C.accent)
            else BigButton("Update (${mb(remote.size)})", kind = "gold") { s.install(remote) }
        }
    }
}

@Composable
private fun AvailableCard(s: AppState, p: CatalogPack) {
    val dl = s.downloads[p.id]
    CardBox(border = C.gold.copy(alpha = 0.5f)) {
        Badge("New")
        Text(p.title, fontWeight = FontWeight.SemiBold, fontSize = 15.sp)
        Text("${p.lessons} lessons · ${mb(p.size)}" + (p.requires.firstOrNull()?.let { r ->
            " · start with " + (s.catalog?.packs?.firstOrNull { it.id == r }?.title ?: r) } ?: ""), color = C.muted, fontSize = 12.sp)
        if (p.description.isNotBlank()) Text(p.description, color = C.ink2, fontSize = 13.sp, modifier = Modifier.padding(top = 4.dp))
        Spacer(Modifier.height(10.dp))
        if (dl != null) {
            LinearProgressIndicator(progress = { dl }, Modifier.fillMaxWidth(), color = C.accent)
            Text("Downloading… ${pct(dl.toDouble())}", color = C.muted, fontSize = 12.sp)
        } else BigButton("Download") { s.install(p) }
    }
}

@Composable
fun Badge(text: String) = Text(text.uppercase(), fontSize = 10.5.sp, fontWeight = FontWeight.Bold, color = C.goldInk,
    modifier = Modifier.padding(bottom = 4.dp).clip(RoundedCornerShape(6.dp)).background(C.gold).padding(horizontal = 7.dp, vertical = 2.dp))

fun mb(bytes: Long) = if (bytes >= 1_000_000) "%.0f MB".format(bytes / 1e6) else "%.1f MB".format(bytes / 1e6)

// ------------------------------------------------------------------------------------------------ review
@Composable
private fun ReviewTab(s: AppState) {
    s.tick
    val due = remember(s.tick) { s.dueItems() }
    val upcoming = remember(s.tick) { s.progress.review.values.filter { it.due > s.progress.today() }.minOfOrNull { it.due } }
    Column(Modifier.fillMaxSize()) {
        Header({ Title("Review", 22) })
        Column(Modifier.padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            CardBox {
                Text(if (due.isEmpty()) "Nothing due today" else "${due.size} question${if (due.size == 1) "" else "s"} due",
                    fontFamily = Serif, fontSize = 22.sp, fontWeight = FontWeight.SemiBold)
                Text("Questions you missed come back after 1 day, then 3, then 7. Get one right three times in a row and it retires.",
                    color = C.ink2, fontSize = 13.sp)
                upcoming?.let { Text("Next one due in ${it - s.progress.today()} day(s).", color = C.muted, fontSize = 12.sp) }
            }
            due.groupBy { it.pack to it.lesson }.forEach { (k, v) ->
                val title = s.ref(k.first, k.second)?.title ?: k.second
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text(title, Modifier.weight(1f), color = C.ink2, fontSize = 14.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text("${v.size}", color = C.ink2, fontSize = 14.sp)
                }
            }
            if (due.isNotEmpty()) BigButton("Review ${due.size} question${if (due.size == 1) "" else "s"}") {
                s.go(Screen.Quiz(due, -1, "Review"))
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ progress
@Composable
private fun ProgressTab(s: AppState) {
    val ctx = LocalContext.current
    s.tick
    data class Row1(val label: String, val score: Double)
    val rows = remember(s.tick, s.installed) {
        s.installed.flatMap { m ->
            m.lessons.mapNotNull { r -> s.lesson(m.id, r.id)?.let { l -> s.bestScore(m.id, l)?.let { Row1(r.id.uppercase(), it) } } }
        }
    }
    val total = s.installed.sumOf { it.lessons.size }
    val weak = remember(s.tick, s.installed) {
        s.installed.flatMap { m ->
            m.lessons.flatMap { r ->
                val l = s.lesson(m.id, r.id) ?: return@flatMap emptyList()
                val a = s.lp(m.id, r.id).attempts.lastOrNull { it.finished != null } ?: return@flatMap emptyList()
                l.concepts.map { c -> "${c.title} · ${l.title}" to c.exercises.map { a.first[it.id] ?: 0.0 }.average() }
            }
        }.filter { it.second < 0.8 }.sortedBy { it.second }.take(5)
    }
    val export = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        uri ?: return@rememberLauncherForActivityResult
        runCatching { ctx.contentResolver.openOutputStream(uri)?.use { it.write(s.progress.toJson().toString(1).toByteArray()) } }
            .onSuccess { s.message = "Progress exported." }.onFailure { s.message = "Export failed: ${it.message}" }
    }
    val import = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@rememberLauncherForActivityResult
        runCatching { ctx.contentResolver.openInputStream(uri)?.use { s.progress.importFrom(it.readBytes().decodeToString()) } }
            .onSuccess { s.message = "Progress restored."; s.changed() }.onFailure { s.message = "That file isn't a StepStone backup." }
    }
    LazyColumn(Modifier.fillMaxSize().padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { Header({ Title("Progress", 22) }) }
        item {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Tile("${rows.size} / $total", "lessons completed", Modifier.weight(1f))
                    Tile(if (rows.isEmpty()) "–" else pct(rows.map { it.score }.average()), "average best score", Modifier.weight(1f))
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Tile("${s.progress.dueReview().size}", "questions due for review", Modifier.weight(1f))
                    Tile("${s.progress.streak()} day${if (s.progress.streak() == 1) "" else "s"}", "study streak", Modifier.weight(1f))
                }
            }
        }
        if (rows.isNotEmpty()) item {
            Label("Best score per completed lesson (tap a bar)")
            ScoreChart(rows.map { it.label to it.score })
        }
        if (weak.isNotEmpty()) item {
            CardBox {
                Label("Weakest concepts (last attempt)")
                weak.forEach { (n, v) ->
                    Row(Modifier.padding(top = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                        Text(n, Modifier.weight(1f), fontSize = 12.5.sp, color = C.ink2, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Meter(v.toFloat(), Modifier.width(70.dp).padding(horizontal = 8.dp))
                        Text(pct(v), fontSize = 12.5.sp, color = C.ink2)
                    }
                }
            }
        }
        item {
            CardBox {
                Label("Backup")
                Text("Your progress lives only on this phone. Export it to a file to keep it safe or move it to another phone.",
                    color = C.ink2, fontSize = 13.sp)
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    BigButton("Export", Modifier.weight(1f), kind = "ghost") { export.launch("stepstone-progress.json") }
                    BigButton("Import", Modifier.weight(1f), kind = "ghost") { import.launch(arrayOf("application/json", "*/*")) }
                }
            }
        }
        item {
            Text("StepStone ${BuildConfig.VERSION_NAME} · lessons from $CATALOG_URL", color = C.muted, fontSize = 11.sp,
                modifier = Modifier.padding(vertical = 10.dp))
        }
    }
}

@Composable
private fun Tile(value: String, label: String, modifier: Modifier) {
    Surface(modifier, color = C.surface, shape = RoundedCornerShape(14.dp), border = androidx.compose.foundation.BorderStroke(1.dp, C.line)) {
        Column(Modifier.padding(horizontal = 12.dp, vertical = 10.dp)) {
            Text(value, fontFamily = Serif, fontWeight = FontWeight.SemiBold, fontSize = 24.sp)
            Text(label, color = C.muted, fontSize = 11.5.sp)
        }
    }
}

/** One series: best score per completed lesson. Thin bars, rounded tops, recessive grid, tap for the value. */
@Composable
private fun ScoreChart(data: List<Pair<String, Double>>) {
    var sel by remember { mutableStateOf<Int?>(null) }
    Column {
        Text(sel?.let { "${data[it].first}: ${pct(data[it].second)} best score" } ?: " ", color = C.ink, fontSize = 13.sp)
        Canvas(Modifier.fillMaxWidth().height(150.dp).pointerInput(data) {
            detectTapGestures { o ->
                val left = 30.dp.toPx(); val bw = minOf((size.width - left) / data.size, 28.dp.toPx())
                sel = ((o.x - left) / bw).toInt().takeIf { it in data.indices }
            }
        }) {
            val left = 30.dp.toPx(); val bottom = size.height - 4.dp.toPx(); val top = 6.dp.toPx()
            fun y(v: Double) = top + (bottom - top) * (1 - v).toFloat()
            listOf(0.5, 0.75, 1.0).forEach { v -> drawLine(C.line, Offset(left, y(v)), Offset(size.width, y(v)), 1f) }
            val bw = minOf((size.width - left) / data.size, 28.dp.toPx())      // thin bars, even with one lesson
            val gap = minOf(4.dp.toPx(), bw * 0.25f)
            data.forEachIndexed { i, (_, v) ->
                val x = left + i * bw + gap / 2
                drawRoundRect(if (sel == i) C.gold else C.accent, Offset(x, y(v)), Size(bw - gap, bottom - y(v)),
                    CornerRadius(minOf(4.dp.toPx(), (bw - gap) / 2)))
            }
        }
        Row(Modifier.fillMaxWidth().padding(start = 30.dp)) {
            Text(data.first().first, color = C.muted, fontSize = 10.sp, modifier = Modifier.weight(1f))
            if (data.size > 1) Text(data.last().first, color = C.muted, fontSize = 10.sp)
        }
    }
}
