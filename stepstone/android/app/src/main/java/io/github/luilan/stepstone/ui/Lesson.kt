package io.github.luilan.stepstone.ui

import android.content.Intent
import android.net.Uri
import android.widget.MediaController
import android.widget.VideoView
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import io.github.luilan.stepstone.data.Grading
import io.github.luilan.stepstone.data.Lesson
import io.github.luilan.stepstone.data.PASS_SHARE
import java.text.DateFormat
import java.util.Date

@Composable
fun TopBar(s: AppState, title: String, actions: @Composable RowScope.() -> Unit = {}) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
        IconButton(onClick = { s.back() }) { Text("←", fontSize = 20.sp, color = C.ink2) }
        Text(title, Modifier.weight(1f), fontWeight = FontWeight.SemiBold, fontSize = 16.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
        actions()
    }
}

@Composable
fun Page(s: AppState, title: String, foot: (@Composable ColumnScope.() -> Unit)? = null, actions: @Composable RowScope.() -> Unit = {},
         body: @Composable ColumnScope.() -> Unit) {
    Column(Modifier.fillMaxSize()) {
        TopBar(s, title, actions)
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(horizontal = 16.dp).padding(bottom = 16.dp), content = body)
        if (foot != null) {
            HorizontalDivider(color = C.line)
            Column(Modifier.padding(horizontal = 16.dp, vertical = 10.dp), verticalArrangement = Arrangement.spacedBy(8.dp), content = foot)
        }
    }
}

@Composable
fun Missing(s: AppState) = Page(s, "") { Text("This lesson isn't installed any more.", color = C.ink2) }

// ------------------------------------------------------------------------------------------------ series
@Composable
fun SeriesScreen(s: AppState, pack: String) {
    s.tick
    val m = s.manifest(pack) ?: return Missing(s)
    val offline = pack in s.progress.offlineSeries
    val videoTotal = m.lessons.sumOf { it.video.offline?.size ?: 0L }
    var confirmRemove by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxSize()) {
        TopBar(s, m.title) { TextButton(onClick = { confirmRemove = true }) { Text("Remove", color = C.muted) } }
        LazyColumn(Modifier.fillMaxSize().padding(horizontal = 16.dp)) {
            item {
                if (m.description.isNotBlank()) Text(m.description, color = C.ink2, fontSize = 14.sp, modifier = Modifier.padding(bottom = 10.dp))
                if (videoTotal > 0) CardBox {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text("Videos offline", fontWeight = FontWeight.SemiBold, fontSize = 14.sp)
                            val have = s.repo.videoBytes(pack)
                            Text(if (offline) "${mb(have)} of ${mb(videoTotal)} on this phone" else "${mb(videoTotal)} · plays without internet",
                                color = C.muted, fontSize = 12.sp)
                        }
                        Switch(offline, { s.setOffline(pack, it) }, colors = SwitchDefaults.colors(checkedTrackColor = C.accent))
                    }
                    s.downloads.entries.firstOrNull { it.key.startsWith("video:$pack/") }?.let { (k, v) ->
                        Text("Downloading ${k.substringAfterLast('/')}… ${pct(v.toDouble())}", color = C.muted, fontSize = 12.sp)
                        LinearProgressIndicator(progress = { v }, Modifier.fillMaxWidth().padding(top = 4.dp), color = C.accent)
                    }
                }
                Spacer(Modifier.height(6.dp))
            }
            items(m.lessons, key = { it.id }) { r ->
                val lp = s.lp(pack, r.id)
                val lesson = remember(r.id, s.tick) { if (lp.completed) s.lesson(pack, r.id) else null }
                val best = lesson?.let { s.bestScore(pack, it) }
                val cur = lp.current
                Row(Modifier.fillMaxWidth().clickable { s.go(Screen.LessonStart(pack, r.id)) }.padding(vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    val (bg, fg) = when { cur != null -> C.gold.copy(alpha = 0.14f) to C.gold; lp.completed -> C.accent to C.accentInk; else -> C.surface2 to C.muted }
                    Box(Modifier.size(width = 36.dp, height = 26.dp).clip(RoundedCornerShape(50)).background(bg), contentAlignment = Alignment.Center) {
                        Text("%02d".format(r.number), color = fg, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                    Spacer(Modifier.width(11.dp))
                    Column(Modifier.weight(1f)) {
                        Text(r.title, fontWeight = FontWeight.SemiBold, fontSize = 14.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        val st = when {
                            cur != null -> " · concept ${cur.passed.size + 1} of ${r.concepts}"
                            best != null -> " · best ${pct(best)}"
                            else -> ""
                        }
                        Text(r.duration + st + if (s.repo.hasVideo(pack, r.id)) " · ⤓" else "", color = C.muted, fontSize = 12.sp)
                    }
                    Text(if (cur != null) "▶" else if (lp.completed) "✓" else "", color = C.ink2)
                }
                HorizontalDivider(color = C.line)
            }
        }
    }
    if (confirmRemove) AlertDialog(onDismissRequest = { confirmRemove = false },
        title = { Text("Remove ${m.title}?") },
        text = { Text("Deletes the lessons and any offline videos from this phone. Your scores are kept, and you can download it again.") },
        confirmButton = { TextButton(onClick = { confirmRemove = false; s.remove(pack); s.back() }) { Text("Remove", color = C.bad) } },
        dismissButton = { TextButton(onClick = { confirmRemove = false }) { Text("Cancel") } })
}

// ------------------------------------------------------------------------------------------------ lesson start
@Composable
fun LessonStartScreen(s: AppState, pack: String, id: String) {
    s.tick
    val l = s.lesson(pack, id) ?: return Missing(s)
    val lp = s.lp(pack, id)
    val stones = s.stones(pack, l)
    val open = s.firstOpen(pack, l)
    val ctx = LocalContext.current
    val nEx = l.concepts.sumOf { it.exercises.size }
    Page(s, "", foot = {
        when {
            lp.current != null -> BigButton("Continue · concept ${open + 1}") { s.go(Screen.ConceptView(pack, id, open)) }
            lp.completed -> {
                BigButton("Retake lesson", kind = "ghost") { s.progress.startAttempt(pack, id); s.changed(); s.go(Screen.ConceptView(pack, id, 0)) }
                BigButton("See results") { s.go(Screen.LessonDone(pack, id)) }
            }
            else -> BigButton("Start lesson") { s.progress.startAttempt(pack, id); s.changed(); s.go(Screen.ConceptView(pack, id, 0)) }
        }
    }) {
        Label("${l.label.ifBlank { "Lesson ${l.number}" }} · ${l.duration} · ${l.concepts.size} concepts · $nEx questions")
        Spacer(Modifier.height(4.dp))
        Title(l.title, 23)
        Text(l.tagline, color = C.ink2, fontSize = 14.sp)
        Spacer(Modifier.height(10.dp))
        Blocks(l.intro, s.packDir(pack))
        if (l.prereq.isNotEmpty()) { Spacer(Modifier.height(8.dp)); Blocks(l.prereq, s.packDir(pack), textSize = 14) }
        Spacer(Modifier.height(14.dp))
        Label("Your path")
        Stones(stones) { i -> s.go(Screen.ConceptView(pack, id, i)) }
        Spacer(Modifier.height(6.dp))
        if (lp.current == null && !lp.completed) Text("Master one concept before the next: each stone unlocks when you pass the one before it (${pct(PASS_SHARE)} of its questions).",
            color = C.muted, fontSize = 12.5.sp)
        Spacer(Modifier.height(10.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Chip("▶ Watch the video") { s.go(Screen.Player(pack, id, 0.0)) }
            l.studyPdf?.let { u -> Chip("📄 Study guide PDF") { ctx.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(u))) } }
            l.code?.let { u -> Chip("{ } Code") { ctx.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(u))) } }
        }
    }
}

// ------------------------------------------------------------------------------------------------ concept
@Composable
fun ConceptScreen(s: AppState, pack: String, id: String, idx: Int) {
    s.tick
    val l = s.lesson(pack, id) ?: return Missing(s)
    val c = l.concepts[idx]
    val stones = s.stones(pack, l)
    val inAttempt = s.lp(pack, id).current != null
    val state = stones[idx]
    Page(s, l.title, foot = {
        when {
            state == StoneState.LOCKED -> BigButton("Locked: pass concept ${stones.indexOf(StoneState.NOW) + 1} first", enabled = false) {}
            inAttempt && state == StoneState.NOW -> {
                val todo = s.gateItems(pack, l, idx)
                BigButton("Check your understanding · ${todo.size} question${if (todo.size == 1) "" else "s"}") {
                    s.go(Screen.Quiz(todo, idx, "Concept ${idx + 1} · check"))
                }
            }
            else -> {
                if (idx < l.concepts.lastIndex && stones[idx + 1] != StoneState.LOCKED)
                    BigButton("Next concept") { s.replace(Screen.ConceptView(pack, id, idx + 1)) }
                BigButton("Practise these questions", kind = "ghost") {
                    s.go(Screen.Quiz(c.exercises.map { QItem(pack, id, it) }, -1, "Practice · ${c.title}"))
                }
            }
        }
    }) {
        Stones(stones) { i -> s.replace(Screen.ConceptView(pack, id, i)) }
        Label("Concept ${idx + 1} of ${l.concepts.size}")
        Title(c.title)
        Spacer(Modifier.height(6.dp))
        if (c.segment.second > c.segment.first) Chip("▶ Watch this part · ${mmss(c.segment.first)}–${mmss(c.segment.second)}", accent = true) {
            s.go(Screen.Player(pack, id, c.segment.first))
        }
        Spacer(Modifier.height(8.dp))
        if (state == StoneState.LOCKED) Text("This concept unlocks when you pass the previous one.", color = C.muted)
        else Blocks(c.blocks, s.packDir(pack))
    }
}

// ------------------------------------------------------------------------------------------------ gate
@Composable
fun GateScreen(s: AppState, pack: String, id: String, idx: Int) {
    val l = s.lesson(pack, id) ?: return Missing(s)
    val c = l.concepts[idx]
    val (pts, n) = s.conceptPoints(pack, l, idx)
    val a = s.lp(pack, id).current
    val missed = c.exercises.filter { (a?.best?.get(it.id) ?: 0.0) < 1.0 }
    val key = c.blocks.filterIsInstance<io.github.luilan.stepstone.data.Block.Callout>().firstOrNull { it.style == "key" }
    Page(s, l.title, foot = {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            BigButton("Re-read", Modifier.weight(1f), kind = "ghost") { s.replace(Screen.ConceptView(pack, id, idx)) }
            BigButton("Retry missed", Modifier.weight(1f), kind = "gold") {
                s.replace(Screen.Quiz(missed.map { QItem(pack, id, it) }, idx, "Concept ${idx + 1} · retry"))
            }
        }
    }) {
        Stones(s.stones(pack, l))
        Column(Modifier.fillMaxWidth().padding(top = 16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Chip("Not yet")
            Text("${fmt(pts)} / $n", fontFamily = Serif, fontWeight = FontWeight.SemiBold, fontSize = 54.sp, modifier = Modifier.padding(top = 6.dp))
            Text("You need ${pct(PASS_SHARE)} of the points (${fmt(PASS_SHARE * n)} of $n) to lay this stone.", color = C.muted, fontSize = 13.sp,
                textAlign = TextAlign.Center)
        }
        Spacer(Modifier.height(16.dp))
        key?.let { Callout("Key idea, once more", it.html) }
        Spacer(Modifier.height(10.dp))
        CardBox {
            Label("Missed")
            missed.forEach { e ->
                val p = e.prompt.filterIsInstance<io.github.luilan.stepstone.data.Block.Text>().firstOrNull()?.html ?: e.id
                Text(html("✗ $p"), fontSize = 14.sp, color = C.ink2, maxLines = 2, overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(top = 4.dp))
            }
        }
    }
}

fun fmt(x: Double) = if (x == Math.floor(x)) "%.0f".format(x) else "%.2f".format(x).trimEnd('0')

// ------------------------------------------------------------------------------------------------ passed
@Composable
fun PassedScreen(s: AppState, pack: String, id: String, idx: Int) {
    val l = s.lesson(pack, id) ?: return Missing(s)
    val (pts, n) = s.conceptPoints(pack, l, idx)
    val next = l.concepts.getOrNull(idx + 1)
    Page(s, l.title, foot = {
        if (next != null) BigButton("Continue · concept ${idx + 2}") { s.replace(Screen.ConceptView(pack, id, idx + 1)) }
    }) {
        Stones(s.stones(pack, l))
        Column(Modifier.fillMaxWidth().padding(top = 34.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Box(Modifier.size(width = 64.dp, height = 44.dp).clip(RoundedCornerShape(percent = 50)).background(C.accent),
                contentAlignment = Alignment.Center) { Text("✓", color = C.accentInk, fontSize = 22.sp, fontWeight = FontWeight.Bold) }
            Spacer(Modifier.height(10.dp))
            Title("Stone laid", 22)
            Text("Concept ${idx + 1} passed · ${fmt(pts)} of $n", color = C.muted, fontSize = 13.sp)
        }
        Spacer(Modifier.height(26.dp))
        if (next != null) CardBox {
            Label("Unlocked")
            Text("${idx + 2} · ${next.title}", fontWeight = FontWeight.SemiBold)
            Text("${next.exercises.size} questions" + if (next.segment.second > 0) " · ${mmss(next.segment.first)}–${mmss(next.segment.second)} in the video" else "",
                color = C.muted, fontSize = 12.5.sp)
        }
        Text("Missed questions come back in your review queue after 1, 3 and 7 days.", color = C.muted, fontSize = 12.sp,
            textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth().padding(top = 10.dp))
    }
}

// ------------------------------------------------------------------------------------------------ lesson done
@Composable
fun LessonDoneScreen(s: AppState, pack: String, id: String) {
    s.tick
    val l = s.lesson(pack, id) ?: return Missing(s)
    val lp = s.lp(pack, id)
    val finished = lp.attempts.filter { it.finished != null }
    val last = finished.lastOrNull() ?: return Missing(s)
    val (pts, n) = s.attemptScore(l, last)
    val best = s.bestScore(pack, l) ?: 0.0
    val missed = l.concepts.flatMap { it.exercises }.filter { (last.first[it.id] ?: 0.0) < 1.0 }
    val m = s.manifest(pack)
    val next = m?.lessons?.let { ls -> ls.getOrNull(ls.indexOfFirst { it.id == id } + 1) }
    Page(s, l.title, foot = {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            BigButton("Retake", Modifier.weight(1f), kind = "ghost") {
                s.progress.startAttempt(pack, id); s.changed(); s.replace(Screen.ConceptView(pack, id, 0))
            }
            BigButton("Missed (${missed.size})", Modifier.weight(1f), kind = "ghost", enabled = missed.isNotEmpty()) {
                s.go(Screen.Quiz(missed.map { QItem(pack, id, it) }, -1, "Practice · missed"))
            }
        }
        if (next != null) BigButton("Next: ${next.title}") { s.replace(Screen.LessonStart(pack, next.id)) }
    }) {
        Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
            Label("Lesson ${l.number} complete")
            Text(pct(if (n == 0) 1.0 else pts / n), fontFamily = Serif, fontWeight = FontWeight.SemiBold, fontSize = 54.sp)
            Text("${fmt(pts)} of $n points on first try · best ${pct(best)}", color = C.muted, fontSize = 13.sp)
        }
        Spacer(Modifier.height(16.dp))
        Label("Score by concept (first try)")
        l.concepts.forEach { c ->
            val v = c.exercises.map { last.first[it.id] ?: 0.0 }.let { if (it.isEmpty()) 1.0 else it.average() }
            Row(Modifier.padding(top = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(c.title, Modifier.width(150.dp), fontSize = 12.5.sp, color = C.ink2, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Meter(v.toFloat(), Modifier.weight(1f).padding(horizontal = 8.dp))
                Text(pct(v), fontSize = 12.5.sp, color = C.ink2, modifier = Modifier.width(38.dp), textAlign = TextAlign.End)
            }
        }
        Spacer(Modifier.height(12.dp))
        CardBox {
            Label("Attempts")
            finished.reversed().forEach { a ->
                val (p, nn) = s.attemptScore(l, a)
                Row(Modifier.fillMaxWidth().padding(top = 4.dp)) {
                    Text(DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(a.finished!!)),
                        Modifier.weight(1f), fontSize = 14.sp)
                    Text(pct(if (nn == 0) 1.0 else p / nn), fontWeight = FontWeight.SemiBold, fontSize = 14.sp)
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ player
@Composable
fun PlayerScreen(s: AppState, pack: String, id: String, start: Double) {
    val ref = s.ref(pack, id) ?: return Missing(s)
    val ctx = LocalContext.current
    val local = s.repo.videoFile(pack, id).takeIf { it.exists() }
    val yt = ref.video.youtube
    val stream = ref.video.offline?.url
    var useStream by remember { mutableStateOf(false) }
    val dl = s.downloads[s.videoKey(pack, id)]
    Column(Modifier.fillMaxSize().background(Color.Black)) {
        TopBar(s, ref.title)
        if (local != null || useStream) {
            AndroidView(factory = { c ->
                VideoView(c).apply {
                    val mc = MediaController(c); mc.setAnchorView(this); setMediaController(mc)
                    if (local != null) setVideoPath(local.path) else setVideoURI(Uri.parse(stream))
                    setOnPreparedListener { seekTo((start * 1000).toInt()); start() }
                }
            }, modifier = Modifier.fillMaxWidth().weight(1f))
        } else {
            Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(if (start > 0) "From ${mmss(start)}" else "Whole video", color = C.ink2)
                if (yt != null) BigButton("Watch on YouTube") {
                    ctx.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://www.youtube.com/watch?v=$yt&t=${start.toInt()}s")))
                }
                if (stream != null) {
                    BigButton("Stream here", kind = if (yt == null) "primary" else "ghost") { useStream = true }
                    if (dl != null) LinearProgressIndicator(progress = { dl }, Modifier.fillMaxWidth(), color = C.accent)
                    else BigButton("Download for offline (${mb(ref.video.offline!!.size)})", kind = "ghost") { s.downloadVideos(pack, listOf(ref)) }
                }
                if (yt == null && stream == null) Text("No video for this lesson.", color = C.muted)
            }
        }
    }
}
