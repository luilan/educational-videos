package io.github.luilan.stepstone.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGesturesAfterLongPress
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.zIndex
import io.github.luilan.stepstone.data.*
import kotlin.math.roundToInt
import kotlin.random.Random

private val KIND = mapOf("mc" to "choice", "tf" to "true or false", "number" to "number", "order" to "order",
    "short" to "your words", "code" to "code")

@Composable
fun QuizScreen(s: AppState, q: Screen.Quiz) {
    var index by remember(q) { mutableIntStateOf(0) }
    val results = remember(q) { mutableStateListOf<Double>() }
    if (q.items.isEmpty() || index >= q.items.size) {
        if (q.concept >= 0) LaunchedEffect(q) {               // gate quiz finished: decide pass / not yet
            val it0 = q.items.firstOrNull()
            val l = if (it0 != null) s.lesson(it0.pack, it0.lesson) else null
            if (l != null) s.afterGate(it0!!.pack, l, q.concept) else s.back()
        } else PracticeDone(s, results)
        return
    }
    val item = q.items[index]
    // one composition per question, so each question starts with fresh input state
    key(q, index) {
        Question(s, q, item, index, results,
            onScored = { score, text ->
                results.add(score)
                if (q.concept >= 0) s.progress.record(item.pack, item.lesson, item.ex.id, score, text)
                else s.progress.recordReview(item.pack, item.lesson, item.ex.id, score)
                s.changed()
            },
            onNext = { index++ })
    }
}

@Composable
private fun PracticeDone(s: AppState, results: List<Double>) {
    Page(s, "Practice", foot = { BigButton("Done") { s.back() } }) {
        Column(Modifier.fillMaxWidth().padding(top = 40.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text("${fmt(results.sum())} / ${results.size}", fontFamily = Serif, fontWeight = FontWeight.SemiBold, fontSize = 54.sp)
            Text("Right answers move a question to a later review; misses come back tomorrow.", color = C.muted, fontSize = 13.sp,
                modifier = Modifier.padding(16.dp))
        }
    }
}

/** One question. onScored is called exactly once, when the answer is graded; onNext moves on. */
@Composable
private fun Question(s: AppState, q: Screen.Quiz, item: QItem, index: Int, results: List<Double>,
                     onScored: (Double, String?) -> Unit, onNext: () -> Unit) {
    val ex = item.ex
    val dir = s.packDir(item.pack)
    var score by remember { mutableStateOf<Double?>(null) }
    // inputs
    var choice by remember { mutableStateOf<Int?>(null) }
    var tf by remember { mutableStateOf<Boolean?>(null) }
    val nums = remember { mutableStateListOf<String>().apply { repeat((ex.key as? Key.Numbers)?.parts?.size ?: 1) { add("") } } }
    val order = remember {
        mutableStateListOf<String>().apply {
            (ex.key as? Key.Order)?.items?.let { items ->
                val r = Random(ex.id.hashCode())
                var sh = items.shuffled(r)
                var guard = 0
                while (sh == items && items.size > 1 && guard++ < 10) sh = items.shuffled(r)
                addAll(sh)
            }
        }
    }
    var text by remember { mutableStateOf("") }
    var revealed by remember { mutableStateOf(false) }

    val key = ex.key
    val canCheck = when (key) {
        is Key.Choice -> choice != null
        is Key.TrueFalse -> tf != null
        is Key.Numbers -> nums.all { it.isNotBlank() }
        is Key.Order -> true
        null -> true
    }
    fun grade() {
        val sc = when (key) {
            is Key.Choice -> Grading.choiceScore(key, choice)
            is Key.TrueFalse -> Grading.trueFalseScore(key, tf)
            is Key.Numbers -> Grading.numberScore(key, nums)
            is Key.Order -> Grading.orderScore(key, order.toList())
            null -> 0.0
        }
        score = sc; onScored(sc, null)
    }
    fun selfGrade(g: SelfGrade) {
        score = g.score
        onScored(g.score, text.takeIf { it.isNotBlank() })
        onNext()
    }

    Page(s, q.title, foot = {
        when {
            key == null && score == null && !revealed -> BigButton("Show the model answer") { revealed = true }
            key == null && score == null -> {
                Label("How did you do?")
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    GradeButton("✓ Got it", C.good, Modifier.weight(1f)) { selfGrade(SelfGrade.GOT_IT) }
                    GradeButton("◐ Partly", C.gold, Modifier.weight(1f)) { selfGrade(SelfGrade.PARTLY) }
                    GradeButton("✗ Missed", C.bad, Modifier.weight(1f)) { selfGrade(SelfGrade.MISSED) }
                }
            }
            score == null -> BigButton("Check", enabled = canCheck) { grade() }
            else -> BigButton(if (index == q.items.lastIndex) "Finish" else "Next question") { onNext() }
        }
    }) {
        Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Label("Question ${index + 1} of ${q.items.size} · ${KIND[ex.kind] ?: ex.kind}")
            Spacer(Modifier.weight(1f))
            q.items.indices.forEach { i ->
                val col = when {
                    i < results.size -> if (results[i] >= 1.0) C.good else if (results[i] > 0) C.gold else C.bad
                    i == index -> C.gold.copy(alpha = 0.6f)
                    else -> C.surface2
                }
                Box(Modifier.padding(start = 4.dp).size(width = if (q.items.size > 8) 10.dp else 20.dp, height = 5.dp).clip(RoundedCornerShape(3.dp)).background(col))
            }
        }
        if (q.concept < 0) Text(s.ref(item.pack, item.lesson)?.title ?: "", color = C.muted, fontSize = 12.sp)
        Spacer(Modifier.height(6.dp))
        Blocks(ex.prompt, dir, textSize = 17)
        ex.code?.takeIf { it.isNotBlank() }?.let { Spacer(Modifier.height(10.dp)); CodeView(it) }
        Spacer(Modifier.height(14.dp))

        when (key) {
            is Key.Choice -> ex.options.forEachIndexed { i, o ->
                val st = when {
                    score != null && i == key.index -> "right"
                    score != null && i == choice -> "wrong"
                    i == choice -> "sel"
                    else -> ""
                }
                OptionRow(('A' + i).toString(), o, st) { if (score == null) choice = i }
            }
            is Key.TrueFalse -> Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf(true to "True", false to "False").forEach { (v, l) ->
                    val st = when {
                        score != null && v == key.value -> "right"
                        score != null && v == tf -> "wrong"
                        v == tf -> "sel"
                        else -> ""
                    }
                    Box(Modifier.weight(1f)) { OptionRow(null, l, st) { if (score == null) tf = v } }
                }
            }
            is Key.Numbers -> key.parts.forEachIndexed { i, p ->
                val ok = score?.let { Grading.numberPartRight(p, nums[i]) }
                OutlinedTextField(nums[i], { if (score == null) nums[i] = it }, Modifier.fillMaxWidth().padding(bottom = 8.dp),
                    label = p.label?.let { { Text(it) } } ?: (if (p.unit == "%") ({ Text("%") }) else null),
                    singleLine = true, readOnly = score != null,
                    textStyle = LocalTextStyle.current.copy(fontFamily = Mono, fontSize = 18.sp),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                    trailingIcon = ok?.let { { Text(if (it) "✓" else "✗", color = if (it) C.good else C.bad) } },
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = ok?.let { if (it) C.good else C.bad } ?: C.accent,
                        unfocusedBorderColor = ok?.let { if (it) C.good else C.bad } ?: C.line))
            }
            is Key.Order -> {
                OrderList(order, enabled = score == null)
                if (score == null) Text("Long-press and drag, or use the arrows, then check.", color = C.muted, fontSize = 12.5.sp)
            }
            null -> {
                OutlinedTextField(text, { if (score == null) text = it }, Modifier.fillMaxWidth().heightIn(min = 96.dp),
                    placeholder = { Text("Your answer (optional, saved with your attempt)") }, readOnly = score != null,
                    colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = C.accent, unfocusedBorderColor = C.line))
                if (revealed || score != null) {
                    Spacer(Modifier.height(10.dp))
                    Surface(color = C.surface2, shape = RoundedCornerShape(12.dp)) {
                        Column(Modifier.fillMaxWidth().padding(12.dp)) {
                            Label("Model answer")
                            Blocks(ex.answer, dir, textSize = 15)
                            if (ex.why.isNotEmpty()) { Spacer(Modifier.height(6.dp)); Blocks(ex.why, dir, textSize = 14) }
                        }
                    }
                }
            }
        }

        // feedback for auto-graded questions
        val sc = score
        if (sc != null && key != null) {
            Spacer(Modifier.height(12.dp))
            val (col, head) = when {
                sc >= 1.0 -> C.good to "✓ Right"
                sc > 0 -> C.gold to "◐ Partly right"
                else -> C.bad to "✗ Not quite"
            }
            Surface(color = col.copy(alpha = 0.14f), shape = RoundedCornerShape(12.dp), border = BorderStroke(1.dp, col.copy(alpha = 0.4f))) {
                Column(Modifier.fillMaxWidth().padding(12.dp)) {
                    Text(head, color = col, fontWeight = FontWeight.Bold)
                    if (sc < 1.0) { Text("Answer:", color = C.ink2, fontSize = 13.sp); Blocks(ex.answer, dir, textSize = 15) }
                    if (ex.why.isNotEmpty()) { Spacer(Modifier.height(4.dp)); Blocks(ex.why, dir, textSize = 15) }
                }
            }
        }
    }
}

@Composable
private fun GradeButton(text: String, col: Color, modifier: Modifier, onClick: () -> Unit) {
    OutlinedButton(onClick, modifier.heightIn(min = 48.dp), shape = RoundedCornerShape(12.dp), border = BorderStroke(2.dp, col.copy(alpha = 0.6f)),
        contentPadding = PaddingValues(4.dp)) { Text(text, color = col, fontWeight = FontWeight.SemiBold, fontSize = 13.sp) }
}

@Composable
private fun OptionRow(letter: String?, textHtml: String, state: String, onClick: () -> Unit) {
    val border = when (state) { "right" -> C.good; "wrong" -> C.bad; "sel" -> C.accent; else -> C.line }
    val bg = when (state) { "right" -> C.good.copy(alpha = 0.14f); "wrong" -> C.bad.copy(alpha = 0.14f); "sel" -> C.accent.copy(alpha = 0.12f); else -> C.surface }
    Row(Modifier.fillMaxWidth().padding(bottom = 8.dp).clip(RoundedCornerShape(12.dp)).background(bg)
        .border(2.dp, border, RoundedCornerShape(12.dp)).clickable(onClick = onClick).padding(horizontal = 12.dp, vertical = 11.dp),
        verticalAlignment = Alignment.CenterVertically) {
        if (letter != null) {
            Box(Modifier.size(24.dp).clip(RoundedCornerShape(50)).background(if (state == "right") C.good else if (state == "wrong") C.bad else C.surface2),
                contentAlignment = Alignment.Center) { Text(letter, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = if (state in setOf("right", "wrong")) C.bg else C.ink2) }
            Spacer(Modifier.width(10.dp))
        }
        Text(html(textHtml), fontSize = 15.sp, color = C.ink, modifier = Modifier.weight(1f))
        if (state == "right") Text("✓", color = C.good) else if (state == "wrong") Text("✗", color = C.bad)
    }
}

/** Reorderable list: long-press and drag an item, or use its arrows. */
@Composable
private fun OrderList(items: MutableList<String>, enabled: Boolean) {
    var dragging by remember { mutableStateOf<Int?>(null) }
    var offset by remember { mutableFloatStateOf(0f) }
    var rowH by remember { mutableIntStateOf(1) }
    val gapPx = with(androidx.compose.ui.platform.LocalDensity.current) { 8.dp.roundToPx() }
    Column {
        items.forEachIndexed { i, t ->
            val lifted = dragging == i
            Row(Modifier.fillMaxWidth().padding(bottom = 8.dp).zIndex(if (lifted) 1f else 0f)
                .offset { IntOffset(0, if (lifted) offset.roundToInt() else 0) }
                .onSizeChanged { rowH = it.height + gapPx }
                .clip(RoundedCornerShape(12.dp)).background(C.surface)
                .border(if (lifted) 2.dp else 1.dp, if (lifted) C.gold else C.line, RoundedCornerShape(12.dp))
                .pointerInput(enabled, items.size) {
                    if (!enabled) return@pointerInput
                    detectDragGesturesAfterLongPress(
                        onDragStart = { dragging = items.indexOf(t); offset = 0f },
                        onDragEnd = { dragging = null; offset = 0f },
                        onDragCancel = { dragging = null; offset = 0f },
                    ) { change, d ->
                        change.consume()
                        val cur = dragging ?: return@detectDragGesturesAfterLongPress
                        offset += d.y
                        if (offset > rowH / 2 && cur < items.lastIndex) { items.add(cur + 1, items.removeAt(cur)); dragging = cur + 1; offset -= rowH }
                        else if (offset < -rowH / 2 && cur > 0) { items.add(cur - 1, items.removeAt(cur)); dragging = cur - 1; offset += rowH }
                    }
                }
                .padding(horizontal = 10.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(22.dp).clip(RoundedCornerShape(50)).background(C.surface2), contentAlignment = Alignment.Center) {
                    Text("${i + 1}", fontSize = 12.sp, color = C.ink2)
                }
                Text(html(t), Modifier.weight(1f).padding(horizontal = 10.dp), fontSize = 14.5.sp)
                if (enabled) Column {
                    Text("▲", fontSize = 13.sp, color = if (i > 0) C.ink2 else C.surface2,
                        modifier = Modifier.clickable(enabled = i > 0) { items.add(i - 1, items.removeAt(i)) }.padding(horizontal = 8.dp, vertical = 2.dp))
                    Text("▼", fontSize = 13.sp, color = if (i < items.lastIndex) C.ink2 else C.surface2,
                        modifier = Modifier.clickable(enabled = i < items.lastIndex) { items.add(i + 1, items.removeAt(i)) }.padding(horizontal = 8.dp, vertical = 2.dp))
                }
            }
        }
    }
}
