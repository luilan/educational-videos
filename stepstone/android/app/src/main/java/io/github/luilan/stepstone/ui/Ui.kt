package io.github.luilan.stepstone.ui

import android.annotation.SuppressLint
import android.graphics.BitmapFactory
import android.webkit.WebView
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.*
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.BaselineShift
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import io.github.luilan.stepstone.data.Block
import java.io.File

object C {
    val bg = Color(0xFF0F1115); val surface = Color(0xFF181B22); val surface2 = Color(0xFF20242D); val line = Color(0xFF2C313B)
    val ink = Color(0xFFECEBE7); val ink2 = Color(0xFFB9BCC2); val muted = Color(0xFF7F858F)
    val accent = Color(0xFF2A9D8F); val accentInk = Color(0xFF06221F); val accentText = Color(0xFF7FD3C8)
    val gold = Color(0xFFE9C46A); val goldInk = Color(0xFF2A2109)
    val good = Color(0xFF5BBF84); val bad = Color(0xFFE76F51)
}

val Serif = FontFamily.Serif
val Mono = FontFamily.Monospace

@Composable
fun StepStoneTheme(content: @Composable () -> Unit) = MaterialTheme(
    colorScheme = darkColorScheme(
        primary = C.accent, onPrimary = C.accentInk, secondary = C.gold, onSecondary = C.goldInk,
        background = C.bg, onBackground = C.ink, surface = C.surface, onSurface = C.ink, surfaceVariant = C.surface2,
        onSurfaceVariant = C.ink2, outline = C.line, error = C.bad,
    ),
    content = content,
)

// ------------------------------------------------------------------------------------------------ inline HTML
private val entity = mapOf("amp" to "&", "lt" to "<", "gt" to ">", "quot" to "\"", "#39" to "'", "nbsp" to " ")

/** The packs' inline HTML (b, i, code, sub, sup, br) as styled text. Unknown tags are dropped, their text kept. */
fun html(src: String): AnnotatedString = buildAnnotatedString {
    var i = 0
    val stack = ArrayDeque<String>()
    while (i < src.length) {
        val ch = src[i]
        if (ch == '<') {
            val end = src.indexOf('>', i)
            if (end < 0) { append(src.substring(i)); break }
            val raw = src.substring(i + 1, end).trim()
            val closing = raw.startsWith("/")
            val name = raw.removePrefix("/").split(' ', '/').first().lowercase()
            when {
                name == "br" -> append('\n')
                closing -> if (name in stack) { while (stack.isNotEmpty()) { val t = stack.removeLast(); pop(); if (t == name) break } }
                name in setOf("b", "strong", "i", "em", "code", "tt", "sub", "sup") -> {
                    pushStyle(when (name) {
                        "b", "strong" -> SpanStyle(fontWeight = FontWeight.SemiBold, color = C.ink)
                        "i", "em" -> SpanStyle(fontStyle = FontStyle.Italic)
                        "code", "tt" -> SpanStyle(fontFamily = Mono, background = C.surface2, fontSize = 0.92.em)
                        "sub" -> SpanStyle(baselineShift = BaselineShift.Subscript, fontSize = 0.75.em)
                        else -> SpanStyle(baselineShift = BaselineShift.Superscript, fontSize = 0.75.em)
                    })
                    stack.addLast(name)
                }
            }
            i = end + 1
        } else if (ch == '&') {
            val end = src.indexOf(';', i)
            val e = if (end in (i + 1)..(i + 8)) entity[src.substring(i + 1, end)] else null
            if (e != null) { append(e); i = end + 1 } else { append(ch); i++ }
        } else { append(ch); i++ }
    }
    while (stack.isNotEmpty()) { stack.removeLast(); pop() }
}

private val Double.em get() = androidx.compose.ui.unit.TextUnit(this.toFloat(), androidx.compose.ui.unit.TextUnitType.Em)

// ------------------------------------------------------------------------------------------------ blocks
@Composable
fun Blocks(blocks: List<Block>, packDir: File, modifier: Modifier = Modifier, textSize: Int = 16) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(10.dp)) {
        blocks.forEach { BlockView(it, packDir, textSize) }
    }
}

@Composable
fun BlockView(b: Block, packDir: File, textSize: Int = 16) {
    when (b) {
        is Block.Text -> Text(html(b.html), color = C.ink, fontSize = textSize.sp, lineHeight = (textSize * 1.45).sp,
            textAlign = if (b.center) TextAlign.Center else TextAlign.Start, modifier = Modifier.fillMaxWidth())
        is Block.Callout -> Callout(b.title ?: if (b.style == "key") "Key idea" else null, b.html, b.style == "key")
        is Block.Figures -> Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            b.items.forEach { f -> FigureView(File(packDir, f.src), f.caption) }
        }
        is Block.Table -> TableView(b)
        is Block.Code -> CodeView(b.text)
        is Block.Html -> HtmlView(b.html)
        is Block.Unknown -> Text("This part needs a newer version of StepStone.", color = C.muted, fontSize = 13.sp)
    }
}

@Composable
fun Callout(title: String?, body: String, key: Boolean = true) {
    val col = if (key) C.accent else C.muted
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(col.copy(alpha = 0.16f)).height(IntrinsicSize.Min)) {
        Box(Modifier.width(4.dp).fillMaxHeight().background(col))
        Column(Modifier.padding(horizontal = 11.dp, vertical = 9.dp)) {
            if (title != null) Text(title.uppercase(), color = if (key) C.accentText else C.ink2, fontSize = 11.sp,
                fontWeight = FontWeight.Bold, letterSpacing = 0.8.sp)
            Text(html(body), color = C.ink, fontSize = 15.sp, lineHeight = 21.sp)
        }
    }
}

@Composable
fun FigureView(file: File, caption: String) {
    val bmp = remember(file) {
        runCatching {
            val o = BitmapFactory.Options().apply { inSampleSize = 2 }   // frames are 1920 px wide: half is plenty
            BitmapFactory.decodeFile(file.path, o)?.asImageBitmap()
        }.getOrNull()
    }
    Column {
        if (bmp != null) Image(bmp, caption, Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)), contentScale = ContentScale.FillWidth)
        if (caption.isNotBlank()) Text(html(caption), color = C.muted, fontSize = 12.5.sp, lineHeight = 17.sp, modifier = Modifier.padding(top = 5.dp))
    }
}

@Composable
fun TableView(t: Block.Table) {
    val rows = listOfNotNull(t.header) + t.rows
    val cols = rows.maxOfOrNull { it.size } ?: 0
    Column(Modifier.horizontalScroll(rememberScrollState()).border(1.dp, C.line, RoundedCornerShape(8.dp)).clip(RoundedCornerShape(8.dp))) {
        rows.forEachIndexed { r, row ->
            val head = t.header != null && r == 0
            Row(Modifier.background(if (head) C.surface2 else if (r % 2 == 0) C.surface else C.bg)) {
                for (c in 0 until cols) {
                    Text(html(row.getOrElse(c) { "" }), color = if (head) C.ink else C.ink2, fontSize = 13.sp,
                        fontWeight = if (head) FontWeight.SemiBold else FontWeight.Normal,
                        modifier = Modifier.widthIn(min = 64.dp, max = 220.dp).padding(horizontal = 9.dp, vertical = 6.dp))
                }
            }
        }
    }
}

@Composable
fun CodeView(code: String) {
    val clip = LocalClipboardManager.current
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(Color(0xFF272822))) {
        Row(Modifier.fillMaxWidth().padding(start = 10.dp, end = 4.dp, top = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("code", color = C.muted, fontSize = 11.sp, modifier = Modifier.weight(1f))
            TextButton(onClick = { clip.setText(AnnotatedString(code)) }) { Text("Copy", fontSize = 12.sp, color = C.accentText) }
        }
        Text(code, fontFamily = Mono, fontSize = 13.sp, color = Color(0xFFF8F8F2), lineHeight = 18.sp, softWrap = false,
            modifier = Modifier.horizontalScroll(rememberScrollState()).padding(start = 10.dp, end = 10.dp, bottom = 10.dp))
    }
}

@SuppressLint("SetJavaScriptEnabled")
@Composable
fun HtmlView(src: String) {
    val page = """<html><head><meta name="viewport" content="width=device-width"><style>
        body{background:#0F1115;color:#ECEBE7;font:15px sans-serif;margin:0} td{color:#ECEBE7}</style></head><body>$src</body></html>"""
    AndroidView(factory = { ctx -> WebView(ctx).apply { setBackgroundColor(0xFF0F1115.toInt()) } },
        update = { it.loadDataWithBaseURL(null, page, "text/html", "utf-8", null) },
        modifier = Modifier.fillMaxWidth().heightIn(min = 60.dp))
}

// ------------------------------------------------------------------------------------------------ pieces
enum class StoneState { DONE, NOW, LOCKED }

/** The concept path: one stone per concept. */
@Composable
fun Stones(states: List<StoneState>, onTap: ((Int) -> Unit)? = null) {
    Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        states.forEachIndexed { i, s ->
            if (i > 0) Box(Modifier.weight(1f).height(2.dp).padding(horizontal = 2.dp).background(C.line))
            val shape = RoundedCornerShape(percent = 50)
            val mod = Modifier.size(width = 30.dp, height = 22.dp).clip(shape).let { m ->
                when (s) {
                    StoneState.DONE -> m.background(C.accent)
                    StoneState.NOW -> m.background(C.gold.copy(alpha = 0.14f)).border(2.dp, C.gold, shape)
                    StoneState.LOCKED -> m.background(C.surface2)
                }
            }.let { m -> if (onTap != null && s != StoneState.LOCKED) m.clickable { onTap(i) } else m }
            Box(mod, contentAlignment = Alignment.Center) {
                Text(when (s) { StoneState.DONE -> "✓"; StoneState.NOW -> "${i + 1}"; StoneState.LOCKED -> "🔒" },
                    fontSize = if (s == StoneState.LOCKED) 9.sp else 11.sp, fontWeight = FontWeight.Bold,
                    color = when (s) { StoneState.DONE -> C.accentInk; StoneState.NOW -> C.gold; StoneState.LOCKED -> C.muted })
            }
        }
    }
}

@Composable
fun BigButton(text: String, modifier: Modifier = Modifier, kind: String = "primary", enabled: Boolean = true, onClick: () -> Unit) {
    val (bg, fg) = when (kind) { "gold" -> C.gold to C.goldInk; "ghost" -> C.surface2 to C.ink; else -> C.accent to C.accentInk }
    Button(onClick, modifier.fillMaxWidth().heightIn(min = 50.dp), enabled = enabled, shape = RoundedCornerShape(14.dp),
        colors = ButtonDefaults.buttonColors(containerColor = bg, contentColor = fg, disabledContainerColor = C.surface2, disabledContentColor = C.muted)) {
        Text(text, fontWeight = FontWeight.SemiBold, fontSize = 15.sp, textAlign = TextAlign.Center)
    }
}

@Composable
fun CardBox(modifier: Modifier = Modifier, border: Color = C.line, onClick: (() -> Unit)? = null, content: @Composable ColumnScope.() -> Unit) {
    Surface(color = C.surface, shape = RoundedCornerShape(16.dp), border = BorderStroke(1.dp, border),
        modifier = modifier.fillMaxWidth().let { if (onClick != null) it.clip(RoundedCornerShape(16.dp)).clickable(onClick = onClick) else it }) {
        Column(Modifier.padding(12.dp), content = content)
    }
}

@Composable
fun Meter(share: Float, modifier: Modifier = Modifier, color: Color = C.accent) {
    Box(modifier.height(6.dp).fillMaxWidth().clip(RoundedCornerShape(4.dp)).background(C.surface2)) {
        Box(Modifier.fillMaxWidth(share.coerceIn(0f, 1f)).fillMaxHeight().clip(RoundedCornerShape(4.dp)).background(color))
    }
}

@Composable
fun Chip(text: String, accent: Boolean = false, onClick: (() -> Unit)? = null) {
    Text(text, fontSize = 12.5.sp, color = if (accent) C.accentText else C.ink2, maxLines = 1, overflow = TextOverflow.Ellipsis,
        modifier = Modifier.clip(RoundedCornerShape(50)).background(if (accent) C.accent.copy(alpha = 0.16f) else C.surface2)
            .let { if (onClick != null) it.clickable(onClick = onClick) else it }.padding(horizontal = 11.dp, vertical = 6.dp))
}

@Composable
fun Label(text: String) = Text(text.uppercase(), color = C.muted, fontSize = 11.5.sp, letterSpacing = 0.6.sp)

@Composable
fun Title(text: String, size: Int = 20) = Text(text, fontFamily = Serif, fontWeight = FontWeight.SemiBold, fontSize = size.sp,
    lineHeight = (size * 1.25).sp, color = C.ink)

fun pct(x: Double) = "${Math.round(x * 100)}%"
fun mmss(t: Double): String { val s = t.toInt(); return "%d:%02d".format(s / 60, s % 60) }
