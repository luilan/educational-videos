package io.github.luilan.stepstone

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import io.github.luilan.stepstone.ui.*

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent { StepStoneTheme { Root() } }
    }
}

@Composable
private fun Root() {
    val ctx = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    val s = remember { AppState(ctx.applicationContext, scope) }
    val snack = remember { SnackbarHostState() }
    LaunchedEffect(Unit) { s.refresh() }
    LaunchedEffect(s.message) { s.message?.let { snack.showSnackbar(it); s.message = null } }
    BackHandler(enabled = s.stack.size > 1) { s.back() }
    // messages show at the top, so they never cover the action buttons in the footers
    Scaffold(containerColor = C.bg) { pad ->
        Box(Modifier.fillMaxSize().background(C.bg).padding(pad)) {
            when (val sc = s.screen) {
                is Screen.Home -> HomeScreen(s, sc.tab)
                is Screen.Series -> SeriesScreen(s, sc.pack)
                is Screen.LessonStart -> LessonStartScreen(s, sc.pack, sc.lesson)
                is Screen.ConceptView -> ConceptScreen(s, sc.pack, sc.lesson, sc.idx)
                is Screen.Quiz -> QuizScreen(s, sc)
                is Screen.Gate -> GateScreen(s, sc.pack, sc.lesson, sc.idx)
                is Screen.Passed -> PassedScreen(s, sc.pack, sc.lesson, sc.idx)
                is Screen.LessonDone -> LessonDoneScreen(s, sc.pack, sc.lesson)
                is Screen.Player -> PlayerScreen(s, sc.pack, sc.lesson, sc.start)
            }
            SnackbarHost(snack, Modifier.align(androidx.compose.ui.Alignment.TopCenter).padding(top = 56.dp))
        }
    }
}
