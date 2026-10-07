package io.github.luilan.stepstone

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import java.io.File

/** Draws the launcher icon layers to build/ui-shots/icon_*.png so the real vector drawables can be checked by eye. */
@RunWith(AndroidJUnit4::class)
@Config(sdk = [35])
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class IconRenderTest {
    @Test fun renderIconLayers() {
        val ctx = ApplicationProvider.getApplicationContext<android.content.Context>()
        File("build/ui-shots").mkdirs()
        for ((name, res, bg) in listOf(Triple("color", R.drawable.ic_launcher_foreground, 0xFF0F1115.toInt()),
                                       Triple("mono", R.drawable.ic_launcher_monochrome, Color.rgb(60, 70, 58)))) {
            val d = ctx.getDrawable(res)!!
            val bmp = Bitmap.createBitmap(432, 432, Bitmap.Config.ARGB_8888)
            val c = Canvas(bmp)
            c.drawColor(bg)
            d.setBounds(0, 0, 432, 432)
            d.draw(c)
            var painted = 0
            for (x in 0 until 432 step 4) for (y in 0 until 432 step 4) if (bmp.getPixel(x, y) != bg) painted++
            assertTrue("$name icon draws nothing", painted > 200)
            File("build/ui-shots/icon_$name.png").outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        }
    }
}
