
package com.rudra.visionv4
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke

data class Detection(
    val label: String,
    val confidence: Float,
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float
)

@Composable
fun Overlay(
    detections: List<Detection>
) {
    Canvas(modifier = Modifier.fillMaxSize()) {

        val scaleX = size.width / 640f
        val scaleY = size.height / 480f

        detections.forEach { det ->

            val left = det.x * scaleX
            val top = det.y * scaleY
            val w = det.width * scaleX
            val h = det.height * scaleY

            drawRect(
                color = Color(0xFF00E676),
                topLeft = Offset(left, top),
                size = Size(w, h),
                style = Stroke(width = 3f)
            )

            drawRect(
                color = Color(0xCC00E676),
                topLeft = Offset(left, (top - 42f).coerceAtLeast(0f)),
                size = Size(180f, 40f)
            )

            drawContext.canvas.nativeCanvas.drawText(
                "${det.label} ${"%.0f".format(det.confidence * 100)}%",
                left + 10f,
                (top - 14f).coerceAtLeast(28f),
                android.graphics.Paint().apply {
                    color = android.graphics.Color.BLACK
                    textSize = 30f
                    isFakeBoldText = true
                }
            )
        }
    }
}
