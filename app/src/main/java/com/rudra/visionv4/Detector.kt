package com.rudra.visionv4

import android.content.Context
import androidx.camera.core.ImageProxy

class Detector(context: Context) {

    private val yolo = Yolo26Ncnn()

    private val loaded = yolo.loadModel(
        context.assets,
        0,
        0
    )

    fun testModel(): String =
        if (loaded) "YOLO26 NCNN Ready ✓"
        else "Model failed to load ✗"

    fun detect(image: ImageProxy): List<Detection> {

        val bitmap = image.toBitmap()

        val objs = yolo.detect(bitmap) ?: emptyArray()

        android.util.Log.d("VisionV4", "Detections = ${objs.size}")

        return objs.map {
            Detection(
                label = it.label,
                confidence = it.prob,
                x = it.x,
                y = it.y,
                width = it.w,
                height = it.h
            )
        }
    }
}