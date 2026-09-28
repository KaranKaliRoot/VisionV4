package com.rudra.visionv4

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.rudra.visionv4.ui.theme.VisionV4Theme

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            VisionV4Theme {
                CameraScreen()
            }
        }
    }
}