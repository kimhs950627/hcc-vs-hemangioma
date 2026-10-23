package com.kimhs.liverdx

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.kimhs.liverdx.ui.AnalyzeRoute
import com.kimhs.liverdx.ui.theme.LiverDxTheme
import com.kimhs.liverdx.ui.theme.Palette

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            LiverDxTheme {
                Surface(color = Palette.Ink, modifier = Modifier.fillMaxSize()) {
                    AnalyzeRoute()
                }
            }
        }
    }
}
