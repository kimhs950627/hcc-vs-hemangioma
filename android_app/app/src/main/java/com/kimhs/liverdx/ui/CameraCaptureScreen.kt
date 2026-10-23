package com.kimhs.liverdx.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.kimhs.liverdx.ui.components.clickableNoRipple
import com.kimhs.liverdx.ui.theme.AppType
import com.kimhs.liverdx.ui.theme.Dimens
import com.kimhs.liverdx.ui.theme.Palette
import com.kimhs.liverdx.ui.theme.Shapes
import com.kimhs.liverdx.util.CaptureFiles
import java.io.File

/**
 * In-app capture. The CAMERA runtime permission is required; the frame is written straight to a
 * private cache file and handed back. Nothing is uploaded, and the file is deleted once decoded.
 */
@Composable
fun CameraCaptureScreen(
    onCaptured: (File) -> Unit,
    onFailure: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val currentOnCaptured by rememberUpdatedState(onCaptured)
    val previewView = remember { PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER } }

    var granted by remember { mutableStateOf(hasCameraPermission(context)) }
    var provider by remember { mutableStateOf<ProcessCameraProvider?>(null) }
    var camera by remember { mutableStateOf<Camera?>(null) }
    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var torch by remember { mutableStateOf(false) }
    var capturing by remember { mutableStateOf(false) }

    val requestPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        granted = isGranted
        if (!isGranted) onFailure("카메라 권한이 거부되었습니다")
    }
    LaunchedEffect(granted) {
        if (!granted) requestPermission.launch(Manifest.permission.CAMERA)
    }

    DisposableEffect(granted, lifecycleOwner) {
        if (!granted) return@DisposableEffect onDispose { }
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener(
            {
                val bound = runCatching { future.get() }.getOrNull() ?: return@addListener
                provider = bound
                val capture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val cameraInstance = runCatching {
                    bound.unbindAll()
                    bound.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        capture,
                    )
                }.getOrNull()
                camera = cameraInstance
                imageCapture = capture
            },
            ContextCompat.getMainExecutor(context),
        )
        onDispose { provider?.unbindAll() }
    }

    Box(
        Modifier
            .fillMaxSize()
            .background(Palette.Ink),
    ) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { previewView },
        )

        // Framing guide: the model squeezes the whole frame, so keep the lesion centred instead.
        Box(
            Modifier
                .fillMaxSize()
                .padding(Dimens.breathing),
            contentAlignment = Alignment.Center,
        ) {
            Box(
                Modifier
                    .fillMaxWidth(GUIDE_WIDTH_FRACTION)
                    .aspectRatio(1f)
                    .clip(Shapes.card)
                    .border(Dimens.hairline, Palette.Benign.copy(alpha = 0.6f), Shapes.card)
            )
        }

        Column(
            Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .padding(Dimens.gutter),
            verticalArrangement = Arrangement.spacedBy(Dimens.hairGap),
        ) {
            Text("CAPTURE", style = AppType.gaugeLabel, color = Palette.TextSecondary)
            Text(
                "화면 반사와 손떨림을 피하고, 병변을 네모 중앙에 두세요. 반사광은 촬영 후 제거됩니다.",
                style = AppType.gaugeNote,
                color = Palette.TextMuted,
            )
        }

        Row(
            Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .padding(Dimens.gutter),
            horizontalArrangement = Arrangement.spacedBy(Dimens.tight),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            CaptureAction(label = "닫기", onClick = onDismiss)
            CaptureAction(
                label = if (torch) "라이트 끄기" else "라이트 켜기",
                onClick = {
                    val next = !torch
                    torch = next
                    camera?.cameraControl?.enableTorch(next)
                },
            )
            Text(
                text = if (capturing) "저장 중…" else "촬영",
                style = AppType.buttonLabel,
                color = Palette.Ink,
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .size(width = Dimens.shutter, height = Dimens.touchTarget)
                    .clip(Shapes.control)
                    .background(Palette.Benign)
                    .clickableNoRipple {
                        val capture = imageCapture
                        if (capture == null) {
                            onFailure("카메라를 아직 준비하지 못했습니다")
                            return@clickableNoRipple
                        }
                        if (capturing) return@clickableNoRipple
                        capturing = true
                        val target = CaptureFiles.nextFile(context)
                        val output = ImageCapture.OutputFileOptions.Builder(target).build()
                        capture.takePicture(
                            output,
                            ContextCompat.getMainExecutor(context),
                            object : ImageCapture.OnImageSavedCallback {
                                override fun onImageSaved(results: ImageCapture.OutputFileResults) {
                                    capturing = false
                                    currentOnCaptured(target)
                                }

                                override fun onError(exception: ImageCaptureException) {
                                    capturing = false
                                    onFailure(exception.message ?: "촬영에 실패했습니다")
                                }
                            },
                        )
                    },
            )
        }
    }
}

@Composable
private fun CaptureAction(label: String, onClick: () -> Unit) {
    Text(
        text = label,
        style = AppType.buttonLabel,
        color = Palette.TextSecondary,
        modifier = Modifier
            .clip(Shapes.control)
            .background(Palette.Surface)
            .border(Dimens.hairline, Palette.Hairline, Shapes.control)
            .clickableNoRipple(onClick)
            .padding(horizontal = Dimens.gutter, vertical = Dimens.controlVertical),
    )
}

private const val GUIDE_WIDTH_FRACTION = 0.72f

private fun hasCameraPermission(context: Context): Boolean =
    ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
