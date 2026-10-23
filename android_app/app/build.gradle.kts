plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Which flatbuffer gets packaged. `-PmodelPrecision=fp16` halves the asset (27.2 MB -> 13.8 MB) but
// moves the cosine markers by up to 9.5e-3; the app shows the measured drift on the model chip.
val modelPrecision = (findProperty("modelPrecision") as String?) ?: "fp32"
val stagedModelDir = layout.projectDirectory.dir("src/main/model")
val generatedAssetDir = layout.buildDirectory.dir("generated/modelAssets")

/**
 * Copies the selected flatbuffer, its meta file and the prototype banks into a generated asset
 * directory, renaming them to the names the app expects. Keeping the models out of `assets/` means
 * one export serves both builds and neither can be selected by accident.
 */
val stageModelAssets by tasks.registering(Sync::class) {
    from(stagedModelDir.file("hcc_classifier_$modelPrecision.tflite")) { rename { "hcc_classifier.tflite" } }
    from(stagedModelDir.file("model_meta_$modelPrecision.json")) { rename { "model_meta.json" } }
    from(stagedModelDir) { include("proto_*.bin") }
    into(generatedAssetDir)
    doFirst {
        val missing = stagedModelDir.asFile.listFiles().orEmpty().map { it.name }
        check("hcc_classifier_$modelPrecision.tflite" in missing) {
            "model asset for precision '$modelPrecision' is missing. Run: python android_app/tools/export_tflite.py"
        }
    }
}

android {
    namespace = "com.kimhs.liverdx"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.kimhs.liverdx"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.2.0"
    }

    // The flatbuffer ships uncompressed so it can be memory-mapped.
    androidResources { noCompress += "tflite" }

    buildFeatures { compose = true }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions { jvmTarget = "17" }

    sourceSets["main"].assets.srcDir(generatedAssetDir)

    buildTypes {
        debug {
            isMinifyEnabled = false
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
        }
        // Signed with the debug key so it installs for debugging, but shrunk and arm64-only so it is
        // small enough to hand out.
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            signingConfig = signingConfigs.getByName("debug")
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            ndk { abiFilters += listOf("arm64-v8a") }
        }
    }

    packaging {
        resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" }
    }
}

tasks.named("preBuild") { dependsOn(stageModelAssets) }
tasks.configureEach {
    if (name.startsWith("merge") && name.endsWith("Assets")) dependsOn(stageModelAssets)
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.exifinterface:exifinterface:1.3.7")
    implementation("org.tensorflow:tensorflow-lite:2.16.1")

    val cameraX = "1.4.1"
    implementation("androidx.camera:camera-camera2:$cameraX")
    implementation("androidx.camera:camera-lifecycle:$cameraX")
    implementation("androidx.camera:camera-view:$cameraX")

    debugImplementation("androidx.compose.ui:ui-tooling")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}
