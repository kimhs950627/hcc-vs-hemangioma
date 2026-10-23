# TensorFlow Lite reaches its native interpreter through the Java API; keep the whole package.
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

# CameraX, AndroidX and Compose ship their own consumer rules, so nothing else is needed here.
