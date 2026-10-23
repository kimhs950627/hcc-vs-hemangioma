# Toolchain setup — Linux (or WSL) without root

Installs everything `./gradlew :app:assembleDebug` needs into `$HOME/android-dev`, using plain
tarballs so no package manager or admin rights are required. Verified on 2026-10-18: JDK 17
(Temurin 17.0.20.1), Android cmdline-tools 11076708, SDK platform 35, build-tools 35.0.0,
Gradle 8.11.1.

```bash
bash tools/setup-android-toolchain.sh
```

Then build with:

```bash
cd android_app
JAVA_HOME=$HOME/android-dev/jdk17 ANDROID_HOME=$HOME/android-dev/sdk \
  $HOME/android-dev/gradle-8.11.1/bin/gradle :app:assembleDebug
```

Notes:

- `python3` is used to unzip the archives (no `unzip` dependency).
- Zipfile extraction loses the exec bit, so the script re-applies `chmod +x` to `gradle` and the
  `sdkmanager` binaries.
- Re-running the script is safe: each step is skipped when its target directory already exists.
