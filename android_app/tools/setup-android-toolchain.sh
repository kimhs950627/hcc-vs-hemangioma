#!/usr/bin/env bash
set -euo pipefail
DEV="$HOME/android-dev"
cd "$DEV"

echo "[1/6] extract JDK"
JDK_DIR=$(ls -d jdk-17* 2>/dev/null | head -1 || true)
if [ -z "$JDK_DIR" ]; then
  mkdir -p jdk17 && tar xzf jdk17.tar.gz -C jdk17 --strip-components=1
  JDK_DIR=jdk17
fi
export JAVA_HOME="$DEV/$JDK_DIR"
echo "JAVA_HOME=$JAVA_HOME"
"$JAVA_HOME/bin/java" -version

echo "[2/6] extract cmdline-tools"
if [ ! -d sdk/cmdline-tools/latest ]; then
  mkdir -p sdk/cmdline-tools
  python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" cmdline-tools.zip sdk/cmdline-tools
  mv sdk/cmdline-tools/cmdline-tools sdk/cmdline-tools/latest
fi
echo "[3/6] gradle"
if [ ! -d gradle-8.11.1 ]; then
  curl -sL -o gradle.zip https://services.gradle.org/distributions/gradle-8.11.1-bin.zip
  python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" gradle.zip .
fi
chmod +x gradle-8.11.1/bin/gradle
chmod +x sdk/cmdline-tools/latest/bin/* 2>/dev/null || true
export PATH="$DEV/gradle-8.11.1/bin:$PATH"
gradle --version | head -6

echo "[4/6] licenses"
yes | sdk/cmdline-tools/latest/bin/sdkmanager --sdk_root="$DEV/sdk" --licenses > licenses.log 2>&1 || true
tail -3 licenses.log

echo "[5/6] install packages"
sdk/cmdline-tools/latest/bin/sdkmanager --sdk_root="$DEV/sdk" \
  "platform-tools" "platforms;android-35" "build-tools;35.0.0" > sdk_install.log 2>&1
tail -5 sdk_install.log

echo "[6/6] done"
ls "$DEV/sdk"
echo "SETUP_OK"
