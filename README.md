# Radiation Detection Web

Live link (GitHub Pages):
- https://aprowaz1-crypto.github.io/Radiation-detection-/

## Quick Start

```bash
npm install
npm run dev
```

Open local app:
- http://localhost:5173/

## Android APK (Capacitor)

This project can be packaged as a native Android app (APK) using Capacitor.

### Prerequisites

- Node.js + npm
- Java 21 (recommended for current Android Gradle setup)
- Android Studio with Android SDK installed

### Build flow

1. Install dependencies:

```bash
npm install
```

2. Build web assets for Android (relative base path):

```bash
npm run build:android
```

3. Sync web build into native Android project:

```bash
npm run cap:sync
```

4. Open Android project in Android Studio:

```bash
npm run cap:open:android
```

5. In Android Studio:
- Let Gradle sync finish
- Build `app` variant `debug` or `release`
- APK output is usually in `android/app/build/outputs/apk/`

### One-command refresh before native build

```bash
npm run android
```

This command rebuilds web assets and syncs them into `android/`.

### Build APK via GitHub Actions

Repository includes workflow:
- `.github/workflows/android-apk.yml`

How to use:
1. Open GitHub repo -> `Actions` -> `Build Android APK`
2. Click `Run workflow` (or push to `main`)
3. After run completes, download artifact `radiation-detection-debug-apk`

APK path inside artifact:
- `android/app/build/outputs/apk/debug/app-debug.apk`

## Deploy (auto)

This repo is configured with GitHub Actions workflow:
- `.github/workflows/deploy-pages.yml`

On every push to `main`, it builds and deploys `dist` to GitHub Pages.

## Enable GitHub Pages once

In GitHub repository settings:
1. Go to `Settings` -> `Pages`
2. In `Build and deployment`, set `Source` to `GitHub Actions`

After this, push to `main` and the live link above will start working.

## Ambient Light Sensor (native)

For native ALS mode on Android Chrome:
1. Open app over HTTPS (GitHub Pages link above)
2. In Chrome open `chrome://flags/#enable-generic-sensor-extra-classes`
3. Enable the flag and restart Chrome
4. Open the app again and use `Ambient Light Sensor`

If API is not available in your Chrome/ROM build, this mode may remain unavailable even after enabling the flag.
