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
