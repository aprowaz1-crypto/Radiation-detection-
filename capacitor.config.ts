import type { CapacitorConfig } from '@capacitor/cli'

const config: CapacitorConfig = {
  appId: 'com.aprowaz1crypto.raddetection',
  appName: 'Radiation Detection',
  webDir: 'dist',
  server: {
    androidScheme: 'https'
  }
}

export default config
