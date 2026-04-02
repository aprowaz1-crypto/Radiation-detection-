import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig(({ mode }) => {
  const repoBase = '/Radiation-detection-/'
  const isProd = mode === 'production'
  const isAndroid = mode === 'android'
  const base = isAndroid ? './' : (isProd ? repoBase : '/')

  return {
    base,
    plugins: [
      react(),
      VitePWA({
        registerType: 'autoUpdate',
        includeAssets: ['icon.svg'],
        manifest: {
          name: 'Radiation Detection Web',
          short_name: 'RadDetect',
          description: 'Mobile-first web app for detecting short bright camera events in a darkened sensor stream.',
          theme_color: '#101814',
          background_color: '#0b100d',
          display: 'standalone',
          start_url: base,
          icons: [
            {
              src: 'icon.svg',
              sizes: 'any',
              type: 'image/svg+xml',
              purpose: 'any maskable'
            }
          ]
        }
      })
    ],
    worker: {
      format: 'es'
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            'tensorflow': ['@tensorflow/tfjs'],
            'vendor': ['react', 'react-dom']
          }
        }
      }
    }
  }
})