// Dedicated Web Worker for AI inference to avoid blocking main thread
let aiModule: any = null
let tf: any = null
let initialized = false

async function initializeTensorFlow() {
  if (initialized) return
  
  try {
    tf = await import('@tensorflow/tfjs')
    await initializeAi()
    initialized = true
    self.postMessage({ type: 'ready' })
  } catch (error) {
    self.postMessage({ type: 'error', error: (error as Error).message })
  }
}

async function initializeAi() {
  if (!tf) return

  // Copy MiniAi class logic here to avoid importing App code
  const LABELS = ['noise', 'gamma-quantum', 'beta-particle', 'alpha-star']

  class MiniAiWorker {
    eventModel: any
    driftModel: any

    constructor() {
      this.eventModel = this.buildEventModel()
      this.driftModel = this.buildDriftModel()
    }

    classifyEvent(window10x10: number[]) {
      if (window10x10.length !== 100) {
        return { label: 'noise', confidence: 1, logits: [1, 0, 0, 0] }
      }

      return tf.tidy(() => {
        const tensor = tf.tensor4d(window10x10, [1, 10, 10, 1], 'float32')
        const normalized = tf.div(tensor, 255)
        const prediction = this.eventModel.predict(normalized)
        const logits = Array.from(prediction.dataSync() as Float32Array)
        const maxIndex = logits.reduce((bestIndex: number, value: number, index: number) => 
          value > logits[bestIndex] ? index : bestIndex, 0)
        return {
          label: LABELS[maxIndex],
          confidence: logits[maxIndex],
          logits
        }
      })
    }

    analyzeDrift(
      brightnessMean: number,
      sigma: number,
      hotPixelRatio: number,
      currentBrightnessThreshold: number,
      currentDeltaThreshold: number
    ) {
      return tf.tidy(() => {
        const features = tf.tensor2d(
          [[brightnessMean / 255, sigma / 128, hotPixelRatio, currentBrightnessThreshold / 255, currentDeltaThreshold / 255]],
          [1, 5],
          'float32'
        )
        const output = this.driftModel.predict(features)
        const [overheatingScore, brightnessShift, deltaShift] = Array.from(output.dataSync() as Float32Array)

        const adjustedBrightnessThreshold = clamp(Math.round(currentBrightnessThreshold + brightnessShift * 24), 110, 255)
        const adjustedDeltaThreshold = clamp(Math.round(currentDeltaThreshold + deltaShift * 18), 8, 255)

        return {
          adjustedBrightnessThreshold,
          adjustedDeltaThreshold,
          overheatingScore: clamp(overheatingScore, 0, 1)
        }
      })
    }

    private buildEventModel() {
      const input = tf.input({ shape: [10, 10, 1] })
      const conv1 = tf.layers.conv2d({ filters: 8, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(input)
      const pool1 = tf.layers.maxPooling2d({ poolSize: 2 }).apply(conv1)
      const conv2 = tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(pool1)
      const flat = tf.layers.flatten().apply(conv2)
      const dense = tf.layers.dense({ units: 32, activation: 'relu' }).apply(flat)
      const output = tf.layers.dense({ units: 4, activation: 'softmax' }).apply(dense)

      const model = tf.model({ inputs: input, outputs: output })

      const kernel = tf.tensor4d(buildPatternKernels(), [3, 3, 1, 8], 'float32')
      const bias = tf.zeros([8])
      const convLayer = model.layers[1]
      convLayer.setWeights([kernel, bias])

      const denseLayer = model.layers[5]
      const outputLayer = model.layers[6]
      denseLayer.setWeights([
        tf.randomUniform([400, 32], -0.08, 0.08, 'float32', 11),
        tf.randomUniform([32], -0.02, 0.02, 'float32', 12)
      ])
      outputLayer.setWeights([
        tf.randomUniform([32, 4], -0.1, 0.1, 'float32', 13),
        tf.tensor1d([0.7, 0.2, 0.2, 0.15], 'float32')
      ])

      return model
    }

    private buildDriftModel() {
      const input = tf.input({ shape: [5] })
      const dense1 = tf.layers.dense({ units: 8, activation: 'relu' }).apply(input)
      const output = tf.layers.dense({ units: 3, activation: 'tanh' }).apply(dense1)
      const model = tf.model({ inputs: input, outputs: output })

      model.layers[1].setWeights([
        tf.tensor2d([
          [0.9, 0.2, -0.1, 0.7, 0.1, 0.4, -0.2, 0.3],
          [0.6, 0.8, 0.1, 0.5, -0.4, 0.1, 0.2, -0.1],
          [0.4, 0.7, 0.9, 0.6, 0.2, -0.2, 0.3, 0.1],
          [0.1, 0.5, -0.2, 0.2, 0.3, 0.2, -0.3, 0.4],
          [0.2, 0.4, 0.8, 0.2, 0.2, -0.1, 0.1, 0.1]
        ], [5, 8], 'float32'),
        tf.tensor1d([0, 0, 0, 0, 0, 0, 0, 0], 'float32')
      ])

      model.layers[2].setWeights([
        tf.tensor2d([
          [0.7, 0.6, 0.4],
          [0.5, 0.5, 0.3],
          [0.8, 0.6, 0.7],
          [0.4, 0.3, 0.2],
          [0.2, 0.3, 0.4],
          [0.3, 0.2, 0.3],
          [0.1, 0.3, 0.2],
          [0.2, 0.2, 0.1]
        ], [8, 3], 'float32'),
        tf.tensor1d([0.05, 0.03, 0.02], 'float32')
      ])

      return model
    }
  }

  aiModule = new MiniAiWorker()
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function buildPatternKernels(): number[] {
  const kernels = [
    [0, 1, 0, 1, 4, 1, 0, 1, 0],
    [1, 0, -1, 2, 0, -2, 1, 0, -1],
    [1, 2, 1, 0, 0, 0, -1, -2, -1],
    [0, 0, 1, 0, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, -1, 0, -1, 5, -1, 0, -1, 0],
    [0.5, 0, 0, 0, 1, 0, 0, 0, 0.5],
    [0, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 0]
  ]
  return kernels.flat()
}

// Message handler for main thread
self.onmessage = async (event: MessageEvent) => {
  const {
    type,
    window10x10,
    brightnessMean,
    sigma,
    hotPixelRatio,
    currentBrightnessThreshold,
    currentDeltaThreshold,
    size,
    width,
    height,
    peak,
    intensity,
    id
  } = event.data

  if (type === 'init') {
    await initializeTensorFlow()
    return
  }

  if (!initialized) {
    self.postMessage({ type: 'error', error: 'AI not initialized', id })
    return
  }

  try {
    if (type === 'classify') {
      const result = aiModule.classifyEvent(window10x10)
      self.postMessage({ type: 'classify-result', result, id })
    } else if (type === 'classify-shape') {
      const aspectRatio = Math.max(width, height) / Math.max(1, Math.min(width, height))
      const concentration = peak / Math.max(intensity, 1)
      let label: 'round' | 'line-zigzag' | 'star-burst' | 'unknown' = 'unknown'
      let confidence = 0.55

      if (size >= 8 && concentration < 0.2 && aspectRatio < 2.4) {
        label = 'star-burst'
        confidence = 0.72
      } else if (aspectRatio >= 3 || size >= 12) {
        label = 'line-zigzag'
        confidence = 0.76
      } else if (size >= 2 && aspectRatio < 3) {
        label = 'round'
        confidence = 0.74
      }

      self.postMessage({ type: 'shape-result', result: { label, confidence }, id })
    } else if (type === 'analyze-drift') {
      const result = aiModule.analyzeDrift(
        brightnessMean,
        sigma,
        hotPixelRatio,
        currentBrightnessThreshold,
        currentDeltaThreshold
      )
      self.postMessage({ type: 'drift-result', result, id })
    }
  } catch (error) {
    self.postMessage({ type: 'error', error: (error as Error).message, id })
  }
}

// Initialize on startup
initializeTensorFlow()
