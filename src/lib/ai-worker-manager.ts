// Manager for AI Web Worker communication
export type AiEventClass = 'noise' | 'gamma-quantum' | 'beta-particle' | 'alpha-star'
export type ShapeClass = 'round' | 'line-zigzag' | 'star-burst' | 'unknown'

export type EventAiResult = {
  label: AiEventClass
  confidence: number
  logits: number[]
}

export type DriftResult = {
  adjustedBrightnessThreshold: number
  adjustedDeltaThreshold: number
  overheatingScore: number
}

export type ShapeResult = {
  label: ShapeClass
  confidence: number
}

class AiWorkerManager {
  private worker: Worker | null = null
  private pendingRequests = new Map<number, { resolve: (value: any) => void; reject: (error: any) => void }>()
  private requestId = 0
  private initPromise: Promise<void> | null = null

  async initialize(): Promise<void> {
    if (this.initPromise) return this.initPromise

    this.initPromise = new Promise((resolve, reject) => {
      try {
        // Lazy-load worker only when needed
        this.worker = new Worker(new URL('./ai.worker.ts', import.meta.url), { type: 'module' })

        this.worker.onmessage = (event: MessageEvent) => {
          const { type, id, result, error } = event.data

          if (type === 'ready') {
            resolve()
            return
          }

          if (type === 'error') {
            const pending = this.pendingRequests.get(id)
            if (pending) {
              pending.reject(new Error(error))
              this.pendingRequests.delete(id)
            }
            return
          }

          if (type === 'classify-result' || type === 'drift-result' || type === 'shape-result') {
            const pending = this.pendingRequests.get(id)
            if (pending) {
              pending.resolve(result)
              this.pendingRequests.delete(id)
            }
          }
        }

        this.worker.onerror = (error) => {
          reject(error)
        }

        // Send init message
        this.worker.postMessage({ type: 'init' })
      } catch (error) {
        reject(error)
      }
    })

    return this.initPromise
  }

  async classifyEvent(window10x10: number[]): Promise<EventAiResult> {
    if (!this.worker) {
      await this.initialize()
    }

    const id = ++this.requestId
    const promise = new Promise<EventAiResult>((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject })
    })

    this.worker!.postMessage({
      type: 'classify',
      window10x10,
      id
    })

    return promise
  }

  async analyzeDrift(
    brightnessMean: number,
    sigma: number,
    hotPixelRatio: number,
    currentBrightnessThreshold: number,
    currentDeltaThreshold: number
  ): Promise<DriftResult> {
    if (!this.worker) {
      await this.initialize()
    }

    const id = ++this.requestId
    const promise = new Promise<DriftResult>((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject })
    })

    this.worker!.postMessage({
      type: 'analyze-drift',
      brightnessMean,
      sigma,
      hotPixelRatio,
      currentBrightnessThreshold,
      currentDeltaThreshold,
      id
    })

    return promise
  }

  async classifyShape(size: number, width: number, height: number, peak: number, intensity: number): Promise<ShapeResult> {
    if (!this.worker) {
      await this.initialize()
    }

    const id = ++this.requestId
    const promise = new Promise<ShapeResult>((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject })
    })

    this.worker!.postMessage({
      type: 'classify-shape',
      size,
      width,
      height,
      peak,
      intensity,
      id
    })

    return promise
  }

  terminate(): void {
    if (this.worker) {
      this.worker.terminate()
      this.worker = null
    }
    this.pendingRequests.clear()
  }
}

export const aiWorkerManager = new AiWorkerManager()
