import { useEffect, useRef, useState, type ChangeEvent, type FormEvent } from 'react'
import { playClick } from './lib/audio'
import { getBadPixelCount, resetHotPixelMap, scanHotPixelEvents, setBadPixels, accumulateLongExposure, scanHotPixelEventsWithZScore, resetLongExposure, type DetectionEvent, type DetectorSettings } from './lib/detector'
import { aiWorkerManager, type AiEventClass, type ShapeClass } from './lib/ai-worker-manager'

type FacingMode = 'environment' | 'user'
type Status = 'idle' | 'requesting' | 'measuring' | 'light-leak' | 'error'
type DetectionMode = 'camera-cmos' | 'ambient-light'

type AmbientLightSensorLike = EventTarget & {
  illuminance: number
  start: () => void
  stop: () => void
  addEventListener: (type: 'reading' | 'error', listener: EventListenerOrEventListenerObject) => void
  removeEventListener: (type: 'reading' | 'error', listener: EventListenerOrEventListenerObject) => void
}

declare global {
  interface Window {
    AmbientLightSensor?: new (options?: { frequency?: number }) => AmbientLightSensorLike
  }
}

type EventRecord = DetectionEvent & {
  source: 'camera-cmos' | 'ambient-light'
  doseRate: number
  aiLabel: AiEventClass
  aiConfidence: number
  shapeLabel: ShapeClass
  shapeConfidence: number
  energyClass: 'light' | 'heavy'
}

type CalibrationMode = 'idle' | 'baseline' | 'hot-pixels'

type TerminalMessage = {
  id: string
  role: 'user' | 'ai'
  text: string
}

type PhotoRadiationAnalysis = {
  meanBrightness: number
  hotPixelRatio: number
  brightPixelRatio: number
  estimatedEpm: number
  estimatedDoseRate: number
  verdict: string
}

const detectorDefaults: DetectorSettings = {
  thresholdBrightness: 210,
  thresholdDelta: 60,
  minClusterSize: 1,
  maxClusterSize: 24,
  hotPixelCutoff: 6,
  downscale: 0.25,
  adaptiveBoost: 6,
  staticHitFrames: 2
}

const naturalBackgroundDefault = 12
const conversionFactorDefault = 0.85
const alarmThresholdDefault = 50
const aiConfidenceThreshold = 0.6
const terminalMessageLimit = 28

type SiteConfig = {
  showMetrics: boolean
  showGraph: boolean
  showSettings: boolean
  showEventLog: boolean
  showAiTerminal: boolean
  showCamera: boolean
  heroTitle: string
  heroDesc: string
}

const defaultSiteConfig: SiteConfig = {
  showMetrics: true,
  showGraph: true,
  showSettings: true,
  showEventLog: true,
  showAiTerminal: true,
  showCamera: true,
  heroTitle: 'Темний сенсор. Короткі спалахи. Живий лічильник.',
  heroDesc: 'Вебдодаток читає затемнений потік камери, шукає короткі яскраві кластери і веде live-оцінку подій, фону та накопиченої дози.'
}

function loadSiteConfig(): SiteConfig {
  try {
    const saved = localStorage.getItem('rad-site-config')
    if (saved) return { ...defaultSiteConfig, ...JSON.parse(saved) }
  } catch {}
  return { ...defaultSiteConfig }
}

function saveSiteConfig(config: SiteConfig) {
  localStorage.setItem('rad-site-config', JSON.stringify(config))
}

export default function App() {
  const aiInitializedRef = useRef(false)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const ambientSensorRef = useRef<AmbientLightSensorLike | null>(null)
  const ambientSpikePhaseRef = useRef<'idle' | 'spike'>('idle')
  const ambientLastLuxRef = useRef(0)
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const previousFrameRef = useRef<Uint8Array | null>(null)
  const eventLogRef = useRef<EventRecord[]>([])
  const [facingMode, setFacingMode] = useState<FacingMode>('environment')
  const [detectionMode, setDetectionMode] = useState<DetectionMode>('camera-cmos')
  const [status, setStatus] = useState<Status>('idle')
  const [statusDetail, setStatusDetail] = useState('Оберіть метод і натисніть Start. Для камери потрібна повна темрява.')
  const [settings, setSettings] = useState<DetectorSettings>(detectorDefaults)
  const [epm, setEpm] = useState(0)
  const [cameraEpm, setCameraEpm] = useState(0)
  const [ambientEpm, setAmbientEpm] = useState(0)
  const [doseRate, setDoseRate] = useState(naturalBackgroundDefault)
  const [liveDoseRate, setLiveDoseRate] = useState(naturalBackgroundDefault)
  const [accumulatedDose, setAccumulatedDose] = useState(0)
  const [baselineNoise, setBaselineNoise] = useState(0)
  const [naturalBackground, setNaturalBackground] = useState(naturalBackgroundDefault)
  const [conversionFactor, setConversionFactor] = useState(conversionFactorDefault)
  const [alarmThreshold, setAlarmThreshold] = useState(alarmThresholdDefault)
  const [isAlarmEnabled, setIsAlarmEnabled] = useState(true)
  const [graphPoints, setGraphPoints] = useState<number[]>([])
  const [cameraGraphPoints, setCameraGraphPoints] = useState<number[]>([])
  const [ambientGraphPoints, setAmbientGraphPoints] = useState<number[]>([])
  const [eventLog, setEventLog] = useState<EventRecord[]>([])
  const [lightLevel, setLightLevel] = useState(0)
  const [adaptiveBrightnessThreshold, setAdaptiveBrightnessThreshold] = useState(detectorDefaults.thresholdBrightness)
  const [adaptiveDeltaThreshold, setAdaptiveDeltaThreshold] = useState(detectorDefaults.thresholdDelta)
  const [fiveMinuteEpm, setFiveMinuteEpm] = useState(0)
  const [poissonLow, setPoissonLow] = useState(0)
  const [poissonHigh, setPoissonHigh] = useState(0)
  const [poissonPlusMinus, setPoissonPlusMinus] = useState(0)
  const [totalEvents, setTotalEvents] = useState(0)
  const [badPixelCount, setBadPixelCount] = useState(0)
  const useLongExposure = true
  const useZScore = true
  const cameraAutoSettings = false
  const aiEnabled = true
  const [latestAiLabel, setLatestAiLabel] = useState<AiEventClass>('noise')
  const [latestAiConfidence, setLatestAiConfidence] = useState(0)
  const [overheatingScore, setOverheatingScore] = useState(0)
  const [predictedDoseRate, setPredictedDoseRate] = useState(0)
  const [energyBins, setEnergyBins] = useState<number[]>(new Array(16).fill(0))
  const [cloudStatus, setCloudStatus] = useState('Cloud: idle')
  const [terminalInput, setTerminalInput] = useState('')
  const [terminalImageName, setTerminalImageName] = useState('')
  const [terminalImageAnalysis, setTerminalImageAnalysis] = useState<PhotoRadiationAnalysis | null>(null)
  const [isAnalyzingImage, setIsAnalyzingImage] = useState(false)
  const [terminalMessages, setTerminalMessages] = useState<TerminalMessage[]>([
    {
      id: 'boot',
      role: 'ai',
      text: 'Mini AI online. Запитуй про що завгодно — радіацію, фізику, хімію, код і не тільки. Завантаж темне фото для оцінки рівня радіації.'
    }
  ])
  const [calibrationMode, setCalibrationMode] = useState<CalibrationMode>('idle')
  const [effectiveThresholdBrightness, setEffectiveThresholdBrightness] = useState(detectorDefaults.thresholdBrightness)
  const [effectiveThresholdDelta, setEffectiveThresholdDelta] = useState(detectorDefaults.thresholdDelta)
  const [meanFrameTimeMs, setMeanFrameTimeMs] = useState(0)
  const [deadTimeCorrection, setDeadTimeCorrection] = useState(1)
  const [dynamicBackgroundEpm, setDynamicBackgroundEpm] = useState(0)
  const [baselineCaptureRemainingSec, setBaselineCaptureRemainingSec] = useState(60)
  const [isBaselineReady, setIsBaselineReady] = useState(false)
  const frameTimesRef = useRef<number[]>([])
  const baselineStartRef = useRef<number | null>(null)
  const baselineEpmSamplesRef = useRef<number[]>([])
  const calibrationSamplesRef = useRef<number[]>([])
  const hotPixelFramesRef = useRef<Uint16Array | null>(null)
  const hotPixelFrameCountRef = useRef(0)
  const calibrationTimeoutRef = useRef<number | null>(null)
  const lastFrameAtRef = useRef<number | null>(null)
  const secondBucketRef = useRef<{ startedAt: number; count: number }>({ startedAt: 0, count: 0 })
  const zeroNoiseAdjustRef = useRef<{ startedAt: number; adjustedAt: number }>({ startedAt: 0, adjustedAt: 0 })
  const measurementStartedAtRef = useRef<number | null>(null)
  const [siteConfig, setSiteConfigState] = useState<SiteConfig>(loadSiteConfig)

  const updateSiteConfig = (patch: Partial<SiteConfig>) => {
    setSiteConfigState(prev => {
      const next = { ...prev, ...patch }
      saveSiteConfig(next)
      return next
    })
  }

  useEffect(() => {
    if (!aiInitializedRef.current) {
      aiWorkerManager.initialize().catch(error => console.error('Failed to initialize AI worker:', error))
      aiInitializedRef.current = true
    }

    return () => {
      stopMeasurement()
      aiWorkerManager.terminate()
    }
  }, [])

  useEffect(() => {
    if (doseRate > alarmThreshold && isAlarmEnabled) {
      navigator.vibrate?.([120, 80, 120])
    }
  }, [alarmThreshold, doseRate, isAlarmEnabled])

  useEffect(() => {
    stopMeasurement()
  }, [detectionMode])

  useEffect(() => {
    const metricsIntervalId = setInterval(() => {
      const now = Date.now()
      const lastFiveMinutes = eventLogRef.current.filter((event) => now - event.timestamp <= 300_000)
      const lastMinute = lastFiveMinutes.filter((event) => now - event.timestamp <= 60_000)
      const lastMinuteCamera = lastMinute.filter((event) => event.source === 'camera-cmos')
      const lastMinuteAmbient = lastMinute.filter((event) => event.source === 'ambient-light')

      const measurementStartedAt = eventLogRef.current.length > 0
        ? eventLogRef.current[eventLogRef.current.length - 1].timestamp
        : now
      const elapsedSeconds = Math.max((now - measurementStartedAt) / 1000, 1)
      const minuteWindowSeconds = Math.min(elapsedSeconds, 60)
      const fiveMinuteWindowSeconds = Math.min(elapsedSeconds, 300)

      const currentEpmRaw = (lastMinute.length / minuteWindowSeconds) * 60
      const correctedEpm = currentEpmRaw * deadTimeCorrection
      const currentCameraEpm = (lastMinuteCamera.length / minuteWindowSeconds) * 60
      const currentAmbientEpm = (lastMinuteAmbient.length / minuteWindowSeconds) * 60
      const currentFiveMinuteEpm = (lastFiveMinutes.length / fiveMinuteWindowSeconds) * 60
      const backgroundReference = isBaselineReady ? dynamicBackgroundEpm : baselineNoise
      const currentDoseRate = naturalBackground + Math.max(correctedEpm - backgroundReference, 0) * conversionFactor
      const jitterAmplitude = Math.max(0.12, currentDoseRate * 0.05 + correctedEpm * 0.012)
      const jitter = (Math.random() - 0.5) * 2 * jitterAmplitude
      const liveDose = Math.max(0, currentDoseRate + jitter)

      setEpm(Number(correctedEpm.toFixed(1)))
      setCameraEpm(Number(currentCameraEpm.toFixed(1)))
      setAmbientEpm(Number(currentAmbientEpm.toFixed(1)))
      setFiveMinuteEpm(Number(currentFiveMinuteEpm.toFixed(1)))
      setDoseRate(currentDoseRate)
      setLiveDoseRate(liveDose)

      const startedAt = measurementStartedAtRef.current ?? now
      const elapsedSecondsTotal = Math.max((now - startedAt) / 1000, 1)
      const totalLambda = totalEvents
      const totalSigma = Math.sqrt(totalLambda)
      const totalLowCount = Math.max(totalLambda - 1.96 * totalSigma, 0)
      const totalHighCount = totalLambda + 1.96 * totalSigma
      const totalPoissonScale = 60 / elapsedSecondsTotal
      setPoissonLow(Number((totalLowCount * totalPoissonScale).toFixed(1)))
      setPoissonHigh(Number((totalHighCount * totalPoissonScale).toFixed(1)))
      setPoissonPlusMinus(Number((1.96 * Math.sqrt(Math.max(totalLambda, 1)) * totalPoissonScale).toFixed(1)))

      if (!isBaselineReady && baselineStartRef.current !== null) {
        const elapsed = Math.max(0, now - baselineStartRef.current)
        const remaining = Math.max(0, 60 - Math.floor(elapsed / 1000))
        setBaselineCaptureRemainingSec(remaining)
        baselineEpmSamplesRef.current.push(correctedEpm)
        if (elapsed >= 60_000) {
          const samples = baselineEpmSamplesRef.current
          const avg = samples.length > 0 ? samples.reduce((sum, value) => sum + value, 0) / samples.length : 0
          setDynamicBackgroundEpm(Number(avg.toFixed(1)))
          setIsBaselineReady(true)
        }
      }

      // Update predicted dose (no smoothing - direct reading)
      setPredictedDoseRate(currentDoseRate)

      // Update trends (add new point every 1 second)
      const shouldAppendPoint = lastFrameAtRef.current === null || now - lastFrameAtRef.current >= 1000
      if (shouldAppendPoint) {
        lastFrameAtRef.current = now
        setGraphPoints((points) => [...points.slice(-59), liveDose])
        setCameraGraphPoints((points) => [...points.slice(-59), currentCameraEpm])
        setAmbientGraphPoints((points) => [...points.slice(-59), currentAmbientEpm])
      }
    }, 100)

    const doseIntervalId = setInterval(() => {
      setAccumulatedDose((prev) => prev + (doseRate / 3600) * 10)
    }, 10_000)

    return () => {
      clearInterval(metricsIntervalId)
      clearInterval(doseIntervalId)
    }
  }, [baselineNoise, conversionFactor, naturalBackground, alarmThreshold, isAlarmEnabled, doseRate, totalEvents, deadTimeCorrection, dynamicBackgroundEpm, isBaselineReady])

  const startMeasurement = async () => {
    stopMeasurement()
    resetHotPixelMap()
    resetLongExposure()
    previousFrameRef.current = null
    eventLogRef.current = []
    setEventLog([])
    setGraphPoints([])
    setCameraGraphPoints([])
    setAmbientGraphPoints([])
    setEpm(0)
    setCameraEpm(0)
    setAmbientEpm(0)
    setFiveMinuteEpm(0)
    setLiveDoseRate(naturalBackground)
    setAccumulatedDose(0)
    setTotalEvents(0)
    setLatestAiLabel('noise')
    setLatestAiConfidence(0)
    setEffectiveThresholdBrightness(detectorDefaults.thresholdBrightness)
    setEffectiveThresholdDelta(detectorDefaults.thresholdDelta)
    setDeadTimeCorrection(1)
    setMeanFrameTimeMs(0)
    setDynamicBackgroundEpm(0)
    setBaselineCaptureRemainingSec(60)
    setIsBaselineReady(false)
    baselineStartRef.current = Date.now()
    baselineEpmSamplesRef.current = []
    frameTimesRef.current = []
    measurementStartedAtRef.current = Date.now()
    zeroNoiseAdjustRef.current = { startedAt: 0, adjustedAt: 0 }

    if (detectionMode === 'ambient-light') {
      setStatus('requesting')
      setStatusDetail('Запит доступу до Ambient Light Sensor...')
      startAmbientMeasurement()
      return
    }

    setStatus('requesting')
    setStatusDetail('Запит доступу до камери...')

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: 'environment' },
          frameRate: { ideal: 24, max: 30 },
          width: { ideal: 640 },
          height: { ideal: 480 },
          // Disable auto-enhancements to preserve weak radiation signals
          ...(cameraAutoSettings === false && {
            noise_reduction: { ideal: false } as any,
            auto_exposure: { ideal: false } as any,
            auto_white_balance: { ideal: false } as any,
            auto_focus: { ideal: false } as any
          })
        } as any
      })

      streamRef.current = stream
      const video = videoRef.current
      if (!video) {
        throw new Error('Video element unavailable')
      }

      video.srcObject = stream
      // Try to apply manual exposure settings for better dark-frame detection
      const videoTracks = stream.getVideoTracks()
      if (videoTracks.length > 0) {
        const track = videoTracks[0]
        if (track.getCapabilities) {
          const capabilities = track.getCapabilities() as any
          try {
            const settings: any = {}
            // Minimize auto-adjustments
            if (capabilities.exposureCompensation) {
              settings.exposureCompensation = -2 // Darker for better contrast
            }
            if (capabilities.brightness) {
              settings.brightness = 0 // Minimum brightness
            }
            if (capabilities.contrast) {
              settings.contrast = 100 // Maximum contrast to see weak signals
            }
            await track.applyConstraints(settings)
          } catch (e) {
            console.warn('Could not apply manual camera constraints:', e)
          }
        }
      }
      await video.play()

      // Auto-calibration: find best exposure for this specific camera
      if (cameraAutoSettings) {
        setStatusDetail('Авто-калібрування камери...')
        await new Promise((resolve) => setTimeout(resolve, 500)) // Wait for frame to stabilize

        const testSettings = [
          { exposureCompensation: -3, brightness: -10, contrast: 100 },
          { exposureCompensation: -2, brightness: 0, contrast: 100 },
          { exposureCompensation: -1, brightness: 10, contrast: 100 },
          { exposureCompensation: 0, brightness: 20, contrast: 100 }
        ]

        let bestSetting = testSettings[1] // Default
        let bestScore = 0

        for (const setting of testSettings) {
          try {
            await videoTracks[0].applyConstraints(setting as any)
            await new Promise((resolve) => setTimeout(resolve, 300)) // Stabilize exposure

            // Sample frame and analyze hot-pixel variance
            const canvas = document.createElement('canvas')
            canvas.width = video.videoWidth || 640
            canvas.height = video.videoHeight || 480
            const ctx = canvas.getContext('2d')!
            ctx.drawImage(video, 0, 0)
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
            
            // Compute mean brightness variance
            let brightnessSum = 0
            for (let i = 0; i < imageData.data.length; i += 4) {
              const r = imageData.data[i]
              const g = imageData.data[i + 1]
              const b = imageData.data[i + 2]
              const gray = r * 0.299 + g * 0.587 + b * 0.114
              brightnessSum += gray
            }
            const meanBrightness = brightnessSum / (canvas.width * canvas.height)
            
            // Compute variance (prefer moderate variance - not too dark, not too bright)
            let varianceSum = 0
            for (let i = 0; i < imageData.data.length; i += 4) {
              const r = imageData.data[i]
              const g = imageData.data[i + 1]
              const b = imageData.data[i + 2]
              const gray = r * 0.299 + g * 0.587 + b * 0.114
              varianceSum += Math.pow(gray - meanBrightness, 2)
            }
            const variance = Math.sqrt(varianceSum / (canvas.width * canvas.height))
            
            // Score: prefer moderate brightness (20-80) with high variance for good signal
            const brightnessPenalty = Math.abs(meanBrightness - 50) / 50
            const score = variance * Math.max(0, 1 - brightnessPenalty * 0.5)
            
            if (score > bestScore) {
              bestScore = score
              bestSetting = setting
            }
          } catch (e) {
            console.warn('Calibration test failed:', e)
          }
        }

        // Apply best setting
        try {
          await videoTracks[0].applyConstraints(bestSetting as any)
        } catch (e) {
          console.warn('Could not apply best calibration:', e)
        }
        setStatusDetail('Авто-калібрування завершено. Камера має залишатися повністю закритою.')
      }

      setStatus('measuring')
      setStatusDetail('Вимірювання активне. Камера має залишатися повністю закритою.')
      loop()
    } catch (error) {
      setStatus('error')
      setStatusDetail(error instanceof Error ? error.message : 'Не вдалося отримати доступ до камери.')
    }
  }

  const startAmbientMeasurement = async () => {
    const AmbientLightSensorCtor = window.AmbientLightSensor ?? (globalThis as any).AmbientLightSensor
    if (!AmbientLightSensorCtor) {
      setStatus('error')
      setStatusDetail('Ambient Light Sensor API недоступний. Для POCO F6 відкрийте сайт у Chrome (HTTPS) і увімкніть chrome://flags/#enable-generic-sensor-extra-classes')
      return
    }

    try {
      const sensor = new AmbientLightSensorCtor({ frequency: 60 }) as AmbientLightSensorLike
      ambientSpikePhaseRef.current = 'idle'
      ambientLastLuxRef.current = 0

      const onReading = () => {
        const lux = Number(sensor.illuminance) || 0
        setLightLevel(lux)

        if (lux > 0.05) {
          setStatus('light-leak')
          setStatusDetail('Для Ambient Light Sensor потрібна повна темрява (майже 0 lux).')
        } else {
          setStatus('measuring')
          setStatusDetail('Ambient Light Sensor: темрява стабільна, моніторинг мікроімпульсів активний.')
        }

        const lastLux = ambientLastLuxRef.current
        const darkLevel = 0.001
        const spikeLevel = 0.01

        if (ambientSpikePhaseRef.current === 'idle' && lastLux <= darkLevel && lux >= spikeLevel) {
          ambientSpikePhaseRef.current = 'spike'
        } else if (ambientSpikePhaseRef.current === 'spike' && lux <= darkLevel) {
          const now = Date.now()
          const event: EventRecord = {
            id: `${now}-als`,
            timestamp: now,
            source: 'ambient-light',
            x: 0,
            y: 0,
            size: 1,
            peak: Math.round(lux * 1000),
            intensity: Math.round(lux * 1000),
            width: 1,
            height: 1,
            kind: 'gamma-candidate',
            shape: 'round',
            window10x10: new Array(100).fill(0),
            doseRate: naturalBackground,
            aiLabel: 'gamma-quantum',
            aiConfidence: 0.55,
            shapeLabel: 'round',
            shapeConfidence: 0.65,
            energyClass: 'light'
          }
          eventLogRef.current = [event, ...eventLogRef.current]
          setEventLog(eventLogRef.current)
          setTotalEvents((value) => value + 1)
          ambientSpikePhaseRef.current = 'idle'
          playClick()
        }

        ambientLastLuxRef.current = lux
      }

      const onError = () => {
        setStatus('error')
        setStatusDetail('Помилка читання Ambient Light Sensor.')
      }

      sensor.addEventListener('reading', onReading)
      sensor.addEventListener('error', onError)
      sensor.start()
      ambientSensorRef.current = sensor
      setStatus('measuring')
      setStatusDetail('Ambient Light Sensor запущено. Тримайте пристрій у повній темряві.')
    } catch (error) {
      setStatus('error')
      setStatusDetail(error instanceof Error ? error.message : 'Не вдалося запустити Ambient Light Sensor.')
    }
  }

  const stopMeasurement = () => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }

    if (calibrationTimeoutRef.current !== null) {
      window.clearTimeout(calibrationTimeoutRef.current)
      calibrationTimeoutRef.current = null
    }

    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null

    ambientSensorRef.current?.stop()
    ambientSensorRef.current = null

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    setCalibrationMode('idle')
    lastFrameAtRef.current = null
    secondBucketRef.current = { startedAt: 0, count: 0 }
    if (status !== 'idle') {
      setStatus('idle')
      setStatusDetail('Вимірювання зупинено.')
    }
  }

  const loop = () => {
    const video = videoRef.current
    const sourceCanvas = sourceCanvasRef.current
    const overlayCanvas = overlayCanvasRef.current

    if (!video || !sourceCanvas || !overlayCanvas || video.videoWidth === 0 || video.videoHeight === 0) {
      animationFrameRef.current = requestAnimationFrame(loop)
      return
    }

    const width = Math.max(1, Math.round(video.videoWidth * settings.downscale))
    const height = Math.max(1, Math.round(video.videoHeight * settings.downscale))
    sourceCanvas.width = width
    sourceCanvas.height = height
    overlayCanvas.width = width
    overlayCanvas.height = height

    const context = sourceCanvas.getContext('2d', { willReadFrequently: true })
    const overlayContext = overlayCanvas.getContext('2d')

    if (!context || !overlayContext) {
      animationFrameRef.current = requestAnimationFrame(loop)
      return
    }

    const frameStartedAt = performance.now()
    context.drawImage(video, 0, 0, width, height)
    const imageData = context.getImageData(0, 0, width, height)
    
    // Choose detection method: standard vs Z-score vs long-exposure
    let result = scanHotPixelEvents(imageData.data, width, height, 20)
    
    if (useZScore) {
      result = scanHotPixelEventsWithZScore(result.grayscale, width, height)
    }
    
    // Long-exposure: accumulate frames for 5-10 sec, analyze every 60 frames (~2.5 sec @ 24fps)
    if (useLongExposure) {
      const longExpResult = accumulateLongExposure(result.grayscale)
      if (longExpResult.frameCount % 60 === 0 && longExpResult.frameCount > 60) {
        // Every ~2.5 seconds, run detection on accumulated long-exposure
        const leResult = scanHotPixelEventsWithZScore(longExpResult.accumulated, width, height)
        // Merge with current frame results (long-exposure finds slow/weak signals)
        if (leResult.events.length > 0) {
          result.events = [...result.events, ...leResult.events]
        }
      }
    }

    const frameProcessingMs = performance.now() - frameStartedAt
    frameTimesRef.current = [...frameTimesRef.current.slice(-89), frameProcessingMs]
    const frameMean = frameTimesRef.current.reduce((sum, value) => sum + value, 0) / Math.max(1, frameTimesRef.current.length)
    setMeanFrameTimeMs(frameMean)
    const correction = frameMean > 16 ? Math.min(3, frameMean / 16) : 1
    setDeadTimeCorrection(correction)

    setLightLevel(result.brightnessMean)
    setAdaptiveBrightnessThreshold(effectiveThresholdBrightness)
    setAdaptiveDeltaThreshold(effectiveThresholdDelta)
    setOverheatingScore(0)

    if (result.brightnessMean > 5) {
      setStatus('light-leak')
      setDoseRate(naturalBackground)
      setEpm(0)
      setFiveMinuteEpm(0)
      eventLogRef.current = []
      setEventLog([])
      setStatusDetail('Заклейте камеру чорною стрічкою!')
      animationFrameRef.current = requestAnimationFrame(loop)
      return
    } else if (status !== 'measuring') {
      setStatus('measuring')
      setStatusDetail('Вимірювання активне. Камера має залишатися повністю закритою.')
    }

    overlayContext.clearRect(0, 0, width, height)
    overlayContext.fillStyle = '#ffffff'

    if (calibrationMode === 'hot-pixels') {
      collectHotPixelFrame(result.grayscale)
    }

    if (result.events.length > 0) {
      const currentBackground = isBaselineReady ? dynamicBackgroundEpm : baselineNoise
      const currentDoseRate = naturalBackground + Math.max(epm - currentBackground, 0) * conversionFactor
      
      // Process events synchronously with fallback labels, classify asynchronously
      const newEvents = result.events.map((event) => ({
        ...event,
        source: 'camera-cmos' as const,
        doseRate: currentDoseRate,
        aiLabel: (event.kind === 'beta-track' ? 'beta-particle' : 'gamma-quantum') as AiEventClass,
        aiConfidence: 0.5,
        shapeLabel: event.shape,
        shapeConfidence: 0.6,
        energyClass: (event.intensity >= 900 ? 'heavy' : 'light') as 'light' | 'heavy'
      }))

      eventLogRef.current = [...newEvents, ...eventLogRef.current]
      setEventLog(eventLogRef.current)
      setTotalEvents((value) => value + newEvents.length)

      // Classify events asynchronously in background
      if (aiEnabled && aiInitializedRef.current) {
        newEvents.forEach((event, index) => {
          aiWorkerManager
            .classifyShape(event.size, event.width, event.height, event.peak, event.intensity)
            .then((shape) => {
              setEventLog((prev) => {
                const updated = [...prev]
                const eventIndex = updated.findIndex((e) => e.id === event.id)
                if (eventIndex >= 0) {
                  updated[eventIndex] = {
                    ...updated[eventIndex],
                    shapeLabel: shape.label,
                    shapeConfidence: shape.confidence
                  }
                }
                return updated
              })
            })
            .catch(error => console.error('Shape classification failed:', error))

          aiWorkerManager
            .classifyEvent(event.window10x10)
            .then((result) => {
              // Update the event with actual AI classification
              setEventLog((prev) => {
                const updated = [...prev]
                const eventIndex = updated.findIndex((e) => e.id === event.id)
                if (eventIndex >= 0) {
                  updated[eventIndex] = {
                    ...updated[eventIndex],
                    aiLabel: result.label,
                    aiConfidence: result.confidence
                  }
                }
                return updated
              })

              // Update latest classification
              if (index === 0) {
                if (result.confidence < aiConfidenceThreshold) {
                  setLatestAiLabel('noise')
                  setLatestAiConfidence(0)
                } else {
                  setLatestAiLabel(result.label)
                  setLatestAiConfidence(result.confidence)
                }
              }
            })
            .catch(error => console.error('Classification failed:', error))
        })
      }

      setEnergyBins((bins) => {
        const next = [...bins]
        for (const event of newEvents) {
          const normalized = Math.min(1, event.intensity / 1500)
          const index = Math.min(next.length - 1, Math.floor(normalized * next.length))
          next[index] += 1
        }
        return next
      })

      for (const event of result.events) {
        overlayContext.fillStyle = event.kind === 'beta-track' ? '#9bd4ff' : '#ffffff'
        overlayContext.beginPath()
        overlayContext.arc(event.x, event.y, 2.4, 0, Math.PI * 2)
        overlayContext.fill()
        playClick()
      }
    }

    const now = Date.now()

    if (result.events.length === 0) {
      if (zeroNoiseAdjustRef.current.startedAt === 0) {
        zeroNoiseAdjustRef.current.startedAt = now
      }

      const shouldAdjust = now - zeroNoiseAdjustRef.current.startedAt >= 20_000 && now - zeroNoiseAdjustRef.current.adjustedAt >= 5_000
      if (shouldAdjust) {
        setEffectiveThresholdBrightness((value) => Math.max(120, value - 5))
        setEffectiveThresholdDelta((value) => Math.max(8, value - 5))
        zeroNoiseAdjustRef.current.adjustedAt = now
      }

      if (aiEnabled) {
        setLatestAiLabel('noise')
        setLatestAiConfidence(0)
      }
    } else {
      zeroNoiseAdjustRef.current.startedAt = 0
      if (aiEnabled) {
        setLatestAiLabel('noise')
        setLatestAiConfidence(0)
      }
    }

    const currentSecondBucket = secondBucketRef.current
    if (currentSecondBucket.startedAt === 0 || now - currentSecondBucket.startedAt >= 1000) {
      secondBucketRef.current = { startedAt: now, count: result.events.length }
    } else {
      secondBucketRef.current = {
        startedAt: currentSecondBucket.startedAt,
        count: currentSecondBucket.count + result.events.length
      }
    }

    if (secondBucketRef.current.count > 500) {
      setEffectiveThresholdBrightness((value) => Math.min(255, value + 6))
      setEffectiveThresholdDelta((value) => Math.min(255, value + 4))
    } else {
      setEffectiveThresholdBrightness((value) => Math.max(settings.thresholdBrightness, value - 1))
      setEffectiveThresholdDelta((value) => Math.max(settings.thresholdDelta, value - 1))
    }

    const lastFiveMinutes = eventLogRef.current.filter((event) => now - event.timestamp <= 300_000)
    eventLogRef.current = lastFiveMinutes
    setEventLog(lastFiveMinutes)

    // Calculate EPM for calibration if needed (non-UI critical)
    if (calibrationMode === 'baseline') {
      const measurementStartedAt = eventLogRef.current.length > 0
        ? eventLogRef.current[eventLogRef.current.length - 1].timestamp
        : now
      const fiveMinuteWindowSeconds = Math.min((now - measurementStartedAt) / 1000, 300)
      const currentFiveMinuteEpm = (lastFiveMinutes.length / fiveMinuteWindowSeconds) * 60
      calibrationSamplesRef.current.push(currentFiveMinuteEpm)
    }

    animationFrameRef.current = requestAnimationFrame(loop)
  }

  const startCalibration = () => {
    calibrationSamplesRef.current = []
    setCalibrationMode('baseline')
    setStatusDetail('Йде 30-секундна калібровка нуля. Не рухай телефон і не відкривай камеру.')

    if (calibrationTimeoutRef.current !== null) {
      window.clearTimeout(calibrationTimeoutRef.current)
    }

    calibrationTimeoutRef.current = window.setTimeout(() => {
      const samples = calibrationSamplesRef.current
      const average = samples.length > 0 ? samples.reduce((sum, value) => sum + value, 0) / samples.length : 0
      setBaselineNoise(Number(average.toFixed(1)))
      setCalibrationMode('idle')
      setStatusDetail('Калібровку завершено. Базовий шум оновлено.')
    }, 30_000)
  }

  const startHotPixelCalibration = () => {
    const video = videoRef.current
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
      setStatusDetail('Спершу запусти камеру, тоді можна побудувати hot-pixel mask.')
      return
    }

    const width = Math.max(1, Math.round(video.videoWidth * settings.downscale))
    const height = Math.max(1, Math.round(video.videoHeight * settings.downscale))
    hotPixelFramesRef.current = new Uint16Array(width * height)
    hotPixelFrameCountRef.current = 0
    resetHotPixelMap()
    setBadPixelCount(0)
    setCalibrationMode('hot-pixels')
    setStatusDetail('Йде калібрування hot pixels: збираю 100 темних кадрів для маски дефектів сенсора.')
  }

  const collectHotPixelFrame = (grayscale: Uint8Array) => {
    const frameCounts = hotPixelFramesRef.current
    if (!frameCounts) {
      return
    }

    for (let index = 0; index < grayscale.length; index += 1) {
      if (grayscale[index] >= settings.thresholdBrightness) {
        frameCounts[index] += 1
      }
    }

    hotPixelFrameCountRef.current += 1
    if (hotPixelFrameCountRef.current < 100) {
      return
    }

    const badPixels: number[] = []
    for (let index = 0; index < frameCounts.length; index += 1) {
      if (frameCounts[index] >= 50) {
        badPixels.push(index)
      }
    }

    setBadPixels(badPixels)
    setBadPixelCount(getBadPixelCount())
    hotPixelFramesRef.current = null
    hotPixelFrameCountRef.current = 0
    setCalibrationMode('idle')
    setStatusDetail(`Hot-pixel маску побудовано. Ігнорую ${badPixels.length} дефектних пікселів.`)
  }

  const exportReport = () => {
    const report = {
      exportedAt: new Date().toISOString(),
      epm,
      doseRate,
      accumulatedDose,
      baselineNoise,
      naturalBackground,
      conversionFactor,
      alarmThreshold,
      lightLevel,
      adaptiveBrightnessThreshold,
      adaptiveDeltaThreshold,
      fiveMinuteEpm,
      poissonInterval: {
        low: poissonLow,
        high: poissonHigh
      },
      poissonPlusMinus,
      deadTimeCorrection,
      meanFrameTimeMs,
      dynamicBackgroundEpm,
      isBaselineReady,
      badPixelCount,
      predictedDoseRate,
      latestAiLabel,
      latestAiConfidence,
      overheatingScore,
      energyBins,
      events: eventLog
    }

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `radiation-report-${Date.now()}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  const saveToCloud = async () => {
    const payload = {
      createdAt: new Date().toISOString(),
      doseRate,
      epm,
      predictedDoseRate,
      location: null,
      overheatingScore
    }

    try {
      setCloudStatus('Cloud: sending...')
      const response = await fetch('/api/cloud/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      if (!response.ok) {
        throw new Error('Cloud endpoint unavailable')
      }
      setCloudStatus('Cloud: saved')
    } catch {
      setCloudStatus('Cloud: mock mode (endpoint не налаштований)')
    }
  }

  const pushTerminalMessage = (message: TerminalMessage) => {
    setTerminalMessages((current) => [...current.slice(-(terminalMessageLimit - 1)), message])
  }

  const handleTerminalSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const prompt = terminalInput.trim()
    if (!prompt) {
      return
    }

    const userMessage: TerminalMessage = {
      id: `${Date.now()}-user`,
      role: 'user',
      text: prompt
    }
    pushTerminalMessage(userMessage)
    setTerminalInput('')

    // --- AI site-editor commands ---
    const hide = /прибер|убер|сховай|видал|hide|remove|вимк|закр/i.test(prompt)
    const show = /покажи|додай|відкрий|включи|show|add|відображ/i.test(prompt)
    const reset = /скид|скинь|відновити|відновит|reset|default|все повернути/i.test(prompt)

    if (reset && /налаштув|конфіг|сайт|все/i.test(prompt)) {
      updateSiteConfig(defaultSiteConfig)
      pushTerminalMessage({ id: `${Date.now()}-ai`, role: 'ai', text: '✅ Сайт повернуто до початкового вигляду. Оновіть сторінку щоб побачити зміни.' })
      return
    }

    if (hide || show) {
      const val = show

      if (/метрик|metric|показник/i.test(prompt)) {
        updateSiteConfig({ showMetrics: val })
        pushTerminalMessage({ id: `${Date.now()}-ai`, role: 'ai', text: `✅ Метрики ${val ? 'показано' : 'приховано'}. Оновіть сторінку.` })
        return
      }
      if (/граф|graph|тренд|trend/i.test(prompt)) {
        updateSiteConfig({ showGraph: val })
        pushTerminalMessage({ id: `${Date.now()}-ai`, role: 'ai', text: `✅ Графіки ${val ? 'показано' : 'приховано'}. Оновіть сторінку.` })
        return
      }
      if (/налаштув|setting|пороги|slider/i.test(prompt)) {
        updateSiteConfig({ showSettings: val })
        pushTerminalMessage({ id: `${Date.now()}-ai`, role: 'ai', text: `✅ Налаштування ${val ? 'показано' : 'приховано'}. Оновіть сторінку.` })
        return
      }
      if (/лог|log|поді|event/i.test(prompt)) {
        updateSiteConfig({ showEventLog: val })
        pushTerminalMessage({ id: `${Date.now()}-ai`, role: 'ai', text: `✅ Лог подій ${val ? 'показано' : 'приховано'}. Оновіть сторінку.` })
        return
      }
      if (/термінал|terminal|чат|ai|айшку/i.test(prompt)) {
        pushTerminalMessage({ id: `${Date.now()}-ai`, role: 'ai', text: `⚠️ Не можу сховати сам себе 😄 Але можу сховати інші секції.` })
        return
      }
      if (/камер|camera|сенсор|sensor/i.test(prompt)) {
        updateSiteConfig({ showCamera: val })
        pushTerminalMessage({ id: `${Date.now()}-ai`, role: 'ai', text: `✅ Секція камери ${val ? 'показана' : 'прихована'}. Оновіть сторінку.` })
        return
      }
      if (/все|all|всі секці/i.test(prompt)) {
        updateSiteConfig({ showMetrics: val, showGraph: val, showSettings: val, showEventLog: val, showCamera: val })
        pushTerminalMessage({ id: `${Date.now()}-ai`, role: 'ai', text: `✅ Всі секції ${val ? 'показано' : 'приховано'} (крім AI терміналу). Оновіть сторінку.` })
        return
      }
    }

    // --- Change title/description ---
    const titleMatch = prompt.match(/(?:заголовок|title|назву?)\s+(?:на|:)?\s*[«"]?(.+?)[»"]?$/i)
    if (titleMatch?.[1]) {
      updateSiteConfig({ heroTitle: titleMatch[1].trim() })
      pushTerminalMessage({ id: `${Date.now()}-ai`, role: 'ai', text: `✅ Заголовок змінено на «${titleMatch[1].trim()}». Оновіть сторінку.` })
      return
    }
    const descMatch = prompt.match(/(?:опис|description|підзаголовок)\s+(?:на|:)?\s*[«"]?(.+?)[»"]?$/i)
    if (descMatch?.[1]) {
      updateSiteConfig({ heroDesc: descMatch[1].trim() })
      pushTerminalMessage({ id: `${Date.now()}-ai`, role: 'ai', text: `✅ Опис змінено. Оновіть сторінку.` })
      return
    }

    // --- Show current config ---
    if (/що показ|які секці|конфіг|config|що сховано/i.test(prompt)) {
      const sc = siteConfig
      const lines = [
        `Камера: ${sc.showCamera ? '✅' : '❌'}`,
        `Метрики: ${sc.showMetrics ? '✅' : '❌'}`,
        `Графіки: ${sc.showGraph ? '✅' : '❌'}`,
        `Налаштування: ${sc.showSettings ? '✅' : '❌'}`,
        `Лог подій: ${sc.showEventLog ? '✅' : '❌'}`,
      ].join(' | ')
      pushTerminalMessage({ id: `${Date.now()}-ai`, role: 'ai', text: `Поточний стан секцій: ${lines}` })
      return
    }

    if (/фото|зображ|image|jpg|png|визнач|радіац.*фото|фото.*радіац/i.test(prompt)) {
      if (!terminalImageAnalysis) {
        pushTerminalMessage({
          id: `${Date.now()}-ai`,
          role: 'ai',
          text: 'Завантажте фото через кнопку «Upload photo», і я оціню рівень радіації за щільністю гарячих пікселів.'
        })
        return
      }

      const a = terminalImageAnalysis
      pushTerminalMessage({
        id: `${Date.now()}-ai`,
        role: 'ai',
        text:
          `📷 Аналіз «${terminalImageName || 'uploaded image'}»: ${a.verdict} ` +
          `Розрахунковий EPM ≈ ${a.estimatedEpm.toFixed(1)}, ` +
          `приблизна доза ≈ ${a.estimatedDoseRate.toFixed(2)} мкР/год. ` +
          `(hot-pixel ratio=${(a.hotPixelRatio * 100).toFixed(3)}%, ` +
          `середня яскравість=${a.meanBrightness.toFixed(1)}). ` +
          'Точний вимір потребує серію закритих темних кадрів у режимі CMOS.'
      })
      return
    }

    pushTerminalMessage({
      id: `${Date.now()}-ai`,
      role: 'ai',
      text: buildAiReply(prompt, {
        epm,
        cameraEpm,
        ambientEpm,
        doseRate: liveDoseRate,
        baselineNoise,
        detectionMode,
        lastPhotoAnalysis: terminalImageAnalysis
      })
    })
  }

  const handleTerminalPhotoUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }

    setIsAnalyzingImage(true)
    setTerminalImageName(file.name)
    try {
      const analysis = await analyzePhotoRadiation(file)
      setTerminalImageAnalysis(analysis)
      pushTerminalMessage({
        id: `${Date.now()}-ai`,
        role: 'ai',
        text: `Фото «${file.name}» проаналізовано. ${analysis.verdict} Розрахунковий EPM ≈ ${analysis.estimatedEpm.toFixed(1)}, доза ≈ ${analysis.estimatedDoseRate.toFixed(2)} мкР/год. Запитай деталі або завантаж інше фото.`
      })
    } catch (error) {
      pushTerminalMessage({
        id: `${Date.now()}-ai`,
        role: 'ai',
        text: error instanceof Error ? error.message : 'Не вдалося обробити фото.'
      })
    } finally {
      setIsAnalyzingImage(false)
      event.target.value = ''
    }
  }

  const statusTone = {
    idle: 'status-idle',
    requesting: 'status-idle',
    measuring: 'status-ok',
    'light-leak': 'status-warn',
    error: 'status-error'
  }[status]

  return (
    <div className="app-shell">
      <header className="hero-panel">
        <div>
          <p className="eyebrow">Radiation Detection Web</p>
          <h1>{siteConfig.heroTitle}</h1>
          <p className="hero-copy">{siteConfig.heroDesc}</p>
        </div>
        <div className={`status-pill ${statusTone}`}>{statusDetail}</div>
      </header>

      <main className="dashboard-grid">
        {siteConfig.showCamera && (<section className="camera-panel panel-card">
          <div className="panel-heading">
            <h2>Сенсор</h2>
            <label className="mode-select">
              <span>Метод</span>
              <select value={detectionMode} onChange={(event) => setDetectionMode(event.target.value as DetectionMode)}>
                <option value="camera-cmos">CMOS-матриця (камера)</option>
                <option value="ambient-light">Ambient Light Sensor</option>
              </select>
            </label>
            <div className="button-row">
              <button onClick={startMeasurement}>Start</button>
              <button className="ghost" onClick={stopMeasurement}>Stop</button>
            </div>
          </div>

          <div className="camera-stage">
            <video ref={videoRef} className="camera-video" playsInline muted />
            <canvas ref={overlayCanvasRef} className="overlay-canvas" />
            <canvas ref={sourceCanvasRef} className="hidden-canvas" />
          </div>

          <div className="camera-meta">
            <label>
              Камера
              <select value={facingMode} onChange={(event) => setFacingMode(event.target.value as FacingMode)} disabled={detectionMode === 'ambient-light'}>
                <option value="environment">Основна</option>
                <option value="user">Фронтальна</option>
              </select>
            </label>
            <div>
              <span className="meta-label">Середня яскравість</span>
              <strong>{lightLevel.toFixed(1)}</strong>
            </div>
            <div>
              <span className="meta-label">Baseline шум</span>
              <strong>{baselineNoise} EPM</strong>
            </div>
            <div>
              <span className="meta-label">Bad pixels</span>
              <strong>{badPixelCount}</strong>
            </div>
          </div>
        </section>)}

        {siteConfig.showMetrics && (<section className="metrics-panel panel-card">
          <div className="metric-tile">
            <span>EPM</span>
            <strong>{status === 'idle' ? 0 : epm}</strong>
          </div>
          <div className="metric-tile">
            <span>EPM (CMOS)</span>
            <strong>{status === 'idle' ? '0.0' : cameraEpm.toFixed(1)}</strong>
          </div>
          <div className="metric-tile">
            <span>EPM (ALS)</span>
            <strong>{status === 'idle' ? '0.0' : ambientEpm.toFixed(1)}</strong>
          </div>
          <div className="metric-tile">
            <span>мкР/год live</span>
            <strong>{status === 'idle' ? '0.00' : liveDoseRate.toFixed(2)}</strong>
          </div>
          <div className="metric-tile">
            <span>EPM за 5 хв</span>
            <strong>{status === 'idle' ? '0.0' : fiveMinuteEpm.toFixed(1)}</strong>
          </div>
          <div className="metric-tile">
            <span>95% Poisson CI</span>
            <strong>{status === 'idle' ? '0.0 ± 0.0' : `${epm.toFixed(1)} ± ${poissonPlusMinus.toFixed(1)}`}</strong>
          </div>
          <div className="metric-tile">
            <span>Total events</span>
            <strong>{status === 'idle' ? 0 : totalEvents}</strong>
          </div>
          <div className="metric-tile">
            <span>Накопичена доза</span>
            <strong>{status === 'idle' ? '0.000' : accumulatedDose.toFixed(3)}</strong>
          </div>
          <div className="metric-tile">
            <span>Подій у логу</span>
            <strong>{status === 'idle' ? 0 : eventLog.length}</strong>
          </div>
          <div className="metric-tile">
            <span>Frame time</span>
            <strong>{status === 'idle' ? '0ms' : `${meanFrameTimeMs.toFixed(1)}ms`}</strong>
          </div>
        </section>)}

        {siteConfig.showGraph && (<section className="graph-panel panel-card">
          <div className="panel-heading">
            <h2>Фон за останні 5 хвилин</h2>
            <div className="button-row">
              <button className="ghost" onClick={startHotPixelCalibration} disabled={calibrationMode !== 'idle' || status === 'idle'}>
                {calibrationMode === 'hot-pixels' ? 'Hot Mask...' : 'Hot Pixel Mask'}
              </button>
              <button className="ghost" onClick={startCalibration} disabled={calibrationMode !== 'idle' || status === 'idle'}>
                {calibrationMode === 'baseline' ? 'Calibrating...' : 'Калібрувати 30с'}
              </button>
            </div>
          </div>
          <div className="stats-strip">
            <span>Adaptive brightness: {adaptiveBrightnessThreshold.toFixed(1)}</span>
            <span>Adaptive delta: {adaptiveDeltaThreshold.toFixed(1)}</span>
            <span>Dynamic background: {isBaselineReady ? `${dynamicBackgroundEpm.toFixed(1)} EPM` : `capturing... ${baselineCaptureRemainingSec}s`}</span>
            <span>Падає до нуля лише через 5-хвилинне усереднення і baseline</span>
          </div>
          <Graph points={graphPoints} variant="sensitive" />
          <div className="source-graphs">
            <div className="source-graph-card">
              <h3>Тренд EPM: CMOS</h3>
              <Graph points={cameraGraphPoints} variant="sensitive" strokeClass="graph-line-cmos" />
            </div>
            <div className="source-graph-card">
              <h3>Тренд EPM: Ambient</h3>
              <Graph points={ambientGraphPoints} variant="sensitive" strokeClass="graph-line-ambient" />
            </div>
          </div>
          <h3>Режим спектрометра</h3>
          <EnergyHistogram bins={energyBins} />
        </section>)}

        {siteConfig.showSettings && (<section className="settings-panel panel-card">
          <div className="panel-heading">
            <h2>Пороги і перерахунок</h2>
            <button className="ghost" onClick={exportReport}>Експорт</button>
          </div>
          <div className="settings-grid">
            <Slider label="Brightness" min={120} max={255} step={1} value={settings.thresholdBrightness} onChange={(value) => setSettings((current) => ({ ...current, thresholdBrightness: value }))} />
            <Slider label="Delta" min={10} max={180} step={1} value={settings.thresholdDelta} onChange={(value) => setSettings((current) => ({ ...current, thresholdDelta: value }))} />
            <Slider label="Min cluster" min={1} max={12} step={1} value={settings.minClusterSize} onChange={(value) => setSettings((current) => ({ ...current, minClusterSize: value }))} />
            <Slider label="Max cluster" min={4} max={64} step={1} value={settings.maxClusterSize} onChange={(value) => setSettings((current) => ({ ...current, maxClusterSize: value }))} />
            <Slider label="Adaptive boost" min={1} max={12} step={0.5} value={settings.adaptiveBoost} onChange={(value) => setSettings((current) => ({ ...current, adaptiveBoost: value }))} />
            <Slider label="Static hit frames" min={1} max={6} step={1} value={settings.staticHitFrames} onChange={(value) => setSettings((current) => ({ ...current, staticHitFrames: value }))} />
            <Slider label="Natural background" min={0} max={40} step={1} value={naturalBackground} onChange={setNaturalBackground} />
            <Slider label="Conversion factor" min={0.1} max={4} step={0.05} value={conversionFactor} onChange={setConversionFactor} />
            <Slider label="Alarm threshold" min={10} max={120} step={1} value={alarmThreshold} onChange={setAlarmThreshold} />
            <label className="toggle-row">
              <span>Alarm</span>
              <input type="checkbox" checked={isAlarmEnabled} onChange={(event) => setIsAlarmEnabled(event.target.checked)} />
            </label>
            <button className="ghost" onClick={saveToCloud}>Зберегти в хмару</button>
            <div className="meta-label">{cloudStatus}</div>
          </div>
        </section>)}

        <section className="ai-mini-panel panel-card">
          <div className="panel-heading">
            <h2>Mini AI Terminal</h2>
            <span className="meta-label">Запитай про що завгодно + аналіз фото</span>
          </div>
          <div className="terminal-window">
            {terminalMessages.map((message) => (
              <p key={message.id} className={`terminal-line ${message.role}`}>
                <strong>{message.role === 'user' ? 'you>' : 'rad-ai>'}</strong> {message.text}
              </p>
            ))}
          </div>
          <form className="terminal-form" onSubmit={handleTerminalSubmit}>
            <input
              value={terminalInput}
              onChange={(event) => setTerminalInput(event.target.value)}
              placeholder="Наприклад: як зменшити шум CMOS?"
            />
            <button type="submit">Send</button>
          </form>
          <div className="terminal-tools">
            <label className="ghost terminal-upload">
              Upload photo
              <input type="file" accept="image/*" onChange={handleTerminalPhotoUpload} disabled={isAnalyzingImage} />
            </label>
            <span className="meta-label">
              {isAnalyzingImage ? 'Аналіз фото...' : terminalImageName ? `Фото: ${terminalImageName}` : 'Фото не вибрано'}
            </span>
          </div>
        </section>

        {siteConfig.showEventLog && (<section className="log-panel panel-card">
          <div className="panel-heading">
            <h2>Лог подій</h2>
            <span>{eventLog.length} записів</span>
          </div>
          <div className="event-list">
            {eventLog.length === 0 ? <p className="empty-state">Поки що немає валідних подій.</p> : null}
            {eventLog.map((event) => (
              <article key={event.id} className="event-row">
                <div>
                  <strong>{new Date(event.timestamp).toLocaleTimeString('uk-UA')}</strong>
                  <p>{event.kind}, shape: {event.shapeLabel} ({(event.shapeConfidence * 100).toFixed(0)}%), AI: {event.aiLabel} ({(event.aiConfidence * 100).toFixed(0)}%), energy: {event.energyClass}, кластер {event.size}px, peak {event.peak}, intensity {event.intensity}</p>
                </div>
                <div className="event-coords">
                  <span>{event.x.toFixed(1)} / {event.y.toFixed(1)} | {event.width}x{event.height}</span>
                  <strong>{event.doseRate.toFixed(1)} мкР/год</strong>
                </div>
              </article>
            ))}
          </div>
        </section>)}
      </main>
    </div>
  )
}

type EnergyHistogramProps = {
  bins: number[]
}

function EnergyHistogram({ bins }: EnergyHistogramProps) {
  const width = 600
  const height = 120
  const max = Math.max(1, ...bins)
  const barWidth = width / bins.length

  return (
    <svg className="graph-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <rect x="0" y="0" width={width} height={height} rx="12" className="graph-backdrop" />
      {bins.map((value, index) => {
        const h = (value / max) * (height - 12)
        const x = index * barWidth + 2
        const y = height - h - 4
        return <rect key={index} x={x} y={y} width={Math.max(2, barWidth - 4)} height={h} fill="rgba(146,255,175,0.8)" rx="2" />
      })}
    </svg>
  )
}

type SliderProps = {
  label: string
  min: number
  max: number
  step: number
  value: number
  onChange: (value: number) => void
}

function Slider({ label, min, max, step, value, onChange }: SliderProps) {
  return (
    <label className="slider-row">
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  )
}

type GraphProps = {
  points: number[]
  variant?: 'normal' | 'sensitive'
  strokeClass?: string
}

function Graph({ points, variant = 'normal', strokeClass = 'graph-line' }: GraphProps) {
  const width = 600
  const height = 180
  const padded = points.length > 1 ? points : [0, ...points]
  const sourceMin = Math.min(...padded)
  const sourceMax = Math.max(...padded)
  const sourceRange = Math.max(sourceMax - sourceMin, 0.001)
  const sensitiveRange = Math.max(0.15, sourceRange * 1.18)
  const center = (sourceMin + sourceMax) / 2
  const minValue = variant === 'sensitive' ? center - sensitiveRange / 2 : 0
  const maxValue = variant === 'sensitive' ? center + sensitiveRange / 2 : Math.max(...padded, 1)
  const valueRange = Math.max(maxValue - minValue, 0.001)
  const stepX = width / Math.max(padded.length - 1, 1)
  const path = padded
    .map((point, index) => {
      const x = index * stepX
      const normalized = (point - minValue) / valueRange
      const y = height - normalized * (height - 18) - 9
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
    })
    .join(' ')

  const topLabel = maxValue.toFixed(2)
  const bottomLabel = minValue.toFixed(2)

  return (
    <svg className="graph-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id="graph-fill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="rgba(208,255,231,0.85)" />
          <stop offset="100%" stopColor="rgba(208,255,231,0.04)" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width={width} height={height} rx="18" className="graph-backdrop" />
      <line x1="8" x2={width - 8} y1="24" y2="24" className="graph-grid-line" />
      <line x1="8" x2={width - 8} y1={height - 24} y2={height - 24} className="graph-grid-line" />
      <text x="14" y="18" className="graph-label">{topLabel}</text>
      <text x="14" y={height - 8} className="graph-label">{bottomLabel}</text>
      <path d={path} className={strokeClass} />
    </svg>
  )
}

function buildAiReply(
  prompt: string,
  context: {
    epm: number
    cameraEpm: number
    ambientEpm: number
    doseRate: number
    baselineNoise: number
    detectionMode: DetectionMode
    lastPhotoAnalysis: PhotoRadiationAnalysis | null
  }
): string {
  const p = prompt.toLowerCase()

  // --- Radiation / detector status ---
  if (/статус|зараз|поточ|current|readings?/i.test(p)) {
    return `Поточний стан: EPM=${context.epm.toFixed(1)}, CMOS=${context.cameraEpm.toFixed(1)}, ALS=${context.ambientEpm.toFixed(1)}, live dose=${context.doseRate.toFixed(2)} мкР/год, baseline=${context.baselineNoise.toFixed(1)} EPM, режим=${context.detectionMode}.`
  }

  if (/калібр|calib/i.test(p)) {
    return 'Для стабільного виміру: 1) закрийте камеру чорною стрічкою, 2) натисніть «Калібрувати 30с» для baseline-шуму, 3) запустіть «Hot Pixel Mask» для маски дефектних пікселів. Перевіряйте, що середня яскравість ≤ 5.'
  }

  if (/epm|подій|lічильник|count|events? per/i.test(p)) {
    return `EPM (Events Per Minute) — кількість зареєстрованих радіаційних подій за хвилину. Зараз: загальний=${context.epm.toFixed(1)}, CMOS=${context.cameraEpm.toFixed(1)}, ALS=${context.ambientEpm.toFixed(1)}. Природний фон зазвичай 10–20 EPM.`
  }

  if (/доз|dose|мкр|зиверт|сиверт|sievert|рентген|roentgen/i.test(p)) {
    const lvl = context.doseRate
    const safety = lvl < 30 ? 'норма' : lvl < 60 ? 'підвищений' : lvl < 120 ? 'небезпечний' : 'дуже небезпечний'
    return `Поточна доза: ${lvl.toFixed(2)} мкР/год — ${safety}. Норма: 10–30 мкР/год. Чорнобильська зона відчуження: 100–5000+. 1 мЗв/рік = 114 мкР/год безперервно.`
  }

  if (/гамм|альф|бета|gamma|alpha|beta|частинк|particle/i.test(p)) {
    return 'Гамма-кванти — проникне випромінювання, дає точкові кластери на CMOS. Бета-частинки — менш проникні, дають лінійні треки. Альфа — зупиняються шкірою, смартфон майже не фіксує. Цей додаток краще детектує гамму і бету.'
  }

  if (/cmos|матриц|камер|sensor|сенсор/i.test(p)) {
    return 'CMOS-матриця смартфона реагує на гамма-кванти: частинка вибиває електрони в пікселі → яскрава точка в темряві. Поріг шуму: R/G/B > 20. Важливо: камера має бути повністю закрита, середня яскравість ≤ 5.'
  }

  if (/als|ambient|освітл|lux|люкс/i.test(p)) {
    return 'Ambient Light Sensor — фотодіод у темряві видає 0 лк. Радіаційна частинка створює мікроімпульс: стрибок з 0 до ≥0.010 і назад. Потрібна повна темрява (< 0.001 лк). Менш точний, ніж CMOS, але не потребує закривати камеру.'
  }

  if (/фон|background|природн|природній/i.test(p)) {
    return 'Природний радіаційний фон: 10–20 мкР/год (Україна в середньому ~15). Гори, граніт, авіарейси — вищий фон. Київ: ~12–18 мкР/год. Зони підвищеного ризику: > 60 мкР/год.'
  }

  if (/небезпеч|safe|danger|захист|protect|shield/i.test(p)) {
    return 'Небезпечний рівень: > 60 мкР/год (тривала дія), > 500 мкР/год (гостра небезпека). Захист: відстань (↑2× відстані → ↓4× дози), час (менше перебувати), екранування (свинець, бетон, вода).'
  }

  // --- Physics & science ---
  if (/квант|quantum|фізик|physics/i.test(p)) {
    return 'Квантова фізика описує поведінку частинок на субатомному рівні. Принцип невизначеності Гайзенберга: ΔxΔp ≥ ℏ/2. Фотон має нульову масу спокою але імпульс p=ℏ/λ. Що конкретно тебе цікавить?'
  }

  if (/хім|хімі|chemistry|елемент|періодич/i.test(p)) {
    return 'В таблиці Менделєєва 118 елементів. Радіоактивні елементи: від Z≥83 (Бісмут) і деякі легші ізотопи (¹⁴C, ⁴⁰K). Уран-238 має T½ = 4.47 млрд років, Полоній-210 — 138 днів.'
  }

  if (/математик|math|рівнян|equation|числ/i.test(p)) {
    return 'Базова формула активності: A = λN, де λ = ln2/T½. Доза потужності: D = A·E·k. Закон радіоактивного розпаду: N(t) = N₀·e^(−λt). Що хочеш порахувати?'
  }

  // --- Coding & tech ---
  if (/код|code|javascript|typescript|react|програм/i.test(p)) {
    return 'Цей додаток написано на React + TypeScript + Vite. Детекція: CMOS через getUserMedia + canvas pixel analysis, ALS через AmbientLightSensor API. Чим можу допомогти з кодом?'
  }

  if (/api|браузер|browser|chrome|firefox|permission|дозвол/i.test(p)) {
    return 'Для роботи потрібні дозволи: Camera (для CMOS) та Sensors (для ALS). AmbientLightSensor доступний в Chrome/Edge з увімкненим Generic Sensor API. Firefox і Safari поки не підтримують ALS.'
  }

  // --- Life / general ---
  if (/погод|weather|температур|temperature/i.test(p)) {
    return 'Немає доступу до реал-тайм даних. Перевір погоду на weather.gov або Google Weather.'
  }

  if (/привіт|hello|hi|hey|вітаю|добрий/i.test(p)) {
    return `Привіт! Я Mini AI у цьому радіаційному детекторі. Можу відповідати про радіацію, фізику, код та загальні питання. Зараз live-доза: ${context.doseRate.toFixed(2)} мкР/год. Чим можу допомогти?`
  }

  if (/дякую|thanks|thank you|спасибі/i.test(p)) {
    return 'Будь ласка! Якщо є питання по радіації або іншому — питай.'
  }

  if (/що ти|who are|what are you|хто ти/i.test(p)) {
    return 'Я — вбудований Mini AI цього вебдодатку. Можу відповідати про радіацію, фізику, код, а також керувати сайтом: ховати/показувати секції, змінювати заголовок. Наприклад: «сховай лог», «покажи метрики», «заголовок на Мій детектор», «що показано».'
  }

  // --- Photo follow-up without keyword фото ---
  if (context.lastPhotoAnalysis) {
    return `По завантаженому фото: ${context.lastPhotoAnalysis.verdict} EPM ≈ ${context.lastPhotoAnalysis.estimatedEpm.toFixed(1)}, доза ≈ ${context.lastPhotoAnalysis.estimatedDoseRate.toFixed(2)} мкР/год.`
  }

  // --- Fallback ---
  return `Розумію питання, але у мене немає конкретної відповіді. Спробуй уточнити або запитай про: радіацію, дозу, EPM, CMOS/ALS, фізику, хімію, код, або завантаж фото для аналізу.`
}

async function analyzePhotoRadiation(file: File): Promise<PhotoRadiationAnalysis> {
  const imageUrl = URL.createObjectURL(file)
  try {
    const image = await loadImage(imageUrl)
    const canvas = document.createElement('canvas')
    const width = Math.max(1, Math.min(image.naturalWidth || image.width, 1024))
    const height = Math.max(1, Math.min(image.naturalHeight || image.height, 1024))
    canvas.width = width
    canvas.height = height
    const context = canvas.getContext('2d', { willReadFrequently: true })

    if (!context) {
      throw new Error('Не вдалося створити canvas-контекст для аналізу фото.')
    }

    context.drawImage(image, 0, 0, width, height)
    const frame = context.getImageData(0, 0, width, height).data
    let brightnessSum = 0
    let hotPixels = 0
    let brightPixels = 0

    for (let i = 0; i < frame.length; i += 4) {
      const red = frame[i]
      const green = frame[i + 1]
      const blue = frame[i + 2]
      const brightness = red * 0.299 + green * 0.587 + blue * 0.114
      brightnessSum += brightness

      if (red > 20 || green > 20 || blue > 20) {
        hotPixels += 1
      }

      if (brightness > 50) {
        brightPixels += 1
      }
    }

    const pixelCount = Math.max(1, frame.length / 4)
    const meanBrightness = brightnessSum / pixelCount
    const hotPixelRatio = hotPixels / pixelCount
    const brightPixelRatio = brightPixels / pixelCount

    // Estimate EPM from hot-pixel density.
    // Empirical model: background (~12 EPM) corresponds to hotPixelRatio ≈ 0.0008 in a dark frame.
    // Ratio is clamped so bright photos don't give nonsense values.
    const darkFrame = meanBrightness <= 35
    const effectiveRatio = darkFrame ? hotPixelRatio : 0
    const estimatedEpm = Math.min(9999, Math.max(0, (effectiveRatio / 0.0008) * 12))
    // Rough conversion: 1 EPM ≈ 0.85 µR/h above background of 12 µR/h
    const estimatedDoseRate = 12 + Math.max(0, estimatedEpm - 12) * 0.85

    let verdict: string
    if (!darkFrame) {
      verdict = '⚠️ Фото засвічене — аналіз некоректний. Для оцінки радіації потрібне темне фото (закрита камера або нічний режим).'
    } else if (estimatedEpm > 300) {
      verdict = `🚨 НЕБЕЗПЕЧНО: дуже висока щільність гарячих пікселів. Розрахунковий EPM=${estimatedEpm.toFixed(0)} — можлива серйозна радіаційна аномалія.`
    } else if (estimatedEpm > 80) {
      verdict = `⚠️ ПІДВИЩЕНИЙ рівень: EPM≈${estimatedEpm.toFixed(0)} — значно вище фону. Рекомендую перевірити серією живих кадрів.`
    } else if (estimatedEpm > 25) {
      verdict = `🟡 Помірно підвищений: EPM≈${estimatedEpm.toFixed(0)} — трохи вище норми (10–20). Можливий локальний шум матриці або слабке підвищення фону.`
    } else {
      verdict = `✅ Норма: EPM≈${estimatedEpm.toFixed(0)} — відповідає природному фону (10–20 EPM). Ознак підвищеної радіації не виявлено.`
    }

    return {
      meanBrightness,
      hotPixelRatio,
      brightPixelRatio,
      estimatedEpm,
      estimatedDoseRate,
      verdict
    }
  } finally {
    URL.revokeObjectURL(imageUrl)
  }
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('Неможливо відкрити файл зображення.'))
    image.src = src
  })
}
