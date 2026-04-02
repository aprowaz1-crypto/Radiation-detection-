// ─── Physical constants ───────────────────────────────────────────────────────
// Silicon dark-current Arrhenius model
const EA_SILICON = 0.65     // Activation energy [eV]
const KB_EV = 8.617e-5      // Boltzmann constant [eV/K]
const T_REF = 298           // Reference temperature [K] = 25 °C

// 3D-CFAR temporal ring-buffer depth (number of past frames used)
const CFAR_FRAMES = 4

// Temporal ring buffer for 3D-CFAR
const frameHistory: Float32Array[] = []

// Permanently lit "hot" pixels identified during calibration
const badPixelSet = new Set<number>()

// ─── Public types ─────────────────────────────────────────────────────────────

export type CalibrationMaps = {
  dcnu: Float32Array    // Dark Current Non-Uniformity: per-pixel dark bias
  prnu: Float32Array    // Photo Response Non-Uniformity: per-pixel σ/μ ratio
  width: number
  height: number
  frameCount: number
}

export type RadiationEvent = {
  id: string
  timestamp: number
  cx: number            // centroid x (scaled pixels)
  cy: number            // centroid y (scaled pixels)
  size: number          // number of pixels in cluster
  peak: number          // brightest pixel value (0-255)
  meanIntensity: number
  aspectRatio: number
  kind: 'gamma-candidate' | 'beta-track'
}

export type FrameResult = {
  events: RadiationEvent[]
  meanBrightness: number
  sigma: number
  threshold: number     // effective MLE+CFAR threshold used
  tempFactor: number    // Arrhenius correction factor applied
}

// ─── Internal cluster type ────────────────────────────────────────────────────
type Cluster = {
  pixels: { x: number; y: number; v: number }[]
  cx: number; cy: number; peak: number
  w: number; h: number; aspect: number
}

// ─── State management ─────────────────────────────────────────────────────────

export function resetDetectorState(): void {
  frameHistory.length = 0
  badPixelSet.clear()
}

export function markBadPixels(indices: number[]): void {
  badPixelSet.clear()
  for (const i of indices) badPixelSet.add(i)
}

export function getBadPixelCount(): number {
  return badPixelSet.size
}

/**
 * Rolling dark-frame update via EMA.
 * Refreshes DCNU/PRNU maps using a new dark-like frame (typically every 5 min).
 */
export function updateCalibrationMapsEMA(
  calibration: CalibrationMaps,
  darkLikeFrame: Float32Array,
  alpha: number
): void {
  if (!calibration || darkLikeFrame.length !== calibration.dcnu.length) return

  const a = Math.max(0.001, Math.min(0.25, alpha))
  const b = 1 - a
  const n = calibration.dcnu.length
  let globalSum = 0

  for (let i = 0; i < n; i++) {
    const prevDc = calibration.dcnu[i]
    const nextDc = b * prevDc + a * darkLikeFrame[i]
    calibration.dcnu[i] = nextDc
    globalSum += nextDc

    // Robust PRNU proxy update from normalized absolute deviation.
    const dev = Math.abs(darkLikeFrame[i] - nextDc)
    const ratio = nextDc > 1 ? dev / nextDc : calibration.prnu[i]
    const clipped = Math.max(0.05, Math.min(4.0, ratio))
    calibration.prnu[i] = b * calibration.prnu[i] + a * clipped
  }

  // Rebuild hot-pixel mask from updated dark map.
  badPixelSet.clear()
  const globalMean = globalSum / Math.max(1, n)
  const hotThreshold = globalMean * 5
  for (let i = 0; i < n; i++) {
    if (calibration.dcnu[i] > hotThreshold) badPixelSet.add(i)
  }
}

// ─── Calibration ──────────────────────────────────────────────────────────────

/**
 * Build DCNU and PRNU correction maps from N dark frames (complete darkness).
 * DCNU: mean pixel value across frames → dark-current bias to subtract.
 * PRNU: σ/μ ratio → relative noise per pixel (used in threshold weighting).
 * Also identifies permanently lit hot pixels (excluded from future detection).
 */
export function buildCalibrationMaps(
  darkFrames: Float32Array[],
  width: number,
  height: number
): CalibrationMaps {
  const n = width * height
  const dcnu = new Float32Array(n)
  const prnu = new Float32Array(n).fill(1.0)

  if (darkFrames.length === 0) {
    return { dcnu, prnu, width, height, frameCount: 0 }
  }

  const sum = new Float32Array(n)
  const sumSq = new Float32Array(n)

  for (const f of darkFrames) {
    for (let i = 0; i < n; i++) {
      sum[i] += f[i]
      sumSq[i] += f[i] * f[i]
    }
  }

  const fc = darkFrames.length
  let globalSum = 0

  for (let i = 0; i < n; i++) {
    const mean = sum[i] / fc
    dcnu[i] = mean
    globalSum += mean
    const variance = Math.max(sumSq[i] / fc - mean * mean, 0)
    prnu[i] = mean > 1 ? Math.sqrt(variance) / mean : 1.0
  }

  // Mark hot pixels: dark-frame mean > 5× global mean  →  persistently lit
  const globalMean = globalSum / n
  const hotThreshold = globalMean * 5
  for (let i = 0; i < n; i++) {
    if (dcnu[i] > hotThreshold) badPixelSet.add(i)
  }

  return { dcnu, prnu, width, height, frameCount: fc }
}

// ─── Temperature compensation ─────────────────────────────────────────────────

/**
 * Arrhenius-model correction factor.
 * Dark current doubles roughly every 6-8 °C; this compensates the threshold.
 * Returns a multiplier for the detection threshold.
 *   Cold device  (T < T_ref): factor < 1 → threshold can be lower.
 *   Hot device   (T > T_ref): factor > 1 → raise threshold to avoid false hits.
 */
export function arrhenius(tempC: number): number {
  const T = tempC + 273.15
  // Dark current scales as exp(-Ea/kT); ratio relative to reference temp:
  const darkCurrentRatio = Math.exp((EA_SILICON / KB_EV) * (1 / T_REF - 1 / T))
  // Threshold should scale proportionally with dark current increase
  return Math.max(0.5, Math.min(3.0, darkCurrentRatio))
}

// ─── Core algorithms ──────────────────────────────────────────────────────────

function toGrayscale(rgba: Uint8ClampedArray, n: number): Float32Array {
  const g = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    g[i] = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]
  }
  return g
}

function applyDCNU(gray: Float32Array, dcnu: Float32Array): Float32Array {
  const out = new Float32Array(gray.length)
  for (let i = 0; i < gray.length; i++) {
    out[i] = Math.max(0, gray[i] - dcnu[i])
  }
  return out
}

/**
 * Poisson-Gaussian MLE: derive adaptive baseline threshold.
 * Background thermal noise is Gaussian(μ, σ). The threshold is set at
 * μ + k·σ where k = 3.5 gives ~99.97 % noise rejection before CFAR.
 * Temperature factor scales the threshold up for hot sensors.
 */
function mleThreshold(mean: number, sigma: number, tempFactor: number): number {
  const k = 3.5
  const base = mean + k * sigma
  return Math.max(3, Math.min(120, base * tempFactor))
}

/**
 * 3D-CFAR (Constant False Alarm Rate) detector.
 * Operates on three dimensions: x, y (spatial) and t (temporal via history).
 *
 * For each pixel:
 *  - Spatial training ring: guard radius 1 px, training radius 3 px
 *  - Temporal training: all frames in frameHistory ring buffer
 *  - Local threshold: localMean + 3.5 × localSigma
 *
 * This adapts to local sensor non-uniformities and slow drift,
 * rejecting 99 %+ of thermal noise while passing transient bright events.
 */
function cfar3D(
  corrected: Float32Array,
  width: number,
  height: number,
  _globalMean: number,
  globalSigma: number,
  baseThreshold: number
): Uint8Array {
  const candidates = new Uint8Array(corrected.length)
  const guardR = 1
  const trainR = 3

  for (let y = trainR; y < height - trainR; y++) {
    for (let x = trainR; x < width - trainR; x++) {
      const idx = y * width + x

      if (badPixelSet.has(idx)) continue
      if (corrected[idx] <= baseThreshold) continue  // fast pre-filter

      // Collect spatial training cells (annular region, skip guard zone)
      let ts = 0, tsq = 0, tc = 0
      for (let dy = -trainR; dy <= trainR; dy++) {
        for (let dx = -trainR; dx <= trainR; dx++) {
          if (Math.abs(dy) <= guardR && Math.abs(dx) <= guardR) continue
          const nx = x + dx, ny = y + dy
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue
          const v = corrected[ny * width + nx]
          ts += v; tsq += v * v; tc++
        }
      }

      // Temporal dimension: all frames in history ring buffer
      for (const pf of frameHistory) {
        const v = pf[idx]
        ts += v; tsq += v * v; tc++
      }

      if (tc < 6) continue

      const localMean = ts / tc
      const localVar = Math.max(tsq / tc - localMean * localMean, 0)
      const localSigma = Math.sqrt(localVar)

      // CFAR threshold: use local stats, bounded from below by global estimate
      const localThreshold = localMean + 3.5 * Math.max(localSigma, globalSigma * 0.2)

      if (corrected[idx] > localThreshold) {
        candidates[idx] = 1
      }
    }
  }

  return candidates
}

/**
 * BFS flood-fill: group 8-connected candidate pixels into clusters.
 */
function clusterBFS(
  candidates: Uint8Array,
  corrected: Float32Array,
  width: number,
  height: number
): Cluster[] {
  const visited = new Uint8Array(candidates.length)
  const clusters: Cluster[] = []

  for (let i = 0; i < candidates.length; i++) {
    if (!candidates[i] || visited[i]) continue

    const queue: number[] = [i]
    visited[i] = 1
    const pixels: { x: number; y: number; v: number }[] = []
    let head = 0

    while (head < queue.length) {
      const idx = queue[head++]
      const x = idx % width, y = Math.floor(idx / width)
      pixels.push({ x, y, v: corrected[idx] })

      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue
          const nx = x + dx, ny = y + dy
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue
          const ni = ny * width + nx
          if (!visited[ni] && candidates[ni]) {
            visited[ni] = 1
            queue.push(ni)
          }
        }
      }
    }

    if (pixels.length === 0) continue

    let xMin = pixels[0].x, xMax = pixels[0].x
    let yMin = pixels[0].y, yMax = pixels[0].y
    let cx = 0, cy = 0, peak = 0

    for (const p of pixels) {
      if (p.x < xMin) xMin = p.x; if (p.x > xMax) xMax = p.x
      if (p.y < yMin) yMin = p.y; if (p.y > yMax) yMax = p.y
      cx += p.x; cy += p.y
      if (p.v > peak) peak = p.v
    }

    cx /= pixels.length; cy /= pixels.length
    const w = xMax - xMin + 1, h = yMax - yMin + 1
    const aspect = Math.max(w, h) / Math.max(1, Math.min(w, h))

    clusters.push({ pixels, cx, cy, peak, w, h, aspect })
  }

  return clusters
}

/**
 * Spatial cluster validation.
 *
 * Physical profile of a single ionising-particle hit on a CMOS sensor:
 *   - Size: 1–6 pixels  (larger blobs → multiple hits or artefact)
 *   - Shape: compact, nearly round (aspect ratio < 4)
 *   - Gamma/X-ray:  mostly 1–3 compact pixels
 *   - Beta track:   1–6 pixels, slightly elongated (aspect ≤ 2–3)
 *   - Reject: lines, blobs > 6 px, pure-noise fluctuations below peak floor
 */
function validateCluster(c: Cluster): { valid: boolean; kind: RadiationEvent['kind'] } {
  if (c.pixels.length < 1 || c.pixels.length > 6) return { valid: false, kind: 'gamma-candidate' }
  if (c.aspect > 4.0) return { valid: false, kind: 'gamma-candidate' }
  if (c.peak < 4) return { valid: false, kind: 'gamma-candidate' }

  const kind: RadiationEvent['kind'] =
    c.pixels.length <= 3 && c.aspect < 1.8
      ? 'gamma-candidate'
      : 'beta-track'

  return { valid: true, kind }
}

// ─── Main pipeline ─────────────────────────────────────────────────────────────

/**
 * Full single-frame detection pipeline:
 *   RGBA → grayscale → DCNU correction → global stats (MLE threshold)
 *   → temperature compensation → 3D-CFAR → cluster BFS → spatial validation
 *   → update temporal history
 *
 * @param rgba        Raw RGBA pixels from ImageData
 * @param width       Frame width (after downscaling)
 * @param height      Frame height (after downscaling)
 * @param calibration DCNU/PRNU maps from buildCalibrationMaps(); pass null during calibration
 * @param tempC       Current device temperature from Battery API or estimate
 */
export function detectFrame(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  calibration: CalibrationMaps | null,
  tempC: number
): FrameResult {
  const n = width * height

  // 1. Convert to single-channel grayscale
  const gray = toGrayscale(rgba, n)

  // 2. Subtract DCNU (dark-current bias per pixel)
  const corrected = calibration ? applyDCNU(gray, calibration.dcnu) : gray

  // 3. Global frame statistics for Poisson-Gaussian MLE
  let sum = 0, sumSq = 0
  for (let i = 0; i < n; i++) { sum += corrected[i]; sumSq += corrected[i] * corrected[i] }
  const mean = sum / n
  const sigma = Math.sqrt(Math.max(sumSq / n - mean * mean, 0))

  // 4. Temperature compensation (Arrhenius)
  const tempFactor = arrhenius(tempC)
  const threshold = mleThreshold(mean, sigma, tempFactor)

  // 5. 3D-CFAR detection
  const candidates = cfar3D(corrected, width, height, mean, sigma, threshold)

  // 6. Group candidates into spatial clusters
  const rawClusters = clusterBFS(candidates, corrected, width, height)

  // 7. Spatial validation — reject artefacts, keep only particle-like events
  const events: RadiationEvent[] = []
  for (const c of rawClusters) {
    const { valid, kind } = validateCluster(c)
    if (!valid) continue
    const meanIntensity = c.pixels.reduce((s, p) => s + p.v, 0) / c.pixels.length
    events.push({
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      timestamp: Date.now(),
      cx: c.cx,
      cy: c.cy,
      size: c.pixels.length,
      peak: c.peak,
      meanIntensity,
      aspectRatio: c.aspect,
      kind,
    })
  }

  // 8. Push to temporal history for next frame's CFAR
  frameHistory.push(new Float32Array(corrected))
  if (frameHistory.length > CFAR_FRAMES) frameHistory.shift()

  return { events, meanBrightness: mean, sigma, threshold, tempFactor }
}
