export type DetectorSettings = {
  thresholdBrightness: number
  thresholdDelta: number
  minClusterSize: number
  maxClusterSize: number
  hotPixelCutoff: number
  downscale: number
  adaptiveBoost: number
  staticHitFrames: number
}

export type EventKind = 'noise' | 'gamma-candidate' | 'beta-track'
export type EventShape = 'round' | 'line-zigzag' | 'star-burst' | 'unknown'

export type DetectionEvent = {
  id: string
  timestamp: number
  x: number
  y: number
  size: number
  peak: number
  intensity: number
  width: number
  height: number
  kind: EventKind
  shape: EventShape
  window10x10: number[]
}

type ClusterPoint = {
  index: number
  x: number
  y: number
  value: number
}

export type FrameStats = {
  brightnessMean: number
  hotPixelCount: number
  adaptiveBrightnessThreshold: number
  adaptiveDeltaThreshold: number
}

export type DetectionResult = {
  events: DetectionEvent[]
  stats: FrameStats
  grayscale: Uint8Array
  width: number
  height: number
}

export type HotPixelScanResult = {
  brightnessMean: number
  events: DetectionEvent[]
  grayscale: Uint8Array
}

export type LongExposureResult = {
  accumulated: Uint8Array
  frameCount: number
  mean: number
}

const badPixelSet = new Set<number>()
const repeatedHitMap = new Map<number, number>()
const accumulatedFrames: Uint8Array[] = []
const maxAccumulatedFrames = 120 // ~5 sec @ 24fps

export function resetHotPixelMap(): void {
  badPixelSet.clear()
  repeatedHitMap.clear()
}

export function setBadPixels(indices: Iterable<number>): void {
  badPixelSet.clear()
  for (const index of indices) {
    badPixelSet.add(index)
  }
}

export function getBadPixelCount(): number {
  return badPixelSet.size
}

export function accumulateLongExposure(grayscale: Uint8Array): LongExposureResult {
  // Store frames for long-exposure averaging (~5 sec @ 24fps = 120 frames)
  accumulatedFrames.push(new Uint8Array(grayscale))
  if (accumulatedFrames.length > maxAccumulatedFrames) {
    accumulatedFrames.shift()
  }

  const accumulated = new Uint8Array(grayscale.length)
  let sum = 0
  for (const frame of accumulatedFrames) {
    for (let i = 0; i < frame.length; i++) {
      accumulated[i] += frame[i]
      sum += frame[i]
    }
  }

  const count = accumulatedFrames.length
  const mean = count > 0 ? sum / (count * grayscale.length) : 0
  for (let i = 0; i < accumulated.length; i++) {
    accumulated[i] = Math.round(accumulated[i] / count)
  }

  return { accumulated, frameCount: count, mean }
}

export function scanHotPixelEventsWithZScore(
  grayscale: Uint8Array,
  width: number,
  height: number
): HotPixelScanResult {
  const candidates = new Uint8Array(width * height)
  let brightnessSum = 0

  for (let i = 0; i < grayscale.length; i++) {
    brightnessSum += grayscale[i]
  }

  const meanBrightness = brightnessSum / Math.max(1, grayscale.length)

  // Z-score neighbor analysis: detect pixels that differ significantly from neighbors
  for (let index = 0; index < grayscale.length; index++) {
    const x = index % width
    const y = Math.floor(index / width)
    const pixelValue = grayscale[index]

    let neighborSum = 0
    let neighborCount = 0
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue
        const nx = x + dx
        const ny = y + dy
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          neighborSum += grayscale[ny * width + nx]
          neighborCount += 1
        }
      }
    }

    if (neighborCount === 0) continue

    const neighborMean = neighborSum / neighborCount
    const deviation = pixelValue - neighborMean

    // In total darkness, even 5% deviation is significant
    const threshold = Math.max(5, neighborMean * 0.05)
    if (deviation > threshold) {
      candidates[index] = 1
    }
  }

  const events = clusterCandidates(
    candidates,
    grayscale,
    width,
    height,
    {
      thresholdBrightness: 0,
      thresholdDelta: 0,
      minClusterSize: 1,
      maxClusterSize: 64,
      hotPixelCutoff: 0,
      downscale: 1,
      adaptiveBoost: 0,
      staticHitFrames: 0
    }
  ).map((event) => ({
    ...event,
    kind: 'gamma-candidate' as EventKind,
    shape: event.shape === 'unknown' ? 'round' : event.shape
  }))

  return {
    brightnessMean: meanBrightness,
    events,
    grayscale
  }
}

export function resetLongExposure(): void {
  accumulatedFrames.length = 0
}

export function scanHotPixelEvents(
  frame: Uint8ClampedArray,
  width: number,
  height: number,
  noiseThreshold = 20
): HotPixelScanResult {
  const grayscale = new Uint8Array(width * height)
  const candidates = new Uint8Array(width * height)
  let brightnessSum = 0

  for (let sourceIndex = 0, targetIndex = 0; sourceIndex < frame.length; sourceIndex += 4, targetIndex += 1) {
    const red = frame[sourceIndex]
    const green = frame[sourceIndex + 1]
    const blue = frame[sourceIndex + 2]
    const value = Math.round(red * 0.299 + green * 0.587 + blue * 0.114)
    grayscale[targetIndex] = value
    brightnessSum += value

    if (red > noiseThreshold || green > noiseThreshold || blue > noiseThreshold) {
      candidates[targetIndex] = 1
    }
  }

  const events = clusterCandidates(
    candidates,
    grayscale,
    width,
    height,
    {
      thresholdBrightness: 0,
      thresholdDelta: 0,
      minClusterSize: 1,
      maxClusterSize: 64,
      hotPixelCutoff: 0,
      downscale: 1,
      adaptiveBoost: 0,
      staticHitFrames: 0
    }
  ).map((event) => ({
    ...event,
    kind: 'gamma-candidate' as EventKind,
    shape: event.shape === 'unknown' ? 'round' : event.shape
  }))

  return {
    brightnessMean: brightnessSum / Math.max(1, grayscale.length),
    events,
    grayscale
  }
}


export function detectEvents(
  frame: Uint8ClampedArray,
  width: number,
  height: number,
  previous: Uint8Array | null,
  settings: DetectorSettings
): DetectionResult {
  const grayscale = new Uint8Array(width * height)
  let brightnessSum = 0
  let brightnessSquareSum = 0

  for (let sourceIndex = 0, targetIndex = 0; sourceIndex < frame.length; sourceIndex += 4, targetIndex += 1) {
    const red = frame[sourceIndex]
    const green = frame[sourceIndex + 1]
    const blue = frame[sourceIndex + 2]
    const value = Math.round(red * 0.299 + green * 0.587 + blue * 0.114)
    grayscale[targetIndex] = value
    brightnessSum += value
    brightnessSquareSum += value * value
  }

  const candidates = new Uint8Array(width * height)
  let hotPixelCount = 0
  const brightnessMean = brightnessSum / grayscale.length
  const variance = Math.max(brightnessSquareSum / grayscale.length - brightnessMean * brightnessMean, 0)
  const brightnessSigma = Math.sqrt(variance)
  const adaptiveBrightnessThreshold = Math.min(
    255,
    Math.max(settings.thresholdBrightness, brightnessMean + settings.adaptiveBoost * brightnessSigma)
  )
  const adaptiveDeltaThreshold = Math.max(settings.thresholdDelta, Math.round(brightnessSigma * 1.5))

  if (previous) {
    for (let index = 0; index < grayscale.length; index += 1) {
      const brightness = grayscale[index]
      const delta = brightness - previous[index]
      const isHotPixel = badPixelSet.has(index)

      if (isHotPixel) {
        hotPixelCount += 1
        continue
      }

      if (brightness >= adaptiveBrightnessThreshold && delta >= adaptiveDeltaThreshold) {
        const staticHits = (repeatedHitMap.get(index) ?? 0) + 1
        repeatedHitMap.set(index, staticHits)

        if (staticHits > settings.staticHitFrames) {
          continue
        }

        candidates[index] = 1
      } else if (brightness >= adaptiveBrightnessThreshold) {
        repeatedHitMap.set(index, (repeatedHitMap.get(index) ?? 0) + 1)
      } else if (repeatedHitMap.has(index)) {
        repeatedHitMap.delete(index)
      }
    }
  }

  const events = clusterCandidates(candidates, grayscale, width, height, settings)

  return {
    events,
    stats: {
      brightnessMean,
      hotPixelCount,
      adaptiveBrightnessThreshold,
      adaptiveDeltaThreshold
    },
    grayscale,
    width,
    height
  }
}

function clusterCandidates(
  candidates: Uint8Array,
  grayscale: Uint8Array,
  width: number,
  height: number,
  settings: DetectorSettings
): DetectionEvent[] {
  const visited = new Uint8Array(candidates.length)
  const events: DetectionEvent[] = []
  const neighbors = [
    [-1, -1],
    [0, -1],
    [1, -1],
    [-1, 0],
    [1, 0],
    [-1, 1],
    [0, 1],
    [1, 1]
  ]

  for (let index = 0; index < candidates.length; index += 1) {
    if (candidates[index] === 0 || visited[index] === 1) {
      continue
    }

    const queue: ClusterPoint[] = [{ index, x: index % width, y: Math.floor(index / width), value: grayscale[index] }]
    const cluster: ClusterPoint[] = []
    visited[index] = 1

    while (queue.length > 0) {
      const point = queue.shift()!
      cluster.push(point)

      for (const [dx, dy] of neighbors) {
        const nx = point.x + dx
        const ny = point.y + dy

        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
          continue
        }

        const neighborIndex = ny * width + nx
        if (visited[neighborIndex] === 1 || candidates[neighborIndex] === 0) {
          continue
        }

        visited[neighborIndex] = 1
        queue.push({ index: neighborIndex, x: nx, y: ny, value: grayscale[neighborIndex] })
      }
    }

    if (cluster.length < settings.minClusterSize || cluster.length > settings.maxClusterSize) {
      continue
    }

    let weightedX = 0
    let weightedY = 0
    let peak = 0
    let intensity = 0
    let minX = width
    let maxX = 0
    let minY = height
    let maxY = 0

    for (const point of cluster) {
      weightedX += point.x * point.value
      weightedY += point.y * point.value
      intensity += point.value
      peak = Math.max(peak, point.value)
      minX = Math.min(minX, point.x)
      maxX = Math.max(maxX, point.x)
      minY = Math.min(minY, point.y)
      maxY = Math.max(maxY, point.y)
    }

    const safeIntensity = Math.max(intensity, 1)
    const clusterWidth = maxX - minX + 1
    const clusterHeight = maxY - minY + 1
    const aspectRatio = Math.max(clusterWidth, clusterHeight) / Math.max(1, Math.min(clusterWidth, clusterHeight))
    const shape = classifyClusterShape(cluster.length, aspectRatio, peak, intensity)
    const kind = mapShapeToKind(shape)

    if (kind === 'noise') {
      continue
    }

    events.push({
      id: `${Date.now()}-${index}`,
      timestamp: Date.now(),
      x: weightedX / safeIntensity,
      y: weightedY / safeIntensity,
      size: cluster.length,
      peak,
      intensity,
      width: clusterWidth,
      height: clusterHeight,
      kind,
      shape,
      window10x10: extractWindow10x10(grayscale, width, height, Math.round(weightedX / safeIntensity), Math.round(weightedY / safeIntensity))
    })
  }

  return events
}

function extractWindow10x10(grayscale: Uint8Array, width: number, height: number, centerX: number, centerY: number): number[] {
  const output: number[] = []
  for (let dy = -5; dy < 5; dy += 1) {
    for (let dx = -5; dx < 5; dx += 1) {
      const x = centerX + dx
      const y = centerY + dy
      if (x < 0 || y < 0 || x >= width || y >= height) {
        output.push(0)
      } else {
        output.push(grayscale[y * width + x])
      }
    }
  }
  return output
}

function classifyClusterShape(size: number, aspectRatio: number, peak: number, intensity: number): EventShape {
  if (size <= 1) {
    return 'unknown'
  }

  const concentration = peak / Math.max(intensity, 1)

  if (size >= 8 && concentration < 0.2 && aspectRatio < 2.4) {
    return 'star-burst'
  }

  if (aspectRatio >= 3 || size >= 12) {
    return 'line-zigzag'
  }

  if (size >= 2 && aspectRatio < 3) {
    return 'round'
  }

  return 'unknown'
}

function mapShapeToKind(shape: EventShape): EventKind {
  if (shape === 'line-zigzag') {
    return 'beta-track'
  }

  if (shape === 'round' || shape === 'star-burst') {
    return 'gamma-candidate'
  }

  return 'noise'
}
