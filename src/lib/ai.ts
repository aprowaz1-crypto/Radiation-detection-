// ─── Gamma-Poisson Bayesian Inference for CPM / Dose estimation ──────────────
//
// Statistical model:
//   Observed counts:  n events in t seconds  ~  Poisson(λ · t)
//   Prior on rate λ:  Gamma(α₀, β₀)          (conjugate prior)
//   Posterior:        Gamma(α₀ + n,  β₀ + t)
//   Posterior mean:   (α₀ + n) / (β₀ + t)    [events / second]
//
// Prior hyperparameters encode a weak belief around natural background:
//   α₀/β₀ ≈ 10 CPM = 10/60 events/s  with broad spread
//   α₀ = 2,  β₀ = 12 s  →  prior mean = 10 CPM, but very diffuse.
//
// As observations accumulate the posterior narrows around the true rate
// and the 95 % credible interval shrinks, matching the Poisson √n uncertainty.

const ALPHA0 = 2.0
const BETA0 = 12.0     // seconds equivalent of prior observations

// ─── Persistent accumulator ───────────────────────────────────────────────────
let accumN = 0         // total observed events
let accumT = 0         // total observation time [seconds]

// ─── Public types ─────────────────────────────────────────────────────────────

export type BayesEstimate = {
  cpm: number            // Posterior mean [events per minute]
  ci95Low: number        // 95 % credible interval lower bound [CPM]
  ci95High: number       // 95 % credible interval upper bound [CPM]
  uncertaintyPct: number // (hi − lo) / mean × 100 %
  confidenceRatio: number// 0 … 1; grows as √n accumulates
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function resetBayes(): void {
  accumN = 0
  accumT = 0
}

/**
 * Add new observations to the Bayesian accumulator.
 * @param events  Number of validated radiation events in this window
 * @param seconds Length of the observation window in seconds
 */
export function addObservation(events: number, seconds: number): void {
  accumN += events
  accumT += seconds
}

/**
 * Compute the current Bayesian CPM estimate with 95 % credible interval.
 * @param efficiencyCoeff  User-adjustable scale factor (default 1.0).
 *                          Set to sensor/geometry calibration value once
 *                          compared against a reference dosimeter.
 */
export function estimateCPM(efficiencyCoeff: number): BayesEstimate {
  const alpha = ALPHA0 + accumN
  const beta  = BETA0  + accumT   // β in "per second" units

  // Posterior mean rate [events/s] → [events/min]
  const rateSec = alpha / beta
  const cpm = rateSec * 60 * efficiencyCoeff

  // 95 % credible interval via Wilson-Hilferty gamma quantile approximation
  const lo = gammaQuantile(0.025, alpha, beta) * 60 * efficiencyCoeff
  const hi = gammaQuantile(0.975, alpha, beta) * 60 * efficiencyCoeff

  // Relative uncertainty: narrows as √n
  const uncertaintyPct = cpm > 0 ? Math.min(((hi - lo) / cpm) * 100, 999) : 100

  // Confidence proxy: 0 at start, approaches 1 as n grows
  const relErr = accumN > 0 ? 1 / Math.sqrt(accumN) : 1
  const confidenceRatio = Math.max(0, Math.min(1, 1 - relErr))

  return { cpm, ci95Low: lo, ci95High: hi, uncertaintyPct, confidenceRatio }
}

/**
 * Convert CPM to dose rate in μSv/h.
 * @param cpm            Events per minute
 * @param naturalBgMicroSvH  Regional natural background (μSv/h, e.g. 0.12)
 * @param sensorFactor   μSv/h per CPM calibration coefficient for this device
 */
export function cpmToDose(
  cpm: number,
  naturalBgMicroSvH: number,
  sensorFactor: number
): number {
  return naturalBgMicroSvH + Math.max(cpm, 0) * sensorFactor
}

// ─── Gamma distribution quantile (Wilson–Hilferty approximation) ─────────────
//
// For X ~ Gamma(α, rate = β), the p-th quantile is approximated via
// the normal distribution and the WH cube-root transformation.

function gammaQuantile(p: number, alpha: number, beta: number): number {
  const scale = 1 / beta
  const z = normalQuantile(p)
  const a = alpha
  const term = 1 - 1 / (9 * a) + z / Math.sqrt(9 * a)
  const wh = a * Math.pow(Math.max(0, term), 3)
  return wh * scale
}

// Beasley–Springer–Moro rational approximation for the normal quantile Φ⁻¹(p)
function normalQuantile(p: number): number {
  if (p <= 0) return -6
  if (p >= 1) return 6

  const a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
  const b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
  const c = [
    0.3374754822726, 0.9761690190917, 0.1607979714918,
    0.0276438810334, 0.0038405729374, 0.0003951896511,
    0.0000321767882, 0.0000002888167, 0.0000003960315,
  ]

  const x = p - 0.5
  if (Math.abs(x) < 0.42) {
    const r = x * x
    return (
      (x * (((a[3] * r + a[2]) * r + a[1]) * r + a[0])) /
      ((((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1)
    )
  }

  const r = x < 0 ? p : 1 - p
  const s = Math.log(-Math.log(r))
  let t = c[0]
  for (let i = 1; i < c.length; i++) t += c[i] * Math.pow(s, i)
  return x < 0 ? -t : t
}
