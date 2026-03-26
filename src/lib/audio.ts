let audioContext: AudioContext | null = null

export function playClick(volume = 0.04): void {
  const AudioContextCtor = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext

  if (!AudioContextCtor) {
    return
  }

  audioContext ??= new AudioContextCtor()

  if (audioContext.state === 'suspended') {
    void audioContext.resume()
  }

  const oscillator = audioContext.createOscillator()
  const gain = audioContext.createGain()
  oscillator.type = 'square'
  oscillator.frequency.value = 1850
  gain.gain.value = volume
  oscillator.connect(gain)
  gain.connect(audioContext.destination)
  oscillator.start()
  oscillator.stop(audioContext.currentTime + 0.018)
}
