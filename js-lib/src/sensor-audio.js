export default function init(sn) {
    const A = Object.assign
    return A(class Audio extends sn.Sensor {
        static docs() { return `Sound that's already playing.

Options:
- \`source\`: TODO:
- \`fftSize = 2048\`: the window size: how many values are exposed per packet (or twice that if \`frequency\`). [Must be a power-of-2.](https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode/fftSize)
- \`frequency = {minDecibels:-100, maxDecibels:-30}\`: \`null\` to expose \`fftSize\` time-domain numbers, or an object to expose \`fftSize/2\` frequency-domain numbers.` }
        static options() { return {
            // TODO: Uhh, what do we want?
            //   fftSize, frequency. (`source` is not an option.)
        } }
        resume(opts) {
            if (opts) {
                const name = Array.isArray(opts.name) ? opts.name : typeof opts.name == 'string' ? [opts.name] : []
                const fftSize = opts.fftSize || 2048
                const frequency = opts.frequency || {minDecibels:-100, maxDecibels:-30}
                // TODO: Also opts.source.
                sn._assertCounts('', fftSize), sn._assert([32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768].includes(fftSize), "Not an acceptable power-of-2")
                sn._assert(frequency == null || typeof frequency.minDecibels == 'number' && typeof frequency.maxDecibels == 'number')
                opts.name = [
                    'text',
                    ...name,
                    (dStart, dEnd, dLen) => 2 * dEnd / dLen - 1,
                ]
                opts.values = frequency ? fftSize/2|0 : fftSize
                opts.onValues = Audio.onValues
                opts.noFeedback = true
                this.frequency = frequency
                if (this.ctx) this.ctx.close()
                this.ctx = null
                const ctx = opts.source instanceof AudioContext ? opts.source : (this.ctx = new AudioContext)
                const node = this.node = ctx.createAnalyser()
                node.fftSize = fftSize
                node.smoothingTimeConstant = 0
                Object.defineProperty(ctx, 'destination', {
                    configurable: true,
                    enumerable: true,
                    value: node,
                    writable: true,
                })
            }
            return super.resume(opts)
        }
        static onValues(sensor, data) {
            const node = sensor.node
            if (!node) return
            if (sensor.frequency) {
                node.getFloatFrequencyData(data)
                const min = sensor.frequency.minDecibels, max = sensor.frequency.maxDecibels, rlen = 1 / (max - min)
                for (let i = 0; i < data.length; ++i) { // Rescale to -1…1 manually.
                    const v01 = (data[i] - min) * rlen
                    data[i] = Math.max(-1, Math.min(v01*2-1, 1))
                }
            } else
                node.getFloatTimeDomainData(data)
            sensor.sendCallback(null, data)
        }
    }, {
        // TODO: `Audio.DOM`.
    })
}