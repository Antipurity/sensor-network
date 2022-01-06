export default function init(sn) {
    const A = Object.assign
    return A(class Audio extends sn.Sensor {
        static docs() { return `Sound that's already playing.

If you need to test what this sensor records, here:

<audio controls crossorigin src="https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3"></audio>

Options:
- \`source = Audio.DOM(Audio)\`: \`<video>\` or \`<audio>\` (with the \`crossorigin\` attribute if crossorigin), \`MediaStream\` \`function(Audio)(audioContext)\`, or the \`AudioContext\` whose \`.destination\` is augmented.
- \`fftSize = 2048\`: the window size: how many values are exposed per packet (or twice that if \`frequency\`). [Must be a power-of-2.](https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode/fftSize)
- \`frequency = {minDecibels:-100, maxDecibels:-30}\`: \`null\` to expose \`fftSize\` time-domain numbers, or an object to expose \`fftSize/2\` frequency-domain numbers.` }
        static options() { return {
            fftSize:{
                ['2048 ']: () => 2048,
                ['512 ']: () => 512,
                ['16384 ']: () => 16384,
            },
            frequency:{
                ['Frequency-domain']: () => null,
                ['Time-domain']: () => ({minDecibels:-100, maxDecibels:-30}),
            },
        } }
        resume(opts) {
            if (opts) {
                const name = Array.isArray(opts.name) ? opts.name : typeof opts.name == 'string' ? [opts.name] : []
                const fftSize = opts.fftSize || 2048
                const frequency = opts.frequency || {minDecibels:-100, maxDecibels:-30}
                const source = opts.source || Audio.DOM(Audio)
                sn._assert(source instanceof AudioContext || source instanceof MediaStreamTrack || source instanceof Element && (source.tagName === 'VIDEO' || source.tagName === 'AUDIO') || typeof source == 'function', "Bad source")
                sn._assertCounts('', fftSize), sn._assert([32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768].includes(fftSize), "Not an acceptable power-of-2")
                sn._assert(frequency == null || typeof frequency.minDecibels == 'number' && typeof frequency.maxDecibels == 'number')
                opts.name = [
                    'text',
                    ...name,
                    (dStart, dEnd, dLen) => 2 * dEnd / dLen - 1,
                ]
                opts.values = frequency ? fftSize/2|0 : fftSize
                opts.onValues = this.onValues
                opts.noFeedback = true
                this.source = source
                this.frequency = frequency
                if (this.ctx) this.ctx.close()
                this.ctx = null
                const ctx = source instanceof AudioContext ? source : (this.ctx = new AudioContext)
                const node = this.node = ctx.createAnalyser()
                node.fftSize = fftSize
                node.smoothingTimeConstant = 0
                const oldDestination = ctx.destination
                Object.defineProperty(ctx, 'destination', { // Augment the destination with our analyser.
                    configurable: true,
                    enumerable: true,
                    value: node,
                    writable: true,
                })
                node.connect(oldDestination)
                if (!(source instanceof AudioContext) && typeof source != 'function')
                    Audio._connect(opts.source, ctx)
            }
            return super.resume(opts)
        }
        onValues(data) {
            const node = this.node
            if (!node) return
            if (typeof this.source == 'function' && this.ctx) this.source(this.ctx)
            if (this.frequency) {
                node.getFloatFrequencyData(data)
                const min = this.frequency.minDecibels, max = this.frequency.maxDecibels, rlen = 1 / (max - min)
                for (let i = 0; i < data.length; ++i) { // Rescale to -1…1 manually.
                    const v01 = (data[i] - min) * rlen
                    data[i] = Math.max(-1, Math.min(v01*2-1, 1))
                }
            } else
                node.getFloatTimeDomainData(data)
            this.sendCallback(null, data)
        }
        static _connect(source, ctx) {
            const dst = ctx.destination
            const src = source instanceof Element ? ctx.createMediaElementSource(source) : source instanceof MediaStream ? ctx.createMediaStreamSource(source) : sn._assert(false)
            src.connect(dst)
            return src
        }
    }, {
        DOM: A(function DOM(Audio) {
            let prevCtx = null
            let nodes = null // elem → node
            return function getDOM(ctx) {
                if (prevCtx !== ctx) nodes = new WeakMap
                prevCtx = ctx
                if (!ctx) return
                const videos = document.getElementsByTagName('video'), audios = document.getElementsByTagName('audio')
                Array.prototype.forEach.call(videos, connect)
                Array.prototype.forEach.call(audios, connect)
                function connect(elem) {
                    if (!nodes.has(elem)) nodes.set(elem, Audio._connect(elem, ctx))
                }
            }
        }, {}),
    })
}