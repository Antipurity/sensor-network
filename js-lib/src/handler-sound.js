export default function init(sn) {
    class IFFT {
        // From frequency to samples.
        // Taken from https://github.com/oampo/Audiolet/blob/master/src/dsp/IFFT.js
        constructor(N) {
            this.N = N
            this.reverseTable = new Uint32Array(N)
            this.calculateReverseTable()
        }
        calculateReverseTable() {
            let limit = 1, bit = this.N >> 1
            while (limit < this.N) {
                for (let i = 0; i < limit; ++i)
                    this.reverseTable[i + limit] = this.reverseTable[i] + bit
                limit = limit << 1
                bit = bit >> 1
            }
        }
        do(inReal, inImag) { // Performs IFFT and returns the real component.
            sn._assert(inReal.length === this.N)
            sn._assert(inReal.length === inImag.length)

            const revReal = sn._allocF32(inReal.length)
            const revImag = sn._allocF32(inImag.length)
            for (let i = 0; i < this.N; ++i) {
                revReal[i] = inReal[this.reverseTable[i]]
                revImag[i] = inImag[this.reverseTable[i]]
            }

            let halfSize = 1
            while (halfSize < this.N) {
                const phaseShiftStepReal = Math.cos(-Math.PI / halfSize)
                const phaseShiftStepImag = Math.sin(-Math.PI / halfSize)
                let phaseShiftReal = 1
                let phaseShiftImag = 0
                for (let fftStep = 0; fftStep < halfSize; ++fftStep) {
                    const R = phaseShiftReal, I = phaseShiftImag
                    for (let i = fftStep; i+halfSize < this.N; i += halfSize << 1) {
                        const j = i + halfSize
                        const tr = R * revReal[j] - I * revImag[j]
                        const ti = R * revImag[j] + I * revReal[j]
                        revReal[j] = revReal[i] - tr, revReal[i] += tr
                        revImag[j] = revImag[i] - ti, revImag[i] += ti
                    }
                    phaseShiftReal = R * phaseShiftStepReal - I * phaseShiftStepImag
                    phaseShiftImag = R * phaseShiftStepImag + I * phaseShiftStepReal
                }
                halfSize <<= 1
            }
            for (let i = 0; i < this.N; ++i) revReal[i] /= this.N
            sn._deallocF32(revImag)
            return revReal
        }
    }
    function ifft(freq, phase = 0) {
        if (ifft.n !== freq.length) {
            ifft.o = new IFFT(freq.length)
            ifft.n = freq.length
        }
        const imag = sn._allocF32(freq.length).fill(phase)
        try { return ifft.o.do(freq, imag) }
        finally { sn._deallocF32(imag) }
    }

    return class Sound extends sn.Handler {
        static docs() { return `Exposes data as sound, for humans to listen.

In Chrome, users might have to first click on the page for sound to play.

(This handler normalizes output frequencies to be zero-mean, so provide a diverse picture rather than a single number with a varying magnitude.)

- Extra options, for \`constructor\` and \`resume\`:
    - \`volume = .3\`: amplitude of sound output.
    - \`minFrequency = 1000\`, \`maxFrequency = 13000\`: how well you can hear. [From 20 or 50, to 16000 or 20000 is reasonable.](https://en.wikipedia.org/wiki/Hearing_range) The wider the range, the higher the bandwidth.
    - \`nameImportance = .5\`: multiplier of cell names. Non-1 to make it easier on your ears, and emphasize data.
    - \`foregroundOnly = false\`: if set, switching away from the tab will stop the sound.
    - \`debug = false\`: if set, visualizes frequency data in a \`<canvas>\`. (Usable for quickly testing \`.Sensor.Video\`.)
` }
        static options() {
            return {
                volume: {
                    ['100%']: () => 1,
                    ['30%']: () => .3,
                    ['10%']: () => .1,
                    ['3%']: () => .03,
                    ['1%']: () => .01,
                },
                minFrequency: {
                    ['1 kHz']: () => 1000,
                    ['0 Hz']: () => 0,
                    ['5 kHz']: () => 5000,
                },
                maxFrequency: {
                    ['13 kHz']: () => 13000,
                    ['16 kHz']: () => 16000,
                    ['20 kHz']: () => 20000,
                    ['???']: () => 999999999,
                },
                nameImportance: {
                    ['50%']: () => .5,
                    ['100%']: () => 1,
                    ['0%']: () => 0,
                },
                foregroundOnly: {
                    No: false,
                    Yes: true,
                },
                debug: {
                    No: false,
                    Yes: true,
                },
            }
        }
        resume(opts) {
            if (opts) {
                opts.onValues = Sound.onValues
                opts.noFeedback = true
                this.volume = typeof opts.volume == 'number' && opts.volume >= 0 && opts.volume <= 1 ? opts.volume : .3
                this.minFrequency = opts.minFrequency !== undefined ? opts.minFrequency : 1000
                this.maxFrequency = opts.maxFrequency || 13000
                this.nameImportance = opts.nameImportance !== undefined ? opts.nameImportance : .5
                this.debug = opts.debug
                this.avgGap = 0
                this.foregroundOnly = !!opts.foregroundOnly
            }
            return super.resume(opts)
        }
        onlyIfNoExtension() { return true }
        static bench() {
            let loud = 1
            const sensorCounts = new Array(3).fill().map((_,i) => 1 + i*10)
            return sensorCounts.map(river)
            function river(sensors) { // Listen to random named data.
                const dataSize = 64
                return function start() {
                    function changeLoud(evt) { loud = Math.random() }
                    addEventListener('pointerdown', changeLoud)
                    const froms = new Array(sensors).fill().map(() => {
                        return new sn.Sensor({
                            name: ['some', 'kinda', 'name'],
                            values: 1*dataSize,
                            onValues(data) {
                                for (let i = 0; i < data.length; ++i)
                                    data[i] = (Math.random()*2-1) * loud
                                this.send(data)
                            },
                        })
                    })
                    const to = new Sound({
                        volume: .01,
                    })
                    return function stop() {
                        froms.forEach(from => from.pause()), to.pause()
                        removeEventListener('pointerdown', changeLoud)
                    }
                }
            }
        }
        static onValues(then, {data, cellShape}) {
            if (this.foregroundOnly && document.visibilityState === 'hidden')
                return setTimeout(then, 4000)
            if (!data || !data.length) return then()
            if (!Sound.ctx) {
                Sound.ctx = new AudioContext()
                Sound.next = Sound.ctx.currentTime
                Sound.overshoot = 0

                Sound.dst = Sound.ctx.createAnalyser()
                Sound.dst.fftSize = 2048
                Sound.dst.smoothingTimeConstant = .0
                Sound.dst.connect(Sound.ctx.destination)
            }
            if (this.debug && !Sound.debug) {
                const canvas = Sound.debug = document.createElement('canvas')
                canvas.width = 1024
                canvas.ctx = canvas.getContext('2d')
                document.body.append(canvas)
                draw()
                function draw() {
                    requestAnimationFrame(draw)
                    const data = new Float32Array(Sound.dst.frequencyBinCount)
                    const time = false
                    time ? Sound.dst.getFloatTimeDomainData(data) : Sound.dst.getFloatFrequencyData(data)

                    canvas.ctx.fillStyle = 'rgb(200, 200, 200)'
                    canvas.ctx.fillRect(0, 0, canvas.width, canvas.height)

                    canvas.ctx.lineWidth = 2
                    canvas.ctx.strokeStyle = 'rgb(0, 0, 0)'
                    canvas.ctx.beginPath()
                    const sliceWidth = canvas.width / data.length
                    let max = .000001
                    for (let i = 0; i < data.length; ++i) max = Math.max(max, Math.abs(data[i]))
                    for (let i = 0, x = 0; i < data.length; ++i, x += sliceWidth) {
                        const v = time ? (data[i]/max/2+.5) : (data[i] + 105) / 65
                        const y = canvas.height - v * canvas.height
                        !i ? canvas.ctx.moveTo(x, y) : canvas.ctx.lineTo(x, y)
                    }
                    canvas.ctx.stroke()
                }
            }
            const cellSize = cellShape.reduce((a,b)=>a+b)
            const channels = 1
            const sampleRate = Sound.ctx.sampleRate
            const maxFrequency = Math.min(this.maxFrequency, sampleRate)
            const soundLen = Math.ceil(data.length * sampleRate / maxFrequency)
            // Firefox has skips all over the sound if we don't delay.
            const delay = (Sound.ctx.outputLatency || Sound.ctx.baseLatency)
            const buf = Sound.ctx.createBuffer(channels, soundLen, sampleRate)
            const start = Math.max(Sound.next, Sound.ctx.currentTime + delay)
            const offset = this.minFrequency / sampleRate * soundLen
            writeData.call(this, data, buf.getChannelData(0), this.volume, offset, this.nameImportance, 0)
            const src = Sound.ctx.createBufferSource()
            src.buffer = buf
            const gap = Sound.ctx.currentTime - Sound.next
            this.avgGap = .9*this.avgGap + .1*gap
            sn.meta.metric('gap in sound, bool', start !== Sound.next ? 1 : 0)
            sn.meta.metric('latency, s', -gap)
            src.connect(Sound.dst), src.start(start)
            Sound.next = start + soundLen / sampleRate

            let needToWait = Math.max((Sound.next-.03 - Sound.ctx.currentTime - (start === Sound.next ? 0 : delay)) * 1000 - 5 - Sound.overshoot, 0)
            if (this.avgGap > 0) needToWait -= this.avgGap * 1000
            if (needToWait > Sound.overshoot) {
                const willLikelyEndAt = performance.now() + needToWait
                setTimeout(() => {
                    Sound.overshoot = .75*Sound.overshoot + .25*Math.max(performance.now() - willLikelyEndAt, 0)
                    then(needToWait)
                }, needToWait)
            } else then()

            function writeData(src, dst, volume, offset, p, phase) {
                const nameSize = cellSize - cellShape[cellShape.length-1]
                dst.fill(0)
                const off = Math.floor(offset)
                if (p !== 1) {
                    const avgPerCell = sn._allocF32(src.length / cellSize | 0)
                    for (let i = 0; i < src.length; ++i) {
                        const isName = (i % cellSize) < nameSize, cell = i / cellSize | 0
                        if (!isName) avgPerCell[cell] += src[i]
                    }
                    for (let i = 0; i < avgPerCell.length; ++i)
                        avgPerCell[i] /= cellSize - nameSize
                    for (let i = 0; i < src.length; ++i) {
                        const isName = (i % cellSize) < nameSize, cell = i / cellSize | 0
                        const v = isName ? p*src[i]+(1-p)*avgPerCell[cell] : src[i]
                        dst[off + i] = v
                    }
                    sn._deallocF32(avgPerCell)
                } else if (off + src.length <= src.length)
                    dst.set(src, off)
                else
                    for (let i = 0; i < src.length && off + i < dst.length; ++i)
                        dst[off + i] = src[i]
                // Normalize `dst` to have mean=0. (Too much clicking otherwise.)
                let mean = (dst.length > 2 ? dst.reduce((a,b)=>a+b) : 1) / dst.length
                if (!this.mean) this.mean = mean // Gradually.
                else this.mean = .9*this.mean + .1*mean
                for (let i = 0; i < dst.length; ++i)
                    dst[i] -= this.mean

                const real = ifft(dst, phase)
                real[0] = 0
                dst.set(real)
                for (let i = 0; i < dst.length; ++i) dst[i] *= volume
                sn._deallocF32(real)
            }
        }
    }
}