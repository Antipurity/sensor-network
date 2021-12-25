export default function init(sn) {
    return class Sound extends sn.Handler {
        docs() { return `Exposes data as sound, for humans to listen.

In Chrome, users might have to first click on the page for sound to play.

- Extra options, for \`constructor\` and \`resume\`:
    - \`volume = 1\`: multiplier of sound output.
` }
        resume(opts) {
            if (opts) {
                opts.onValues = Sound.onValues
                opts.noFeedback = true
                this.volume = opts.volume >= 0 && opts.volume <= 1 ? opts.volume : 1
            }
            super.resume(opts)
        }
        onlyIfNoExtension() { return true } // TODO: Make the extension suppress handlers with this, by flipping a bool in `sn`.
        // TODO: Also, `visualize({data, error}, elem)`, which draws on a canvas.
        //   (May not want actual sound here, even, but an actual *visual*ization.)
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
                            onValues(sensor, data) {
                                for (let i = 0; i < data.length; ++i)
                                    data[i] = (Math.random()*2-1) * loud
                                sensor.send(data)
                            },
                        })
                    })
                    const to = new Sound({
                        volume: .03,
                    })
                    return function stop() {
                        froms.forEach(from => from.pause()), to.pause()
                        removeEventListener('pointerdown', changeLoud)
                    }
                }
            }
        }
        static async onValues({data, error, cellShape}) {
            if (!data || !data.length) return
            if (!Sound.ctx) {
                Sound.ctx = new AudioContext()
                Sound.next = Sound.ctx.currentTime

                Sound.dst = Sound.ctx.createAnalyser()
                Sound.dst.fftSize = 2048
                Sound.dst.connect(Sound.ctx.destination)
                Sound.overshoot = 0
                // Metrics.

                // TODO: Remove this silliness. (Only do it on visualization.)
                const canvas = document.createElement('canvas')
                canvas.width = 1024
                canvas.ctx = canvas.getContext('2d')
                document.body.append(canvas)
                const T = this
                draw()
                function draw() {
                    requestAnimationFrame(draw)
                    const data = new Float32Array(Sound.dst.frequencyBinCount)
                    Sound.dst.getFloatTimeDomainData(data)

                    canvas.ctx.fillStyle = 'rgb(200, 200, 200)'
                    canvas.ctx.fillRect(0, 0, canvas.width, canvas.height)

                    canvas.ctx.lineWidth = 2
                    canvas.ctx.strokeStyle = 'rgb(0, 0, 0)'
                    canvas.ctx.beginPath()
                    const sliceWidth = canvas.width / data.length
                    const mid = canvas.height / 2
                    for (let i = 0, x = 0; i < data.length; ++i, x += sliceWidth) {
                        const y = mid + data[i] / T.volume * mid
                        !i ? canvas.ctx.moveTo(x, y) : canvas.ctx.lineTo(x, y)
                    }
                    canvas.ctx.lineTo(canvas.width, mid)
                    canvas.ctx.stroke()
                }
            }
            const channels = 1
            const sampleRate = Sound.ctx.sampleRate
            const soundLen = dataLenToSoundLen(data.length)
            // Firefox has skips all over the sound if we don't delay.
            const delay = Sound.ctx.outputLatency || Sound.ctx.baseLatency
            const buf = Sound.ctx.createBuffer(channels, soundLen, sampleRate)
            writeData(data, buf.getChannelData(0), this.volume)
            const src = Sound.ctx.createBufferSource()
            src.buffer = buf
            const start = Math.max(Sound.next, Sound.ctx.currentTime + delay)
            sn.meta.metric('latency, s', start - Sound.ctx.currentTime)
            src.connect(Sound.dst), src.start(start)
            Sound.next = start + soundLen / sampleRate

            const needToWait = (Sound.next-.01 - Sound.ctx.currentTime - (start === Sound.next ? 0 : delay)) * 1000 - Sound.overshoot
            if (needToWait > Sound.overshoot) {
                const A = performance.now(), B = A + needToWait
                await new Promise(then => setTimeout(then, needToWait))
                Sound.overshoot = Math.max(performance.now() - B, 0)
            }

            function dataLenToSoundLen(n) { return n * 4 }
            function writeData(src, dst, volume) { // TODO: An option to use IFFT.
                for (let i = 0; i < src.length; ++i) {
                    const v = src[i]
                    dst[i*4+0] = v * volume
                    dst[i*4+1] = v * volume
                    dst[i*4+2] = -v * volume
                    dst[i*4+3] = -v * volume
                }
            }
        }
    }
}