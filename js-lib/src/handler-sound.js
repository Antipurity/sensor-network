export default function init(sn) {
    return class Sound extends sn.Handler { // TODO: Integrate it into `sn` already, through backpatching. ...Wait, even just importing it throws... I thought testing showed viability?? Do we need to actually export an initializing function, maybe? And don't even import `sn`.
        constructor(...a) { return super(...a) }
        docs() { return `// TODO: Docs.
    ` }
        resume(opts) {
            if (opts) {
                opts.onValues = Sound.onValues
                opts.noFeedback = true
            }
            super.resume(opts)
        }
        onlyIfNoExtension() { return true } // TODO: Make the extension suppress handlers with this, by flipping a bool in `sn`.
        // TODO: Also, `visualize({data, error}, elem)`, which draws on a canvas.
        static onValues({data, error}, cellShape) {
            // TODO: Test.
            // console.log(data.length) // TODO: Why 0?! ...And why is the sensor not even called?! BUG
            //   ...Is it because we have no main handler, and so no one will auto-call our sensors...
            //   TODO: If no shape in a channel has a main handler, auto-call sensors anyway.
            //     Maybe, maintain `mainSensor`?
            if (!data || !data.length) return
            if (!Sound.ctx) {
                Sound.ctx = new AudioContext()
                Sound.next = Sound.ctx.currentTime
                // Firefox has skips if we don't delay.
                Sound.delay = Sound.ctx.outputLatency || Sound.ctx.baseLatency

                Sound.dst = Sound.ctx.createAnalyser()
                Sound.dst.fftSize = 2048
                Sound.dst.connect(Sound.ctx.destination)
            }
            const channels = 1
            const sampleRate = Sound.ctx.sampleRate
            const soundLen = dataLenToSoundLen(data.length)
            const buf = Sound.ctx.createBuffer(channels, soundLen, sampleRate)
            writeData(data, buf.getChannelData(0))
            const src = Sound.ctx.createBufferSource()
            src.buffer = buf
            const start = Math.max(Sound.next, Sound.ctx.currentTime + Sound.delay)
            src.connect(Sound.dst), src.start(start)
            Sound.next = start + soundLen / sampleRate
            function dataLenToSoundLen(n) { return n * 4 }
            function writeData(src, dst) {
                for (let i = 0; i < src.length; ++i) {
                    const v = src[i]
                    dst[i*4+0] = v
                    dst[i*4+1] = v
                    dst[i*4+2] = v
                    dst[i*4+3] = v
                }
            }
        }
    }
}