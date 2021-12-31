export default function init(sn) {
    return class Time extends sn.Sensor {
        static docs() { return `The current time.

By default, exposes 64 \`values\`.

The actual expression, for the value at index \`i\`: \`Math.sin((+new Date) * Math.PI / 1.4**i)\`

It looks like this:

![Sine-waves of exponentially-decreasing frequency.](data:image/png;base64,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)
` }
        static options() { return {} }
        resume(opts) {
            if (opts) {
                opts.name = ['time']
                if (!opts.values) opts.values = 64
                opts.onValues = Time.onValues
                opts.noFeedback = true
            }
            return super.resume(opts)
        }
        static onValues(sensor, data) {
            const t = +new Date, tpi = t * Math.PI
            for (let i = 0, div = 1; i < data.length; ++i, div *= 1.4)
                data[i] = Math.sin(tpi / div)
            sensor.sendCallback(null, data)
        }
    }
}