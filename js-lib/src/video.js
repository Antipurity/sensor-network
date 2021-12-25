export default function init(sn) { // TODO: Set `.Sensor.Video` in `sn`.
    return class Video extends sn.Sensor {
        docs() { return `// TODO:
` }
        // TODO: How do we do this?
        resume(opts) {
            if (opts) {
                opts.extraValues = 0
                opts.onValues = Video.onValues
                // TODO: Also set `name` and `values`.
                //   ...What's `values`?
                // TODO: What props do we want?
            }
            super.resume(opts)
        }
        // TODO: First, at least make it work for , right?

        static onValues(sensor, data) {
            const targetShape = sensor.cellShape()
            if (!targetShape) return
            const cellData = targetShape[targetShape.length-1]
            // TODO: How to fill `data`?
            //   Can we at least get some imageData?… From where? …Okay, so, source is next, right?
            // TODO: sensor.sendCallback(Video.onFeedback)
        }
        static onFeedback(feedback, sensor) {
            if (!feedback) return
            // TODO: What do we do here?
        }
    }
}