export default function init(sn) { // TODO: Set `.Sensor.Video` in `sn`.
    return class Video extends sn.Sensor {
        // TODO: How do we do this?
        resume(opts) {
            if (opts) {
                opts.onValues = Video.onValues
                opts.noFeedback = true // TODO: This is pass-through, because we would actually have feedback, right?
                // TODO: What props do we want?
            }
            super.resume(opts)
        }
    }
}