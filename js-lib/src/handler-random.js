export default function init(sn) {
    return class Random extends sn.Handler {
        static docs() { return `Random -1…1 feedback, for debugging.` }
        static options() { return {} }
        resume(opts) {
            if (opts)
                opts.onValues = Random.onValues
            return super.resume(opts)
        }
        static onValues(then, opts, feedback) {
            try {
                if (feedback)
                    for (let i = 0; i < feedback.length; ++i)
                        feedback[i] = Math.random()*2-1
            } finally { then() }
        }
    }
}