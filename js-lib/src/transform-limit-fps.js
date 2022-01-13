export default function init(sn) {
    return class LimitFPS extends sn.Transform {
        static docs() { return `Limits steps-per-second.

Useful if you don't want to record ten thousand steps per second, almost none of which convey any useful information.

Options:
- \`hz = 60\`: max frequency.` }
        static options() { return {
            hz: {
                ['60 ']: () => 60,
                ['30 ']: () => 30,
                ['10 ']: () => 10,
                ['3 ']: () => 3,
            },
        } }
        resume(opts) {
            if (opts)
                opts.onValues = LimitFPS.onValues, this.hz = opts.hz || 3, this.last = performance.now()
            return super.resume(opts)
        }
        static onValues(then, input) {
            const now = performance.now()
            const period = 1000 / this.hz
            if (now - this.last > period) this.last = now, then()
            else { // Delay if needed.
                const delay = Math.max(0, period - (now - this.last) - 5)
                setTimeout(LimitFPS.resume, delay, then, this.last = period + this.last)
            }
        }
        static resume(then, end) {
            // Could do a spinlock until `end` has passed, but why the need for such precision?
            then()
        }
    }
}