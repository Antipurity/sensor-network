export default function init(sn) {
    return class Keyboard extends sn.Sensor {
        static docs() { return `Tracks or controls keyboard state.

Options:
- \`name\`: heeded, augmented.
- \`noFeedback = true\`: \`true\` to track, \`false\` to control.
- \`keys = 4\`: max simultaneously-pressed [keys](https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/key/Key_Values) to report.
- \`keySize = 16\`: numbers per key.
- \`tokenToData = sn.Sensor.Text.tokenToDataMD5\`: converts a token to actual numbers.
    - \`function(token, data, start, end)\`
    - \`.feedback(feedback, start, end) → token\`
` }
        static options() {
            return {
                noFeedback: {
                    Yes: true,
                    No: false,
                },
                keys: {
                    ['4×']: () => 4,
                    ['3×']: () => 3,
                    ['2×']: () => 2,
                    ['1×']: () => 1,
                    ['8×']: () => 8,
                    ['16×']: () => 16,
                },
                keySize: {
                    ['16 ']: () => 16,
                    ['64 ']: () => 64,
                    ['256']: () => 256,
                },
            }
        }
        pause() {
            removeEventListener('keydown', this.onkeydown, {capture:true})
            removeEventListener('keyup', this.onkeyup, {capture:true})
            removeEventListener('blur', this.onkeyclear, {capture:true})
            return super.pause()
        }
        resume(opts) {
            if (!this.currentKeys)
                this.currentKeys = new Set,
                this.pastKeys = new Set, // For reporting blink-and-you'll-miss-it keypresses.
                this.onkeydown = this.onkeydown.bind(this),
                this.onkeyup = this.onkeyup.bind(this),
                this.onkeyclear = this.onkeyclear.bind(this)
            if (opts) {
                const keys = opts.keys || 4
                const keySize = opts.keySize || 16
                const tokenToData = sn.Sensor.Text.tokenToDataMD5
                sn._assertCounts('', keys, keySize)
                sn._assert(typeof tokenToData == 'function')
                Keyboard.prefill(tokenToData)
                const name = Array.isArray(opts.name) ? opts.name : typeof opts.name == 'string' ? [opts.name] : []
                opts.onValues = Keyboard.onValues
                opts.values = keys * keySize
                opts.name = [
                    'keyboard',
                    ...name,
                ]
                this.keys = keys, this.keySize = keySize
                this.tokenToData = tokenToData
            }
            try {
                return super.resume(opts)
            } finally {
                addEventListener('keydown', this.onkeydown, {capture:true})
                addEventListener('keyup', this.onkeyup, {capture:true})
                addEventListener('blur', this.onkeyclear, {capture:true})
            }
        }

        onkeydown(evt) {
            this.onkeyup(evt), this.currentKeys.add(evt.key), this.pastKeys.add(evt.key)
        }
        onkeyup(evt) {
            const m = this.currentKeys
            m.delete(evt.key), m.delete(evt.key.toLowerCase()), m.delete(evt.key.toUpperCase())
        }
        onkeyclear(evt) { this.currentKeys.clear() }

        static onValues(sensor, data) {
            const keys = sensor.keys, keySize = sensor.keySize
            let i = 0
            const cur = sensor.currentKeys, past = sensor.pastKeys
            cur.forEach(key => past.delete(key))
            past.forEach(key => cur.add(key))
            cur.forEach(key => {
                if (i++ < keys) sensor.tokenToData(key, data, i*keySize, (i+1)*keySize)
            }), data.fill(0, cur.size * keySize)
            past.forEach(key => cur.delete(key))
            past.clear()
            sensor.sendCallback(Keyboard.onFeedback, data)
        }
        static onFeedback(feedback, sensor) {
            if (!feedback || sensor.noFeedback) return
            // TODO: What do we do here?
        }

        static prefill(tokenToData) {
            // MD5 feedback has to have seen text to be able to invert it.
            // So, here we feed it the entire keyboard.
            if (tokenToData._keyboardFilled) return
            tokenToData._keyboardFilled = true
            "".split(' ').forEach(s => tokenToData(s))
            // TODO: https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/key/Key_Values
        }
    }
}