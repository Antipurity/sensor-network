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
                    ['256 ']: () => 256,
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
                this.onkeyclear = this.onkeyclear.bind(this),
                this._pastFBKeys = new Set // For feedback.
            if (opts) {
                const keys = opts.keys || 4
                const keySize = opts.keySize || 16
                const tokenToData = sn.Sensor.Text.tokenToDataMD5
                sn._assertCounts('', keys, keySize)
                sn._assert(typeof tokenToData == 'function')
                if (!opts.noFeedback) ;
                Keyboard.prefill(tokenToData)
                const name = Array.isArray(opts.name) ? opts.name : typeof opts.name == 'string' ? [opts.name] : []
                opts.onValues = this.onValues
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

        onValues(data) {
            const keys = this.keys, keySize = this.keySize
            let i = 0
            const cur = this.currentKeys, past = this.pastKeys, fn = this.tokenToData
            cur.forEach(key => past.delete(key))
            past.forEach(key => cur.add(key))
            cur.forEach(key => {
                if (i < keys) fn(key, data, i*keySize, (i+1)*keySize)
                ++i
            }), data.fill(0, cur.size * keySize)
            past.forEach(key => cur.delete(key))
            past.clear()
            this.sendCallback(this.onFeedback, data)
        }
        onFeedback(feedback) {
            // Differences from browser behavior:
            //   - No `.isTrusted` (no way to set it).
            //   - No `.repeat`.
            //   - No `onkeypress` (it's deprecated).
            //   - No `oninput`.
            //   - No .ctrlKey/.shiftKey/.altKey/.metaKey on *other* events (such as `Pointer` events).
            if (!feedback || this.noFeedback) return
            // Get what keys are currently pressed.
            const keys = this.keys, keySize = this.keySize
            const fn = this.tokenToData.feedback
            const currentKeys = new Set
            for (let i = 0; i < keys && i*keySize < feedback.length; ++i) {
                const key = fn(feedback, i*keySize, (i+1)*keySize)
                if (key) currentKeys.add(key)
            }
            // Dispatch events.
            const kbdOpts = {
                bubbles: true,
                key: '',
                ctrlKey: currentKeys.has('Control'),
                shiftKey: currentKeys.has('Shift'),
                altKey: currentKeys.has('Alt'),
                metaKey: currentKeys.has('Meta'),
            }
            const pastKeys = this._pastFBKeys
            const el = document.activeElement || document.body
            currentKeys.forEach(key => {
                if (!pastKeys.has(key)) {
                    kbdOpts.key = key, el.dispatchEvent(new KeyboardEvent('keydown', kbdOpts))
                    if ([...key].length === 1)
                        document.execCommand('insertText', false, key)
                    else if (key === 'Backspace')
                        document.execCommand('delete', false, null)
                    else if (key === 'Delete')
                        document.execCommand('forwardDelete', false, null)
                }
            })
            pastKeys.forEach(key => {
                if (!currentKeys.has(key))
                    kbdOpts.key = key, el.dispatchEvent(new KeyboardEvent('keyup', kbdOpts))
            })
        }

        static prefill(tokenToData) {
            // MD5 feedback has to have seen text to be able to invert it.
            // So, here we feed it practically the entire keyboard.
            if (tokenToData._keyboardFilled) return
            tokenToData._keyboardFilled = true
            // For a fuller list see: https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/key/Key_Values
            ;[' ', ...("! \" # $ % & ' ( ) * + , - . / 0 1 2 3 4 5 6 7 8 9 : ; < = > ? @ A Alt ArrowDown ArrowLeft ArrowRight ArrowUp B Backspace C CapsLock Control D Delete E End Enter Escape F G H Home I J K L M N NumLock O P PageDown PageUp Q R S Shift T Tab U V W X Y Z [ \\ ] ^ _ ` a b c d e f g h i j k l m n o p q r s t u v w x y z { | } ~".split(' '))].forEach(s => tokenToData(s))
        }
    }
}