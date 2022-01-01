export default function init(sn) { // TODO: Assign this to `sn.Sensor`.
    const A = Object.assign
    return A(class Text extends sn.Sensor {
        // TODO: Docs.
        // TODO: Options.
        resume(opts) {
            if (opts) {
                const tokens = opts.tokens || 64
                const tokenSize = opts.tokenSize || 64
                const name = Array.isArray(name) ? name : typeof name == 'string' ? [name] : []
                const text = opts.text
                const textToTokens = opts.textToTokens
                const tokenToData = opts.tokenToData
                sn._assertCounts('', tokens, tokenSize)
                sn._assert(typeof text == 'function' || typeof text.feedback == 'function' && typeof textToTokens.feedback == 'function' && typeof tokenToData.feedback == 'function')
                sn._assert(typeof textToTokens == 'function' && typeof tokenToData == 'function')
                opts.values = tokens * tokenSize
                opts.name = [
                    'text',
                    ...name,
                    (dStart, dEnd, dLen) => dLen > dStart ? 2 * dStart / (dLen - dEnd) - 1 : 1
                ]
                this.tokens = tokens, this.tokenSize = tokenSize
                if (typeof text == 'function') opts.onValues = Text.onValues
                if (typeof text.feedback == 'function') opts.onFeedback = Text.onFeedback // TODO: ...Uh... `sensor.sendCallback` accepts a callback; we don't set it here.
                this.textToTokens = textToTokens, this.tokenToData = tokenToData
            }
            return super.resume(opts)
        }
        // TODO: `static onValues(sensor, data)`; what does it do?
        //   this.textToTokens(this.text(), this.tokens) → tokens
        //   For each token: this.tokenToData(tokens[i], data, i*this.tokenSize, (i+1)*this.tokenSize)
        //     Zero-fill the non-provided tokens.
        // TODO: `static onFeedback(feedback, sensor)`; what does it do?
        //   For each token: this.tokenToData.feedback()
        // TODO: `static readSelection(n=2048)`; what does it do, exactly?
        // TODO: `static readHover(n=2048)`; what does it do, exactly?
        // TODO: `readChanges(n=2048)`; what does it do, exactly?
        // TODO: `writeSelection()`
    }, {
        // TODO: `_textToTokens`, splitting every char.
        //   TODO: `.feedback`.
        // TODO: `_tokenToData`, one-hot-encoding base64 or MD5-ing.
        //   TODO: `.feedback`.
    })
}