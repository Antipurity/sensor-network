export default function init(sn) {
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
                opts.emptyValues = 0
                if (typeof text == 'function') opts.onValues = Text.onValues
                this.onFeedback = typeof text.onFeedback == 'function' ? Text.onFeedback : null
                this.textToTokens = textToTokens, this.tokenToData = tokenToData
                this.tokens = tokens, this.tokenSize = tokenSize
            }
            return super.resume(opts)
        }
        static onValues(sensor, data) {
            const cellShape = sensor.cellShape()
            if (!cellShape) return
            const dataSize = cellShape[cellShape.length-1]
            const valuesPerCell = dataSize // Since sensor.emptyValues === 0.

            const str = sensor.text()
            const tokens = sensor.textToTokens(str, sensor.tokens)
            for (let i = 0; i < tokens.length; ++i)
                sensor.tokenToData(tokens[i], data, i*valuesPerCell, (i+1)*valuesPerCell)
            data.fill(0, tokens.length)
            sensor.send(sensor.onFeedback, data)
        }
        static onFeedback(feedback, sensor) {
            const cellShape = sensor.cellShape()
            if (!cellShape) return
            const dataSize = cellShape[cellShape.length-1]
            const valuesPerCell = dataSize // Since sensor.emptyValues === 0.

            const tokens = [], end = Math.min(sensor.tokens, feedback.length / dataSize | 0)
            for (let i = 0; i < end; ++i)
                tokens.push(sensor.tokenToData.feedback(feedback, i*valuesPerCell, (i+1)*valuesPerCell))
            const str = sensor.textToTokens.feedback(tokens)
            sensor.text.feedback(str)
        }
        // TODO: `static readSelection(n=2048)`; what does it do, exactly?
        // TODO: `static readHover(n=2048)`; what does it do, exactly?
        // TODO: `readChanges(n=2048)`; what does it do, exactly?
        // TODO: `writeSelection()`: what does it do, exactly?
    }, {
        // TODO: `_textToTokens`, splitting every char.
        //   TODO: `.feedback`.
        // TODO: `_tokenToData`, one-hot-encoding base64 or MD5-ing.
        //   TODO: `.feedback`.
    })
}