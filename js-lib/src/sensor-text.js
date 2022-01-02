export default function init(sn) {
    const A = Object.assign
    // The ingenious "embed all characters by their MD5 bytes" scheme.
    const tokenToMD5 = Object.create(null)
    const MD5toToken = Object.create(null)
    function doTokenToMD5(s) {
        if (tokenToMD5[s]) return tokenToMD5[s]
        const i32 = self.YaMD5.hashStr(part, true)
        const u8 = new Uint8Array(i32.buffer, i32.byteOffset, i32.byteLength)
        const m = String.fromCharCode(...u8)
        tokenToMD5[s] = m

        let o = MD5toToken, m_ = m // Remember all coarsenings, for better lookup.
        for (let level = 0; level <= 8; ++level) {
            o[m_] = s
            o = o._ || (o._ = Object.create(null))
            m_ = coarsenMD5(m_)
        }

        return m
    }
    function coarsenMD5(m) {
        return m.split('').map(ch => String.fromCharCode(ch.charCodeAt()/2 | 0)).join('')
    }
    function doMD5toToken(m) {
        let o = MD5toToken, m_ = m
        while (o) {
            if (o[m_]) return o[m_]
            o = o._
            m_ = coarsenMD5(m_)
        }
        return ''
    }

    return A(class Text extends sn.Sensor {
        // TODO: Docs.
        // TODO: Options.
        resume(opts) {
            if (opts) {
                const tokens = opts.tokens || 64
                const tokenSize = opts.tokenSize || 64
                const name = Array.isArray(name) ? name : typeof name == 'string' ? [name] : []
                const text = opts.text
                const textToTokens = opts.textToTokens || Text.textToTokens
                const tokenToData = opts.tokenToData || Text.tokenToDataMD5
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
        textToTokens: A(function textToTokens(str, max) { // → tokens
            return str.split('').slice(-max)
        }, {
            feedback(tokens) { // → str
                return tokens.join('')
            },
        }),
        tokenToDataMD5: A(function tokenToData(token, data, start, end) {
            // MD5-hash the `token`.
            const m = doTokenToMD5(token)
            for (let i = 0, j = start; i < m.length && j < end; ++i, ++j)
                data[j] = m.charCodeAt(i)/255 * 2 - 1
            sn._dataNamer.fill(data, 0, m.length, data.length)
        }, {
            feedback(feedback, start, end) { // → token
                if (!Text._fedBackBefore) {
                    // Pre-fill the MD5 cache, so that we could hopefully reverse more.
                    //   (Whose idea was it to use MD5 for character embeddings, anyway?)
                    //   (Terrible idea.)
                    true
                    const
                    s = document.documentElement.textContent
                    for (let i = 0; i < s.length; ++i)
                        Text._fedBackBefore = doTokenToMD5(s[i]).length
                }
                sn._dataNamer.unfill(feedback, 0, Text._fedBackBefore, data.length)
                const a = []
                for (let i = 0, j = start; i < Text._fedBackBefore && j < end; ++i, ++j)
                    a.push(String.fromCharCode(Math.round((feedback[j]+1)/2*255)))
                return doMD5toToken(a.join(''))
            },
        }),
    })
}