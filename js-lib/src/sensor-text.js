export default function init(sn) {
    const A = Object.assign
    // The ingenious "embed all characters by their MD5 bytes" scheme.
    let fedBackMD5 = null
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
        // TODO: Test everything.
        resume(opts) {
            if (opts) {
                const tokens = opts.tokens || 64
                const tokenSize = opts.tokenSize || 64
                const name = Array.isArray(name) ? name : typeof name == 'string' ? [name] : []
                const text = opts.text || Text.readSelection()
                const textToTokens = opts.textToTokens || Text.textToTokens
                const tokenToData = opts.tokenToData || Text.tokenToDataMD5
                sn._assertCounts('', tokens, tokenSize)
                // TODO: Allow `text` to be a string or input or textarea.
                sn._assert(typeof text == 'function' || typeof text.feedback == 'function' && typeof textToTokens.feedback == 'function' && typeof tokenToData.feedback == 'function')
                sn._assert(typeof textToTokens == 'function' && typeof tokenToData == 'function')
                opts.values = tokens * tokenSize
                opts.name = [
                    'text',
                    ...name,
                    (dStart, dEnd, dLen) => dLen > dStart ? 2 * dStart / (dLen - dEnd) - 1 : 1
                ]
                opts.emptyValues = 0
                if (typeof text != 'object') opts.onValues = Text.onValues
                this.onFeedback = typeof text.onFeedback == 'function' ? Text.onFeedback : null
                this.text = text, this.textToTokens = textToTokens, this.tokenToData = tokenToData
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
    }, {
        readSelection: A(function readSelection(n=2048) {
            return function read() {
                const selection = getSelection().toString()
                if (selection) return selection.slice(-n)
                const el = document.activeElement
                if (el && (el.tagName === 'INPUT' && typeof el.value == 'string' && typeof el.selectionStart == 'number' || el.tagName === 'TEXTAREA')) { // TODO: Should this check be extracted to a function, so that even `resume` can use it?
                    const start = el.selectionStart, end = el.selectionEnd
                    return el.value.slice(start === end ? 0 : end, end).slice(-n)
                }
                return ''
            }
        }, {
            docs:`Reads the selection, in document and \`<input>\`s and \`<textarea>\`s.

Can pass the maximum returned string length, 2048 by default.`,
        }),
        readHover: A(function readHover(pos = sn.Sensor.Video.pointers(), n=2048) {
            return function read() {
                let p = pos
                if (Array.isArray(p) && !p.length) return p
                if (Array.isArray(p)) p = p[0]
                sn._assert(typeof p.x == 'number' && typeof p.y == 'number')
                const x = p.x * innerWidth, y = p.y * innerHeight
                let node, offset
                if (document.caretRangeFromPoint) {
                    const range = document.caretRangeFromPoint(x,y)
                    node = range.startContainer, offset = range.startOffset
                } else if (document.caretPositionFromPoint) {
                    const caret = document.caretPositionFromPoint(x,y)
                    node = caret.offsetNode, offset = caret.offset
                }
                if (node) {
                    let s = node.textContent
                    if (!(node instanceof Element)) { // Go to the word's end.
                        const ws = /\s|[!@#$%^&*()\[\]{},.<>/?;:'"]|$/
                        ws.lastIndex = offset
                        const wordEnd = s.search(ws)
                        s = s.slice(0, Math.max(wordEnd, offset))
                    }
                    while (s.length < n && node) { // Collect text before the hover-point too.
                        while (node && !node.previousSibling && node.parentNode)
                            node = node.parentNode
                        node = node.previousSibling
                        if (node) s = node.textContent + s
                    }
                    return s.slice(-n)
                }
            }
        }, {
            docs:`Reads the text under the pointer.

Can pass the \`{x,y}\` object (\`Video.pointers\` by default), and maximum returned string length, 2048 by default.`,
        }),
        // TODO: `writeSelection()`: what does it do, exactly?

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
                if (!fedBackMD5) {
                    // Pre-fill the MD5 cache, so that we could hopefully reverse more.
                    //   (Whose idea was it to use MD5 for character embeddings, anyway?)
                    //   (Terrible idea.)
                    true
                    const
                    s = document.documentElement.textContent
                    for (let i = 0; i < s.length; ++i)
                        fedBackMD5 = doTokenToMD5(s[i]).length
                }
                sn._dataNamer.unfill(feedback, 0, fedBackMD5, data.length)
                const a = []
                for (let i = 0, j = start; i < fedBackMD5 && j < end; ++i, ++j)
                    a.push(String.fromCharCode(Math.round((feedback[j]+1)/2*255)))
                return doMD5toToken(a.join(''))
            },
        }),
    })
}