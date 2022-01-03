export default function init(sn) { // TODO: OHHH NOOO: `Text` should at least define `.save:{a}`, so that it could be `save`d and later loaded...
    const A = Object.assign
    // The ingenious "embed all characters by their MD5 bytes" scheme.
    let fedBackMD5 = null // TODO: All these should be on some func.
    const tokenToMD5 = Object.create(null)
    const MD5toToken = Object.create(null)
    function doTokenToMD5(s) {
        if (tokenToMD5[s]) return tokenToMD5[s]
        const i32 = self.YaMD5.hashStr(s, true)
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
    function MD5isEmpty(m) {
        for (let i = 0; i < m.length; ++i) if (m.charCodeAt(i) !== 0) return false
        return true
    }
    function doMD5toToken(m) {
        let o = MD5toToken, m_ = m
        while (o) {
            if (MD5isEmpty(m_)) return '' // Allow emptiness.
            if (o[m_]) return o[m_]
            o = o._
            m_ = coarsenMD5(m_)
        }
        return ''
    }

    return A(class Text extends sn.Sensor {
        static docs() { return `Observe text, or suggest completions.

Text: abstract, compressed, easy for humans to create. You are reading it.    
Split into tokens, which are presumably interconnected, and defined by each other.

Options:
- \`name\`: heeded, augmented.
- \`tokens = 64\`: max token count, in characters by default.
- \`tokenSize = 64\`: how many numbers each token takes up. Ideally, should match the handler's \`dataSize\`.
- \`text = Text.readSelection()\`: the actual text observation.
    - A string, or \`<input>\` or \`<textarea>\`, or a function that returns a string.
    - Optional \`.feedback(string)\`.
- \`textToTokens = Text.textToTokens\`: splits text into tokens, characters by default.
    - \`function(string, tokens) → [...token]\`
    - \`.feedback([...token]) → string\`
- \`tokenToData = Text.tokenToDataMD5\`: converts a token to actual numbers.
    - \`function(token, data, start, end)\`
    - \`.feedback(feedback, start, end) → token\`
`}
        static options() {
            return {
                tokens: {
                    ['64×']: () => 64,
                    ['128×']: () => 128,
                    ['256×']: () => 256,
                    ['512×']: () => 512,
                    ['1024×']: () => 1024,
                    ['8×']: () => 8,
                },
                tokenSize: {
                    ['64 ']: () => 64,
                    ['16 ']: () => 16,
                    ['256 ']: () => 256,
                },
                text: {
                    ['Read selection']: () => sn.Sensor.Text.readSelection(),
                    ['Read hovered-over text']: () => sn.Sensor.Text.readHover(sn.Sensor.Video.pointers()),
                    ['Write selection']: () => sn.Sensor.Text.writeSelection(),
                },
            }
        }
        resume(opts) {
            if (opts) {
                const tokens = opts.tokens || 64
                const tokenSize = opts.tokenSize || 64
                const name = Array.isArray(opts.name) ? opts.name : typeof opts.name == 'string' ? [opts.name] : []
                const text = opts.text || Text.readSelection()
                const textToTokens = opts.textToTokens || Text.textToTokens
                const tokenToData = opts.tokenToData || Text.tokenToDataMD5
                const haveText = typeof text == 'string' || typeof text == 'function' || isInputy(text)
                const haveTextFeedback = (typeof text.feedback == 'function' || isInputy(text.feedback)) && typeof textToTokens.feedback == 'function' && typeof tokenToData.feedback == 'function'
                sn._assertCounts('', tokens, tokenSize)
                sn._assert(haveText || haveTextFeedback)
                sn._assert(typeof textToTokens == 'function' && typeof tokenToData == 'function')
                opts.values = tokens * tokenSize
                opts.name = [
                    'text',
                    ...name,
                    (dStart, dEnd, dLen) => 2 * dEnd / dLen - 1,
                ]
                opts.emptyValues = 0
                opts.onValues = Text.onValues
                opts.noFeedback = !haveTextFeedback
                this.onFeedback = typeof text.feedback == 'function' ? Text.onFeedback : null
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

            const txt = sensor.text
            const str = typeof txt == 'string' ? txt : isInputy(txt) ? txt.value : typeof txt == 'function' ? txt() : null
            if (str != null) {
                sn._assert(typeof str == 'string')
                const tokens = sensor.textToTokens(str, sensor.tokens)
                for (let i = 0; i < tokens.length; ++i)
                    sensor.tokenToData(tokens[i], data, i*valuesPerCell, (i+1)*valuesPerCell)
                data.fill(0, tokens.length)
                sensor.sendCallback(sensor.onFeedback, data)
            } else
                sensor.sendCallback(sensor.onFeedback, null)
        }
        static onFeedback(feedback, sensor) {
            if (!feedback) return
            const cellShape = sensor.cellShape()
            if (!cellShape || !sensor.text || !sensor.text.feedback) return
            const dataSize = cellShape[cellShape.length-1]
            const valuesPerCell = dataSize // Since sensor.emptyValues === 0.

            const tokens = [], end = Math.min(sensor.tokens, feedback.length / dataSize | 0)
            for (let i = 0; i < end; ++i)
                tokens.push(sensor.tokenToData.feedback(feedback, i*valuesPerCell, (i+1)*valuesPerCell))
            const str = sensor.textToTokens.feedback(tokens)
            const txt = sensor.text.feedback
            isInputy(txt) ? (txt.value = str) : txt(str)
        }
    }, {
        readSelection: A(function readSelection(n=2048) {
            return function read() {
                const el = document.activeElement
                if (isInputy(el)) {
                    let start = el.selectionStart, end = el.selectionEnd
                    ;[start, end] = [Math.min(start, end), Math.max(start, end)]
                    return el.value.slice(start === end ? 0 : start, end).slice(-n)
                }
                const selection = getSelection().toString()
                if (selection) return selection.slice(-n)
                return ''
            }
        }, {
            docs:`Reads the selection, in document and \`<input>\`s and \`<textarea>\`s.

Can pass the maximum returned string length, 2048 by default.`,
        }),
        readHover: A(function readHover(pos = sn.Sensor.Video.pointers(), n=2048) {
            return function read() {
                let p = pos
                if (typeof p == 'function') p = p()
                if (Array.isArray(p) && !p.length) return ''
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
                        const ws = /\s|[!@#$%^&*()\[\]{},\.\-<>/?;:'"\|\&]|$/
                        // String.prototype.search doesn't support sticky regexes, for whatever reason.
                        const wordEnd = offset + s.slice(offset).search(ws)
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

Can pass the \`{x,y}\` object or array/function to that object (\`Video.pointers()\` by default), and maximum returned string length, 2048 by default.`,
        }),
        writeSelection: A(function writeSelection() {
            return { feedback: function write(str) {
                if (!str) return
                const el = document.activeElement
                if (isInputy(el)) {
                    if (el.disabled || el.readOnly) return
                    let start = el.selectionStart, end = el.selectionEnd
                    ;[start, end] = [Math.min(start, end), Math.max(start, end)]
                    el.setRangeText(str, start, end, 'select')
                    return
                }
                const selection = getSelection()
                if (selection.rangeCount) {
                    const range = selection.getRangeAt(0)
                    let editable = false
                    for (let p = range.commonAncestorContainer; p; p = p.parentNode)
                        if (p.isContentEditable) editablee = true
                    if (!editable) return
                    range.deleteContents()
                    range.insertNode(document.createTextNode(str))
                }
            } }
        }, {
            docs:`Modifies the selection to be the feedback, if possible.

The new text will still be selected, so it can function as autocomplete or autocorrect.`,
        }),

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
                sn._dataNamer.unfill(feedback, 0, fedBackMD5, feedback.length)
                const a = []
                for (let i = 0, j = start; i < fedBackMD5 && j < end; ++i, ++j)
                    a.push(String.fromCharCode(Math.round((feedback[j]+1)/2*255)))
                return doMD5toToken(a.join(''))
            },
        }),
    })
    function isInputy(el) {
        return el && el instanceof Element && (el.tagName === 'INPUT' && typeof el.value == 'string' && typeof el.selectionStart == 'number' || el.tagName === 'TEXTAREA')
    }
}