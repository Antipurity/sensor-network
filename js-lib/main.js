import { YaMD5 } from "yamd5.js"

;(function(exports) {
    // Browser compatibility (import):
    //   Chrome  61
    //   Edge    16
    //   Firefox 60
    //   Opera   48
    //   Safari  10.1
    //   WebView Android     61
    //   Chrome Android      61
    //   Firefox for Android 60
    //   Opera Android       45
    //   Safari on iOS       10.3
    //   Samsung Internet    8.0
    const E = exports, A = Object.assign

    A(E, {
        naming:{
            hasher: A(function hasher(name, partCount, partSize) {
                let parts = [], lastPartWasNumber = false
                flattenName(name), name = null
                return function specifyCell(...args) {
                    const numbers = fillParts(new Float32Array(partCount * partSize), partCount, partSize, ...args)
                    return function hash(dst, offset) { dst.set(numbers, offset) }
                }
                function flattenName(name) {
                    if (Array.isArray(name))
                        name.forEach(flattenName)
                    else if (typeof part == 'string') {
                        const i32 = YaMD5.hashStr(part, true)
                        const u8 = new Uint8Array(i32.buffer, i32.byteOffset, i32.byteLength)
                        const bytes = new Array(u8.length)
                        for (let i = 0; i < u8.length; ++i) bytes[i] = (u8 / 255) * 2 - 1
                        parts.push(bytes), lastPartWasNumber = false
                    } else if (typeof part == 'number' || typeof part == 'function') {
                        if (typeof part == 'number' && (part !== part || part < -1 || part > 1))
                            error("Name parts must be -1..1, got", part)
                        if (!lastPartWasNumber || parts[parts.length-1].length >= partSize)
                            parts.push([]), lastPartWasNumber = true
                        parts[parts.length-1].push(part)
                    } else error("Unrecognized name part:", part)
                }
                function fillParts(numbers, partCount, partSize, ...args) {
                    const end = Math.min(parts.length, partCount)
                    for (let p = 0; p < end; ++p) {
                        const part = parts[p]
                        for (let i = 0; i < part.length; ++i) {
                            let x = part[i]
                            if (typeof x == 'function') {
                                x = x(...args)
                                if (typeof x != 'number' || (x !== x || x < -1 || x > 1))
                                    error("Name parts must be -1..1, got", x)
                            }
                            numbers[p * partSize + i] = x
                        }
                        E.naming.fill(numbers, p * partSize, part.length, partSize)
                    }
                    E.naming.fill(numbers, 0, end, numbers.length)
                    return numbers
                }
            }, {
                docs:`Creates a function that allocates \`name\` into \`partCount\` parts, each with \`partSize\` numbers.
The result takes \`dataStart, dataEnd, dataLen\` (for filling function values) and returns a closure, which takes \`dst\` and \`dstOffset\`, and will write \`partCount * partSize\` -1..1 f32 numbers there when called.

\`name\`:
- A string. Hashed with MD5 and put in byte-by-byte into one part, rescaled to \`-1\`…\`1\`.
- A number, \`-1\`…\`1\`.
- A number-returning function, given \`dataStart, dataEnd, dataLen\`.
- An array of these.

If you want a different hashing strategy, replace this at runtime.`,
                fill:A(function fill(dst, offset, haveNumbers, needNumbers) {
                    if (haveNumbers >= needNumbers || !haveNumbers) return
                    for (let i = offset + haveNumbers; i < offset + needNumbers; ++i)
                        dst[i] = 1 - 2 * Math.abs(dst[i - haveNumbers])
                }, {
                    docs:`Where there is free space, this will put an end to it.

Increases sensitivity to variations, by fractally folding available values onto free space via \`x → 1-2*abs(x)\`.

Even if your AI model can only accept and return bits, it can still be used with the Sensor Network by having lots of free space.`,
                }),
                unfill:A(function unfill(dst, offset, haveNumbers, needNumbers) {
                    if (haveNumbers >= needNumbers || !haveNumbers) return
                    for (let i = offset + needNumbers - haveNumbers - 1; i >= offset; --i)
                        dst[i] = Math.sign(dst[i]) * Math.abs(dst[i + haveNumbers] - 1) * .5
                    return dst
                }, { docs:`Reverses \`.naming.fill\`, enhancing low-frequency numbers with best guesses from high-frequency numbers.

Makes only the sign matter for low-frequency numbers.` }),
                // TODO: Also have `.moveDataToNamed(src, dst, )` --- ...what are the args... (Also, maybe have a closure, with the name given already?) (Distibute evenly, to maximize the benefit of fractal-folding.)
                // TODO: ...Also need `.moveNamedToData(src, dst, )` --- ...what are the args...?
            }),
        },
    })
    function error(...msg) { throw new Error(msg.join(' ')) }
})(self.sn = Object.create(null))