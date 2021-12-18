// Creates the global `sn`, through which to access the sensor network.

import "./yamd5.js"

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
        namedData: A(function namedData({ reward=0, user='self', name, values, emptyValues=0, nameSize=64, namePartSize=16, dataSize=64, hasher=E.namedData.hasher }) {
            assert(typeof reward == 'number' && reward >= -1 && reward <= 1 || typeof reward == 'function')
            assert(typeof name == 'string' || Array.isArray(name), 'Must have a name')
            assert(typeof user == 'string' || Array.isArray(user))
            assertCounts('Must have the value-count', values)
            assertCounts('', emptyValues, nameSize, namePartSize, dataSize)
            assert(namePartSize < nameSize && nameSize % namePartSize === 0, 'Cell name must consist of an integer number of parts')
            const namePartCount = nameSize / namePartSize | 0
            const hasherMaker = hasher(name, namePartCount-1, namePartSize)
            // Values are distributed evenly per-cell, to maximize the benefit of fractal-folding.
            //   (The last cell may end up with less values than others. This slight inefficiency is worth the consistency.)
            const cells = Math.ceil((emptyValues + values) / dataSize), valuesPerCell = Math.ceil(values / cells)
            // (This re-hashes the user for each new sensor.)
            const userHasher = hasher(user, 1, namePartSize-1)(0, namePartSize-1, namePartSize-1)
            const nameHashers = new Array(cells).fill().map((_,i) => hasherMaker(i * valuesPerCell, Math.min((i+1) * valuesPerCell, values), values))
            return {
                cells,
                cellShape: [1, namePartSize-1, nameSize - namePartSize, dataSize], // [reward=1, user, name, data]
                name(src, dst, dstOffset) { // flat → named
                    for (let i = 0; i < cells; ++i) { // Fill out the whole cell.
                        const start = dstOffset + i * (nameSize + dataSize)
                        // Reward.
                        if (typeof reward == 'function') {
                            const r = reward()
                            assert(r >= -1 && r <= 1)
                            dst[start] = r
                        } else dst[start] = reward
                        // User.
                        userHasher(dst, start + 1)
                        // Name.
                        nameHashers[i](dst, start + namePartSize)
                        // Data.
                        const srcStart = i * valuesPerCell, srcEnd = Math.min(srcStart + valuesPerCell, src.length)
                        for (let s = srcStart, d = start + nameSize; s < srcEnd; ++s, ++d) dst[d] = src[s]
                        E.namedData.fill(dst, start + nameSize, valuesPerCell, dataSize)
                    }
                    return dstOffset + cells * (nameSize + dataSize) // Return the next `dstOffset`.
                },
                unname(src, srcOffset, dst) { // named → flat; `named` is consumed.
                    for (let i = 0; i < cells; ++i) { // Extract data from the whole cell.
                        const start = srcOffset + i * (nameSize + dataSize)
                        E.namedData.unfill(src, start + nameSize, valuesPerCell, dataSize)
                        const dstStart = i * valuesPerCell, dstEnd = Math.min(dstStart + valuesPerCell, dst.length)
                        for (let s = start + nameSize, d = dstStart; d < dstEnd; ++s, ++d) dst[d] = src[s]
                    }
                    return srcOffset + cells * (nameSize + dataSize) // Return the next `srcOffset`.
                },
            }
        }, {
            docs:`Implementation detail.

Prepares to go between flat number arrays and named ones.

The result is \`{ name(src, dst, dstOffset)→dstOffset, unname(src, srcOffset, dst)→srcOffset }\`. \`name\` goes from flat to named, \`unname\` reverses this.

Main parameters, in the one object that serves as arguments:
- \`name\`: describes this interface to handlers, such as with a string. See \`.namedData.hasher\`.
- \`values\`: how many flat numbers there will be.

Extra parameters:
- \`reward = 0\`: can be a function that will be called without arguments to get the cell's reward.
- \`user = 'self'\`: describes the current user/source/machine to handlers, mainly for when the sensor network encompasses multiple devices across the Internet.
- \`emptyValues = 0\`: how many fake \`values\` to insert, so that values are fractally folded more; see \`.namedData.fill\`.
- \`nameSize = 64\`: each cell is \`nameSize + dataSize\`.
- \`namePartSize = 16\`: the \`name\` can have many parts, and this determines how many parts there are.
- \`dataSize = 64\`: data in a cell.
- \`hasher = .namedData.hasher\`: defines how names are transformed into \`-1\`…\`1\` numbers.
`,
            tests() {
                const F32 = Float32Array
                const r1 = new F32(12)
                return [
                    [
                        new F32([0, 0.96, -0.5, -0.25, 0, 0.5, 0, 0.96, 0.25, 0.5, 0.5, 0]),
                        test(
                            opts => (E.namedData(opts).name(new Float32Array([-.5, -.25, .25, .5]), r1, 0), r1.map(round)),
                            { name:'z', values:4, emptyValues:1, dataSize:4, nameSize:2, namePartSize:1 },
                        ),
                    ],
                    same(1023),
                    same(1024),
                    same(1025),
                ]
                function same(n) { // Assert x = unname(name(x))
                    const src = new F32(n), dst = new F32(n)
                    for (let i=0; i < n; ++i) src[i] = Math.random() * 2 - 1
                    const opts = { name:'matters not', values:n, dataSize:64, nameSize:64, namePartSize:16 }
                    const namer = E.namedData(opts)
                    const cells = new F32(namer.cells * (64+64))
                    namer.name(src, cells, 0), namer.unname(cells, 0, dst)
                    return [src.map(round), dst.map(round)]
                }
                function round(x) { return (x*100|0) / 100 }
            },
            hasher: A(function hasher(name, partCount, partSize) {
                let parts = [], lastPartWasNumber = false
                flattenName(name), name = null
                return function specifyCell(...args) {
                    const numbers = fillParts(new Float32Array(partCount * partSize), ...args)
                    return function hash(dst, offset) { dst.set(numbers, offset) }
                }
                function flattenName(part) {
                    if (Array.isArray(part))
                        part.forEach(flattenName)
                    else if (typeof part == 'string') {
                        const i32 = YaMD5.hashStr(part, true)
                        const u8 = new Uint8Array(i32.buffer, i32.byteOffset, i32.byteLength)
                        const bytes = new Array(Math.min(u8.length, partSize))
                        for (let i = 0; i < bytes.length; ++i) bytes[i] = (u8[i] / 255) * 2 - 1
                        parts.push(bytes), lastPartWasNumber = false
                    } else if (typeof part == 'number' || typeof part == 'function') {
                        if (typeof part == 'number' && (part !== part || part < -1 || part > 1))
                            error("Name parts must be -1..1, got", part)
                        if (!lastPartWasNumber || parts[parts.length-1].length >= partSize)
                            parts.push([]), lastPartWasNumber = true
                        parts[parts.length-1].push(part)
                    } else error("Unrecognized name part:", part)
                }
                function fillParts(numbers, ...args) {
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
                        E.namedData.fill(numbers, p * partSize, part.length, partSize)
                    }
                    E.namedData.fill(numbers, 0, end, numbers.length)
                    return numbers
                }
            }, {
                docs:`Creates a function that allocates \`name\` into \`partCount\` parts, each with \`partSize\` numbers.

The result takes \`dataStart, dataEnd, dataLen\` (for filling function values) and returns a closure, which takes \`dst\` and \`dstOffset\`, and will write \`partCount * partSize\` -1..1 f32 numbers there when called.

\`name\`:
- A string. Hashed with MD5 and put in byte-by-byte into one part, rescaled to \`-1\`…\`1\`.
- A number, \`-1\`…\`1\`.
- A number-returning function, given \`dataStart, dataEnd, dataLen\`.
- An array of these.`,
            }),
            fill: A(function fill(dst, offset, haveNumbers, needNumbers) {
                if (haveNumbers >= needNumbers || !haveNumbers) return
                for (let i = offset + haveNumbers; i < offset + needNumbers; ++i)
                    dst[i] = 1 - 2 * Math.abs(dst[i - haveNumbers])
            }, {
                docs:`Where there is free space, this will put an end to it.

Increases sensitivity to variations, by fractally folding available values onto free space via \`x → 1-2*abs(x)\`.

Even if your AI model can only accept and return ±1 bits, it can still use the Sensor Network by having lots of free space (\`emptyValues\` in \`.namedData\`).`,
            }),
            unfill: A(function unfill(dst, offset, haveNumbers, needNumbers) {
                if (haveNumbers >= needNumbers || !haveNumbers) return
                for (let i = offset + needNumbers - haveNumbers - 1; i >= offset; --i)
                    dst[i] = Math.sign(dst[i]) * Math.abs(dst[i + haveNumbers] - 1) * .5
                return dst
            }, { docs:`Reverses \`.namedData.fill\`, enhancing low-frequency numbers with best guesses from high-frequency numbers.

Makes only the sign matter for low-frequency numbers.` }),
        }),
    })
    function test(func, ...args) {
        try { return func(...args) }
        catch (err) { return err instanceof Error ? [err.message, err.stack] : err }
    }
    function assertCounts(msg, ...xs) { assert(xs.every(x => typeof x == 'number' && x >= 0 && x === x>>>0), msg || 'Must be a non-negative integer') }
    function assert(bool, msg) { if (!bool) error(msg || 'Assertion failed') }
    function error(...msg) { throw new Error(msg.join(' ')) }
})(self.sn = Object.create(null))