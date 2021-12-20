import "./yamd5.js"

export default (function(exports) {
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

    const S = Object.create(null)
    // [channel]:
    //   sensors: Array<Sensor>, but only those that are called automatically.
    //   accumulators: Array<Accumulator>, sorted by priority.
    //   mainHandler: Handler, with max priority.
    //   handlerShapeStrings: Array<String>, for enumeration.
    //   stepsNow: int
    //   waitingOnSteps: Array<function>, called when a step is finished.
    //   [handlerShapeAsString]:
    //     looping: bool
    //     msPerStep: number
    //     cellShape: [reward=1, user, name, data]
    //     handlers: Array<Handler>, sorted by priority.
    //     nextPacket: _Packet

    return A(E, {
        Sensor: A(class Sensor {
            // TODO: `.constructor({ name, values=0, channel='', noFeedback=false, onValues=null })`
            //   TODO: If `onValues` is not `null`, `S[channel].sensors.push(this)`.
            //   TODO: Install getters/setters on the options object, and when anything changes, reinstall it.
            // TODO: `.send(values: Float32Array|null, error: Float32Array|null, reward=0) -> Promise<Float32Array|null>`: send data, receive feedback, once, to all handler shapes. (Reward is not fed back.)
            // TODO: `.deinit()`
            // TODO: For convenience, if [`FinalizationRegistry`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/FinalizationRegistry) is present, stop when the sender is no longer needed (else, set `onValues` to `null`).
        }, {}),
        Accumulator: A(class Accumulator {
            // TODO: `.constructor({ channel='', priority=0, onValues=null, onFeedback=null })`
            //   TODO: `S[channel].accumulators.push(this)`, and sort by priority.
            //   TODO: Install getters/setters on the options object, and when anything changes, reinstall it.
            // TODO: `.deinit()`, which removes it from channels. (Note that packets that are already sent may still call functions.)
            // TODO: For convenience, if [`FinalizationRegistry`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/FinalizationRegistry) is present, stop when the accumulator is no longer needed (else, set `onValues` to `null`).
        }, {}),
        Handler: A(class Handler {
            // TODO: `.constructor({ channel='', priority=0, noFeedback=true, onValues=null, dataSize=64, nameSize=64, namePartSize=16 })`
            //   TODO: Set `.cellShape` = [reward=1, user, name, data].
            //   TODO: Write out the shape as a string.
            //   TODO: `S[channel][shapeAsString].handlers.push(this)` and sort and `_Packet.loopHandle(shapeAsString)`.
            //   TODO: Install getters/setters on the options object, and when anything changes, reinstall it.
        }, {}),
        _Packet: class _Packet {
            constructor(channel, cellShape) {
                Object.assign(this, {
                    channel,
                    cellShape,
                    cellSize: cellShape.reduce((a,b) => a+b),
                    // sensor → accumulator:
                    cells: 0,
                    sensorData: [], // Owns f32a (Float32Array), given to `.data(…)`.
                    sensorError: [], // Owns f32a (Float32Array), given to `.data(…)`.
                    sensorIndices: [], // ints
                    sensorCallbacks: [], // function(data, error, feedback, fbOffset)
                    // accumulator → handler:
                    data: null, // Owned f32a.
                    error: null, // Owned f32a.
                    accumulatorExtra: [], // ints
                    accumulatorCallback: [], // function(feedback, cellShape, extra)
                    // handler → accumulator → sensor:
                    feedback: null, // null | owned f32a.
                })
            }
            data(point, error, callback) {
                // `point` is a named Float32Array of named cells, and this function takes ownership of it.
                // `error` is its error (max abs(true - measurement) - 1) or null.
                // `callback` is a function, from `point` (owned) & `allFeedback` (not owned) & `fbOffset` (int).
                //   The actual number-per-number feedback can be constructed as `allFeedback.subarray(fbOffset, fbOffset + data.length)`
                //     (but that's inefficient; instead, index as `allFeedback[fbOffset + i]`).
                assert(point.length instanceof Float32Array, "Data must be float32")
                assert(point.length % this.cellSize === 0, "Data must be divided into cells")
                assert(error == null || error instanceof Float32Array, "Error must be null or float32")
                assert(error == null || point.length === error.length, "Error must be per-data-point")
                this.sensorData.push(point)
                this.sensorError.push(error)
                this.sensorIndices.push(this.cells)
                this.sensorCallbacks.push(callback)
                this.cells += point.length / this.cellSize | 0
            }
            // TODO: Have a function for de/allocating float32 arrays of a given size. Reuse same-length arrays, and purge cache often.
            // TODO: Have a function for estimating the moving mean of a number stream online, for time-per-step.
            async handle(mainHandler) { // sensors → accumulators → handlers → accumulators → sensors
                if (!mainHandler) return
                const T = this
                // TODO: Prepare `this.data` by concatenating `sensorData`.
                // TODO: Prepare `this.error` by concatenating `sensorError`, -1-initializing where null. Unless everything is null.
                // TODO: For each `a` in `S[channel].accumulators`:
                //   TODO: this.accumulatorExtra.push(await a.onValues(this.data, this.cellShape))
                //   TODO: this.accumulatorCallback.push(a.onFeedback)
                // TODO: Prepare `this.feedback`, `this.data`-initializing it.
                // TODO: Call the main handler to fill `this.feedback`, given this.data and this.error and this.cellShape and this.feedback and feedbackIsWritable=true.
                // TODO: Call all the non-main handlers in `S[this.channel][this.cellShape].handlers`.
                // TODO: Reverse accumulators: while not empty:
                //   TODO: this.accumulatorCallback.pop().call(undefined, this.feedback, this.cellShape, this.accumulatorExtra.pop())
                // TODO: For each remembered sensor:
                //   TODO: this.sensorCallbacks.pop().call(undefined, this.sensorData.pop(), this.sensorError.pop(), this.feedback, this.sensorIndices.pop() * this.cellSize)
                // TODO: Update time-per-step `S[channel][cellShape].msPerStep`.
                // TODO: Notify end-of-step waiters `S[channel].waitingOnSteps.forEach(f => f())`.
            }
            // TODO: _Packet.loopHandle(handlerShapeAsString, maxSimultaneousPackets = 4), which estimates the max speed it could call itself at, and repeatedly does sensing → accumulating → handling.
            //   TODO: If `S[channel][cellShape].looping`, return, else set it to true.
            //   TODO: Infinitely:
            //     TODO: If there are no handlers or sensors, return (and set `looping` to `false`).
            //     TODO: If we are under our time budget, wait until time-per-step ms have expired.
            //     TODO: If we have too many packets, wait until we have enough, being notified via `S[channel].waitingOnSteps`.
            //     TODO: If there are no handlers or sensors, return (and set `looping` to `false`).
            //     TODO: If the main handler shape, call `onValues` of all sensors to get data (for all loops). If not the main, don't have feedback at all.
            //     TODO: Ensure that the nextPacket is replaced with a new one and is only accessible to us.
            //     TODO: nextPacket.handle()
            //   TODO: Be called when there are new handlers or sensors.
        },
        // (TODO: Also have `self` with `tests` and `bench` and `docs`, and `save` and `load` (when a prop is in `self`, it is not `save`d unless instructed to, to save space while saving code).)
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
    // TODO: All these must be in `E`, so that save/load can be aware of them.
    function test(func, ...args) {
        try { return func(...args) }
        catch (err) { return err instanceof Error ? [err.message, err.stack] : err }
    }
    function assertCounts(msg, ...xs) { assert(xs.every(x => typeof x == 'number' && x >= 0 && x === x>>>0), msg || 'Must be a non-negative integer') }
    function assert(bool, msg) { if (!bool) error(msg || 'Assertion failed') }
    function error(...msg) { throw new Error(msg.join(' ')) }
})(Object.create(null))