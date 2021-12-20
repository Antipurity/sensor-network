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
    //   cellShapes: Array<String>, for enumeration.
    //   stepsNow: int
    //   waitingSinceTooManySteps: Array<function>, called when a step is finished.
    //   shaped:
    //     [handlerShapeAsString]:
    //       looping: bool
    //       msPerStep: [n, mean]
    //       cellShape: [reward=1, user, name, data]
    //       handlers: Array<Handler>, sorted by priority.
    //       nextPacket: _Packet
    // To maybe-first read from `S`, use `E._state(channel, cellShape)`.

    return A(E, {
        Sensor: A(class Sensor {
            constructor({ name, values, onValues=null, channel='', noFeedback=false, user='self', emptyValues=0, hasher=undefined }) {
                assert(typeof name == 'string' || Array.isArray(name), 'Must have a name')
                assertCounts('Must have the value-count', values)
                assert(onValues == null || typeof onValues == 'function')
                assert(typeof channel == 'string')
                assert(typeof user == 'string' || Array.isArray(user))
                assertCounts('', emptyValues)
                assert(hasher === undefined || typeof hasher == 'function')
                Object.assign(this, {
                    paused: true,
                    reward: 0,
                    name,
                    values,
                    onValues,
                    channel,
                    noFeedback: !!noFeedback,
                    user,
                    emptyValues,
                    hasher,
                    dataNamers: Object.create(null), // cellShape → _dataNamer({ user='self', name, values, emptyValues=0, nameSize=64, namePartSize=16, dataSize=64, hasher=E._dataNamer.hasher })
                    nameSize: 0,
                    namePartSize: 0,
                    dataSize: 0,
                    feedbackCallbacks: [], // A queue of promise-fulfilling callbacks. Very small, so an array is the fastest option.
                }).resume()
                // TODO: For convenience, if [`FinalizationRegistry`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/FinalizationRegistry) is present, `.pause()` when the sender is no longer needed.
            }
            send(values, error = null, reward = 0) { // Returns a promise of feedback (no reward) or null.
                // Name+send to all handler shapes.
                const ch = E._state(this.channel)
                for (let cellShape of ch.cellShapes) {
                    const dst = ch.shaped[cellShape]
                    if (!dst.handlers.length) continue
                    const namer = this._namer(cellShape)
                    const flatV = _Packet.allocF32(namer.namedSize)
                    const flatE = error ? _Packet.allocF32(namer.namedSize) : null
                    namer.name(values, flatV, 0, reward)
                    flatE && namer.name(error, flatE, 0, reward, -1.)
                    dst.nextPacket.data(this, flatV, flatE, this.noFeedback)
                }
            }
            _gotFeedback(data, error, feedback, fbOffset, cellShape) {
                // Fulfill the promise of `.send`.
                if (feedback && !this.noFeedback) {
                    const flatV = _Packet.allocF32(this.values)
                    this._namer(cellShape).unname(feedback, fbOffset, flatV)
                    this.feedbackCallbacks.shift()(flatV)
                    _Packet.deallocF32(flatV)
                } else
                    this.feedbackCallbacks.shift()(null)
                _Packet.deallocF32(data), _Packet.deallocF32(error)
            }
            _namer(cellShape) {
                const s = ''+cellShape
                if (!this.dataNames[s]) {
                    this.nameSize = this.cellShape[0] + this.cellShape[1] + this.cellShape[2]
                    this.namePartSize = this.cellShape[0] + this.cellShape[1]
                    this.dataSize = this.cellShape[3]
                    this.dataNames[s] = E._dataNamer(this)
                }
                return this.dataNames[s]
            }
            pause() {
                if (this.paused) return
                E._state(this.channel).sensors = E._state(this.channel).sensors.filter(v => v !== this)
                this.paused = true
            }
            resume() {
                if (!this.paused) return
                if (typeof this.onValues == 'function') {
                    E._state(this.channel).sensors.push(this)
                    for (let cellShape of E._state(this.channel).cellShapes)
                        _Packet.handleLoop(this.channel, cellShape)
                }
                this.paused = false
            }
        }, {}), // TODO: Docs.
        Accumulator: A(class Accumulator {
            constructor({ onValues=null, onFeedback=null, priority=0, channel='' }) {
                assert(typeof priority == 'number' && priority === priority)
                assert(onValues == null || typeof onValues == 'function')
                assert(onFeedback == null || typeof onFeedback == 'function')
                assert(onValues || onFeedback, "Why have an accumulator if it does nothing")
                Object.assign(this, {
                    paused: true,
                    onValues,
                    onFeedback,
                    priority,
                    channel,
                }).resume()
                // TODO: For convenience, if [`FinalizationRegistry`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/FinalizationRegistry) is present, `.pause()` when the accumulator is no longer needed (else, set `onValues` to `null`).
            }
            pause() {
                if (this.paused) return
                E._state(this.channel).accumulators = E._state(this.channel).accumulators.filter(v => v !== this)
                this.paused = true
            }
            resume() {
                if (!this.paused) return
                E._state(this.channel).accumulators.push(this)
                E._state(this.channel).accumulators.sort((a,b) => b.priority - a.priority)
                this.paused = false
            }
        }, {}), // TODO: Docs.
        Handler: A(class Handler {
            constructor({ onValues, noFeedback=false, dataSize=64, nameSize=64, namePartSize=16, priority=0, channel='' }) {
                assert(typeof onValues == 'function', "Handlers must have listeners")
                assertCounts('', dataSize, nameSize, namePartSize)
                assert(namePartSize < nameSize && nameSize % namePartSize === 0, 'Cell name must consist of an integer number of parts')
                assert(typeof priority == 'number')
                assert(typeof channel == 'number')
                Object.assign(this, {
                    paused: true,
                    cellShape: [1, namePartSize-1, nameSize-namePartSize, dataSize],
                    onValues,
                    noFeedback: !!noFeedback,
                    priority,
                    channel,
                }).resume()
            }
            pause() {
                if (this.paused) return
                const ch = E._state(this.channel), dst = E._state(this.channel, this.cellShape)
                dst.handlers = dst.handlers.filter(v => v !== this)
                if (ch.mainHandler === this) {
                    ch.mainHandler = null
                    for (let cellShape of ch.cellShapes)
                        for (let h of ch.shaped[cellShape].handlers)
                            if (!h.noFeedback && (ch.mainHandler == null || ch.mainHandler.priority < h.priority)) ch.mainHandler = h
                }
                this.paused = true
            }
            resume() {
                if (!this.paused) return
                const ch = E._state(this.channel), dst = E._state(this.channel, this.cellShape)
                dst.handlers.push(this)
                dst.handlers.sort((a,b) => b.priority - a.priority)
                if (!this.noFeedback && (ch.mainHandler == null || ch.mainHandler.priority < this.priority)) ch.mainHandler = this
                if (this.onValues) _Packet.handleLoop(this.channel, this.cellShape)
                this.paused = false
            }
        }, {}), // TODO: Docs.
        maxSimultaneousPackets: 4,
        _state(channel, cellShape) { // Returns `cellShape != null ? S[channel].shaped[cellShape] : S[channel]`, creating structures if not present.
            if (!S[channel])
                S[channel] = Object.assign(Object.create(null), {
                    sensors: [],
                    accumulators: [],
                    mainHandler: null,
                    cellShapes: [],
                    stepsNow: 0,
                    waitingSinceTooManySteps: [],
                    shaped: Object.create(null),
                })
            const ch = S[channel]
            if (cellShape == null) return ch
            if (!ch.shaped[cellShape])
                ch.shaped[cellShape] = {
                    looping: false,
                    msPerStep: [0,0],
                    cellShape: cellShape,
                    handlers: [],
                    nextPacket: new _Packet(channel, cellShape),
                }, ch.cellShapes.push(cellShape)
            return ch.shaped[cellShape]
        },
        _Packet: class _Packet {
            constructor(channel, cellShape) {
                Object.assign(this, {
                    channel,
                    cellShape,
                    cellSize: cellShape.reduce((a,b) => a+b),
                    // sensor → accumulator:
                    cells: 0,
                    sensorNeedsFeedback: false,
                    sensor: [], // Sensor
                    sensorData: [], // Owns f32a (Float32Array), given to `.data(…)`.
                    sensorError: [], // Owns f32a (Float32Array), given to `.data(…)`.
                    sensorIndices: [], // ints
                    // accumulator → handler:
                    data: null, // Owned f32a.
                    error: null, // Owned f32a.
                    accumulatorExtras: [], // ints
                    accumulatorCallbacks: [], // function(feedback, cellShape, extra)
                    // handler → accumulator → sensor:
                    feedback: null, // null | owned f32a.
                })
            }
            data(sensor, point, error, noFeedback) {
                // `sensor` is a `E.Sensor` with `._gotFeedback(…)`, from `point` (owned) & `error` (owned) & `allFeedback` (not owned) & `fbOffset` (int) & `cellShape`.
                //   The actual number-per-number feedback can be constructed as `allFeedback.subarray(fbOffset, fbOffset + data.length)`
                //     (but that's inefficient; instead, index as `allFeedback[fbOffset + i]`).
                // `point` is a named Float32Array of named cells, and this function takes ownership of it.
                // `error` is its error (max abs(true - measurement) - 1) or null.
                // `noFeedback`: bool. If true, `callback` is still called, possibly even with non-null `allFeedback`.
                assert(point.length instanceof Float32Array, "Data must be float32")
                assert(point.length % this.cellSize === 0, "Data must be divided into cells")
                assert(error == null || error instanceof Float32Array, "Error must be null or float32")
                assert(error == null || point.length === error.length, "Error must be per-data-point")
                if (!noFeedback) this.sensorNeedsFeedback = true
                this.sensor.push(sensor)
                this.sensorData.push(point)
                if (error)
                    (this.sensorError || (this.sensorError = []))[this.sensorError.length-1] = error
                this.sensorIndices.push(this.cells)
                this.cells += point.length / this.cellSize | 0
            }
            static allocF32(len) { return new Float32Array(len) }
            static deallocF32(a) {} // TODO: Reuse same-length arrays via a cache from len to an array of free arrays, unless we already have 16.
            static updateMean(a, value, maxHorizon = 1000) {
                const n1 = a[0], n2 = n1+1
                a[0] = Math.min(n2, maxHorizon)
                a[1] += (value - a[1]) / n2
                if (!isFinite(a[1])) a[0] = a[1] = 0
                return a
            }
            async handle(mainHandler) { // sensors → accumulators → handlers → accumulators → sensors
                const T = this, ch = S[T.channel], dst = ch[T.cellShape]
                const start = performance.now()
                ++ch.stepsNow
                try {
                    // Concat sensors into `.data` and `.error`.
                    T.data = _Packet.allocF32(T.cells * T.cellSize), T.error = !T.sensorError ? null : _Packet.allocF32(T.cells * T.cellSize)
                    for (let i = 0; i < T.sensorData.length; ++i) {
                        const at = T.sensorIndices[i] * T.cellSize
                        T.data.set(T.sensorData[i], at)
                        if (T.error) {
                            if (T.sensorError[i])
                                T.error.set(T.sensorError[i], at)
                            else
                                T.error.fill(-1, at, at + T.sensorData[i].length)
                        }
                    }
                    // Accumulators.
                    for (let a of ch.accumulators)
                        if (typeof a.onValues == 'function' || typeof a.onFeedback == 'function') {
                            T.accumulatorExtra.push(typeof a.onValues == 'function' ? await a.onValues(T.data, T.cellShape) : undefined)
                            T.accumulatorCallback.push(a.onFeedback)
                        }
                    // Handlers.
                    if (mainHandler && !mainHandler.noFeedback && T.sensorNeedsFeedback)
                        T.feedback = _Packet.allocF32(T.cells * T.cellSize), T.feedback.set(T.data)
                    else
                        T.feedback = null
                    //   What does "no feedback" when sending a data piece do? Should it turn off whole-feedback?
                    await mainHandler.onValues(T.data, T.error, T.cellShape, T.feedback ? true : false, T.feedback)
                    for (let h of dst.handlers)
                        if (typeof h.onValues == 'function')
                            await h.onValues(T.data, T.error, T.cellShape, false, T.feedback)
                    // Accumulators.
                    while (T.accumulatorCallbacks.length)
                        T.accumulatorCallbacks.pop().call(undefined, T.feedback, T.cellShape, T.accumulatorExtras.pop())
                    // Sensors.
                    while (T.sensor.length)
                        T.sensor.pop()._gotFeedback(T.sensorData.pop(), T.sensorError.pop(), T.feedback, T.sensorIndices.pop() * T.cellSize, T.cellShape)
                } finally {
                    // Self-reporting.
                    --ch.stepsNow
                    _Packet.updateMean(dst.msPerStep, performance.now() - start)
                    ch.waitingSinceTooManySteps.length && ch.waitingSinceTooManySteps.shift()()
                }
            }
            async static handleLoop(channel, cellShape) {
                const ch = S[channel], dst = ch.shaped[cellShape]
                if (dst.looping) return;  else dst.looping = true
                while (true) {
                    const start = performance.now(), end = start + dst.msPerStep[1]
                    // Don't do too much at once.
                    while (ch.stepsNow > E.maxSimultaneousPackets)
                        await new Promise(then => ch.waitingSinceTooManySteps.push(then))
                    // Pause if no destinations, or no sources & no data to send.
                    if (!dst.handlers.length || !ch.sensors.length && !dst.nextPacket.cells) return dst.looping = false
                    // Get sensor data.
                    const mainHandler = ch.mainHandler && ch.mainHandler.cellShape+'' === cellShape+'' ? ch.mainHandler : null
                    if (mainHandler)
                        for (let s of ch.sensors)
                            await s.onValues(s)
                    // Send it off.
                    const nextPacket = dst.nextPacket;  dst.nextPacket = new _Packet(channel, cellShape)
                    nextPacket.handle(mainHandler)
                    // Don't do it too often.
                    if (performance.now() < end)
                        await new Promise(then => setTimeout(then, end - performance.now()))
                }
                // TODO: Be called when there are new handlers or sensors or sensor data.
            }
        },
        // (TODO: Also have `self` with `tests` and `bench` and `docs`, and `save` and `load` (when a prop is in `self`, it is not `save`d unless instructed to, to save space while saving code).)
        _dataNamer: A(function _dataNamer({ user='self', name, values, emptyValues=0, nameSize=64, namePartSize=16, dataSize=64, hasher=E._dataNamer.hasher }) {
            assertCounts('', nameSize, namePartSize, dataSize)
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
                namedSize: cells * valuesPerCell,
                cellShape: [1, namePartSize-1, nameSize - namePartSize, dataSize], // [reward=1, user, name, data]
                name(src, dst, dstOffset, reward = 0, skipNonData = null) { // flat → named
                    assert(reward >= -1 && reward <= 1)
                    for (let i = 0; i < cells; ++i) { // Fill out the whole cell.
                        const start = dstOffset + i * (nameSize + dataSize)
                        if (skipNonData == null) {
                            // Reward.
                            dst[start] = reward
                            // User.
                            userHasher(dst, start + 1)
                            // Name.
                            nameHashers[i](dst, start + namePartSize)
                        } else dst.fill(skipNonData, start, start + nameSize)
                        // Data.
                        const srcStart = i * valuesPerCell, srcEnd = Math.min(srcStart + valuesPerCell, src.length)
                        for (let s = srcStart, d = start + nameSize; s < srcEnd; ++s, ++d) dst[d] = src[s]
                        E._dataNamer.fill(dst, start + nameSize, valuesPerCell, dataSize)
                    }
                    return dstOffset + cells * (nameSize + dataSize) // Return the next `dstOffset`.
                },
                unname(src, srcOffset, dst) { // named → flat; `named` is consumed.
                    for (let i = 0; i < cells; ++i) { // Extract data from the whole cell.
                        const start = srcOffset + i * (nameSize + dataSize)
                        E._dataNamer.unfill(src, start + nameSize, valuesPerCell, dataSize)
                        const dstStart = i * valuesPerCell, dstEnd = Math.min(dstStart + valuesPerCell, dst.length)
                        for (let s = start + nameSize, d = dstStart; d < dstEnd; ++s, ++d) dst[d] = src[s]
                    }
                    return srcOffset + cells * (nameSize + dataSize) // Return the next `srcOffset`.
                },
            }
        }, {
            docs:`Implementation detail.

Prepares to go between flat number arrays and named ones.

The result is \`{ name(src, dst, dstOffset, reward=0, skipNonData=null)→dstOffset, unname(src, srcOffset, dst)→srcOffset }\`. \`name\` goes from flat to named, \`unname\` reverses this.

Main parameters, in the one object that serves as arguments:
- \`name\`: describes this interface to handlers, such as with a string. See \`._dataNamer.hasher\`.
- \`values\`: how many flat numbers there will be.

Extra parameters:
- \`user = 'self'\`: describes the current user/source/machine to handlers, mainly for when the sensor network encompasses multiple devices across the Internet.
- \`emptyValues = 0\`: how many fake \`values\` to insert, so that values are fractally folded more; see \`._dataNamer.fill\`.
- \`nameSize = 64\`: each cell is \`nameSize + dataSize\`.
- \`namePartSize = 16\`: the \`name\` can have many parts, and this determines how many parts there are.
- \`dataSize = 64\`: data in a cell.
- \`hasher = ._dataNamer.hasher\`: defines how names are transformed into \`-1\`…\`1\` numbers.
`,
            tests() {
                const F32 = Float32Array
                const r1 = new F32(12)
                return [
                    [
                        new F32([0, 0.96, -0.5, -0.25, 0, 0.5, 0, 0.96, 0.25, 0.5, 0.5, 0]),
                        test(
                            opts => (E._dataNamer(opts).name(new Float32Array([-.5, -.25, .25, .5]), r1, 0), r1.map(round)),
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
                    const namer = E._dataNamer(opts)
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
                            if (typeof x == 'function') { // TODO: Functions should be called not once but each time (to be able to simulate moving body parts); here, just remember the index. And `E._dataNamer.fill` should be able to accept the indices array to fill numbers from.
                                x = x(...args)
                                if (typeof x != 'number' || (x !== x || x < -1 || x > 1))
                                    error("Name parts must be -1..1, got", x)
                            }
                            numbers[p * partSize + i] = x
                        }
                        E._dataNamer.fill(numbers, p * partSize, part.length, partSize)
                    }
                    E._dataNamer.fill(numbers, 0, end, numbers.length)
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

Even if your AI model can only accept and return ±1 bits, it can still use the Sensor Network by having lots of free space (\`emptyValues\` in \`._dataNamer\`).`,
            }),
            unfill: A(function unfill(dst, offset, haveNumbers, needNumbers) {
                if (haveNumbers >= needNumbers || !haveNumbers) return
                for (let i = offset + needNumbers - haveNumbers - 1; i >= offset; --i)
                    dst[i] = Math.sign(dst[i]) * Math.abs(dst[i + haveNumbers] - 1) * .5
                return dst
            }, { docs:`Reverses \`._dataNamer.fill\`, enhancing low-frequency numbers with best guesses from high-frequency numbers.

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
})(Object.create(null))