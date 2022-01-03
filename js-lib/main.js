import './yamd5.js' // self.YaMD5
import Sound from './src/handler-sound.js'
import Video from './src/sensor-video.js'
import Time from './src/sensor-time.js'
import Reward from './src/transform-reward.js'
import UI from './src/ui.js'
import Text from './src/sensor-text.js'
import Random from './src/handler-random.js'
import Scroll from './src/sensor-scroll.js'
import Audio from './src/sensor-audio.js'

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
    const S = Object.create(null) // See `state(channel, cellShape, partSize)`.



    // Implementation details. That we put at the top. Because JS does not hoist them.
    let currentBenchmark = null
    const f32aCache = Object.create(null)
    const arrayCache = []
    class _Packet {
        constructor(channel, cellShape, partSize) {
            const noData = [], noFeedback = []
            const summary = shapeSummary(cellShape, partSize)
            Object.assign(this, {
                channel,
                summary,
                cellShape,
                cellSize: cellShape.reduce((a,b) => a+b),
                partSize,
                ch: state(channel),
                dst: state(channel, cellShape, partSize, summary),
                // `handleStateMachine` state:
                stage: 0,
                mainHandler: null,
                handleStart: 0,
                benchAtStart: null,
                transformI: 0, prevCells: 0, handlersLeft: 0,
                handleStateMachine: this.handleStateMachine.bind(this),
                // sensor → transform:
                cells: 0,
                sensorNeedsFeedback: false, // Any sensor.
                sensor: [], // Sensor
                sensorData: [], // Owns f32a (Float32Array), given to `.send(…)`.
                sensorError: [], // Owns f32a (Float32Array), given to `.send(…)`.
                sensorIndices: [], // ints
                noData, // Array<bool>
                noFeedback, // Array<bool>
                noDataIndices: [], // Array<index> where !!noData[index]
                // transform → handler:
                input: { data:null, error:null, noData, noFeedback, cellShape, partSize }, // data&error are owned f32a.
                transformCells: [], // ints
                transformExtra: [], // ints
                transformCallback: [], // function(feedback, cellShape, extra)
                // handler → transform → sensor:
                feedback: null, // null | owned f32a.
            })
        }
        static init(channel, cellShape, partSize, summary) {
            // Is `new _Packet`, but can reuse objects.
            const dst = state(channel, cellShape, partSize, summary)
            return dst.packetCache.length ? dst.packetCache.pop() : new _Packet(channel, cellShape, partSize, summary)
        }
        deinit() { // `this` must not be used after this call.
            // (Allows reuse of this object by `_Packet.init(…)`.)
            const T = this
            T.stage = 0
            T.mainHandler = null
            T.handleStart = 0
            T.benchAtStart = null
            T.transformI = T.prevCells = T.handlersLeft = 0
            T.cells = 0
            T.sensorNeedsFeedback = false
            T.sensor.length = T.sensorData.length = T.sensorError.length = T.sensorIndices.length = T.noData.length = T.noFeedback.length = T.noDataIndices.length = 0
            T.input.data && (deallocF32(T.input.data), T.input.data = null)
            T.input.error && (deallocF32(T.input.error), T.input.error = null)
            T.transformCells.length = T.transformExtra.length = T.transformCallback.length = 0
            T.feedback && (deallocF32(T.feedback), T.feedback = null)
            const dst = state(T.channel, T.cellShape, T.partSize, T.summary)
            if (dst.packetCache.length < 64) dst.packetCache.push(T)
        }
        send(sensor, point, error, noData, noFeedback) {
            // `sensor` is a `E.Sensor`.
            //   The actual number-per-number feedback can be constructed as `allFeedback.subarray(fbOffset, fbOffset + data.length)`
            //     (but that's inefficient; instead, index as `allFeedback[fbOffset + i]`).
            // `point` is a named Float32Array of named cells, and this function takes ownership of it.
            // `error` is its error (max abs(true - measurement) - 1) or null.
            // `noFeedback`: bool. If true, `callback` is still called, possibly even with non-null `allFeedback`.
            assert(point instanceof Float32Array, "Data must be float32")
            assert(point.length % this.cellSize === 0, "Data must be divided into cells")
            assert(error == null || error instanceof Float32Array, "Error must be null or float32")
            assert(error == null || point.length === error.length, "Error must be per-data-point")
            const cells = point.length / this.cellSize | 0
            if (!noFeedback) this.sensorNeedsFeedback = true
            this.sensor.push(sensor)
            this.sensorData.push(point)
            if (error) this.sensorError[this.sensorError.length-1] = error
            this.sensorIndices.push(this.cells)
            noData = !!noData, noFeedback = !!noFeedback
            for (let c = 0; c < cells; ++c)
                this.noData.push(noData), this.noFeedback.push(noFeedback)
            if (noData)
                for (let c = 0; c < cells; ++c)
                    this.noDataIndices.push(this.cells + c)
            this.cells += point.length / this.cellSize | 0
        }
        // TODO: Have median-based `setValue(spot, v, maxHorizon=11)` and `getValue(spot)` (n/2-th largest, sorting a copy each time).
        //   TODO: Use these wherever `_updateMean` and `.msPerStep[1]` are used.
        static updateMean(a, value, maxHorizon = 32) {
            const n1 = a[0], n2 = n1+1
            a[0] = Math.min(n2, maxHorizon)
            a[1] += (value - a[1]) / n2
            if (!isFinite(a[1])) a[0] = a[1] = 0
            return a
        }
        handleStateMachine(A,B,C) {
            // sensors → transforms → handlers → transforms → sensors
            // When first called by `handleLoop`, takes ownership of `this`.
            const T = this, ch = T.ch, dst = T.dst
            if (!ch.shaped[T.summary]) return
            while (true)
                switch (T.stage) {
                    case 0: { // Init variables and concat data & error.
                        T.handleStart = performance.now()
                        T.benchAtStart = currentBenchmark
                        T.input.data = allocF32(T.cells * T.cellSize)
                        T.input.error = !T.sensorError ? null : allocF32(T.cells * T.cellSize)
                        for (let i = 0; i < T.sensorData.length; ++i) {
                            const at = T.sensorIndices[i] * T.cellSize
                            T.input.data.set(T.sensorData[i], at)
                            if (T.input.error) {
                                if (T.sensorError[i])
                                    T.input.error.set(T.sensorError[i], at)
                                else
                                    T.input.error.fill(-1, at, at + T.sensorData[i].length)
                            }
                        }
                        ++dst.stepsNow
                    } T.stage = 1;  case 1: { // Go over transforms in order.
                        if (T.transformI < ch.transforms.length) {
                            const a = ch.transforms[T.transformI]
                            if (typeof a.onValues == 'function' || typeof a.onFeedback == 'function')
                                try {
                                    T.prevCells = T.cells
                                    T.transformCallback.push(a.onFeedback)
                                    T.stage = 9
                                    ++T.transformI
                                    if (typeof a.onValues == 'function')
                                        return a.onValues(T.handleStateMachine, T.input)
                                    else
                                        return T.handleStateMachine()
                                } catch (err) { console.error(err) }
                        }
                    } T.stage = 2;  case 2: { // Set up feedback, and call the main handler.
                        if (T.mainHandler && !T.mainHandler.noFeedback && T.sensorNeedsFeedback)
                            T.feedback = allocF32(T.cells * T.cellSize), T.feedback.set(T.input.data)
                        else
                            T.feedback = null
                        if (T.mainHandler)
                            try {
                                T.stage = 3
                                return T.mainHandler.onValues(T.handleStateMachine, T.input, T.feedback)
                            } catch (err) { console.error(err) }
                    } T.stage = 3;  case 3: { // Replace no-data cells with feedback (AKA record actions).
                        // Undocumented: overscheduling-prevention, for `Sound` latency.
                        if (A) dst.overscheduled = 1
                        else dst.overscheduled *= .9
                        if (T.feedback)
                            for (let i = 0; i < T.noDataIndices.length; ++i) {
                                const start = T.cellSize * T.noDataIndices[i], end = start + T.cellSize
                                for (let j = start; j < end; ++j)
                                    T.input.data[j] = T.feedback[j]
                            }
                    } T.stage = 4;  case 4: { // Call all non-main handlers at once.
                        const hs = dst.handlers
                        T.handlersLeft = 0
                        for (let i = 0; i < hs.length; ++i)
                            if (hs[i] !== T.mainHandler && typeof hs[i].onValues == 'function')
                                ++T.handlersLeft
                        if (T.handlersLeft) {
                            T.stage = 5
                            for (let i = 0; i < hs.length; ++i)
                                if (hs[i] !== T.mainHandler && typeof hs[i].onValues == 'function')
                                    try { hs[i].onValues(T.handleStateMachine, T.input) }
                                    catch (err) { console.error(err) }
                            return
                        } else { T.stage = 6;  continue }
                    } T.stage = 5;  case 5: { // Wait for all non-main handlers.
                        // Undocumented: overscheduling-prevention, for `Sound` latency.
                        if (A) dst.overscheduled = 1
                        else dst.overscheduled *= .9
                        if (--T.handlersLeft) return
                    } T.stage = 6;  case 6: { // Go over transform-feedback in reverse-order.
                        while (T.transformCallback.length) {
                            T.prevCells = T.transformCells.pop()
                            const extra = T.transformExtra.pop()
                            const f = T.transformCallback.pop()
                            if (typeof f == 'function')
                                try {
                                    T.stage = 10
                                    return f(T.handleStateMachine, T.input, T.feedback, extra)
                                } catch (err) { console.error(err) }
                        }
                    } T.stage = 7;  case 7: { // Give feedback to all sensors at once.
                        while (T.sensor.length)
                            try {
                                gotPacketFeedback(T.sensor.pop(), T.sensorData.pop(), T.sensorError.pop(), T.feedback, T.sensorIndices.pop() * T.cellSize, T.cellShape, T.partSize, T.summary)
                            } catch (err) { console.error(err) }
                    } T.stage = 8;  case 8: { // Finalize & self-report.
                        --dst.stepsNow
                        _Packet._handledBytes = (_Packet._handledBytes || 0) + T.cells * T.cellSize * 4
                        _Packet.stepsEnded = (_Packet.stepsEnded || 0) + 1
                        if (T.benchAtStart === currentBenchmark) {
                            let duration = (dst.lastUsed = performance.now()) - T.handleStart
                            if (dst.msPerStep[0]) // Prevent "overeager scheduling of steps makes them interfere with each other, causing overestimation of ms-per-step".
                                duration = Math.min(duration, 1.1 * dst.msPerStep[1] + 11)
                            _Packet.updateMean(dst.msPerStep, duration)
                            E.meta.metric('simultaneous steps', dst.stepsNow+1)
                            E.meta.metric('step processed data, values', T.cells * T.cellSize)
                        }
                        const hadCells = !!T.cells
                        T.deinit()
                        const tooFew = dst.giveNextPacketNow, tooMuch = dst.waitingSinceTooManySteps
                        if (hadCells && dst.stepsNow <= 1 && tooFew) tooFew()
                        return tooMuch && tooMuch()
                    } case 9: { // Assign the next data/error if given by a transform.
                        const extra = A, nextData = B, nextError = C
                        if (nextData && nextData !== T.input.data) {
                            assert(nextData instanceof Float32Array)
                            assert(nextData.length % T.cellSize === 0, "Bad data size")
                            if (T.input.error) {
                                assert(nextError instanceof Float32Array)
                                assert(nextData.length === nextError.length, "Data and error lengths differ")
                            } else assert(!nextError)
                            const nextCells = nextData.length / T.cellSize | 0
                            assert(T.input.noData.length === nextCells, "Must resize `noData` too")
                            assert(T.input.noFeedback.length === nextCells, "Must resize `noFeedback` too")
                            T.cells = nextCells
                            deallocF32(T.input.data), T.input.data = nextData
                            if (T.input.error)
                                deallocF32(T.input.error), T.input.error = nextError
                        } else if (!nextData) assert(!nextError)
                        T.transformCells.push(T.prevCells)
                        T.transformExtra.push(extra)
                        T.stage = 1;  continue
                    } case 10: { // Assign the prev feedback if given.
                        const nextFeedback = A
                        if (!T.feedback) assert(!nextFeedback)
                        if (nextFeedback && nextFeedback !== T.feedback) {
                            assert(nextFeedback instanceof Float32Array)
                            assert(nextFeedback.length === T.prevCells * T.cellSize, "Feedback's length differs from data's")
                            assert(T.input.noData.length === T.prevCells, "Must resize `noData` back too")
                            assert(T.input.noFeedback.length === T.prevCells, "Must resize `noFeedback` back too")
                            T.cells = T.prevCells
                            deallocF32(T.feedback), T.feedback = nextFeedback
                        }
                        T.stage = 6;  continue
                    }
                }
        }
        static async handleLoop(channel, cellShape, partSize, summary) {
            const ch = S[channel], dst = ch.shaped[summary]
            if (!dst || dst.looping) return;  else dst.looping = true
            dst.prevEnd = performance.now()
            let prevWait = dst.prevEnd
            // TODO: Keep track of non-`handle` time. Print it. See how horribly wrong we are. (Even sensor-collection time.)
            while (true) {
                if (!ch.shaped[summary]) return // `Sensor`s might have cleaned us up.
                const timeA = performance.now()
                const lessPackets = dst.overscheduled > .5
                // Get sensor data.
                const mainHandler = ch.mainHandler && ch.mainHandler.summary === summary ? ch.mainHandler : null
                if (!dst.stepsNow || !lessPackets) {
                    const sh = ch.cellShapes
                    if (sh.length > 1 && (!ch.shaped[sh[0].summary] || !ch.shaped[sh[0].summary].handlers.length)) {
                        // This main-ish shape is clearly unused. So don't use it.
                        const i = 0, j = 1 + (Math.random() * (sh.length-1) | 0)
                        ;[sh[i], sh[j]] = [sh[j], sh[i]]
                    }
                    const mainSensor = ch.mainHandler ? !!mainHandler : sh.length && sh[0].summary === summary
                    if (mainSensor && Array.isArray(ch.sensors))
                        for (let i = 0; i < ch.sensors.length; ++i) {
                            const s = ch.sensors[i]
                            const data = allocF32(s.values)
                            try { s.onValues(s, data) }
                            catch (err) { console.error(err) }
                        }
                }
                const timeB = performance.now()
                // console.log('Time-to-sense:', timeB-timeA, 'ms') // TODO:
                // Pause if no destinations, or no sources & no data to send.
                if (!dst.handlers.length || !ch.sensors.length && !dst.nextPacket.sensor.length)
                    return dst.msPerStep[0] = dst.msPerStep[1] = 0, dst.looping = false
                let noData = false
                if (!dst.stepsNow || !lessPackets) {
                    // Send it off.
                    const nextPacket = dst.nextPacket
                    dst.nextPacket = _Packet.init(channel, cellShape, partSize, summary)
                    noData = !nextPacket.cells
                    nextPacket.mainHandler = mainHandler, nextPacket.handleStateMachine()
                    // Benchmark throughput if needed.
                    _Packet._measureThroughput()
                }
                // Don't do too much at once.
                while (dst.stepsNow > E.maxSimultaneousPackets)
                    await new Promise(then => dst.waitingSinceTooManySteps = then)
                // Don't do it too often.
                const now = performance.now(), needToWait = dst.prevEnd - now
                dst.prevEnd += dst.msPerStep[1] + (!dst.stepsNow || !lessPackets ? 0 : 20)
                if (dst.prevEnd < now - 1000)
                    dst.prevEnd = now - 1000 // Don't get too eager after being stalled.
                if (needToWait > 5 || now - prevWait > 100 || noData) {
                    const delay = noData ? 500 : Math.max(needToWait, 0) // TODO: Measure & subtract overshooting. (As the median, of course.)
                    const expectedEnd = performance.now() + delay
                    await new Promise(then => {
                        if (now - prevWait <= 100) dst.giveNextPacketNow = then
                        setTimeout(() => {
                            // console.log('overshoot', performance.now() - expectedEnd) // TODO: Yep, relatively-consistent about-8-ms overshoot.
                            prevWait = performance.now(), then()
                        }, delay)
                    })
                    dst.giveNextPacketNow = null
                }
            }
        }
        static _measureThroughput(reset = false) {
            const now = performance.now()
            if (!_Packet.lastMeasuredThroughputAt)
                _Packet.lastMeasuredThroughputAt = now, _Packet.lastMemory = memory()
            if (reset || now - _Packet.lastMeasuredThroughputAt > 500) { // Filter out noise.
                const s = Math.max(now - _Packet.lastMeasuredThroughputAt, .01) / 1000
                if (!reset)
                    E.meta.metric('throughput, bytes/s', (_Packet._handledBytes || 0) / s),
                    E.meta.metric('allocations, bytes/s', Math.max(memory() - _Packet.lastMemory, 0) / s)
                _Packet.lastMeasuredThroughputAt = now
                _Packet._handledBytes = 0, _Packet.stepsEnded = 0
                _Packet.lastMemory = memory()
            }
        }
    }



    A(E, {
        Sensor: A(class Sensor {
            constructor(opts) { opts && this.resume(opts) }
            needsExtensionAPI() { return null }
            cellShape() {
                const ch = state(this.channel)
                if (ch.mainHandler) return ch.mainHandler.cellShape
                return ch.cellShapes[0] && ch.cellShapes[0].cellShape || null
            }
            sendCallback(then, values = null, error = null, reward = 0) { // In profiling, promises are the leading cause of garbage.
                // Name+send to all handler shapes.
                // Also forget about shapes that are more than 60 seconds old, to not slowly choke over time.
                assert(then === null || typeof then == 'function')
                assert(values === null || values instanceof Float32Array)
                assert(error === null || error instanceof Float32Array)
                values && assert(values.length === this.values, "Data size differs from the one in options")
                error && assert(values && values.length === error.length)
                const ch = state(this.channel)
                const removed = Sensor._removed || (Sensor._removed = new Set) // Probably fine to reuse the same object for this.
                for (let i = 0; i < ch.cellShapes.length; ++i) {
                    const {cellShape, partSize, summary} = ch.cellShapes[i]
                    const dst = ch.shaped[summary]
                    if (!dst.handlers.length) {
                        if (performance.now() - dst.lastUsed > 60000) removed.add(ch.cellShapes[i])
                        continue
                    }
                    const namer = packetNamer(this, this.dataNamers, cellShape, partSize, summary)
                    const flatV = allocF32(namer.namedSize)
                    const flatE = error ? allocF32(namer.namedSize) : null
                    if (!values) flatV.fill(0)
                    flatV && namer.name(values, flatV, 0, reward)
                    flatE && namer.name(error, flatE, 0, 0, -1.)
                    dst.nextPacket.send(this, flatV, flatE, values === null, this.noFeedback)

                    // Wake up.
                    values && dst.stepsNow <= 1 && dst.giveNextPacketNow && dst.giveNextPacketNow() // Wake up.
                }
                if (removed.size) {
                    ch.cellShapes = ch.cellShapes.filter(o => !removed.has(o))
                    removed.forEach(o => delete ch.shaped[o.summary])
                    removed.clear()
                }
                values && deallocF32(values), error && deallocF32(error)
                this.feedbackCallbacks.push(then) // Called even if no feedback is registered, with `null`.
                this.feedbackNoFeedback.push(this.noFeedback)
                this.feedbackNamers.push(this.dataNamers)
            }
            send(values, error = null, reward = 0) { // Returns a promise of feedback (no reward) or null.
                return new Promise((then, reject) => {
                    try { this.sendCallback(then, values, error, reward) }
                    catch (err) { reject(err) }
                })
            }
            pause() {
                if (this.paused !== false) return this
                if (typeof this.onValues == 'function') {
                    const ch = state(this.channel)
                    ch.sensors = ch.sensors.filter(v => v !== this)
                }
                this.paused = true
                return this
            }
            resume(opts) {
                if (opts) {
                    this.pause()
                    const { name, values, onValues=null, channel='', noFeedback=false, userName=[], emptyValues=0, hasher=undefined } = opts
                    assert(typeof name == 'string' || Array.isArray(name), 'Must have a name')
                    assertCounts('Must have the value-count', values)
                    assert(onValues == null || typeof onValues == 'function')
                    assert(typeof channel == 'string')
                    assert(typeof userName == 'string' || Array.isArray(userName))
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
                        userName,
                        emptyValues,
                        hasher,
                        dataNamers: Object.create(null), // cellShape → _dataNamer({ userName=[], userParts=1, nameParts=3, partSize=8, name, values, emptyValues=0, dataSize=64, hasher=E._dataNamer.hasher })
                        partSize: 0,
                        userParts: 0,
                        nameParts: 0,
                        dataSize: 0,
                    })
                    if (!this.feedbackCallbacks) {
                        this.feedbackCallbacks = []
                        this.feedbackNoFeedback = []
                        this.feedbackNamers = []
                    }
                }
                if (!this.paused) return this
                if (typeof this.onValues == 'function') {
                    const ch = state(this.channel)
                    ch.sensors.push(this)
                    for (let {cellShape, partSize, summary} of ch.cellShapes)
                        _Packet.handleLoop(this.channel, cellShape, partSize, summary)
                }
                this.paused = false
                return this
            }
        }, {
            docs:`Generalization of eyes and ears and hands, hotswappable and adjustable.

- \`constructor({ name, values, onValues=null, channel='', noFeedback=false, userName=[], emptyValues=0, hasher=… })\`
    - \`name\`: a human-readable string, or an array of that or a -1…1 number or a function from \`dataStart, dataEnd, dataLen\` to a -1…1 number.
    - \`values\`: how many -1…1 numbers this sensor exposes.
        - Usually a good idea to keep this to powers-of-2, and squares. Such as 64.
    - \`onValues(sensor, data)\`: the regularly-executed function that reports data, by calling \`sensor.send(data, …)\` inside once. Not \`await\`ed.
        - To run faster, use \`sensor.sendCallback(fn(feedback, sensor), data, …)\` with a static function.
    - Extra flexibility:
        - \`channel\`: the human-readable name of the channel. Communication only happens within the same channel.
        - \`noFeedback\`: set to \`true\` if applicable to avoid some processing. Otherwise, feedback is the data that should have been.
        - \`userName\`: the name of the machine that sources data. Makes it possible to reliably distinguish sources.
        - \`emptyValues\`: the guaranteed extra padding, for fractal folding. See \`._dataNamer.fill\`.
        - \`hasher(…)(…)(…)\`: see \`._dataNamer.hasher\`. The default mainly hashes strings in \`userName\`/\`name\` with MD5 and rescales bytes into -1…1.
    - To change any of this, \`resume({…})\`.

- \`cellShape() → [user, name, data] | null\`: returns the target's cell shape. Note that this may change rarely.

- \`send(values = null, error = null, reward = 0) → Promise<null|feedback>\`
    - (Do not override in child classes, only call.)
    - \`values\`: \`null\` or owned flat data, -1…1 \`Float32Array\`. Do not perform ANY operations on it once called.
        - (\`null\` means "feedback only please", meaning, actions.)
        - To mitigate misalignment, try to stick to powers of 2 in all sizes.
        - (Can use \`sn._allocF32(len)\` for efficient reuse.)
    - \`error\`: \`null\` or owned flat data, -1…1 \`Float32Array\` of length \`values.length\`: \`max abs(truth - observation) - 1\`. Do not perform ANY operations on it once called.
    - \`reward\`: every sensor can tell handlers what to maximize, -1…1. (What is closest in meaning for you? Localized pain and pleasure? Satisfying everyone's needs rather than the handler's? …Money? Either way, it sounds like a proper body.)
        - Can be a number or a function from \`valueStart, valueEnd, valuesTotal\` to that.
    - (Result: \`feedback\` is owned by you. Can use \`feedback && sn._deallocF32(feedback)\` once you are done with it, or simply ignore it and let GC collect it.)

- \`sendCallback(then(null|feedback, sensor), values, error = null, reward = 0)\`: exactly like \`send\` but does not have to allocate a promise.

- \`pause()\`, \`resume(opts?)\`: for convenience, these return the object.

- \`needsExtensionAPI() → null|String\`: overridable in child classes. By default, the sensor is entirely in-page in a [content script](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Content_scripts) if injected by an extension. For example, make this return \`'tabs'\` to get access to [\`chrome.tabs\`](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/tabs) in an extension.`,
        }),
        Transform: A(class Transform {
            constructor(opts) { opts && this.resume(opts) }
            pause() {
                if (this.paused !== false) return this
                const ch = state(this.channel)
                ch.transforms = ch.transforms.filter(v => v !== this)
                this.paused = true
                return this
            }
            resume(opts) {
                if (opts) {
                    this.pause()
                    const { onValues=null, onFeedback=null, priority=0, channel='' } = opts
                    assert(typeof priority == 'number' && priority === priority, "Bad `priority`")
                    assert(onValues == null || typeof onValues == 'function', "Bad `onValues`")
                    assert(onFeedback == null || typeof onFeedback == 'function', "Bad `onFeedback`")
                    assert(onValues || onFeedback, "Why have a transform if it does nothing? Pass in either `onValues` or `onFeedback`")
                    assert(typeof channel == 'string', "Bad `channel`")
                    Object.assign(this, {
                        paused: true,
                        onValues,
                        onFeedback,
                        priority,
                        channel,
                    })
                }
                if (!this.paused) return this
                const ch = state(this.channel)
                ch.transforms.push(this)
                ch.transforms.sort((a,b) => b.priority - a.priority)
                this.paused = false
                return this
            }
        }, {
            docs:`Modifies data/feedback, after sensors and before handlers.

- \`constructor({ onValues=null, onFeedback=null, priority=0, channel='' })\`
    - Needs one or both:
        - \`onValues(then, {data, error, noData, noFeedback, cellShape, partSize})\`: can modify \`data\` and the optional \`error\` in-place.
            - ALWAYS do \`then(extra, …)\`, at the end, even on errors. \`extra\` will be seen by \`onFeedback\` if specified.
                - To resize \`data\` and possibly \`error\`, pass the next version (\`._allocF32(len)\`) to \`then(extra, data2)\` or \`then(extra, data2, error2)\`; also resize \`noData\` and \`noFeedback\`; do not deallocate arguments.
            - \`cellShape: [user, name, data]\`
            - Data is split into cells, each made up of \`cellShape.reduce((a,b)=>a+b)\` -1…1 numbers.
            - \`noData\` and \`noFeedback\` are JS arrays, from cell index to boolean.
        - \`onFeedback(then, {data, error, noData, noFeedback, cellShape, partSize}, feedback, extra)\`: can modify \`feedback\` in-place.
            - ALWAYS do \`then()\`, at the end, even on errors.
                - If \`data\` was resized and \`feedback\` was given, must resize it back, by passing the next version (\`._allocF32(len)\`) to \`then(feedback)\`; do not deallocate arguments.
    - Extra flexibility:
        - \`priority\`: transforms run in order, highest priority first.
        - \`channel\`: the human-readable name of the channel. Communication only happens within the same channel.
    - To change any of this, \`resume({…})\`.

- \`pause()\`, \`resume(opts?)\`: for convenience, these return the object.`,
        }),
        Handler: A(class Handler {
            constructor(opts) { opts && this.resume(opts) }
            pause() {
                if (this.paused !== false) return this
                const ch = state(this.channel), dst = state(this.channel, this.cellShape, this.partSize, this.summary)
                dst.handlers = dst.handlers.filter(v => v !== this)
                if (ch.mainHandler === this) {
                    ch.mainHandler = null
                    for (let {cellShape, partSize, summary} of ch.cellShapes)
                        for (let h of ch.shaped[summary].handlers)
                            if (!h.noFeedback && (ch.mainHandler == null || ch.mainHandler.priority < h.priority))
                                ch.mainHandler = h
                }
                this.paused = true
                return this
            }
            resume(opts) {
                if (opts) {
                    this.pause()
                    const { onValues, partSize=8, userParts=1, nameParts=3, dataSize=64, noFeedback=false, priority=0, channel='' } = opts
                    assert(typeof onValues == 'function', "Handlers must have `onValues` listeners")
                    assertCounts('', partSize, userParts, nameParts, dataSize)
                    assert(typeof priority == 'number')
                    assert(typeof channel == 'string')
                    const cellShape = [userParts * partSize, nameParts * partSize, dataSize]
                    Object.assign(this, {
                        paused: true,
                        partSize,
                        cellShape,
                        summary: shapeSummary(cellShape, partSize),
                        onValues,
                        noFeedback: !!noFeedback,
                        priority,
                        channel,
                    })
                }
                if (!this.paused) return this
                const ch = state(this.channel), dst = state(this.channel, this.cellShape, this.partSize, this.summary)
                dst.handlers.push(this)
                dst.handlers.sort((a,b) => b.priority - a.priority)
                if (!this.noFeedback && (ch.mainHandler == null || ch.mainHandler.priority < this.priority)) ch.mainHandler = this
                if (this.onValues) _Packet.handleLoop(this.channel, this.cellShape, this.partSize, this.summary)
                this.paused = false
                return this
            }
        }, {
            docs:`Given data, gives feedback: is a human or AI model.

- \`constructor({ onValues, partSize=8, userParts=1, nameParts=3, dataSize=64, noFeedback=false, priority=0, channel='' })\`
    - \`onValues(then, {data, error, noData, noFeedback, cellShape, partSize}, feedback)\`: process.
        - ALWAYS do \`then()\` when done, even on errors.
        - \`feedback\` is available in the one main handler, which should write to it in-place.
            - In other handlers, data of \`noData\` cells will be replaced by feedback.
        - \`noData\` and \`noFeedback\` are JS arrays, from cell index to boolean.
        - \`data\` and \`error\` are not owned; do not write. \`error\` and \`feedback\` can be \`null\`s.
    - Cell sizes:
        - \`partSize\`: how many numbers each part in the cell ID takes up, where each string in a name takes up a whole part:
            - \`userParts\`
            - \`nameParts\`
        - \`dataSize\`: numbers in the data segment.
    - Extra flexibility:
        - \`noFeedback\`: can't provide feedback if \`true\`, only observe it.
        - \`priority\`: the highest-priority handler without \`noFeedback\` will be the *main* handler, and give feedback.
        - \`channel\`: the human-readable name of the channel. Communication only happens within the same channel.
    - To change any of this, \`resume({…})\`.

- \`pause()\`, \`resume(opts?)\`: for convenience, these return the object.`,
            options() {
                return {
                    partSize: {
                        ['8 ']: () => 8,
                        ['4 ']: () => 4,
                        ['16 ']: () => 16,
                        ['32 ']: () => 32,
                    },
                    userParts: {
                        ['×1']: () => 1,
                        ['×0']: () => 0,
                    },
                    nameParts: {
                        ['×3']: () => 3,
                        ['×2']: () => 2,
                        ['×1']: () => 1,
                        ['×0']: () => 0,
                        ['×5']: () => 5,
                    },
                    dataSize: {
                        ['+64']: () => 64,
                        ['+16']: () => 16,
                        ['+256']: () => 256,
                    },
                }
            },
            bench() {
                const cellCounts = new Array(10).fill().map((_,i) => (i+1)*10)
                return cellCounts.map(river) // See how throughput changes with input size.
                function river(cells) { // 1-filled data → -1-filled feedback
                    const dataSize = 64
                    function onSensorFeedback(feedback) {
                        if (feedback)
                            feedback.fill(.5439828952837), // "Read" it.
                            E._deallocF32(feedback) // Reuse it.
                    }
                    return function start() {
                        const from = new E.Sensor({
                            name: ['some', 'kinda', 'name'],
                            values: cells*dataSize,
                            onValues(sensor, data) {
                                data.fill(1)
                                sensor.sendCallback(onSensorFeedback, data)
                            },
                        })
                        const to = new E.Handler({
                            dataSize,
                            onValues(then, {data, error, cellShape}, feedback) {
                                try {
                                    data.fill(.489018922485) // "Read" it.
                                    if (feedback) feedback.fill(-1)
                                } finally { then() }
                            },
                        })
                        return function stop() { from.pause(), to.pause() }
                    }
                }
            },
        }),
        _allocF32: allocF32,
        _deallocF32: deallocF32,
        _assert: assert,
        _assertCounts: assertCounts,
        maxSimultaneousPackets: 4,
        meta:{
            docs: A(function docs() {
                const markdown = []
                const hierarchy = walk(E, 0)
                markdown.unshift(`<a id=toc></a>` + '\n# Table of contents', 'Sensor network:', ...TOC(hierarchy))
                return markdown.filter(x => x).join('\n\n')
                function walk(x, depth, path = 'sn') {
                    if (!x || typeof x != 'object' && typeof x != 'function') return
                    let haveOwnDocs = false
                    const backpatchHeading = markdown.push(`<a id="${toLinkHash(path)}"></a>` + "\n" + '#'.repeat(depth+1) + " `" + path + "`[ ↑](#toc)")-1
                    if (typeof x.docs == 'string' || typeof x.docs == 'function' && x.docs !== docs) {
                        const md = typeof x.docs == 'function' ? x.docs() : x.docs
                        const sign = funcSignature(x)
                        if (sign) markdown.push("```js\n" + sign + "\n```")
                        markdown.push(md)
                        haveOwnDocs = true
                    }
                    let result = null
                    for (let k of Object.keys(x)) {
                        if (k[0] === '_') continue
                        const subPath = path + "." + k
                        const sub = walk(x[k], depth+1, subPath)
                        if (sub) (result || (result = Object.create(null)))[subPath] = sub
                    }
                    if (!result && !haveOwnDocs) markdown[backpatchHeading] = ''
                    return result || haveOwnDocs
                }
                function toLinkHash(s) { return s.toLowerCase().replace(/[^a-zA-Z0-9]/g, '-') }
                function funcSignature(f) { // A bit janky. Watch out for breakage.
                    if (typeof f != 'function' || String(f).slice(0,6) === 'class ') return
                    const i = String(f).indexOf(' {')
                    return i<0 ? undefined : String(f).slice(0, i)
                }
                function TOC(x, depth = 0, into = []) {
                    if (x && x !== true)
                        for (let k of Object.keys(x)) {
                            into.push('    '.repeat(depth) + `- [${k}](#${toLinkHash(k)})`)
                            TOC(x[k], depth+1, into)
                        }
                    return into
                }
            }, {
                docs:`Returns the Markdown string containing all the sensor network's documentation.

Objects need to define \`.docs\` to be either a string or a function to that.`,
            }),
            tests: A(async function tests() {
                const reports = []
                await walk(E)
                return reports.length ? reports : null
                async function walk(x) {
                    if (!x || typeof x != 'object' && typeof x != 'function') return
                    if (typeof x.tests == 'function' && x.tests !== tests) {
                        try {
                            for (let [name, a, b] of await x.tests())
                                if (''+a !== ''+b)
                                    reports.push([name, a, b])
                        } catch (err) { reports.push(err instanceof Error ? [x.name || '—', err.message, err.stack] : [x.name || '—', !x || typeof x != 'object' && typeof x != 'function' ? ''+x : '<Error>']) }
                    }
                    return Promise.all(Object.values(x).map(walk))
                }
            }, {
                docs:`Runs all sensor-network tests, and returns \`null\` if OK, else an array of \`[failedTestName, value1, value2]\`.

If not \`null\`, things are very wrong.

Internally, it calls \`.tests()\` which return \`[…, [testName, value1, value2], …]\`. String representations must match exactly to succeed.`,
            }),
            metric: A(function metric(key, value) {
                if (!currentBenchmark) return
                if (typeof value == 'string')
                    currentBenchmark[key] = value
                else if (typeof value == 'number') {
                    if (value !== value || !isFinite(value)) return
                    !Array.isArray(currentBenchmark[key]) && (currentBenchmark[key] = []), currentBenchmark[key].push(value)
                } else
                    error("what this: " + value)
            }, {
                docs:`Call this with a string key & string/number value to display/measure something, if \`E.meta.bench\` controls execution.`,
            }),
            bench: A(async function bench(secPerBenchmark = 30, benchFilter=null, onBenchFinished=null) {
                assert(typeof secPerBenchmark == 'number')
                const result = Object.create(null)
                if (typeof onBenchFinished != 'function') onBenchFinished = (obj, id, got, progress) => {
                    // got[key] → result[name][key][id]
                    const name = obj.name || '—'
                    const into = result[name] || (result[name] = Object.create(null))
                    for (let key of Object.keys(got)) {
                        let v = got[key]
                        if (Array.isArray(v)) // Calc the mean.
                            v = v.reduce((a,b) => a+b) / v.length
                        into[key][id] = v
                    }
                }
                const bench = []
                const benchIndex = []
                const benchOwner = []
                walk(E) // Get benchmarks.
                for (let i = 0; i < bench.length; ++i) { // Benchmark.
                    if (typeof benchFilter != 'function' || benchFilter(benchOwner[i]))
                        try {
                            const cb = currentBenchmark = Object.create(null)
                            const stop = bench[i].call()
                            assert(typeof stop == 'function', "BUT HOW DO WE STOP THIS")
                            _Packet._measureThroughput(true)
                            await new Promise((ok, bad) => setTimeout(() => { try { ok(stop()) } catch (err) { bad(err) } }, secPerBenchmark * 1000))
                            _Packet._measureThroughput(true)
                            currentBenchmark = null
                            onBenchFinished(benchOwner[i], benchIndex[i], cb, (i+1) / bench.length)
                        } catch (err) { console.error(err) }
                }
                if (!bench.length)
                    onBenchFinished(null, null, null, 1)
                currentBenchmark = null
                return Object.keys(result).length ? result : undefined
                function walk(x) {
                    if (!x || typeof x != 'object' && typeof x != 'function') return
                    if (typeof x.bench == 'function' && x.bench !== E.meta.bench) {
                        const bs = x.bench()
                        for (let id of Object.keys(bs)) {
                            bench.push(bs[id])
                            benchIndex.push(id)
                            benchOwner.push(x)
                        }
                    }
                    return Object.values(x).map(walk)
                }
            }, {
                docs:`Very slowly, runs all sensor-network benchmarks.

Can \`JSON.stringify\` the result: \`{ …, .name:{ …, key:[...values], … }, … }\`.

Arguments:
- \`secPerBenchmark = 30\`: how many seconds each step should last.
- \`benchFilter(obj) = null\`: return \`true\` to process a benchmark, else skip it.
- \`onBenchFinished(obj, id, { …, key: String | [...values], …}, progress) = null\`: the optional callback. If specified, there is no result.

Note that [Firefox and Safari don't support measuring memory](https://developer.mozilla.org/en-US/docs/Web/API/Performance/memory).

Benchmarks are contained in \`.bench()\` near to code that they benchmark. Report metrics via \`E.meta.metric(key, value)\`.
Those methods return objects (such as arrays) that contain start functions, which return stop functions.
`,
            }),
            save: A(function save(...fs) {
                const entry = fs.map((_,i) => '_mainF'+i)
                const dependencies = Object.create(null)
                fs.forEach((f,i) => walk(entry[i], f))
                const code = Object.keys(dependencies).map(k => `const ${k}=${dependencies[k]}`)
                return code.join('\n') + `\nreturn [${entry}]`
                function walk(k, v) {
                    const s = typeof v == 'string' ? JSON.stringify(v) : ''+v
                    if (typeof v == 'function' && s.slice(0,6) === 'class ') { // Bring classes into our `sn` fold.
                        assert(v instanceof E.Sensor || v instanceof E.Transform || v instanceof E.Handler, "Unrecognized class prototype")
                        const become = 'sn.' + (v instanceof E.Sensor ? 'Sensor' : v instanceof E.Transform ? 'Transform' : 'Handler')
                        s = s.replace(/\s+extends\s+.+\s+{/, become)
                    }
                    const alreadyPresent = dependencies[k] !== undefined
                    if (alreadyPresent) assert(dependencies[k] === s, "Same-name dependencies collide: "+k)
                    dependencies[k] = s
                    if (!alreadyPresent && v.save && typeof v.save == 'object')
                        Object.keys(v.save).forEach(k => walk(k, v.save[k]))
                }
            }, {
                docs:`Preserves closed-over dependencies, so that code can be loaded, via \`const [...funcs] = new Function('sn', result)(sensorNetwork)\`.

Those dependencies do have to be explicitly preserved, such as via \`a => Object.assign(b => a+b, { save:{a} })\`.

Note that directly-referenced methods (\`{ f(){} }\`) have to be written out fully (\`{ f: function(){} }\`), and sensor-network dependencies should not be specified.

Safe to save+load if \`.toString\` is not overriden by any dependency, though not safe to use the loaded functions.`,
                tests() {
                    const fMaker = a => Object.assign(b => a+b, { save:{a} }), f = fMaker(13)
                    return [
                        [
                            "Save-and-load a simple function",
                            42,
                            new Function('sn', E.meta.save(f))(E)[0](29),
                        ],
                    ]
                },
            }),
        },
        _dataNamer: A(function _dataNamer({ userName=[], userParts=1, nameParts=3, partSize=8, name, values, emptyValues=0, dataSize=64, hasher=E._dataNamer.hasher }) {
            assertCounts('', userParts, nameParts, partSize)
            const hasherMaker = hasher(name, nameParts, partSize)
            // Values are distributed evenly per-cell, to maximize the benefit of fractal-folding.
            //   (The last cell may end up with less values than others. This slight inefficiency is made worth by the consistency.)
            const cells = Math.ceil((emptyValues + values) / dataSize), valuesPerCell = Math.ceil(values / cells)
            const cellSize = (userParts + nameParts) * partSize + dataSize
            // (This re-hashes the user for each new sensor.)
            const userHasher = hasher(userName, userParts, partSize)(0, userParts * partSize, userParts * partSize)
            const nameHashers = new Array(cells).fill().map((_,i) => hasherMaker(i * valuesPerCell, Math.min((i+1) * valuesPerCell, values), values))
            return {
                cells,
                namedSize: cells * cellSize,
                cellShape: [userParts * partSize, nameParts * partSize, dataSize], // [user, name, data]
                name(src, dst, dstOffset, reward = 0, skipNonData = null) { // flat → named
                    for (let i = 0; i < cells; ++i) { // Fill out the whole cell.
                        const start = dstOffset + i * cellSize
                        const dataStart = start + (userParts + nameParts) * partSize
                        if (skipNonData == null) {
                            // User.
                            userHasher(dst, start)
                            // Name.
                            nameHashers[i](dst, start + userParts * partSize)
                            // Reward, overwriting the beginning.
                            const r = typeof reward == 'number' ? reward : reward(i * valuesPerCell, Math.min((i+1) * valuesPerCell, values), values)
                            assert(r >= -1 && r <= 1)
                            dst[start] = r
                        } else dst.fill(skipNonData, start, dataStart)
                        // Data.
                        if (src) {
                            const srcStart = i * valuesPerCell, srcEnd = Math.min(srcStart + valuesPerCell, src.length)
                            for (let s = srcStart, d = dataStart; s < srcEnd; ++s, ++d) dst[d] = src[s]
                        }
                        E._dataNamer.fill(dst, dataStart, valuesPerCell, dataSize)
                    }
                    return dstOffset + cells * cellSize // Return the next `dstOffset`.
                },
                unname(src, srcOffset, dst) { // named → flat; `named` is consumed.
                    if (src)
                        for (let i = 0; i < cells; ++i) { // Extract data from the whole cell.
                            const start = srcOffset + i * cellSize, dataStart = start + (userParts + nameParts) * partSize
                            E._dataNamer.unfill(src, dataStart, valuesPerCell, dataSize)
                            const dstStart = i * valuesPerCell, dstEnd = Math.min(dstStart + valuesPerCell, dst.length)
                            for (let s = dataStart, d = dstStart; d < dstEnd; ++s, ++d) dst[d] = src[s]
                        }
                    return srcOffset + cells * cellSize // Return the next `srcOffset`.
                },
            }
        }, {
            docs:`Implementation detail.

Prepares to go between flat number arrays and named ones.

- The result is \`{ name(src, dst, dstOffset, reward=0, skipNonData=null)→dstOffset, unname(src, srcOffset, dst)→srcOffset }\`. \`name\` goes from flat to named, \`unname\` reverses this.
    - \`reward\` is either a -1…1 number or a function from \`valueStart, valueEnd, valuesTotal\` to that.
    - If \`skipNonData\` is a -1…1 number, the non-data portions of each cell are replaced with that. (This is for "naming" data error with "no errors here".)

Main parameters, in the one object that serves as arguments:
- \`name\`: describes this interface to handlers, such as with a string. See \`._dataNamer.hasher\`.
- \`values\`: how many flat numbers there will be.

Extra parameters:
- \`userName = []\`: describes the current user/source/machine to handlers, mainly for when the sensor network encompasses multiple devices across the Internet.
- Cell sizes:
    - \`partSize = 8\`:
        - \`userParts = 1\`
        - \`nameParts = 3\`
    - \`dataSize = 64\`
- \`emptyValues = 0\`: how many fake \`values\` to insert, so that values are fractally folded more; see \`._dataNamer.fill\`.
- \`hasher = ._dataNamer.hasher\`: defines how names are transformed into \`-1\`…\`1\` numbers.
`,
            tests() {
                const F32 = Float32Array
                const buf = new F32(12)
                return [
                    [
                        "Fractal-filling values",
                        new F32([0,  0.97,  -0.5, -0.25, 0, 0.5,     0,  0.97,  0.25, 0.5, 0.5, 0]),
                        get(
                            new F32([-.5, -.25, .25, .5]),
                            { name:'z', values:4, emptyValues:1, dataSize:4, partSize:1, userParts:1, nameParts:1 },
                        ),
                    ],
                    [
                        "Fractal-filling names",
                        new F32([0, 0, 0, 0,    .13, .74, -.48, .04,    -.5, -.25, .25, .5]),
                        get(
                            new F32([-.5, -.25, .25, .5]),
                            { name:[.13], values:4, dataSize:4, partSize:4, userParts:1, nameParts:1 },
                        ),
                    ],
                    [
                        "Filling names, but there are too many strings so `.14` numbers move in",
                        new F32([0,  -.91, .14,  .5, .25, 0]),
                        get(
                            new F32([.5, .25, 0]),
                            { name:['a', 'b', 'c', .14], values:3, dataSize:3, partSize:1, userParts:1, nameParts:2 },
                        ),
                    ],
                    same(1023),
                    same(1024),
                    same(1025),
                ]
                function get(f32, opts) {
                    return test(
                        opts => { const nm = E._dataNamer(opts);  return nm.name(f32, buf, 0), buf.subarray(0, nm.namedSize).map(round) },
                        opts,
                    )
                }
                function same(n) { // Assert x = unname(name(x))
                    const src = new F32(n), dst = new F32(n)
                    for (let i=0; i < n; ++i) src[i] = Math.random() * 2 - 1
                    const opts = { name:'matters not', values:n, dataSize:64, partSize:16, userParts:1, nameParts:3 }
                    const namer = E._dataNamer(opts)
                    const cells = new F32(namer.cells * (16*(1+3)+64))
                    namer.name(src, cells, 0), namer.unname(cells, 0, dst)
                    return ["Name+unname for "+n, src.map(round), dst.map(round)]
                }
                function round(x) { return Math.round(x*100) / 100 }
            },
            hasher: A(function hasher(name, partCount, partSize) {
                let parts = [], lastPartWasNumber = false, firstNumberIndex = null, isConst = true
                flattenName(name), name = null
                if (isConst) {
                    const numbers = fillParts(new Float32Array(partCount * partSize))
                    return function specifyCell(...args) {
                        return function putHash(dst, offset) { dst.set(numbers, offset) }
                    }
                } else
                    return function specifyCell(...args) {
                        const numbers = fillParts(new Float32Array(partCount * partSize), ...args)
                        return function putHash(dst, offset) { dst.set(numbers, offset) }
                    }
                function flattenName(part) {
                    if (Array.isArray(part))
                        part.forEach(flattenName)
                    else if (typeof part == 'string') {
                        const i32 = self.YaMD5.hashStr(part, true)
                        const u8 = new Uint8Array(i32.buffer, i32.byteOffset, i32.byteLength)
                        const bytes = new Array(Math.min(u8.length, partSize))
                        for (let i = 0; i < bytes.length; ++i) bytes[i] = (u8[i] / 255) * 2 - 1
                        parts.push(bytes), lastPartWasNumber = false
                    } else if (typeof part == 'number' || typeof part == 'function') {
                        if (typeof part == 'number' && (part !== part || part < -1 || part > 1))
                            error("Name parts must be -1..1, got", part)
                        if (!lastPartWasNumber || parts[parts.length-1].length >= partSize)
                            parts.push([]), lastPartWasNumber = true
                        if (firstNumberIndex == null) firstNumberIndex = parts.length-1
                        if (typeof part == 'function') isConst = false
                        parts[parts.length-1].push(part)
                    } else error("Unrecognized name part:", part)
                }
                function fillParts(numbers, ...args) {
                    let ps = parts.slice()
                    const end = Math.min(ps.length, partCount)
                    if (firstNumberIndex != null && firstNumberIndex >= end) // Ensure that numbers are always included.
                        ps.splice(end-1, firstNumberIndex - (end-1))
                    for (let p = 0; p < end; ++p) {
                        const part = ps[p]
                        for (let i = 0; i < part.length; ++i) {
                            let x = part[i]
                            if (typeof x == 'function') {
                                x = x(...args)
                                if (typeof x != 'number' || (x !== x || x < -1 || x > 1))
                                    error("Name parts must be -1…1, got", x)
                            }
                            numbers[p * partSize + i] = x
                        }
                        E._dataNamer.fill(numbers, p * partSize, part.length, partSize)
                    }
                    if (ps.length) E._dataNamer.fill(numbers, 0, end * partSize, numbers.length)
                    else numbers.fill(0, numbers.length)
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
    // And set the most-common modules.
    E.meta.UI = UI(E)
    Object.assign(E.Sensor, {
        Text: Text(E),
        Video: Video(E),
        Audio: Audio(E),
        Scroll: Scroll(E),
        Time: Time(E),
    })
    Object.assign(E.Transform, {
        Reward: Reward(E),
    })
    Object.assign(E.Handler, {
        Sound: Sound(E),
        Random: Random(E),
    })
    return E

    function test(func, ...args) {
        try { return func(...args) }
        catch (err) { return err instanceof Error ? [err.message, err.stack] : err }
    }
    function assertCounts(msg, ...xs) { assert(xs.every(x => typeof x == 'number' && x >= 0 && x === x>>>0), msg || 'Must be a non-negative integer') }
    function assert(bool, msg) { if (!bool) error(msg || 'Assertion failed') }
    function error(...msg) { throw new Error(msg.join(' ')) }

    function allocF32(len) {
        const c = f32aCache[len]
        return c && c.length ? c.pop() : new Float32Array(len)
    }
    function deallocF32(a) {
        // Makes `allocF32` re-use `a` when allocating an array of the same size. Usually.
        assert(a instanceof Float32Array)
        const len = a.length
        if (!f32aCache[len]) f32aCache[len] = []
        const c = f32aCache[len]
        if (c.length < 16) c.push(a)
    }
    function allocArray() { return arrayCache.length ? arrayCache.pop() : [] }
    function deallocArray(a) { Array.isArray(a) && arrayCache.length < 16 && (a.length = 0, arrayCache.push(a)) }

    function debug(k, v) {
        // Prints to DOM, to not strain the poor JS console.
        if (!debug.o) debug.o = Object.create(null)
        if (!debug.o[k]) {
            if (!document.body) return
            debug.o[k] = document.createElement('div')
            document.body.append(debug.o[k])
        }
        debug.o[k].textContent = k + ': ' + v
    }

    function memory() {
        // Reports the size of the currently active segment of JS heap in bytes, or NaN.
        // Note that [Firefox and Safari don't support measuring memory](https://developer.mozilla.org/en-US/docs/Web/API/Performance/memory).
        return performance.memory ? performance.memory.usedJSHeapSize : NaN
    }
    function gotPacketFeedback(T, data, error, allFeedback, fbOffset, cellShape, partSize, summary) {
        // Fulfill the promise of `.send`.
        try {
            const then = T.feedbackCallbacks.shift()
            const noFeedback = T.feedbackNoFeedback.shift()
            const namers = T.feedbackNamers.shift()
            if (allFeedback && !noFeedback) {
                const flatV = allocF32(data.length)
                const namer = packetNamer(T, namers, cellShape, partSize, summary)
                namer.unname(allFeedback, fbOffset, flatV)
                then && then(flatV, T)
            } else
                then && then(null, T)
        } finally { deallocF32(data), error && deallocF32(error) }
    }
    function packetNamer(T, dataNamers, cellShape, partSize, summary) {
        if (!dataNamers[summary]) {
            // Create the namer.
            const [user, name, data] = cellShape
            T.partSize = partSize
            T.userParts = user / partSize | 0
            T.nameParts = name / partSize | 0
            T.dataSize = data
            dataNamers[summary] = E._dataNamer(T)
        }
        return dataNamers[summary]
    }
    function shapeSummary(cellShape, partSize) { return partSize + ':' + cellShape }
    function state(channel, cellShape, partSize, summary = cellShape && shapeSummary(cellShape, partSize)) {
        // Returns `cellShape == null ? S[channel] : S[channel].shaped[shapeSummary(cellShape, partSize)]`, creating structures if not present.
        if (!S[channel])
            S[channel] = Object.assign(Object.create(null), {
                sensors: [], // Array<Sensor>, but only those that are called automatically.
                transforms: [], // Array<Transform>, sorted by priority.
                mainHandler: null, // Handler, with max priority.
                cellShapes: [], // Array<{cellShape, partSize, summary}>, for enumeration of `.shaped` just below.
                shaped: Object.create(null), // { [summary] }
                prevEnd: 0, // time
            })
        const ch = S[channel]
        if (cellShape == null) return ch
        if (!ch.shaped[summary]) {
            ch.shaped[summary] = {
                partSize,
                looping: false,
                stepsNow: 0, // int
                overscheduled: 0, // 0…1; if too high, should have less .stepsNow.
                lastUsed: performance.now(),
                giveNextPacketNow: null, // Called when .stepsNow is 0|1, or on `sensor.send`.
                waitingSinceTooManySteps: null, // Called when a step is finished.
                msPerStep: [0,0], // [n, mean]
                cellShape, // [user, name, data]
                handlers: [], // Array<Handler>, sorted by priority.
                nextPacket: null,
                packetCache: [], // Array<_Packet>
            }
            ch.shaped[summary].nextPacket = new _Packet(channel, cellShape, partSize, summary)
            ch.cellShapes.push({cellShape, partSize, summary})
        }
        return ch.shaped[summary]
    }
})(Object.create(null))