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
    const S = Object.create(null) // See `E._state(channel, cellShape)`.
    let currentBenchmark = null

    return A(E, {
        Sensor: A(class Sensor {
            constructor({ name, values, onValues=null, channel='', noFeedback=false, rewardName=[], userName=[], emptyValues=0, hasher=undefined }) {
                assert(typeof name == 'string' || Array.isArray(name), 'Must have a name')
                assertCounts('Must have the value-count', values)
                assert(onValues == null || typeof onValues == 'function')
                assert(typeof channel == 'string')
                assert(typeof rewardName == 'string' || Array.isArray(rewardName))
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
                    rewardName,
                    userName,
                    emptyValues,
                    hasher,
                    dataNamers: Object.create(null), // cellShape → _dataNamer({ rewardName=[], rewardParts=0, userName=[], userParts=1, nameParts=3, partSize=8, name, values, emptyValues=0, dataSize=64, hasher=E._dataNamer.hasher })
                    partSize: 0,
                    rewardParts: 0,
                    userParts: 0,
                    nameParts: 0,
                    feedbackCallbacks: [], // A queue of promise-fulfilling callbacks. Very small, so an array is the fastest option.
                }).resume()
            }
            send(values, error = null, reward = 0) { // Returns a promise of feedback (no reward) or null.
                // Name+send to all handler shapes.
                // Also forget about shapes that are more than 60 seconds old, to not slowly choke over time.
                const ch = E._state(this.channel)
                const removed = Sensor._removed || (Sensor._removed = new Set)
                for (let cellShape of ch.cellShapes) {
                    const dst = ch.shaped[cellShape]
                    if (!dst.handlers.length) {
                        if (performance.now() - dst.lastUsed > 60000) removed.add(cellShape)
                        continue
                    }
                    const namer = this._namer(cellShape)
                    const flatV = E._allocF32(namer.namedSize)
                    const flatE = error ? E._allocF32(namer.namedSize) : null
                    namer.name(values, flatV, 0, reward)
                    flatE && namer.name(error, flatE, 0, 0, -1.)
                    dst.nextPacket.data(this, flatV, flatE, this.noFeedback)
                }
                if (removed.size) {
                    ch.cellShapes = ch.cellShapes.filter(sh => !removed.has(sh))
                    removed.forEach(sh => delete ch[sh])
                    removed.clear()
                }
            }
            _gotFeedback(data, error, feedback, fbOffset, cellShape) {
                // Fulfill the promise of `.send`.
                if (feedback && !this.noFeedback) {
                    const flatV = E._allocF32(this.values)
                    this._namer(cellShape).unname(feedback, fbOffset, flatV)
                    this.feedbackCallbacks.shift()(flatV)
                    E._deallocF32(flatV)
                } else
                    this.feedbackCallbacks.shift()(null)
                E._deallocF32(data), E._deallocF32(error)
            }
            _namer(cellShape) {
                const s = ''+cellShape
                if (!this.dataNames[s]) {
                    // *Guess* handler's `partSize`, based only on `cellShape` for reproducibility. And create the namer.
                    const metaSize = this.cellShape[0] + this.cellShape[1] + this.cellShape[2]
                    this.partSize = gcd(gcd(this.cellShape[0], this.cellShape[1]), this.cellShape[2])
                    this.rewardParts = this.cellShape[0] / this.partSize | 0
                    this.userParts = this.cellShape[0] / this.partSize | 0
                    this.nameParts = this.cellShape[0] / this.partSize | 0
                    this.dataNames[s] = E._dataNamer(this)
                    function gcd(a,b) { return !b ? a : gcd(b, a % b) }
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
        }, {
            docs:`Generalization of eyes and ears and hands, hotswappable and differentiable.

- \`constructor({ name, values, onValues=null, channel='', noFeedback=false, rewardName=[], userName=[], emptyValues=0, hasher=undefined })\`
    - \`name\`: a human-readable string, or an array of that or a -1…1 number or a function from \`dataStart, dataEnd, dataLen\` to a -1…1 number.
    - \`values\`: how many values each packet will have. To mitigate misalignment, try to stick to powers-of-2.
    - \`onValues(sensor)\`: the regularly-executed function that reports data, by calling \`sensor.send\` inside.
        - Can return a promise.
    - Extra flexibility:
        - \`channel\`: the human-readable name of the channel. Communication only happens within the same channel.
        - \`noFeedback\`: set to \`true\` if applicable to avoid some processing. Otherwise, feedback is the data that should have been.
        - \`rewardName\`: the name of the currently-optimized task, in case accumulators want to change it and inform handlers.
        - \`userName\`: the name of the machine that sources data. Makes it possible to reliably distinguish sources.
        - \`emptyValues\`: the guaranteed extra padding, for fractal folding. See \`._dataNamer.fill\`.
        - \`hasher(…)(…)(…)\`: see \`._dataNamer.hasher\`. The default hashes strings in \`rewardName\`/\`userName\`/\`name\` with MD5 and rescales bytes into -1…1.
    - To change any of this, \`pause()\` and recreate.

- \`send(values, error = null, reward = 0) → Promise<null|feedback>\`
    - \`values\`: owned flat data, -1…1 \`Float32Array\` of length \`values\`. Do not perform ANY operations on it once called.
    - \`error\`: can be owned flat data, -1…1 \`Float32Array\` of length \`values\`: \`max abs(truth - observation) - 1\`. Do not perform ANY operations on it once called.
    - \`reward\`: every sensor can tell handlers what to maximize, -1…1. (What is closest in your mind? Localized pain and pleasure? Satisfying everyone's needs rather than the handler's? …Money? Close enough.)
        - Can be a number or a function from \`valueStart, valueEnd, valuesTotal\` to that.

- \`pause()\`, \`resume()\``,
        }),
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
        }, {
            docs:`Modifies data/feedback, after sensors and before handlers.

- \`constructor({ onValues=null, onFeedback=null, priority=0, channel='' })\`
    - Needs one or both:
        - \`onValues(data, error, cellShape) → extra\`: can modify \`data\` and the optional \`error\` in-place.
            - \`cellShape: [reward, user, name, data]\`
            - Data is split into cells, each made up of \`cellShape.reduce((a,b)=>a+b)\` -1…1 numbers.
            - Can return a promise.
        - \`onFeedback(feedback, cellShape, extra)\`: can modify \`feedback\` in-place.
            - Can return a promise.
    - Extra flexibility:
        - \`priority\`: accumulators run in order, highest priority first.
        - \`channel\`: the human-readable name of the channel. Communication only happens within the same channel.
    - To change any of this, \`pause()\` and recreate.

- \`pause()\`, \`resume()\``,
        }),
        Handler: A(class Handler {
            constructor({ onValues, partSize=8, rewardParts=0, userParts=1, nameParts=3, dataSize=64, noFeedback=false, priority=0, channel='' }) {
                assert(typeof onValues == 'function', "Handlers must have listeners")
                assertCounts('', partSize, rewardParts, userParts, nameParts, dataSize)
                assert(typeof priority == 'number')
                assert(typeof channel == 'number')
                Object.assign(this, {
                    paused: true,
                    cellShape: [rewardParts * partSize, userParts * partSize, nameParts * partSize, dataSize],
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
        }, {
            docs:`Given data, gives feedback: human or AI model.

- \`constructor({ onValues, partSize=8, rewardParts=0, userParts=1, nameParts=3, dataSize=64, noFeedback=false, priority=0, channel='' })\`
    - \`onValues(data, error, cellShape, writeFeedback, feedback)\`: process.
        - \`error\` and \`feedback\` can be \`null\`s.
        - If \`writeFeedback\`, write something to \`feedback\`, else read \`feedback\`.
        - At any time, there is only one *main* handler, and only that can write feedback.
    - Cell sizes:
        - \`partSize\`: how many numbers each part in the cell ID takes up, where each string in a name takes up a whole part:
            - \`rewardParts\`
            - \`userParts\`
            - \`nameParts\`
        - \`dataSize\`: numbers in the data segment.
    - Extra flexibility:
        - \`noFeedback\`: can't provide feedback if \`true\`, only observe it.
        - \`priority\`: the highest-priority handler without \`noFeedback\` will be the *main* handler, and give feedback.
        - \`channel\`: the human-readable name of the channel. Communication only happens within the same channel.

- \`pause()\`, \`resume()\``,
            // TODO: `bench()`, which creates a `1`-filling sensor and a `-1`-filling handler, with default params and 0…256 cells (so, that*64 values).
        }),
        maxSimultaneousPackets: 4,
        _state(channel, cellShape) { // Returns `cellShape != null ? S[channel].shaped[cellShape] : S[channel]`, creating structures if not present.
            if (!S[channel])
                S[channel] = Object.assign(Object.create(null), {
                    sensors: [], // Array<Sensor>, but only those that are called automatically.
                    accumulators: [], // Array<Accumulator>, sorted by priority.
                    mainHandler: null, // Handler, with max priority.
                    stepsNow: 0, // int
                    waitingSinceTooManySteps: [], // Array<function>, called when a step is finished.
                    cellShapes: [], // Array<String>, for enumeration of `.shaped` just below.
                    shaped: Object.create(null), // { [handlerShapeAsString] }
                })
            const ch = S[channel]
            if (cellShape == null) return ch
            if (!ch.shaped[cellShape])
                ch.shaped[cellShape] = {
                    looping: false,
                    lastUsed: performance.now(),
                    msPerStep: [0,0], // [n, mean]
                    cellShape: cellShape, // [reward, user, name, data]
                    handlers: [], // Array<Handler>, sorted by priority.
                    nextPacket: new _Packet(channel, cellShape),
                }, ch.cellShapes.push(cellShape)
            return ch.shaped[cellShape]
        },
        _memory: A(function() { return performance.memory ? performance.memory.usedJSHeapSize : 0 }, {
            docs:`Reports the size of the currently active segment of JS heap in bytes, or 0.

Note that [Firefox and Safari don't support measuring memory](https://developer.mozilla.org/en-US/docs/Web/API/Performance/memory).`,
        }),
        _allocF32(len) { return _Packet._f32 && _Packet._f32[len] && _Packet._f32[len].length ? _Packet._f32[len].pop() : new Float32Array(len) },
        _deallocF32(a) {
            // Makes `E._allocF32` re-use `a` when allocating an array of the same size. Usually.
            if (!_Packet._f32) _Packet._f32 = Object.create(null)
            if (!_Packet._f32[len]) _Packet._f32[len] = []
            if (_Packet._f32[len].length > 16) return
            _Packet._f32[len].push(a)
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
                    accumulatorExtra: [], // ints
                    accumulatorCallback: [], // function(feedback, cellShape, extra)
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
            static updateMean(a, value, maxHorizon = 1000) {
                const n1 = a[0], n2 = n1+1
                a[0] = Math.min(n2, maxHorizon)
                a[1] += (value - a[1]) / n2
                if (!isFinite(a[1])) a[0] = a[1] = 0
                return a
            }
            async handle(mainHandler) { // sensors → accumulators → handlers → accumulators → sensors
                const T = this, ch = S[T.channel], dst = ch[T.cellShape]
                const start = performance.now(), startMemory = E._memory()
                ++ch.stepsNow
                try {
                    // Concat sensors into `.data` and `.error`.
                    T.data = E._allocF32(T.cells * T.cellSize), T.error = !T.sensorError ? null : E._allocF32(T.cells * T.cellSize)
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
                            T.accumulatorExtra.push(typeof a.onValues == 'function' ? await a.onValues(T.data, T.error, T.cellShape) : undefined)
                            T.accumulatorCallback.push(a.onFeedback)
                        }
                    // Handlers.
                    if (mainHandler && !mainHandler.noFeedback && T.sensorNeedsFeedback)
                        T.feedback = E._allocF32(T.cells * T.cellSize), T.feedback.set(T.data)
                    else
                        T.feedback = null
                    if (mainHandler) await mainHandler.onValues(T.data, T.error, T.cellShape, T.feedback ? true : false, T.feedback)
                    let tmp
                    for (let h of dst.handlers)
                        if (typeof h.onValues == 'function') {
                            const r = h.onValues(T.data, T.error, T.cellShape, false, T.feedback)
                            if (r instanceof Promise) (tmp || (tmp = [])).push(r)
                        }
                    if (r) await Promise.all(tmp)
                    // Accumulators.
                    while (T.accumulatorCallback.length) {
                        const f = T.accumulatorCallback.pop()
                        if (typeof f == 'function') await f(T.feedback, T.cellShape, T.accumulatorExtra.pop())
                    }
                    // Sensors.
                    while (T.sensor.length)
                        T.sensor.pop()._gotFeedback(T.sensorData.pop(), T.sensorError.pop(), T.feedback, T.sensorIndices.pop() * T.cellSize, T.cellShape)
                    _Packet._handledBytes = (_Packet._handledBytes || 0) + T.cells * T.cellSize * 4
                } finally {
                    // Self-reporting.
                    --ch.stepsNow
                    const duration = (dst.lastUsed = performance.now()) - start
                    E.meta.metric('Step duration, ms', duration)
                    E.meta.metric('Step memory, bytes', E._memory() - startMemory)
                    E.meta.metric('Step processed data, values', T.cells * T.cellSize)
                    _Packet.updateMean(dst.msPerStep, (dst.lastUsed = performance.now()) - start)
                    ch.waitingSinceTooManySteps.length && ch.waitingSinceTooManySteps.shift()()
                }
            }
            static async handleLoop(channel, cellShape) {
                const ch = S[channel], dst = ch.shaped[cellShape]
                if (dst.looping) return;  else dst.looping = true
                const tmp = []
                while (true) {
                    const start = performance.now(), end = start + dst.msPerStep[1]
                    // Don't do too much at once.
                    while (ch.stepsNow > E.maxSimultaneousPackets)
                        await new Promise(then => ch.waitingSinceTooManySteps.push(then))
                    // Pause if no destinations, or no sources & no data to send.
                    if (!dst.handlers.length || !ch.sensors.length && !dst.nextPacket.sensor.length) return dst.looping = false
                    // Get sensor data.
                    const mainHandler = ch.mainHandler && ch.mainHandler.cellShape+'' === cellShape+'' ? ch.mainHandler : null
                    if (mainHandler)
                        for (let s of ch.sensors)
                            tmp.push(s.onValues(s))
                    await Promise.all(tmp), tmp.length = 0
                    // Send it off.
                    const nextPacket = dst.nextPacket;  dst.nextPacket = new _Packet(channel, cellShape)
                    nextPacket.handle(mainHandler)
                    // Benchmark throughput if needed.
                    E.meta.metric('Throughput, bytes/s', (_Packet._handledBytes || 0) / ((performance.now() - start) / 1000))
                    _Packet._handledBytes = 0
                    // Don't do it too often.
                    if (performance.now() < end)
                        await new Promise(then => setTimeout(then, end - performance.now()))
                }
            }
        },
        meta:{
            docs: A(function docs() {
                // TODO: ...How to go over the whole `E`, but only the non-`_` parts?...
                // TODO: ...How to format everything, exactly?...
                //   TODO: Special-case `E.meta`: "???".
            }, {
                docs:`Returns the Markdown string containing all the sensor network's documentation.`,
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
                docs:`Asynchronously, runs all sensor-network tests, and returns \`null\` if OK, else an array of \`[failedTestName, value1, value2]\`.

If not \`null\`, things are very wrong.

Internally, it calls \`.tests()\` which return \`[…, [testName, value1, value2], …]\`. String representations must match exactly to succeed.`,
            }),
            metric: A(function (key, value) {
                if (typeof value == 'string')
                    currentBenchmark[key] = value
                else if (typeof value == 'number')
                    !Array.isArray(currentBenchmark[key]) && (currentBenchmark[key] = []), currentBenchmark[key].push(value)
                else
                    error("what this: " + value)
            }, {
                docs:`Call this with a string key & string/number value to display/measure something, if \`E.meta.bench\` controls execution.`,
            }),
            bench: A(async function bench(secPerBenchmark = 30, benchFilter=null, onBenchFinished=null) {
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
                    currentBenchmark = Object.create(null)
                    if (typeof benchFilter != 'function' || benchFilter(bench[i]))
                        try {
                            const stop = bench[i].call()
                            await new Promise((ok, bad) => setTimeout(() => { try { ok(stop()) } catch (err) { bad(err) } }, secPerBenchmark * 1000))
                            onBenchFinished(benchOwner[i], benchIndex[i], currentBenchmark, i / (bench.length-1))
                        } catch (err) { console.error(err) }
                }
                currentBenchmark = null
                return Object.keys(result).length ? result : undefined
                function walk(x) {
                    if (!x || typeof x != 'object' && typeof x != 'function') return
                    if (typeof x.bench == 'function' && x.bench !== bench) {
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
                docs:`Asynchronously & very slowly, runs all sensor-network benchmarks.

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
            save: A(function save(f) {
                // TODO: Recursively put `save`d dependencies before the last bracket wherever `f` defines `save: […dependencies]`, else return `''+f` locally.
                // TODO: Turn classes into funcs first, which just forward args to the constructor, and replace the `extends <…>` part with the correct SN class.
            }, {
                docs:``, // TODO:
                //   (To load, `new Function('sn', result)(sn)`.)
                //   ("Though including all dependencies with every entry point may seem to lead to code duplication, having too many entry points is impossible to learn, so SN inherently discourages the situation from getting too out of hand.")
            }),
        },
        _dataNamer: A(function _dataNamer({ rewardName=[], rewardParts=0, userName=[], userParts=1, nameParts=3, partSize=8, name, values, emptyValues=0, dataSize=64, hasher=E._dataNamer.hasher }) {
            assertCounts('', rewardParts, userParts, nameParts, partSize)
            const hasherMaker = hasher(name, nameParts, partSize)
            // Values are distributed evenly per-cell, to maximize the benefit of fractal-folding.
            //   (The last cell may end up with less values than others. This slight inefficiency is made worth by the consistency.)
            const cells = Math.ceil((emptyValues + values) / dataSize), valuesPerCell = Math.ceil(values / cells)
            const cellSize = (rewardParts + userParts + nameParts) * partSize + dataSize
            // (This re-hashes the user for each new sensor.)
            const rewardHasher = hasher(rewardName, rewardParts, partSize)(0, rewardParts * partSize, rewardParts * partSize)
            const userHasher = hasher(userName, userParts, partSize)(0, userParts * partSize, userParts * partSize)
            const nameHashers = new Array(cells).fill().map((_,i) => hasherMaker(i * valuesPerCell, Math.min((i+1) * valuesPerCell, values), values))
            return {
                cells,
                namedSize: cells * cellSize,
                cellShape: [rewardParts * partSize, userParts * partSize, nameParts * partSize, dataSize], // [reward, user, name, data]
                name(src, dst, dstOffset, reward = 0, skipNonData = null) { // flat → named
                    for (let i = 0; i < cells; ++i) { // Fill out the whole cell.
                        const start = dstOffset + i * cellSize, dataStart = start + (rewardParts + userParts + nameParts) * partSize
                        if (skipNonData == null) {
                            // Reward name.
                            rewardHasher(dst, start)
                            // User.
                            userHasher(dst, start + rewardParts * partSize)
                            // Name.
                            nameHashers[i](dst, start + (rewardParts + userParts) * partSize)
                            // Reward, overwriting the beginning.
                            const r = typeof reward == 'number' ? reward : reward(i * valuesPerCell, Math.min((i+1) * valuesPerCell, values), values)
                            assert(r >= -1 && r <= 1)
                            dst[start] = r
                        } else dst.fill(skipNonData, start, dataStart)
                        // Data.
                        const srcStart = i * valuesPerCell, srcEnd = Math.min(srcStart + valuesPerCell, src.length)
                        for (let s = srcStart, d = dataStart; s < srcEnd; ++s, ++d) dst[d] = src[s]
                        E._dataNamer.fill(dst, dataStart, valuesPerCell, dataSize)
                    }
                    return dstOffset + cells * cellSize // Return the next `dstOffset`.
                },
                unname(src, srcOffset, dst) { // named → flat; `named` is consumed.
                    for (let i = 0; i < cells; ++i) { // Extract data from the whole cell.
                        const start = srcOffset + i * cellSize, dataStart = start + (rewardParts + userParts + nameParts) * partSize
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
- \`rewardName = []\`: what each cell's first number represents, for if you have extra tasks that you want your AI model to learn, such as next-frame prediction.
- \`userName = []\`: describes the current user/source/machine to handlers, mainly for when the sensor network encompasses multiple devices across the Internet.
- Cell sizes:
    - \`partSize = 8\`:
        - \`rewardParts = 0\`
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
                let parts = [], lastPartWasNumber = false, firstNumberIndex = null
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
                        if (firstNumberIndex == null) firstNumberIndex = parts.length-1
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
    function test(func, ...args) {
        try { return func(...args) }
        catch (err) { return err instanceof Error ? [err.message, err.stack] : err }
    }
    function assertCounts(msg, ...xs) { assert(xs.every(x => typeof x == 'number' && x >= 0 && x === x>>>0), msg || 'Must be a non-negative integer') }
    function assert(bool, msg) { if (!bool) error(msg || 'Assertion failed') }
    function error(...msg) { throw new Error(msg.join(' ')) }
})(Object.create(null))