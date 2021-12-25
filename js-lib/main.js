import './yamd5.js'
import Sound from './src/handler-sound.js'
import Video from './src/video.js'

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
    const S = Object.create(null) // See `state(channel, cellShape)`.



    // Implementation details. That we put at the top. Because JS does not hoist them.
    let currentBenchmark = null
    const f32aCache = Object.create(null)
    const arrayCache = []
    class _Packet {
        constructor(channel, cellShape) {
            Object.assign(this, {
                channel,
                cellShape,
                cellSize: cellShape.reduce((a,b) => a+b),
                // sensor → accumulator:
                cells: 0,
                sensorNeedsFeedback: false,
                sensor: [], // Sensor
                sensorData: [], // Owns f32a (Float32Array), given to `.send(…)`.
                sensorError: [], // Owns f32a (Float32Array), given to `.send(…)`.
                sensorIndices: [], // ints
                // accumulator → handler:
                input: { data:null, error:null, cellShape }, // data&error are owned f32a.
                accumulatorExtra: [], // ints
                accumulatorCallback: [], // function(feedback, cellShape, extra)
                // handler → accumulator → sensor:
                feedback: null, // null | owned f32a.
            })
        }
        static init(channel, cellShape) {
            const dst = state(channel, cellShape)
            return dst.packetCache.length ? dst.packetCache.pop() : new _Packet(channel, cellShape)
        }
        deinit() { // `this` must not be used after this call.
            // (Allows reuse of this object by `_Packet.init({…})`.)
            this.cells = 0
            this.sensorNeedsFeedback = false
            this.sensor.length = this.sensorData.length = this.sensorError.length = this.sensorIndices.length = 0
            this.input.data && (deallocF32(this.input.data), this.input.data = null)
            this.input.error && (deallocF32(this.input.error), this.input.error = null)
            this.accumulatorExtra.length = this.accumulatorCallback.length = 0
            this.feedback && (deallocF32(this.feedback), this.feedback = null)
            const dst = state(this.channel, this.cellShape)
            if (dst.packetCache.length < 64) dst.packetCache.push(this)
        }
        send(sensor, point, error, noFeedback) {
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
            if (!noFeedback) this.sensorNeedsFeedback = true
            this.sensor.push(sensor)
            this.sensorData.push(point)
            if (error)
                (this.sensorError || (this.sensorError = []))[this.sensorError.length-1] = error
            this.sensorIndices.push(this.cells)
            this.cells += point.length / this.cellSize | 0
        }
        static updateMean(a, value, maxHorizon = 100) {
            const n1 = a[0], n2 = n1+1
            a[0] = Math.min(n2, maxHorizon)
            a[1] += (value - a[1]) / n2
            if (!isFinite(a[1])) a[0] = a[1] = 0
            return a
        }
        async handle(mainHandler) { // sensors → accumulators → handlers → accumulators → sensors; `this` must not be used after this call.
            const T = this, ch = S[T.channel]
            const cellShapeStr = String(T.cellShape)
            const dst = ch.shaped[cellShapeStr]
            if (!dst) return
            const start = performance.now(), namedSize = T.cells * T.cellSize
            const benchAtStart = currentBenchmark
            ++ch.stepsNow
            try {
                // Concat sensors into `.input.data` and `.input.error`.
                T.input.data = allocF32(namedSize)
                T.input.error = !T.sensorError ? null : allocF32(namedSize)
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
                // Accumulators.
                for (let i = 0; i < ch.accumulators.length; ++i) {
                    const a = ch.accumulators[i]
                    if (typeof a.onValues == 'function' || typeof a.onFeedback == 'function') {
                        T.accumulatorExtra.push(typeof a.onValues == 'function' ? await a.onValues(T.input) : undefined)
                        T.accumulatorCallback.push(a.onFeedback)
                    }
                }
                // Handlers.
                if (mainHandler && !mainHandler.noFeedback && T.sensorNeedsFeedback)
                    T.feedback = allocF32(namedSize), T.feedback.set(T.input.data)
                else
                    T.feedback = null
                if (mainHandler) {
                    const r = mainHandler.onValues(T.input, T.feedback ? true : false, T.feedback)
                    if (r instanceof Promise) await r
                }
                const hs = dst.handlers
                if (hs.length > 2 || hs.length === 1 && hs[0] !== mainHandler) {
                    const tmp = allocArray()
                    for (let i = 0; i < hs.length; ++i) {
                        const h = hs[i]
                        if (h !== mainHandler && typeof h.onValues == 'function') {
                            const r = h.onValues(T.input, false, T.feedback)
                            if (r instanceof Promise) tmp.push(r)
                        }
                    }
                    for (let i = 0; i < tmp.length; ++i) await tmp[i]
                    deallocArray(tmp)
                }
                // Accumulators.
                while (T.accumulatorCallback.length) {
                    const f = T.accumulatorCallback.pop()
                    if (typeof f == 'function') await f(T.feedback, T.cellShape, T.accumulatorExtra.pop())
                }
                // Sensors.
                while (T.sensor.length)
                    gotPacketFeedback(T.sensor.pop(), T.sensorData.pop(), T.sensorError.pop(), T.feedback, T.sensorIndices.pop() * T.cellSize, T.cellShape, cellShapeStr)
                _Packet._handledBytes = (_Packet._handledBytes || 0) + namedSize * 4
                T.deinit()
            } finally {
                // Self-reporting.
                --ch.stepsNow
                if (!ch.stepsNow && ch.giveNextPacketNow) ch.giveNextPacketNow()
                _Packet.stepsEnded = _Packet.stepsEnded + 1 || 1
                if (benchAtStart === currentBenchmark) {
                    const duration = (dst.lastUsed = performance.now()) - start
                    _Packet.updateMean(dst.msPerStep, duration)
                    E.meta.metric('simultaneous steps', ch.stepsNow+1)
                    E.meta.metric('step processed data, values', namedSize)
                }
                ch.waitingSinceTooManySteps.length && ch.waitingSinceTooManySteps.shift()()
            }
        }
        static async handleLoop(channel, cellShape) {
            const cellShapeStr = cellShape+''
            const ch = S[channel], dst = ch.shaped[cellShapeStr]
            if (!dst || dst.looping) return;  else dst.looping = true
            let prevEnd = performance.now(), prevWait = prevEnd
            while (true) {
                if (!ch.shaped[cellShapeStr]) return // `Sensor`s might have cleaned us up.
                const start = performance.now(), end = start + dst.msPerStep[1]
                // Don't do too much at once.
                while (ch.stepsNow > E.maxSimultaneousPackets)
                    await new Promise(then => ch.waitingSinceTooManySteps.push(then))
                // Get sensor data.
                const mainHandler = ch.mainHandler && ch.mainHandler.cellShape+'' === cellShape+'' ? ch.mainHandler : null
                const mainSensor = ch.mainHandler ? !!mainHandler : ch.cellShapes[0]+'' === cellShape+''
                if (mainSensor && Array.isArray(ch.sensors))
                    for (let i = 0; i < ch.sensors.length; ++i) {
                        const s = ch.sensors[i]
                        const data = allocF32(s.values)
                        s.onValues(s, data)
                    }
                // Pause if no destinations, or no sources & no data to send.
                if (!dst.handlers.length || !ch.sensors.length && !dst.nextPacket.sensor.length)
                    return dst.msPerStep[0] = dst.msPerStep[1] = 0, dst.looping = false
                await Promise.resolve() // Wait a bit.
                //   (Also, commenting this out halves Firefox throughput.)
                // Send it off.
                const nextPacket = dst.nextPacket;  dst.nextPacket = _Packet.init(channel, cellShape)
                nextPacket.handle(mainHandler)
                // Benchmark throughput if needed.
                _Packet._measureThroughput()
                // Don't do it too often.
                const now = performance.now(), needToWait = prevEnd - now
                prevEnd += dst.msPerStep[1]
                if (prevEnd < now - 1000) prevEnd = now - 1000 // Don't get too eager after being stalled.
                if (needToWait > 0 || now - prevWait > 100) {
                    await new Promise(then => setTimeout(ch.giveNextPacketNow = then, Math.max(needToWait, 0)))
                    ch.giveNextPacketNow = null
                    prevWait = performance.now()
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
            constructor(opts) { assert(opts), this.resume(opts) }
            needsExtensionAPI() { return null }
            sendCallback(then, values, error = null, reward = 0) { // In profiling, promises are the leading cause of garbage.
                // Name+send to all handler shapes.
                // Also forget about shapes that are more than 60 seconds old, to not slowly choke over time.
                assert(values instanceof Float32Array)
                assert(error === null || error instanceof Float32Array)
                assert(values.length === this.values, "Data size differs from the one in options")
                error && assert(values.length === error.length)
                const ch = state(this.channel)
                const removed = Sensor._removed || (Sensor._removed = new Set)
                for (let i = 0; i < ch.cellShapes.length; ++i) {
                    const cellShape = ch.cellShapes[i]
                    const cellShapeStr = String(cellShape)
                    const dst = ch.shaped[cellShapeStr]
                    if (!dst.handlers.length) {
                        if (performance.now() - dst.lastUsed > 60000) removed.add(cellShape)
                        continue
                    }
                    const namer = packetNamer(this, this.dataNamers, cellShape, cellShapeStr)
                    const flatV = allocF32(namer.namedSize)
                    const flatE = error ? allocF32(namer.namedSize) : null
                    namer.name(values, flatV, 0, reward)
                    flatE && namer.name(error, flatE, 0, 0, -1.)
                    dst.nextPacket.send(this, flatV, flatE, this.noFeedback)
                }
                if (removed.size) {
                    ch.cellShapes = ch.cellShapes.filter(sh => !removed.has(sh))
                    removed.forEach(sh => delete ch[sh])
                    removed.clear()
                }
                deallocF32(values), error && deallocF32(error)
                this.feedbackCallbacks.push(then)
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
                state(this.channel).sensors = state(this.channel).sensors.filter(v => v !== this)
                this.paused = true
                return this
            }
            resume(opts) {
                if (opts) {
                    this.pause()
                    const { name, values, onValues=null, channel='', noFeedback=false, rewardName=[], userName=[], emptyValues=0, hasher=undefined } = opts
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
                    })
                    if (!this.feedbackCallbacks) {
                        this.feedbackCallbacks = []
                        this.feedbackNoFeedback = []
                        this.feedbackNamers = []
                    }
                }
                if (!this.paused) return this
                if (typeof this.onValues == 'function') {
                    state(this.channel).sensors.push(this)
                    for (let cellShape of state(this.channel).cellShapes)
                        _Packet.handleLoop(this.channel, cellShape)
                }
                this.paused = false
                return this
            }
        }, {
            docs:`Generalization of eyes and ears and hands, hotswappable and differentiable.

- \`constructor({ name, values, onValues=null, channel='', noFeedback=false, rewardName=[], userName=[], emptyValues=0, hasher=undefined })\`
    - \`name\`: a human-readable string, or an array of that or a -1…1 number or a function from \`dataStart, dataEnd, dataLen\` to a -1…1 number.
    - \`onValues(sensor, data)\`: the regularly-executed function that reports data, by calling \`sensor.send(data, …)\` inside once. Not \`await\`ed.
        - To run faster, use \`sensor.sendCallback(fn(feedback), data, …)\` with a static function.
    - Extra flexibility:
        - \`channel\`: the human-readable name of the channel. Communication only happens within the same channel.
        - \`noFeedback\`: set to \`true\` if applicable to avoid some processing. Otherwise, feedback is the data that should have been.
        - \`rewardName\`: the name of the currently-optimized task, in case accumulators want to change it and inform handlers.
        - \`userName\`: the name of the machine that sources data. Makes it possible to reliably distinguish sources.
        - \`emptyValues\`: the guaranteed extra padding, for fractal folding. See \`._dataNamer.fill\`.
        - \`hasher(…)(…)(…)\`: see \`._dataNamer.hasher\`. The default mainly hashes strings in \`rewardName\`/\`userName\`/\`name\` with MD5 and rescales bytes into -1…1.
    - To change any of this, \`pause()\` and recreate.

- \`send(values, error = null, reward = 0) → Promise<null|feedback>\`
    - (Do not override in child classes, only call.)
    - \`values\`: owned flat data, -1…1 \`Float32Array\`. Do not perform ANY operations on it once called.
        - To mitigate misalignment, try to stick to powers of 2 in all sizes.
        - (Can use \`sn._allocF32(len)\` for efficient reuse.)
    - \`error\`: can be \`null\` or owned flat data, -1…1 \`Float32Array\` of length \`values.length\`: \`max abs(truth - observation) - 1\`. Do not perform ANY operations on it once called.
    - \`reward\`: every sensor can tell handlers what to maximize, -1…1. (What is closest in your mind? Localized pain and pleasure? Satisfying everyone's needs rather than the handler's? …Money? Either way, it sounds like a proper body.)
        - Can be a number or a function from \`valueStart, valueEnd, valuesTotal\` to that.
    - (Result: \`feedback\` is owned by you. Can use \`feedback && sn._deallocF32(feedback)\` once you are done with it, or simply ignore it and let GC collect it.)

- \`sendCallback(then(null|feedback), values, error = null, reward = 0)\`: exactly like \`send\` but does not have to allocate a promise.

- \`pause()\`, \`resume()\`: for convenience, these return the object.

- \`needsExtensionAPI() → null|String\`: overridable in child classes. By default, the sensor is entirely in-page in a [content script](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Content_scripts) if injected by an extension. For example, make this return \`'tabs'\` to get access to [\`chrome.tabs\`](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/tabs) in an extension.`,
        }),
        Accumulator: A(class Accumulator {
            constructor(opts) { assert(opts), this.resume(opts) }
            pause() {
                if (this.paused !== false) return this
                const ch = state(this.channel)
                ch.accumulators = ch.accumulators.filter(v => v !== this)
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
                    assert(onValues || onFeedback, "Why have an accumulator if it does nothing; pass in either `onValues` or `onFeedback`")
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
                state(this.channel).accumulators.push(this)
                state(this.channel).accumulators.sort((a,b) => b.priority - a.priority)
                this.paused = false
                return this
            }
        }, {
            docs:`Modifies data/feedback, after sensors and before handlers.

- \`constructor({ onValues=null, onFeedback=null, priority=0, channel='' })\`
    - Needs one or both:
        - \`onValues({data, error, cellShape}) → extra\`: can modify \`data\` and the optional \`error\` in-place.
            - \`cellShape: [reward, user, name, data]\`
            - Data is split into cells, each made up of \`cellShape.reduce((a,b)=>a+b)\` -1…1 numbers.
            - Can return a promise.
        - \`onFeedback(feedback, cellShape, extra)\`: can modify \`feedback\` in-place.
            - Can return a promise.
    - Extra flexibility:
        - \`priority\`: accumulators run in order, highest priority first.
        - \`channel\`: the human-readable name of the channel. Communication only happens within the same channel.
    - To change any of this, \`pause()\` and recreate.

- \`pause()\`, \`resume()\`: for convenience, these return the object.`,
        }),
        Handler: A(class Handler {
            constructor(opts) { assert(opts), this.resume(opts) }
            pause() {
                if (this.paused !== false) return this
                const ch = state(this.channel), dst = state(this.channel, this.cellShape)
                dst.handlers = dst.handlers.filter(v => v !== this)
                if (ch.mainHandler === this) {
                    ch.mainHandler = null
                    for (let cellShape of ch.cellShapes)
                        for (let h of ch.shaped[cellShape].handlers)
                            if (!h.noFeedback && (ch.mainHandler == null || ch.mainHandler.priority < h.priority))
                                ch.mainHandler = h
                }
                this.paused = true
                return this
            }
            resume(opts) {
                if (opts) {
                    this.pause()
                    const { onValues, partSize=8, rewardParts=0, userParts=1, nameParts=3, dataSize=64, noFeedback=false, priority=0, channel='' } = opts
                    assert(typeof onValues == 'function', "Handlers must have `onValues` listeners")
                    assertCounts('', partSize, rewardParts, userParts, nameParts, dataSize)
                    assert(typeof priority == 'number')
                    assert(typeof channel == 'string')
                    Object.assign(this, {
                        paused: true,
                        cellShape: [rewardParts * partSize, userParts * partSize, nameParts * partSize, dataSize],
                        onValues,
                        noFeedback: !!noFeedback,
                        priority,
                        channel,
                    })
                }
                if (!this.paused) return this
                const ch = state(this.channel), dst = state(this.channel, this.cellShape)
                dst.handlers.push(this)
                dst.handlers.sort((a,b) => b.priority - a.priority)
                if (!this.noFeedback && (ch.mainHandler == null || ch.mainHandler.priority < this.priority)) ch.mainHandler = this
                if (this.onValues) _Packet.handleLoop(this.channel, this.cellShape)
                this.paused = false
                return this
            }
        }, {
            docs:`Given data, gives feedback: human or AI model.

- \`constructor({ onValues, partSize=8, rewardParts=0, userParts=1, nameParts=3, dataSize=64, noFeedback=false, priority=0, channel='' })\`
    - \`onValues({data, error, cellShape}, writeFeedback, feedback)\`: process.
        - // TODO: And, don't pass in \`feedback\`, return it instead. Fill data with the main handler's feedback on no-data ourselves. As an arg, only give the bool "needFeedback".
        - (\`data\` and \`error\` are not owned; do not write.)
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

- \`pause()\`, \`resume()\`: for convenience, these return the object.`,
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
                            onValues({data, error, cellShape}, writeFeedback, feedback) {
                                data.fill(.489018922485) // "Read" it.
                                if (writeFeedback) feedback.fill(-1)
                            },
                        })
                        return function stop() { from.pause(), to.pause() }
                    }
                }
            },
        }),
        _allocF32: allocF32,
        _deallocF32: deallocF32,
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
                    if (typeof benchFilter != 'function' || benchFilter(bench[i]))
                        try {
                            currentBenchmark = Object.create(null)
                            const stop = bench[i].call()
                            assert(typeof stop == 'function', "BUT HOW DO WE STOP THIS")
                            _Packet._measureThroughput(true)
                            await new Promise((ok, bad) => setTimeout(() => { try { ok(stop()) } catch (err) { bad(err) } }, secPerBenchmark * 1000))
                            _Packet._measureThroughput(true)
                            onBenchFinished(benchOwner[i], benchIndex[i], currentBenchmark, (i+1) / bench.length)
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
                        assert(v instanceof E.Sensor || v instanceof E.Accumulator || v instanceof E.Handler, "Unrecognized class prototype")
                        const become = 'sn.' + (v instanceof E.Sensor ? 'Sensor' : v instanceof E.Accumulator ? 'Accumulator' : 'Handler')
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
    Object.assign(E.Sensor, {
        Video: Video(E),
    })
    Object.assign(E.Handler, {
        Sound: Sound(E),
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
    function gotPacketFeedback(T, data, error, allFeedback, fbOffset, cellShape, s = String(cellShape)) {
        // Fulfill the promise of `.send`.
        try {
            const then = T.feedbackCallbacks.shift()
            const noFeedback = T.feedbackNoFeedback.shift()
            const namers = T.feedbackNamers.shift()
            if (allFeedback && !noFeedback) {
                const flatV = allocF32(data.length)
                const namer = packetNamer(T, namers, cellShape, s)
                namer.unname(allFeedback, fbOffset, flatV)
                then(flatV)
            } else
                then(null)
        } finally { deallocF32(data), error && deallocF32(error) }
    }
    function packetNamer(T, dataNamers, cellShape, s = String(cellShape)) {
        if (!dataNamers[s]) {
            // *Guess* handler's `partSize`, based only on `cellShape` for reproducibility. And create the namer.
            const [reward, user, name, data] = cellShape
            const metaSize = reward + user + name
            T.partSize = gcd(gcd(reward, user), name)
            T.rewardParts = reward / T.partSize | 0
            T.userParts = user / T.partSize | 0
            T.nameParts = name / T.partSize | 0
            dataNamers[s] = E._dataNamer(T)
            function gcd(a,b) { return !b ? a : gcd(b, a % b) }
        }
        return dataNamers[s]
    }
    function state(channel, cellShape) {
        // Returns `cellShape != null ? S[channel].shaped[cellShape] : S[channel]`, creating structures if not present.
        if (!S[channel])
            S[channel] = Object.assign(Object.create(null), {
                sensors: [], // Array<Sensor>, but only those that are called automatically.
                accumulators: [], // Array<Accumulator>, sorted by priority.
                mainHandler: null, // Handler, with max priority.
                stepsNow: 0, // int
                giveNextPacketNow: null, // Called when .stepsNow is 0.
                waitingSinceTooManySteps: [], // Array<function>, called when a step is finished.
                cellShapes: [], // Array<String>, for enumeration of `.shaped` just below.
                shaped: Object.create(null), // { [handlerShapeAsString] }
            })
        const ch = S[channel]
        if (cellShape == null) return ch
        const cellShapeStr = ''+cellShape
        if (!ch.shaped[cellShapeStr])
            ch.shaped[cellShapeStr] = {
                looping: false,
                lastUsed: performance.now(),
                msPerStep: [0,0], // [n, mean]
                cellShape: cellShape, // [reward, user, name, data]
                handlers: [], // Array<Handler>, sorted by priority.
                nextPacket: new _Packet(channel, cellShape),
                packetCache: [], // Array<_Packet>
            }, ch.cellShapes.push(cellShape)
        return ch.shaped[cellShapeStr]
    }
})(Object.create(null))