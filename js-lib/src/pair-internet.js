export default function init(sn) {
    // (Reading/writing may be quite unoptimized here.)
    const A = Object.assign
    const arrayCache = []



    function importSimplePeer() { // Manual use of WebRTC is just not working out.
        if (importSimplePeer.did) return importSimplePeer.did
        return importSimplePeer.did = new Promise(resolve => {
            const el = document.createElement('script')
            el.src = 'https://unpkg.com/simple-peer@9.11.0/simplepeer.min.js'
            document.head.append(el)
            const id = setInterval(() => {
                if (self.SimplePeer) clearInterval(id), resolve(self.SimplePeer)
            }, 50)
        })
    }



    class InternetSensor extends sn.Sensor {
        static docs() { return `Extends this sensor network over the Internet, to control others.

Options:
- \`iceServers = []\`: the [list](https://gist.github.com/mondain/b0ec1cf5f60ae726202e) of [ICE servers](https://developer.mozilla.org/en-US/docs/Web/API/RTCIceServer/urls) (Interactive Connectivity Establishment).
- \`signaler\`: a convenience: on construction, does \`sensor.signal(signaler(sensor))\` for you. Props of \`sn.Handler.Internet\` go well here.

Methods:
- \`signal(metaChannel: { send(string), close(), onopen, onmessage, onclose }, maxCells=65536)\` for manually establishing a connection: on an incoming connection, someone must notify us of it so that negotiation of a connection can take place, for example, [of a \`WebSocket\` which can be passed directly](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket).

Browser compatibility: [Edge 79.](https://developer.mozilla.org/en-US/docs/Web/API/RTCPeerConnection/createDataChannel)

Imports [100 KiB](https://github.com/feross/simple-peer) on use.
` }
        static options() {
            return {
                iceServers: {
                    ['None']: () => [],
                    ['Some']: () => [{urls:'stun: stun.l.google.com:19302'}, {urls:'stun: stunserver.org:3478'}],
                },
                signaler: {
                    ['None']: () => undefined,
                    ['Browser tabs']: () => sn.Handler.Internet.broadcastChannel,
                    ['JS console (F12)']: () => sn.Handler.Internet.consoleLog,
                },
            }
        }
        static tests() {
            return [
                [
                    "Quantization error is correct, f32",
                    checkError(65536, 0),
                    true,
                ],
                [
                    "Quantization error is correct, u8",
                    checkError(65536, 1),
                    true,
                ],
                [
                    "Quantization error is correct, u16",
                    checkError(65536, 2),
                    true,
                ],
            ]
        }
        static bench() { // Extends a virtual sensor/handler pair over a localhost connection.
            const cellCounts = new Array(10).fill().map((_,i) => (i+1)*10)
            return cellCounts.map(river) // See how throughput changes with input size.
            function river(cells) {
                const dataSize = 64
                const iceServers = []
                function onSensorFeedback(feedback) {
                    if (feedback) {
                        sn.meta.metric('feedback-is-correct fraction', +(this.paused || Math.abs(feedback[0] - -.97) < .01))
                        feedback.fill(.5439828952837) // "Read" it.
                        sn._deallocF32(feedback) // Reuse it.
                    }
                }
                return function start() {
                    const signal1 = {
                        send(msg) { signal2.onmessage && signal2.onmessage({data:msg}) },
                    }, signal2 = {
                        send(msg) { signal1.onmessage && signal1.onmessage({data:msg}) },
                    }
                    const aFrom = new sn.Sensor({
                        channel: 'a',
                        name: ['remote', 'data', 'source'],
                        values: cells*dataSize,
                        onValues(data) {
                            data.fill(.99), this.sendCallback(onSensorFeedback, data)
                        },
                    })
                    const aTo = new sn.Handler.Internet({ channel:'a', iceServers, signaler: () => signal1, untrustedWorkaround: true })
                    const bFrom = new sn.Sensor.Internet({ channel:'b', iceServers })
                    const bTo = new sn.Handler({
                        channel: 'b',
                        dataSize,
                        onValues(then, {data, noData}, feedback) {
                            try {
                                data && data.length && sn._assert(Math.abs(data[64] - .99) < .01, data[64])
                                data && data.fill(.489018922485) // "Read" it.
                                if (feedback) feedback.fill(-.97)
                            } finally { then() }
                        },
                    })
                    bFrom.signal(signal2)
                    signal1.onopen && signal1.onopen()
                    signal2.onopen && signal2.onopen()
                    return function stop() {
                        aFrom.pause()
                        aTo.pause(), bFrom.pause()
                        bTo.pause()
                    }
                }
            }
        }
        resume(opts) {
            if (opts) {
                this.iceServers = opts.iceServers || []
                opts.onValues = this.onValues
                opts.values = 0, opts.emptyValues = 0, opts.name = []
                if (!this._data) { // Only once.
                    this._data = [], this._feedback = [] // The main adjustable data queue.
                    const auto = opts.signaler
                    if (typeof auto == 'function')
                        Promise.resolve().then(() => auto(this))
                }
            }
            return super.resume(opts)
        }
        signal(metaChannel, maxCells=65536) {
            if (!metaChannel) return
            sn._assert(typeof metaChannel.send == 'function')
            let peer, inMetaData = [], outMetaData = []
            const signal = data => {
                if (outMetaData) outMetaData.push(data)
                else metaChannel.send(data)
            }
            metaChannel.onopen = evt => {
                outMetaData && outMetaData.forEach(data => metaChannel.send(data))
                outMetaData = null
            }
            metaChannel.onmessage = evt => {
                if (typeof evt.data != 'string') return console.warn('dropping', evt)
                const d = JSON.parse(evt.data)
                peer ? peer.signal(d) : inMetaData.push(d)
            }
            // Ignore `metaChannel.onclose`.
            importSimplePeer().then(SimplePeer => {
                peer = new SimplePeer({
                    trickle: !metaChannel.noTrickle,
                    config: {iceServers:this.iceServers},
                })
                peer.on('close', () => { peer = null, metaChannel.close() })
                peer.on('error', console.error)
                peer.on('signal', data => {
                    signal(JSON.stringify(data))
                })
                const packer = messagePacker(peer)
                const feedback = (fb, bpv, partSize, cellShape) => { // Float32Array, 0|1|2
                    const quant = quantize(fb, bpv)
                    const header = 2 + 4 + 2+4*cellShape.length
                    const bytes = new Uint8Array(header + quant.length)
                    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
                    dv.setUint16(0, bpv)
                    dv.setUint32(2, partSize)
                    dv.setUint16(6, cellShape.length)
                    let offset = 8
                    for (let i = 0; i < cellShape.length; ++i)
                        dv.setUint32(offset, cellShape[i]), offset += 4
                    sn._assert(offset === header)
                    bytes.set(quant, header)
                    if (!this.paused) packer(bytes)
                }
                peer.on('data', messageUnpacker((data, packetId) => {
                    if (data) {
                        sn._assert(data instanceof Uint8Array)
                        if (data.length < 2) return
                        const dv = new DataView(data.buffer, data.byteOffset, data.byteLength)
                        // Read the shape.
                        const cells = dv.getUint32(0)
                        const partSize = dv.getUint32(4)
                        const cellShapeLen = dv.getUint16(8)
                        if (cellShapeLen !== 3) return
                        const cellShape = allocArray(cellShapeLen)
                        let offset = 10
                        for (let i = 0; i < cellShapeLen; ++i)
                            cellShape[i] = dv.getUint32(offset), offset += 4
                        const bpv = dv.getUint16(offset);  offset += 2
                        if (bpv !== 0 && bpv !== 1 && bpv !== 2) return
                        const namedSize = cells * cellShape.reduce((a,b) => a+b)
                        const namedBytes = data.subarray(offset, offset += namedSize * (bpv||4))
                        const noDataBytes = data.subarray(offset, offset += Math.ceil(cells / 8))
                        const noFeedbackBytes = data.subarray(offset, offset += Math.ceil(cells / 8))
                        if (namedBytes.length > maxCells * (namedSize / cells | 0) * (bpv+1))
                            namedBytes = namedBytes.subarray(0, maxCells * (namedSize / cells | 0) * (bpv+1))
                        const rawData = unquantize(namedBytes, bpv)
                        const rawError = unquantizeError(rawData.length, bpv)
                        const noData = fromBits(noDataBytes)
                        const noFeedback = fromBits(noFeedbackBytes)
                        const a = allocArray(9)
                        ;[a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]] = [partSize, cells, cellShape, bpv, rawData, rawError, noData, noFeedback, feedback]
                        this._data.push(a)
                    } else { // Drop.
                        const a = allocArray(9)
                        a.fill(undefined), a[8] = feedback
                        this._data.push(a)
                    }
                }))
                if (inMetaData.length) {
                    for (let data of inMetaData)
                        peer.signal(data)
                    inMetaData.length = 0
                }
            })
        }
        onValues(data) {
            sn._deallocF32(data)
            this.sendRawCallback(this.onFeedback, this._name, this._unname)
            this._data.length = 0
        }
        _name({cellShape, partSize, summary}, namer, packet, then, unname) {
            // Copy all data points into the actual data stream.
            //   `this.onFeedback` will be called for each sending.
            //   Besides, what if there are currently no handlers and no cell-shapes?...
            for (let i = 0; i < this._data.length; ++i) {
                let [realPartSize, cells, realCellShape, bpv, rawData, rawError, noData, noFeedback, feedback] = this._data[i]
                if (rawData === undefined) { // Dropped packet. Synthetic data.
                    cells = 0
                    rawData = sn._allocF32(0), rawError = null
                    noData = noFeedback = true
                }
                const cellSize = cellShape.reduce((a,b)=>a+b), namedSize = cells * cellSize
                const realCellSize = realCellShape && realCellShape.reduce((a,b)=>a+b)
                const namedV = sn._allocF32(namedSize), namedE = rawError && sn._allocF32(namedSize)
                for (let c = 0; c < cells; ++c) { // Na??vely reshape.
                    const offsetTarg = c * cellSize, offsetReal = c * realCellSize
                    for (let i = 0; i < cellSize; ++i)
                        namedV[offsetTarg + i] = rawData[offsetReal + i]
                }
                packet.send(this, then, unname, namedV, namedE || null, noData || true, noFeedback || true)
                this._feedback.push(this._data[i])
            }
        }
        _unname(namer, allFeedback, fbOffset, dataLen) {
            const vals = sn._allocF32(allFeedback.length) // Technically unnecessary (if `main.js` doesn't deallocate `feedbackData` but `onFeedback` can decide), but unless this becomes a bottleneck, better safe than sorry.
            vals.set(allFeedback)
            return vals
        }
        onFeedback(feedbackData, cellShape, partSize) {
            // Send feedback back.
            const fbPoint = this._feedback.shift()
            if (!fbPoint) return
            let [realPartSize, cells, realCellShape, bpv, rawData, rawError, noData, noFeedback, feedback] = fbPoint
            if (rawData === undefined) { // Data was dropped, so drop feedback.
                const f = sn._allocF32(1)
                f[0] = 0
                feedback(f, bpv, partSize, cellShape)
            } else { // Respond to the packet.
                const cellSize = cellShape.reduce((a,b)=>a+b), namedSize = cells * cellSize
                const realCellSize = realCellShape.reduce((a,b)=>a+b)
                const back = sn._allocF32(namedSize)
                if (feedbackData)
                    for (let c = 0; c < cells; ++c) { // Na??vely reshape back.
                        const offsetTarg = c * cellSize, offsetReal = c * realCellSize
                        for (let i = 0; i < cellSize; ++i)
                            back[offsetReal + i] = feedbackData[offsetTarg + i]
                    }
                else
                    back.fill(0)
                feedback(back, bpv, partSize, cellShape)
            }
            feedbackData && sn._deallocF32(feedbackData)
            deallocArray(fbPoint)
            realCellShape && deallocArray(realCellShape)
            noData && deallocArray(noData)
            noFeedback && deallocArray(noFeedback)
        }
    }
    class InternetHandler extends sn.Handler {
        static docs() { return `Makes this environment a remote part of another sensor network, to be controlled.

Options:
- \`iceServers = []\`: the [list](https://gist.github.com/mondain/b0ec1cf5f60ae726202e) of [ICE servers](https://developer.mozilla.org/en-US/docs/Web/API/RTCIceServer/urls) (Interactive Connectivity Establishment).
- \`signaler = sn.Handler.Internet.broadcastChannel\`: creates the channel over which negotiation of connections takes place. When called, constructs \`{ send(Uint8Array), close(), onopen, onmessage, onclose }\`, for example, [\`new WebSocket(url)\`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket).
- \`bytesPerValue=0\`: 0 to transmit each value as float32, 1 to quantize as uint8, 2 to quantize as uint16. 1 is max-compression min-precision; 0 is the opposite.
- \`autoresume = true\`: whether the connection closing will trigger an attempt to re-establish it.
- \`untrustedWorkaround = false\`: if set, will request a microphone stream and do nothing with it, so that a WebRTC connection can connect. The need for this was determined via alchemy, so its exact need-to-use is unknown.

Imports [100 KiB](https://github.com/feross/simple-peer) on use.
` }
        static options() {
            return {
                iceServers: {
                    ['None']: () => [],
                    ['Some']: () => [{urls:'stun: stun.l.google.com:19302'}, {urls:'stun: stunserver.org:3478'}],
                },
                signaler: {
                    ['Browser tabs']: () => sn.Handler.Internet.broadcastChannel,
                    ['JS console (F12)']: () => sn.Handler.Internet.consoleLog,
                },
                bytesPerValue: {
                    ['float32 (4?? size)']: () => 0,
                    ['uint16 (2?? size)']: () => 2,
                    ['uint8 (1?? size)']: () => 1,
                },
                autoresume: {
                    Yes: true,
                    No: false,
                },
                untrustedWorkaround: {
                    No: false,
                    Yes: true,
                },
            }
        }
        pause(inResume) {
            if (!inResume) {
                if (this.peer) Promise.resolve(this.peer).then(p => (p.destroy(), this.peer = null))
                if (this._feedback && this._feedback.length) {
                    for (let [feedback, then, start] of this._feedback)
                        feedback && feedback.fill(0), then()
                    this._feedback.length = 0
                    this._dataToSend && (this._dataToSend.length = 0)
                }
            }
            return super.pause()
        }
        resume(opts) {
            if (opts) {
                const bpv = opts.bytesPerValue || 0
                sn._assert(bpv === 0 || bpv === 1 || bpv === 2)
                opts.onValues = this.onValues
                this.autoresume = opts.autoresume !== undefined ? opts.autoresume : true
                this.iceServers = opts.iceServers || []
                this.signaler = opts.signaler || sn.Handler.Internet.broadcastChannel
                this.untrustedWorkaround = !!opts.untrustedWorkaround
                if (opts !== this._opts) this._opts = Object.assign(Object.create(null), opts)
                if (!this._feedback) { // Only init once.
                    this._feedback = [] // What receiving a feedback-packet will have to do.
                    this.bytesPerValue = bpv
                    this._dataSend = null, this._dataToSend = []
                }
            }
            this.getPeer()
            // Haven't seen a way to update the ICE-servers list in SimplePeer, so no `.setConfiguration`.
            return super.resume(opts)
        }
        getPeer() {
            // Connects to another sensor network through WebRTC.
            if (this.metaChannel == null) {
                const mc = this.metaChannel = this.signaler()
                this._signal = new Promise((resolve, reject) => {
                    mc.onopen = evt => { this._signal = null, resolve() }
                    mc.onmessage = evt => {
                        if (this.peer == null) return console.warn('dropping signal', evt.data)
                        if (typeof evt.data != 'string') return console.warn('dropping signal', evt.data)
                        const d = JSON.parse(evt.data)
                        this.peer instanceof Promise ? this.peer.then(p => p.signal(d)) : this.peer.signal(d)
                    }
                    mc.onclose = evt => { reject(), this._signal = this.metaChannel = null } // When closed, reopen.
                })
            }
            if (this.peer == null) {
                this._dataSend = null
                const p = this.untrustedWorkaround ? navigator.mediaDevices.getUserMedia({audio:true}) : Promise.resolve()
                this.peer = p.then(() => {throw null}).catch(() => {
                    return importSimplePeer().then(SimplePeer => {
                        const peer = this.peer = new SimplePeer({
                            trickle: !this.metaChannel.noTrickle,
                            initiator: true,
                            config: {iceServers:this.iceServers},
                            channelConfig: {
                                ordered: false, // Unordered
                                maxRetransmits: 0, // Unreliable
                            },
                        })
                        peer.on('close', () => {
                            this.pause()
                            if (this.autoresume) setTimeout(() => this.resume(this._opts), 1000)
                            else console.log('sn.Handler.Internet: close')
                        })
                        peer.on('error', console.error)
                        peer.on('signal', data => {
                            signal.call(this, JSON.stringify(data))
                        })
                        peer.on('connect', () => {
                            const packer = messagePacker(peer)
                            this._dataSend = ({data, noData, noFeedback, cellShape, partSize}, bpv) => {
                                // cells 4b, partSize 4b, cellShapeLen 2b, i ?? cellShapeItem 4b, bpv 2b, quantized data, noData bits, noFeedback bits.
                                const cells = data.length / cellShape.reduce((a,b)=>a+b) | 0
                                const totalSize = 4 + 4 + 2+4*cellShape.length + 2 + (bpv||4) * data.length + 2*Math.ceil(cells / 8)
                                const bytes = new Uint8Array(totalSize), dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
                                let offset = 0
                                dv.setUint32(offset, cells), offset += 4
                                dv.setUint32(offset, partSize), offset += 4
                                dv.setUint16(offset, cellShape.length), offset += 2
                                for (let i = 0; i < cellShape.length; ++i)
                                    dv.setUint32(offset, cellShape[i]), offset += 4
                                dv.setUint16(offset, bpv), offset += 2
                                const qData = quantize(data, bpv)
                                const bNoData = toBits(noData)
                                const bNoFeedback = toBits(noFeedback)
                                bytes.set(qData, offset), offset += qData.length
                                bytes.set(bNoData, offset), offset += bNoData.length
                                bytes.set(bNoFeedback, offset), offset += bNoFeedback.length
                                sn._assert(offset === totalSize, "totalSize miscounts")
                                if (!this.paused) packer(bytes)
                            }
                            if (this._dataToSend.length) {
                                for (let [input, bpv] of this._dataToSend)
                                    this._dataSend(input, bpv)
                                this._dataToSend.length = 0
                            }
                        })
                        // `data` is feedback here.
                        peer.on('data', messageUnpacker((bytes, packetId) => {
                            // bpv 2b, partSize 4b, cellShapeLen 2b, i ?? cellShapeItem 4b, quantized feedback.
                            const fb = this._feedback.shift()
                            if (!fb) return console.warn('got feedback that we did not give data for')
                            let gotFeedback
                            if (bytes && bytes.length > 8) {
                                const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
                                let offset = 0
                                const bpv = dv.getUint16(offset);  offset += 2
                                const partSize = dv.getUint32(offset);  offset += 4
                                const cellShape = allocArray(dv.getUint16(offset));  offset += 2
                                for (let i = 0; i < cellShape.length; ++i)
                                    cellShape[i] = dv.getUint32(offset), offset += 4
                                gotFeedback = unquantize(bytes.subarray(offset), bpv)
                                // Handle what we got.
                                if (partSize !== this._remotePartSize || !arrayEqual(cellShape, this._remoteCellShape)) {
                                    this._remotePartSize = partSize
                                    this._remoteCellShape = cellShape
                                    this._opts.partSize = partSize
                                    this._opts.userParts = cellShape[0] / partSize | 0
                                    this._opts.nameParts = cellShape[1] / partSize | 0
                                    this._opts.dataSize = cellShape[cellShape.length-1]
                                    this.resume(this._opts)
                                }
                            }
                            const [feedback, then, start] = fb
                            if (feedback && gotFeedback) {
                                if (gotFeedback.length > feedback.length)
                                    gotFeedback = gotFeedback.subarray(0, feedback.length)
                                feedback.set(gotFeedback)
                                if (feedback.length > gotFeedback.length)
                                    feedback.fill(0, gotFeedback.length)
                            } else if (feedback) feedback.fill(0)
                            sn.meta.metric('processing latency, ms', performance.now() - start)
                            then()
                        }))
                        function signal(data) {
                            // Send `data`, waiting for the signaling channel to open.
                            if (!this.metaChannel) return console.warn('dropping signal', data)
                            if (this._signal) this._signal.then(() => this.metaChannel.send(data))
                            else this.metaChannel.send(data)
                        }
                        return peer
                    })
                })
            }
        }

        onValues(then, input, feedback) {
            if (this._dataSend)
                this._dataSend(input, this.bytesPerValue)
            else
                this._dataToSend.push([input, this.bytesPerValue])
            this._feedback.push([feedback, then, performance.now()])
        }
    }
    A(InternetHandler, {
        broadcastChannel: A(function signalViaBC(sensor=null) {
            const tabId = signalViaBC.id || (signalViaBC.id = String(Math.random()).slice(2) + String(Math.random()).slice(2))
            const bc = new BroadcastChannel('sn-internet-broadcast-channel')
            let interval, objs
            const obj = {
                send(msg, dst=null) { bc.postMessage({ tabId, dst, msg }) },
                close() {
                    clearInterval(interval)
                    bc.close()
                    objs && objs.forEach(o => o.onclose && o.onclose())
                },
                onopen:null, onmessage:null, onclose:null,
            }
            bc.onmessage = evt => {
                const d = evt.data
                if (d.dst == null || d.dst === tabId)
                    if (d.tabId !== tabId)
                        obj.onmessage && obj.onmessage({data:d.msg}, d.tabId)
            }
            setTimeout(() => obj.onopen && obj.onopen(), 0)
            if (sensor) {
                // One connection per connected tab.
                objs = Object.create(null) // tabId ??? {send(msg), ???}
                interval = setInterval(() => { // Garbage-collect.
                    const now = performance.now()
                    for (let tabId of Object.keys(objs)) {
                        const o = objs[tabId]
                        if (now - o.lastTouched > 60000)
                            delete objs[tabId], o.onclose && o.onclose()
                    }
                }, 60000)
                obj.onmessage = (evt, tabId) => {
                    if (!objs[tabId]) {
                        sensor.signal(objs[tabId] = {
                            tabId,
                            lastTouched: 0,
                            send(msg) { obj.send(msg, this.tabId) },
                            close() { delete objs[this.tabId] },
                            onopen:null, onmessage:null, onclose:null,
                        })
                        setTimeout(() => objs[tabId].onopen && objs[tabId].onopen(), 0)
                    }
                    objs[tabId].lastTouched = performance.now()
                    objs[tabId].onmessage && objs[tabId].onmessage(evt)
                }
            } else return obj
        }, {
            docs:`Connects to all browser tabs, one connection per handler. Signals via a [\`BroadcastChannel\`](https://developer.mozilla.org/en-US/docs/Web/API/Broadcast_Channel_API). (Not in Safari.)`,
        }),
        consoleLog: A(function signalViaConsole(sensor=null) {
            if (!signalViaConsole.did) {
                console.log("Carry around WebRTC signals manually, through the JS console.")
                console.log("    Please triple-click and copy each message, then paste at the other end.")
                signalViaConsole.did = true
            }
            if (sensor) {
                self.internetSensor = msg => {
                    const obj = {
                        send(msg) { console.log(`internetHandler(${msg})`) },
                        close() {},
                        noTrickle: true,
                    }
                    setTimeout(() => {
                        obj.onopen && obj.onopen()
                        obj.onmessage && obj.onmessage({data: JSON.stringify(msg)})
                    }, 0)
                    sensor.signal(obj)
                }
            } else {
                const obj = {
                    send(msg) { console.log(`internetSensor(${msg})`) },
                    close() {},
                    noTrickle: true,
                }
                setTimeout(() => obj.onopen && obj.onopen(), 0)
                self.internetHandler = msg => obj.onmessage && obj.onmessage({data: JSON.stringify(msg)})
                return obj
            }
        }, {
            docs:`[\`console.log\`](https://developer.mozilla.org/en-US/docs/Web/API/Console/log) and \`self.internetSensor(messageString)\` and \`self.internetHandler(messageString)\` is used to make the user carry signals around.`,
        }),
        webSocket: A(function signalViaWS(url) {
            return sensor => new WebSocket(url)
        }, {
            docs:`Signals via a [\`WebSocket\`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket). Have to pass it the URL before passing it as the \`signaler\` option.`,
        }),
    })
    Object.defineProperty(InternetSensor, 'name', {value:'Internet', configurable:true, writable:true})
    Object.defineProperty(InternetHandler, 'name', {value:'Internet', configurable:true, writable:true})
    return {
        sensor: InternetSensor,
        handler: InternetHandler,
    }
    function allocArray(n) { return arrayCache.length ? (arrayCache[arrayCache.length-1].length = n, arrayCache.pop()) : new Array(n) }
    function deallocArray(a) { Array.isArray(a) && arrayCache.length < 16 && (a.length = 0, arrayCache.push(a)) }
    function messagePacker(channel, maxPacketBytes=16*1024*1024) {
        // Sends maybe-large messages over unordered & unreliable channels.
        // Use `messageUnpacker` to re-assemble messages on the other end.
        let nextId = 0
        const maxPartLen = 65536, partBuf = new Uint8Array(maxPartLen)
        const partView = new DataView(partBuf.buffer, partBuf.byteOffset, partBuf.byteLength)
        return function message(data) { // Uint8Array
            sn._assert(data.length <= maxPacketBytes, "Message is too long!")
            if (channel.readyState === 'closing' || channel.readyState === 'closed') return
            const partSize = maxPartLen - 4
            const parts = Math.ceil((data.length - 2) / partSize)
            let sent = 0
            for (let part = 0, atData = 0; part < parts; ++part) {
                partView.setUint16(0, nextId)
                partView.setUint16(2, part)
                if (part === 0) partView.setUint16(4, parts)
                const header = (part === 0 ? 6 : 4)
                let i
                for (i = header; i < partBuf.length && atData < data.length; ++i, ++atData)
                    partBuf[i] = data[atData]
                sent += i
                try {
                    channel.send(i >= partBuf.length ? partBuf : partBuf.subarray(0, i))
                } catch (err) { console.warn('Packet-part-sending failed', err) }
            }
            ++nextId, nextId >= 65536 && (nextId = 0)
            sn.meta.metric('sent, bytes/step', sent)
        }
    }
    function messageUnpacker(onpacket, timeoutMs=50, maxPacketBytes=16*1024*1024) {
        // Set the result of this as `dataChannel.onmessage`.
        // `onpacket(null | Uint8Array, packetId)` will be called with full packet data, in the ID-always-increasing-by-1 order, or `null` if taking too long.
        const packs = Object.create(null) // id ??? [remainingParts, ...partData]
        let nextId = 0, prevAt = null
        function superseded() {
            for (let i = nextId + 1; i <= nextId + 16; ++i)
                if (packs[i & 65535]) return true
            return false
        }
        return function onmessage(evt) {
            // (Allows arbitrary-length packets to be encoded in `maxLength`-sized network packets, unordered and unreliable made ordered and reliable.)
            // (4 bytes of overhead per network packet, plus 2 bytes per actual packet.)
            if (evt instanceof Uint8Array) // SimplePeer
                sn._assert(!evt.byteOffset), evt = {data:evt.buffer}
            if (evt.data instanceof Blob) return evt.data.arrayBuffer().then(ab => onmessage({data:ab}))
            sn._assert(evt.data instanceof ArrayBuffer, "Got non-ArrayBuffer data")
            const dv = new DataView(evt.data)
            const id = dv.getUint16(0), part = dv.getUint16(2)
            if (id < nextId) return // You're too late.
            let p
            if (part === 0) { // Packet start.
                const parts = dv.getUint16(4)
                if (!parts || parts * (dv.length-6) > maxPacketBytes) return
                if (packs[id]) deallocArray(packs[id])
                p = packs[id] = allocArray(1 + parts)
                p[0] = parts
                p[0] && !p[1+part] && --p[0], p[1+part] = dv
            } else { // The rest of the packet.
                if ((1+part+1) * (dv.length-4) > maxPacketBytes) return
                if (!packs[id]) packs[id] = allocArray(1 + part + 1), packs[id][0] = null
                p = packs[id]
                p[0] && !p[1+part] && --p[0], p[1+part] = dv
            }
            if (id > nextId && prevAt == null) prevAt = performance.now()
            // Send off the next packet, in-order or on-timeout if there are packets after this one.
            let tooLong = prevAt != null && performance.now() - prevAt > timeoutMs && superseded()
            while (packs[nextId] && packs[nextId][0] === 0 || tooLong) {
                const p = packs[nextId]
                let ok = Array.isArray(p) && p[0] === 0
                if (ok)
                    for (let i = 1; i < p.length; ++i)
                        if (!p[i]) { ok = false;  break }
                if (ok) { // Copy parts into one buffer.
                    // (A copy. And generates garbage. As garbage as JS is.)
                    let len = 0
                    for (let i = 1; i < p.length; ++i)
                        len += p[i].byteLength - (i === 1 ? 6 : 4)
                    const b = new Uint8Array(len)
                    for (let i = 1, off = 0; i < p.length; ++i) {
                        const header = (i === 1 ? 6 : 4)
                        const sublen = p[i].byteLength - header
                        for (let j = 0; j < sublen; ++j)
                            b[off + j] = p[i].getUint8(header + j)
                        off += sublen
                    }
                    deallocArray(packs[nextId]), packs[nextId] = null
                    onpacket(b, nextId)
                } else {
                    if (packs[nextId]) deallocArray(packs[nextId]), packs[nextId] = null
                    onpacket(null, nextId)
                }
                nextId = (nextId + 1) & 65535
                tooLong = false, prevAt = null
            }
        }
    }
    function quantize(f32a, bpv = 0) {
        // From floats to an array of bytes, quantized to lower-but-still-`-1???1` resolutions.
        sn._assert(f32a instanceof Float32Array)
        if (!bpv) return bigEndian(new Uint8Array(f32a.buffer, f32a.byteOffset, f32a.byteLength), bpv)
        sn._assert(bpv === 1 || bpv === 2)
        const r = bpv === 1 ? new Uint8Array(f32a.length) : new Uint16Array(f32a.length)
        const scale = bpv === 1 ? 255 : 65535
        for (let i = 0; i < r.length; ++i)
            r[i] = Math.max(0, Math.min(Math.round((f32a[i]+1)/2 * scale), scale))
        return bpv === 1 ? r : bigEndian(new Uint8Array(r.buffer, r.byteOffset, r.byteLength), bpv, true)
    }
    function unquantize(a, bpv = 0) {
        sn._assert(a instanceof Uint8Array)
        if (!bpv) {
            a = bigEndian(a, bpv)
            return new Float32Array(a.buffer, a.byteOffset, a.byteLength / 4 | 0)
        }
        sn._assert(bpv === 1 || bpv === 2, bpv)
        if (bpv === 2) a = new Uint16Array(bigEndian(a, bpv).buffer)
        const r = new Float32Array(a.length)
        const scale = bpv === 1 ? 255 : 65535
        for (let i = 0; i < r.length; ++i)
            r[i] = a[i]/scale * 2 - 1
        return r
    }
    function unquantizeError(a, bpv = 0) {
        // `a`: pre-existing error; un/quantization error is added to that.
        if (!bpv) return typeof a == 'number' ? new Float32Array(a).fill(-1) : a
        const scale = bpv === 1 ? 255 : 65535
        if (typeof a == 'number') a = new Float32Array(a), a.fill(1 / scale - 1)
        else for (let i = 0; i < a.length; ++i) a[i] += 1 / scale
        return a
    }
    function checkError(a, bpv = 0) {
        if (typeof a == 'number') {
            a = new Float32Array(a)
            for (let i = 0; i < a.length; ++i) a[i] = Math.random()*2-1
        }
        const b = unquantize(quantize(a, bpv), bpv)
        const e = unquantizeError(a.length, bpv)
        sn._assert(a.length === b.length, "Unequal lengths")
        for (let i = 0; i < a.length; ++i)
            if (!(Math.abs(a[i] - b[i]) <= e[i]+1))
                return a[i] + " ??? " + b[i]
        return true
    }
    function bigEndian(a, bpv, inPlace = false) {
        // `a` is copied unless `inPlace`.
        if (bigEndian.bigEnd === undefined) {
            const x = new ArrayBuffer(2), y = new Uint16Array(x), z = new Uint8Array(x)
            y[0] = 0x0102
            bigEndian.bigEnd = z[0] === 0x01
        }
        sn._assert(a instanceof Uint8Array, "Bad byte-array")
        if (bigEndian.bigEnd || bpv === 1) return a
        if (!inPlace) a = new Uint8Array(a)
        if (bpv === 2)
            for (let i = 0; i < a.length; i += 2)
                [a[i+0], a[i+1]] = [a[i+1], a[i+0]]
        else if (bpv === 0 || bpv === 4)
            for (let i = 0; i < a.length; i += 4)
                [a[i+0], a[i+1], a[i+2], a[i+3]] = [a[i+3], a[i+2], a[i+1], a[i+0]]
        return a
    }
    function toBits(a) { // Array<bool> ??? Uint8Array
        const b = new Uint8Array(Math.ceil(a.length / 8))
        for (let i = 0; i < b.length; ++i) {
            const j = 8*i
            b[i] = (a[j+0]<<7) | (a[j+1]<<6) | (a[j+2]<<5) | (a[j+3]<<4) | (a[j+4]<<3) | (a[j+5]<<2) | (a[j+6]<<1) | (a[j+7]<<0)
        }
        return b
    }
    function fromBits(b) { // Uint8Array ??? Array<bool>
        const a = allocArray(b.length * 8) // The length is a bit inexact, which is not important for us.
        for (let i = 0; i < b.length; ++i) {
            const j = 8*i
            a[j+0] = !!(b[i] & (1<<7))
            a[j+1] = !!(b[i] & (1<<6))
            a[j+2] = !!(b[i] & (1<<5))
            a[j+3] = !!(b[i] & (1<<4))
            a[j+4] = !!(b[i] & (1<<3))
            a[j+5] = !!(b[i] & (1<<2))
            a[j+6] = !!(b[i] & (1<<1))
            a[j+7] = !!(b[i] & (1<<0))
        }
        return a
    }
    function arrayEqual(a,b) {
        if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false
        for (let i = 0; i < a.length; ++i) if (a[i] !== b[i]) return false
        return true
    }
}