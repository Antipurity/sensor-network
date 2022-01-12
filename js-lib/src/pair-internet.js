export default function init(sn) {
    // (Reading/writing may be quite unoptimized here.)
    const A = Object.assign
    const arrayCache = []



    function importSimplePeer() { // Manual use of WebRTC is just not working out.
        // TODO: In docs, mention that we import this on use.
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
        static docs() { return `Extends this network over the Internet, to control others.

Methods:
- \`signal(metaChannel: { send(string), close(), onopen, onmessage, onclose }, maxCells=65536)\`: on an incoming connection, someone must notify us of it so that negotiation of a connection can take place, for example, [over a \`WebSocket\`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket).

Options:
- \`iceServers = []\`: the [list](https://gist.github.com/mondain/b0ec1cf5f60ae726202e) of [ICE servers](https://developer.mozilla.org/en-US/docs/Web/API/RTCIceServer/urls) (Interactive Connectivity Establishment).

Browser compatibility: [Edge 79.](https://developer.mozilla.org/en-US/docs/Web/API/RTCPeerConnection/createDataChannel)
` }
        static options() {
            return {
                iceServers: {
                    ['None']: () => [],
                    ['Some']: () => [{urls:'stun: stun.l.google.com:19302'}, {urls:'stun: stunserver.org:3478'}],
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
            const cellCounts = new Array(1).fill().map((_,i) => (i+1)*10) // TODO: 10, not 1.
            return cellCounts.map(river) // See how throughput changes with input size.
            function river(cells) {
                const dataSize = 64
                const iceServers = []
                function onSensorFeedback(feedback) {
                    if (feedback)
                        console.log(feedback[0]), // TODO: Why is this mostly 0s with only occasional -1s? (…And some rare .5439 from here, and .4890 from handler-data-filling. ...And, wait, it's supposed to be -.97, not -1, so it literally never succeeds in delivering feedback.)
                        feedback.fill(.5439828952837), // "Read" it.
                        sn._deallocF32(feedback) // Reuse it.
                }
                return function start() {
                    const signal1 = {
                        send(msg) { 0&&console.log('s1→s2 msg', msg, !!signal2.onmessage), signal2.onmessage && signal2.onmessage({data:msg}) }, // TODO:
                    }, signal2 = {
                        send(msg) { 0&&console.log('s2→s1 msg', msg, !!signal1.onmessage), signal1.onmessage && signal1.onmessage({data:msg}) }, // TODO:
                    }
                    const aFrom = new sn.Sensor({
                        channel: 'a',
                        name: ['remote', 'data', 'source'],
                        values: cells*dataSize,
                        onValues(data) {
                            data.fill(1), this.sendCallback(onSensorFeedback, data)
                        },
                    })
                    const aTo = new sn.Handler.Internet({ channel:'a', iceServers, signaler: () => signal1, untrustedWorkaround: true })
                    const bFrom = new sn.Sensor.Internet({ channel:'b', iceServers })
                    const bTo = new sn.Handler({
                        channel: 'b',
                        dataSize,
                        onValues(then, {data}, feedback) {
                            try {
                                data.fill(.489018922485) // "Read" it.
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
                opts.values = 0, opts.emptyValues = 0, opts.name = []
                if (!this._data) { // Only once.
                    this._data = [], this._feedback = [] // The main adjustable data queue.
                }
            }
            return super.resume(opts)
        }
        signal(metaChannel, maxCells=65536) { // TODO: Maybe, we should really transition to a `onsignal:fn(data)` + `.signal(data)` model, to make things more intuitive for WebRTC-users? Not like it adds more than like 2 lines of code for a WebSocket connection.
            // (Could double-init because getting `SimplePeer` is async.)
            let peer, inMetaData = [], outMetaData = []
            const signal = data => {
                if (outMetaData) outMetaData.push(data)
                else metaChannel.send(data)
            }
            metaChannel.onopen = evt => {
                outMetaData && outMetaData.forEach(data => metaChannel.send(data))
                outMetaData = null
                metaChannel.onmessage = evt => {
                    if (typeof evt.data != 'string') return
                    const d = JSON.parse(evt.data)
                    peer ? peer.signal(d) : inMetaData.push(d)
                }
                // Ignore `metaChannel.onclose`.
            }
            importSimplePeer().then(SimplePeer => {
                peer = new SimplePeer({
                    config: {iceServers:this.iceServers},
                })
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
                    // console.log('sensor sends feedback', bytes.length) // TODO:
                    packer(bytes)
                }
                peer.on('data', messageUnpacker((data, packetId) => {
                    // console.log('sensor receives data', data && data.length) // TODO:
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
                        const nameSize = cells * (cellShape.reduce((a,b) => a+b) - cellShape[cellShape.length-1])
                        const nameBytes = data.subarray(offset, offset += nameSize)
                        const noDataBytes = data.subarray(offset, offset += Math.ceil(cells / 8))
                        const noFeedbackBytes = data.subarray(offset, offset += Math.ceil(cells / 8))
                        if (nameBytes.length > maxCells * (nameSize / cells | 0) * (bpv+1))
                            nameBytes = nameBytes.subarray(0, maxCells * (nameSize / cells | 0) * (bpv+1))
                        const rawData = unquantize(nameBytes, bpv)
                        const rawError = unquantizeError(rawData.length, bpv)
                        const noData = fromBits(noDataBytes)
                        const noFeedback = fromBits(noFeedbackBytes)
                        const a = allocArray(9)
                        ;[a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]] = [partSize, cells, cellShape, bpv, rawData, rawError, noData, noFeedback, feedback]
                        // console.log('sensor prepares to send feedback', 4+24 + 4 * cells * cellShape.reduce((a,b)=>a+b)) // TODO:
                        this._data.push(a)
                        this.sendRawCallback(this.onFeedback, this._name, this._unname)
                    } else {
                        const a = allocArray(9)
                        a.fill(undefined), a[8] = feedback
                        console.log('sensor drops feedback') // TODO:
                        this._data.push(a)
                        this.sendRawCallback(this.onFeedback, this._name, this._unname)
                    }
                }))
                if (inMetaData.length) {
                    for (let data of inMetaData)
                        peer.signal(data)
                    inMetaData.length = 0
                }
            })
        }
        _name({cellShape, partSize, summary}, namer, packet, then, unname) {
            // Copy all data points into the actual data stream.
            //   `this.onFeedback` will be called for each sending.
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
                for (let c = 0; c < cells; ++c) { // Naïvely reshape.
                    const offsetTarg = c * cellSize, offsetReal = c * realCellSize
                    for (let i = 0; i < cellSize; ++i)
                        namedV[offsetTarg + i] = rawData[offsetReal + i]
                }
                packet.send(this, then, unname, namedV, namedE || null, noData || true, noFeedback || true)
                this._feedback.push(this._data[i])
            }
            this._data.length = 0
        }
        _unname(namer, allFeedback, fbOffset, flatV) {}
        onFeedback(feedbackData, cellShape, partSize) {
            // Send feedback back.
            const fbPoint = this._feedback.shift()
            if (!fbPoint) return
            let [realPartSize, cells, realCellShape, bpv, rawData, rawError, noData, noFeedback, feedback] = fbPoint
            if (rawData === undefined) { // Dropped.
                const f = sn._allocF32(1)
                f[0] = 0
                console.log('####### sensor packet dropped, giving 0 feedback') // TODO:
                feedback(f, bpv, partSize, cellShape)
            } else { // Respond to the packet.
                const cellSize = cellShape.reduce((a,b)=>a+b), namedSize = cells * cellSize
                const realCellSize = realCellShape.reduce((a,b)=>a+b)
                const back = sn._allocF32(namedSize)
                for (let c = 0; c < cells; ++c) { // Naïvely reshape back.
                    const offsetTarg = c * cellSize, offsetReal = c * realCellSize
                    for (let i = 0; i < cellSize; ++i)
                        back[offsetReal + i] = feedbackData[offsetTarg + i]
                }
                feedback(back, bpv, partSize, cellShape)
            }
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
- \`signaler = InternetHandler.consoleLog\`: creates the channel over which negotiation of connections takes place. When called, constructs \`{ send(Uint8Array), close(), onopen, onmessage, onclose }\`, for example, [a \`WebSocket\`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket).
- \`bytesPerValue=0\`: 0 to transmit each value as float32, 1 to quantize as uint8, 2 to quantize as uint16. 1 is max-compression min-precision; 0 is the opposite.
- \`untrustedWorkaround = false\`: if set, will request a microphone stream and do nothing with it, so that a WebRTC connection can connect. The need for this was determined via alchemy, so its exact need-to-use is unknown.
` }
        static options() {
            return {
                iceServers: {
                    ['None']: () => [],
                    ['Some']: () => [{urls:'stun: stun.l.google.com:19302'}, {urls:'stun: stunserver.org:3478'}],
                },
                bytesPerValue: {
                    ['float32 (4× size)']: () => 0,
                    ['uint16 (2× size)']: () => 2,
                    ['uint8 (1× size)']: () => 1,
                },
            }
        }
        pause() {
            if (!this._isInResume) {
                if (this.peer) this.peer.destroy(), this.peer = null
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
                this.iceServers = opts.iceServers || []
                this.signaler = opts.signaler || InternetHandler.consoleLog
                this.untrustedWorkaround = !!opts.untrustedWorkaround
                if (opts !== this._opts) this._opts = Object.assign(Object.create(null), opts)
                if (!this._feedback) { // Only init once.
                    this._feedback = [] // What receiving a feedback-packet will have to do.
                    this.bytesPerValue = bpv
                    this._isInResume = false
                    this._dataSend = null, this._dataToSend = []
                }
                this.getPeer()
                this.peer && this.peer.setConfiguration && this.peer.setConfiguration({iceServers:this.iceServers})
            }
            try { this._isInResume = true
                return super.resume(opts)
            } finally { this._isInResume = false }
        }
        getPeer() {
            // Connects to another sensor network through WebRTC.
            // (Could double-init initially, though.)
            if (this.metaChannel == null) {
                const mc = this.metaChannel = this.signaler(true)
                this._signal = new Promise((resolve, reject) => {
                    mc.onopen = evt => { this._signal = null, resolve() }
                    mc.onmessage = evt => {
                        if (this.peer == null) return console.warn('dropping signal', evt.data)
                        if (typeof evt.data != 'string') return
                        const d = JSON.parse(evt.data)
                        this.peer.signal(d)
                    }
                    mc.onclose = evt => { reject(), this._signal = this.metaChannel = null } // When closed, reopen.
                })
            }
            if (this.peer == null) {
                this._dataSend = null
                const p = this.untrustedWorkaround ? navigator.mediaDevices.getUserMedia({audio:true}) : Promise.resolve()
                p.then(() => {throw null}).catch(() => {
                    importSimplePeer().then(SimplePeer => {
                        const peer = this.peer = new SimplePeer({
                            initiator: true,
                            config: {iceServers:this.iceServers},
                            channelConfig: {
                                ordered: false, // Unordered
                                maxRetransmits: 0, // Unreliable
                            },
                        })
                        peer.on('error', console.error)
                        peer.on('signal', data => {
                            signal.call(this, JSON.stringify(data))
                        })
                        peer.on('connect', () => {
                            const packer = messagePacker(peer)
                            this._dataSend = ({data, noData, noFeedback, cellShape, partSize}, bpv) => {
                                // cells 4b, partSize 4b, cellShapeLen 2b, i × cellShapeItem 4b, bpv 2b, quantized data, noData bits, noFeedback bits.
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
                                // console.log('handler sends data', bytes.length) // TODO:
                                packer(bytes)
                            }
                            if (this._dataToSend.length) {
                                for (let [input, bpv] of this._dataToSend)
                                    this._dataSend(input, bpv)
                                this._dataToSend.length = 0
                            }
                            console.log('handler has connected', this) // TODO:
                        })
                        // `data` is feedback here.
                        peer.on('data', messageUnpacker((bytes, packetId) => {
                            console.log('handler receives feedback', bytes && bytes.length) // TODO: What, are these bytes misshapen or something?
                            // bpv 2b, partSize 4b, cellShapeLen 2b, i × cellShapeItem 4b, quantized feedback.
                            const fb = this._feedback.shift()
                            if (!fb) return console.log('oh no', bytes) // TODO:
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
                                    console.log('handler is reshaped:', this._remotePartSize+':'+this._remoteCellShape, '→', partSize+':'+cellShape) // TODO:
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
                            if (!this.metaChannel) return
                            if (this._signal) this._signal.then(() => this.metaChannel.send(data))
                            else this.metaChannel.send(data)
                        }
                    })
                })
            }
        }
        
        onValues(then, input, feedback) {
            // console.log('handler onValues,', !!this._dataSend ? 'real' : 'skip', input.data.length, 'values') // TODO:
            if (this._dataSend)
                this._dataSend(input, this.bytesPerValue)
            else
                this._dataToSend.push([input, this.bytesPerValue])
            this._feedback.push([feedback, then, performance.now()])
        }
    }
    InternetHandler.consoleLog = A(function signalViaConsole(isHandler = false) {
        const obj = {
            send(msg) { console.log(msg) },
            close() {},
        }
        setTimeout(() => {
            obj.onopen && obj.onopen()
            if (isHandler && !self.internetHandler) {
                console.log("Carry around WebRTC messages manually, through the JS console.")
                console.log("    On the sensor, call \`.signal(sn.Sensor.InternetHandler.consoleLog())\` first.")
                console.log("    On request from the handler, do \`internetSensor(string)\`.")
                console.log("    On response from the sensor, do \`internetHandler(string)\`.")
            }
            const k = isHandler ? 'internetHandler' : 'internetSensor'
            self[k] = msg => obj.onmessage && obj.onmessage({data:msg})
        }, 0)
        return obj
    }, {
        docs:`The simplest option: [\`console.log\`](https://developer.mozilla.org/en-US/docs/Web/API/Console/log) and \`self.snInternet(messageString)\` is used to make the user carry signals around.`,
    })
    InternetHandler.webSocket = A(function signalViaWS(url) {
        return () => new WebSocket(url)
    }, {
        docs:`Signals via a [Web Socket](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket). Have to pass it the URL before passing it as the \`signaler\` option.`,
    })
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
            sn.meta.metric('sent, bytes', sent)
        }
    }
    function messageUnpacker(onpacket, timeoutMs=50, maxPacketBytes=16*1024*1024) {
        // Set the result of this as `dataChannel.onmessage`.
        // `onpacket(null | Uint8Array, packetId)` will be called with full packet data, in the ID-always-increasing-by-1 order, or `null` if taking too long.
        const packs = Object.create(null) // id → [remainingParts, ...partData]
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
                // TODO: Also check that `p` doesn't have nulls, because too-long packets will cause that.
                if (Array.isArray(p) && p[0] === 0) { // Copy parts into one buffer.
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
        // From floats to an array of bytes, quantized to lower-but-still-`-1…1` resolutions.
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
                return a[i] + " ≠ " + b[i]
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
    function toBits(a) { // Array<bool> → Uint8Array
        const b = new Uint8Array(Math.ceil(a.length / 8))
        for (let i = 0; i < b.length; ++i) {
            const j = 8*i
            b[i] = (a[j+0]<<7) | (a[j+1]<<6) | (a[j+2]<<5) | (a[j+3]<<4) | (a[j+4]<<3) | (a[j+5]<<2) | (a[j+6]<<1) | (a[j+7]<<0)
        }
        return b
    }
    function fromBits(b) { // Uint8Array → Array<bool>
        const a = allocArray(b.length * 8) // The length is a bit inexact, which is not important for us.
        for (let i = 0; i < b.length; ++i) {
            const j = 8*i
            a[j+0] = b[i] & (1<<7)
            a[j+1] = b[i] & (1<<6)
            a[j+2] = b[i] & (1<<5)
            a[j+3] = b[i] & (1<<4)
            a[j+4] = b[i] & (1<<3)
            a[j+5] = b[i] & (1<<2)
            a[j+6] = b[i] & (1<<1)
            a[j+7] = b[i] & (1<<0)
        }
        return a
    }
    function arrayEqual(a,b) {
        if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false
        for (let i = 0; i < a.length; ++i) if (a[i] !== b[i]) return false
        return true
    }
}