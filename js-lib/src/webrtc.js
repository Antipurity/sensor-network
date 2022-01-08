export default function init(sn) {
    const arrayCache = []
    class InternetSensor extends sn.Sensor {
        static docs() { return `Extends this network over the Internet.

Methods:
- \`signal(metaChannel: { send(Uint8Array), close(), onopen, onmessage, onclose }, maxCells=65536)\`: on an incoming connection, someone must notify us of it so that negotiation of a connection can take place, for example, [over a \`WebSocket\`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket).

Options:
- \`iceServers = []\`: the [list](https://gist.github.com/mondain/b0ec1cf5f60ae726202e) of [ICE servers](https://developer.mozilla.org/en-US/docs/Web/API/RTCIceServer/urls) (Interactive Connectivity Establishment).
- TODO:

TODO: Make note of browser compatibility.
` }
        static options() {
            return {
                iceServers: {
                    ['None']: () => [],
                    ['Some']: () => [{urls:'stun: stun.l.google.com:19302'}, {urls:'stun: stunserver.org:3478'}],
                },
                // TODO:
            }
        }
        resume(opts) {
            if (opts) {
                this.iceServers = opts.iceServers || []
                opts.onValues = this.onValues
                opts.values = 0, opts.emptyValues = 0
                opts.name = []
                this._data = [], this._feedback = [] // The main adjustable data queue.
                // TODO: What else?
            }
            return super.resume(opts)
        }
        signal(metaChannel, maxCells=65536) {
            let metaData = []
            const signal = data => {
                if (metaData) metaData.push(data)
                else metaChannel.send(data)
            }
            metaChannel.onopen = evt => {
                metaData && metaData.forEach(data => metaChannel.send(data))
                metaData = null
            }
            // Ignore `metaChannel.onclose`.

            const peer = new RTCPeerConnection({iceServers:this.iceServers})
            metaChannel.onmessage = evt => {
                if (typeof evt.data != 'string') return
                const d = JSON.parse(evt.data)
                if (d.icecandidate) {
                    peer.addIceCandidate(new RTCIceCandidate(d.icecandidate) )
                } else if (typeof d.offer == 'string') {
                    peer.setRemoteDescription({ type:'offer', sdp: d.offer })
                    peer.createAnswer().then(answer => {
                        peer.setLocalDescription(answer)
                        signal(JSON.stringify({ answer: answer.sdp }))
                    }).catch(() => peer.close())
                }
            }
            peer.onicecandidate = evt => {
                if (evt.candidate)
                    signal(JSON.stringify({ icecandidate: evt.candidate }))
            }
            peer.onconnectionstatechange = evt => {
                if (!peer) return
                const state = peer.connectionState
                if (state === 'failed' || state === 'closed') 'be not sad, but glad that it happened'
            }
            peer.ondatachannel = evt => {
                const dataChannel = evt.channel
                dataChannel.onmessage = evt => messageUnpacker((data, packetId) => {
                    sn._assert(data instanceof Uint8Array)
                    if (data.length < 2) return
                    // TODO: What if `data` is `null`?
                    //   What do we want to do, abstractly? A failed packet... How to send this failure on?
                    //   TODO: (…Wait: if we'll be treating shape-updates the same as data-packets via `packetId`, then won't this mean that we'll have an extra data+feedback step for each shape-update? Or is this tolerable?)
                    const dv = new DataView(data.buffer)
                    const shapeId = dv.getUint16(0)
                    if (shapeId === 0) {
                        // Read the shape.
                        // TODO: (Maybe, should be 2 functions that read & write the same C struct?… For symmetry…)
                        //   `readCellShape(Uint8Array)→obj`, `writeCellShape(obj)→Uint8Array`.
                        const partSize = dv.getUint32(2)
                        const cells = dv.getUint32(6)
                        const cellShapeLen = dv.getUint16(10)
                        if (cellShapeLen !== 4) return
                        const cellShape = new Array(cellShapeLen)
                        let offset = 12
                        for (let i = 0; i < cellShapeLen; ++i)
                            cellShape[i] = dv.getUint32(), offset += 4
                        const bpv = dv.getUint8(offset);  offset += 1
                        if (bpv !== 0 && bpv !== 1 && bpv !== 2) return
                        const nameSize = cells * (cellShape.reduce((a,b) => a+b) - cellShape[cellShape.length-1])
                        const nameBytes = data.subarray(offset, offset += nameSize)
                        const noDataBytes = data.subarray(offset, offset += Math.ceil(cells / 8))
                        const noFeedbackBytes = data.subarray(offset, offset += Math.ceil(cells / 8))
                        const name = unquantize(nameBytes, bpv)
                        const noData = fromBits(noDataBytes)
                        const noFeedback = fromBits(noFeedbackBytes)
                        // TODO: Remember (need global-ish vars: prev-shape-id and prev-shape).
                        //   ...Don't we want to remember more shapes than 1... Maybe 3 previous shapes... Stored in a [..., shapeId, shapeObject, ...]
                        // Acknowledge.
                        {
                            const dv = new DataView(new ArrayBuffer(12))
                            dv.setUint16(0, packetId)
                            dv.setUint16(2, 0) // part
                            dv.setUint16(4, 1) // parts
                            dv.setUint16(6, 0) // prev packet id // TODO: Maybe, this should be on a separate layer which will wrap our own, after the message is here but before decoding? (This way, we don't *have* to waste space here for functionality that's not guaranteed to be implemented anyway.)
                            dv.setUint16(8, 0) // shape id (ACK)
                            dv.setUint16(10, shapeId) // shape id (what we ACK)
                            signal(new Uint8Array(dv.buffer))
                        }
                    } else {
                        // TODO: Fetch shape ID (or request if we don't know it… Wait, do we need a packet queue to not drop these maybe-packets?)
                        // TODO: Turn reward & data (& shape info) into a raw array of, uh, f32...
                        //   ...No, need to decompress first...
                        // (Also, do `unquantize` on all value arrays.)
                    }
                    // TODO: Do we want a JS object for the shape info?
                    // TODO: Need functions for shapely send/receive, right? (*Then*, we can do shapeliness+compression.)
                    //   They only need cellShape and raw data/feedback. And they can already do un/qunatization.
                    //   ...We really need shape handling here, tho. And those funcs are trivial anyway.
                })
                //   TODO: ...What is its `onpacket`?…
                //     How do we encode prevPacketId and shapeId, and do de/compression, and requesting of shapes and composition of shapes with data to create the actual cells...
                dataChannel.onclose = evt => peer.close() // Hopefully not fired on mere instability.
            }
        }
        onValues(data) {
            sn._deallocF32(data)
            // TODO: How to take all data from the queue?
            //   TODO: Also, this.resize(values) and give per-cell noData/noFeedback bool arrays to this.sendCallback. TODO: Actually, use `this.sendRawCallback` instead.
            //   ...What are the values though? Can be inferred by summing up this._data's lengths, right?
            // TODO: How to remember to adjust what we have?
        }
        onFeedback(feedback) {
            // TODO: How to send feedback to their appropriate connections?
            //   Also, isn't *this* code largely shared between sensing and handling too?
            //     (Not in shape giving/requesting, though.)
        }
    }
    class InternetHandler extends sn.Handler {
        static docs() { return `Makes this environment a remote part of another sensor network.

Options:
- \`iceServers = []\`: the [list](https://gist.github.com/mondain/b0ec1cf5f60ae726202e) of [ICE servers](https://developer.mozilla.org/en-US/docs/Web/API/RTCIceServer/urls) (Interactive Connectivity Establishment).
- \`signaler = …\`: creates the channel over which negotiation of connections takes place. When called, constructs \`{ send(Uint8Array), close(), onopen, onmessage, onclose }\`, for example, [a \`WebSocket\`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket).
- TODO:
` }
        static options() {
            return {
                iceServers: {
                    ['None']: () => [],
                    ['Some']: () => [{urls:'stun: stun.l.google.com:19302'}, {urls:'stun: stunserver.org:3478'}],
                },
                // TODO:
            }
        }
        resume(opts) {
            if (opts) {
                this.iceServers = opts.iceServers || []
                this.signaler = opts.signaler // TODO: What's the default implementation?
                this.getPeer()
                this.peer.setConfiguration({iceServers:this.iceServers})
                this._data = [], this._feedback = [] // The main adjustable data queue.
                // TODO: What else?
            }
            return super.resume(opts)
        }
        getPeer() {
            // Connects to another sensor network through WebRTC.
            if (this.metaChannel == null) {
                const mc = this.metaChannel = this.signaler()
                this._signal = new Promise((resolve, reject) => {
                    mc.onopen = evt => { this._signal = null, resolve() }
                    mc.onmessage = evt => {
                        if (typeof evt.data != 'string') return
                        const d = JSON.parse(evt.data)
                        if (d.icecandidate) {
                            if (this.peer) this.peer.addIceCandidate(new RTCIceCandidate(d.icecandidate) )
                        } else if (typeof d.answer == 'string') {
                            if (this.peer) this.peer.setRemoteDescription({ type:'answer', sdp: d.answer })
                        }
                    }
                    mc.onclose = evt => { reject(), this._signal = this.metaChannel = null } // When closed, reopen.
                })
            }
            if (this.peer == null) {
                const peer = this.peer = new RTCPeerConnection({iceServers:this.iceServers})
                peer.onicecandidate = evt => {
                    if (evt.candidate)
                        this.signal(JSON.stringify({ icecandidate: evt.candidate }))
                }
                peer.onnegotiationneeded = evt => {
                    peer.createOffer().then(offer => peer.setLocalDescription(offer)).then(() => {
                        this.signal(JSON.stringify({offer: peer.localDescription.sdp}))
                    }).catch(() => peer.setRemoteDescription({type:'rollback'}))
                }
                peer.onconnectionstatechange = evt => {
                    if (!peer) return
                    const state = peer.connectionState
                    if (state === 'failed' || state === 'closed') this.peer = null // Reopen.
                }
            }
            if (this.dataChannel == null) {
                const dc = this.dataChannel = this.peer.createDataChannel('sn-internet', {
                    ordered: false, // Unordered
                    maxRetransmits: 0, // Unreliable
                })
                dc.binaryType = 'arraybuffer'
                dc.onmessage = evt => {
                    if (!(evt.data instanceof ArrayBuffer)) return
                    const dv = new DataView(evt.data)
                    // TODO: And how do we parse this packet? Packet ID, packet part, shape ID...
                    // TODO: If we have a continuous path from a previous packet, or it took too long, emit data.
                    // TODO: …Need to parse and add to queue…
                    //   ...This code is kinda shared among got-sensor-data & got-handler-feedback, isn't it?... Can we extract it to a common function?
                }
                dc.onclose = evt => this.dataChannel = null // When closed, reopen.
            }
        }
        signal(data) {
            // Send `data`, waiting for the signaling channel to open.
            if (!this.dataChannel) return
            if (this._signal) this._signal.then(() => this.dataChannel.send(data))
            else this.dataChannel.send(data)
        }
        static onValues(then, {data, error, noData, noFeedback, cellShape, partSize}, feedback) {
            // TODO: Check `this.dataChannel.readyState === 'open'`; if not open, just return.
            // TODO: How to send values across `this.dataChannel`, and get some back?
            //   TODO: How to accumulate out-of-order packets, waiting reasonably?
            //   TODO: When to drop packets?
        }
    }
    // TODO: The default signaling option: "the user will manually copy this".
    //   ...What, from JS console? And into JS console? This is so unrefined…
    //   TODO: But also have a WebSocket option. (Once we have a Rust server to test it on, I mean.)
    return {
        sensor: InternetSensor,
        handler: InternetHandler,
    }
    function allocArray() { return arrayCache.length ? arrayCache.pop() : [] }
    function deallocArray(a) { Array.isArray(a) && arrayCache.length < 16 && (a.length = 0, arrayCache.push(a)) }
    function messagePacker(channel, maxPacketBytes=16*1024*1024) {
        // Sends maybe-large messages over unordered & unreliable channels.
        // Use `messageUnpacker` to re-assemble messages on the other end.
        let nextId = 0
        const maxPartLen = 65536, partBuf = new Uint8Array(maxPartLen)
        const partView = new DataView(partBuf.buffer, partBuf.byteOffset, partBuf.byteLength)
        return function message(data) { // Uint8Array
            sn._assert(data.length <= maxPacketBytes, "Message is too long!")
            const partSize = maxPartLen - 4
            const parts = Math.ceil((data.length - 2) / partSize)
            for (let part = 0, atData = 0; part < parts; ++part) {
                partView.setUint16(0, nextId)
                partView.setUint16(2, part)
                if (part === 0) partView.setUint16(4, parts)
                const header = (part === 0 ? 6 : 4)
                for (let i = header; i < partBuf.length && atData < data.length; ++i, ++atData)
                    partBuf[i] = data[atData]
                channel.send(partBuf)
            }
            ++nextId, nextId >= 65536 && (nextId = 0)
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
            if (!(evt.data instanceof ArrayBuffer)) return
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
            if (id > prevId && prevAt == null) prevAt = performance.now()
            // Send off the next packet, in-order or on-timeout if there are packets after this one.
            let tooLong = prevAt != null && performance.now() - prevAt > timeoutMs && superseded()
            while (packs[nextId] && packs[nextId][0] === 0 || tooLong) {
                const p = packs[nextId]
                if (Array.isArray(p) && p[0] === 0) { // Copy parts into one buffer.
                    // (A copy. And generates garbage. As garbage as JS is.)
                    let len = 0
                    for (let i = 1; i < p.length; ++i)
                        len += (i === 1 ? p[i].byteLength-6 : p[i].byteLength-4)
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
                ++nextId, nextId >= 65536 && (nextId = 0)
                tooLong = false
            }
        }
    }
    function quantize(f32a, bpv = 0) {
        // From floats to an array of bytes, quantized to lower-but-still-`-1…1` resolutions.
        sn._assert(f32a instanceof Float32Array)
        if (!bpv) return bigEndian(new Uint8Array(f32a.buffer, f32a.byteOffset, f32a.byteLength), bpv, false, true)
        sn._assert(bpv === 1 || bpv === 2)
        const r = bpv === 1 ? new Uint8Array(f32a.byteLength) : new Uint16Array(f32a.byteLength / 2 | 0)
        const scale = bpv === 1 ? 255 : 65535
        for (let i = 0; i < r.length; ++i)
            r[i] = Math.max(0, Math.min(Math.round((f32a[i]+1)/2 * scale), 255))
        return bpv === 1 ? r : bigEndian(new Uint8Array(r.buffer, r.byteOffset, r.byteLength), bpv, false, true)
    }
    function unquantize(a, bpv = 0) {
        if (!bpv) {
            a = bigEndian(a, bpv, true)
            return new Float32Array(a.buffer, a.byteOffset, a.byteLength / 4 | 0)
        }
        sn._assert(bpv === 1 || bpv === 2)
        if (bpv === 2) a = new Uint16Array(bigEndian(a, bpv, true))
        const r = new Float32Array(a.length)
        const scale = bpv === 1 ? 255 : 65535
        for (let i = 0; i < r.length; ++i)
            r[i] = a[i]/scale * 2 - 1
        return r
    }
    function bigEndian(a, bpv, backToNative = false, inPlace = false) {
        // `a` is copied unless `inPlace`.
        if (bigEndian.bigEnd === undefined) {
            const x = new ArrayBuffer(2), y = new Uint16Array(x), z = new Uint8Array(x)
            y[0] = 0x0102
            bigEndian.bigEnd = z[0] === 0x01
        }
        sn._assert(a instanceof Uint8Array)
        if (bigEndian.bigEnd !== backToNative || bpv === 1) return a
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
        const a = new Array(b.length * 8) // The length is a bit inexact, which is not important for us.
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
}