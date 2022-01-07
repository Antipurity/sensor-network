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
                const packetData = Object.create(null) // id → [remainingParts, prevPacketId, shapeId, ...partData]
                dataChannel.onmessage = evt => {
                    if (!(evt.data instanceof ArrayBuffer)) return
                    const dv = new DataView(evt.data)
                    const packetId = dv.getUint16(0), packetPart = dv.getUint16(2)
                    if (packetPart === 0) {
                        const packetPartLength = dv.getUint16(4)
                        const prevPacketId = dv.getUint16(6), shapeId = dv.getUint16(8)
                        packetData[packetId] = allocArray(3 + packetPartLength)
                        packetData[packetId][0] = packetPartLength
                        packetData[packetId][1] = prevPacketId
                        packetData[packetId][2] = shapeId
                        // TODO: ...Wait, also set prevPacketId and shapeId...
                        // TODO: Set the remaining data.
                        //   ...How, exactly? Just push `evt.data`? `dv`? Or a new byte-array? What's best for concatenation?
                        --packetData[packetId][0]
                    } else {
                        // TODO: Set the data. And decrement the remaining-packet-length.
                    }
                    // TODO: Also, if no packet parts are remaining, concat into data.
                    // TODO: We want an object from packetId to its data, right? ...What about the completion percentage?…
                    //   What's the exact format... {packetId:[...partData]}.
                    // TODO: If we have a continuous path from a previous packet, or it took too long, decompress & emit data.
                } // TODO: Put `messageUnpacker(…)` here.
                //   TODO: ...What is its `onpacket`?…
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
        // `onpacket(null | Uint8Array)` will be called with full packet data, in the ID-always-increasing-by-1 order, or `null` if taking too long.
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
                    onpacket(b)
                } else {
                    if (packs[nextId]) deallocArray(packs[nextId]), packs[nextId] = null
                    onpacket(null)
                }
                ++nextId, nextId >= 65536 && (nextId = 0)
                tooLong = false
            }
        }
    }
}