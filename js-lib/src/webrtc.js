export default function init(sn) {
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
                // TODO: What else?
                // TODO: We also want this._data=[] (floats) and this._feedback=[] (callbacks), for finalized data waiting for feedback.
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
                dataChannel.onmessage = evt => {
                    if (!(evt.data instanceof ArrayBuffer)) return
                    const dv = new DataView(evt.data)
                    // TODO: And how do we parse this packet? Packet ID, packet part, shape ID...
                    // TODO: If we have a continuous path from a previous packet, or it took too long, emit data.
                }
                dataChannel.onclose = evt => peer.close() // Hopefully not fired on mere instability.
            }
        }
        onValues(data) {
            // TODO: How to take all data from the queue?
            //   TODO: Also, this.resize(values) and give per-cell noData/noFeedback bool arrays to this.sendCallback. TODO: Actually, use `this.sendRawCallback` instead.
        }
        onFeedback(feedback) {
            // TODO: Can we make `sensor.sendRawCallback` accept the deobfuscation function, so that here, we receive raw feedback, not un-named feedback? (We do want to send back reward feedback.)
            // TODO: How to send feedback to their appropriate connections?
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
}