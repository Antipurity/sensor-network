export default function init(sn) {
    class InternetSensor extends sn.Sensor {
        static docs() { return `Extends this network over the Internet.

Options:
- \`iceServers = []\`: the [list](https://gist.github.com/mondain/b0ec1cf5f60ae726202e) of [ICE servers](https://developer.mozilla.org/en-US/docs/Web/API/RTCIceServer/urls) (Interactive Connectivity Establishment).
- \`signaler = …\`: creates the channel over which negotiation of connections takes place. When called, constructs \`{ send(Uint8Array), close(), onmessage, onclose }\`, for example, [a \`WebSocket\`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket).
    - TODO: ...Or should it be some returner-of-these-objects, one for each incoming connection?...
- TODO:
- TODO: (Also, how do we allow developers to implement an authentication mechanism?)
` }
        static options() {
            return {
                iceServers: {
                    ['None']: () => [],
                    ['Some']: () => [{urls:'stun: stun.l.google.com:19302'}, {urls:'stun: stunserver.org:3478'}],
                },
                // TODO: signaler
                // TODO:
            }
        }
        resume(opts) {
            if (opts) {
                this.iceServers = opts.iceServers || []
                this.signaler = opts.signaler // TODO: …What's the default implementation?
                this.getPeer()
                this.peer.setConfiguration({iceServers:this.iceServers})
                opts.onValues = this.onValues
                opts.values = 0
                opts.name = []
                // TODO: What else?
            }
            return super.resume(opts)
        }
        getPeer() {
            if (this.peer) return
            this.peer = new RTCPeerConnection({iceServers:this.iceServers})
            // TODO: …When we're signaled:
            //   TODO: (Need to call this.signaler to set up this.signal, right?)
            //   TODO: this.peer.setRemoteDescription({type:'offer', sdp:offer})
            //   TODO: this.peer.createAnswer().then(answer => { this.peer.setLocalDescription(answer), SIGNAL(answer.sdp) }).catch(console.error)
            // TODO: this.peer.onicecandidate = evt => evt.candidate && SIGNAL(evt.candidate) TODO: How to signal, exactly?
            //   TODO: When signaled, this.peer.addIceCandidate(that)
            // TODO: How to listen to incoming connections, preparing to send data for each one?
            // TODO: On connection's data, remember it.
            //   ...What should we do on dropped, and on out-of-order messages?...
            //   TODO: this.peer.ondatachannel = evt => evt.channel???
            //     TODO: this.dataChannel.onmessage = ???
            //     TODO: this.dataChannel.onclose = ???
        }
        onValues(data) {
            // TODO: How to take all data from the queue?
            //   TODO: Also, this.resize(values) and give per-cell noData/noFeedback bool arrays to this.sendCallback. TODO: Actually, use `this.sendRawCallback` instead.
        }
        onFeedback(feedback) {
            // TODO: Can we make `sensor.sendRawCallback` accept the deobfuscation function, so that here, we receive raw feedback, not un-named feedback?
            // TODO: How to send feedback to their appropriate connections?
        }
    }
    class InternetHandler extends sn.Handler {
        static docs() { return `Makes this environment a remote part of another sensor network.

Options:
- \`iceServers = []\`: the [list](https://gist.github.com/mondain/b0ec1cf5f60ae726202e) of [ICE servers](https://developer.mozilla.org/en-US/docs/Web/API/RTCIceServer/urls) (Interactive Connectivity Establishment).
- \`signaler = …\`: creates the channel over which negotiation of connections takes place. When called, constructs \`{ send(Uint8Array), close(), onmessage, onclose }\`, for example, [a \`WebSocket\`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket).
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
                this.signaler = opts.signaler // TODO: …What's the default implementation?
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
                mc.onmessage = evt => {
                    if (typeof evt.data != 'string') return
                    const d = JSON.parse(evt.data)
                    if (d.icecandidate) {
                        if (this.peer) this.peer.addIceCandidate(new RTCIceCandidate(d.icecandidate) )
                    } else if (typeof d.answer == 'string') {
                        if (this.peer) this.peer.setRemoteDescription({ type:'answer', sdp: d.answer })
                    }
                }
                mc.onclose = evt => this.metaChannel = null // When closed, reopen.
            }
            if (this.peer == null) {
                this.peer = new RTCPeerConnection({iceServers:this.iceServers})
                this.peer.onicecandidate = evt => {
                    if (evt.candidate && this.metaChannel.readyState === 'open') // TODO: ...Wait, but web sockets have `1` for 'open'... How to overcome this limitation?...
                        //   Surely we cannot listen to mc.onopen, and return a promise if needed?........
                        //     (With a function that handles sending meta-data for us........)
                        this.metaChannel.send(JSON.stringify({ icecandidate: evt.candidate }))
                }
                this.peer.onnegotiationneeded = evt => {
                    if (!this.peer) return
                    const p = this.peer
                    p.createOffer().then(offer => p.setLocalDescription(offer)).then(() => {
                        // TODO: Have a func on `this` that takes a message, waits for this.metaChannel's onopen if needed, and call that here.
                        this.metaChannel.send(p.localDescription)
                    }).catch(() => p.setRemoteDescription({type:'rollback'}))
                }
                this.peer.onconnectionstatechange = evt => {
                    if (!this.peer) return
                    const state = this.peer.connectionState
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
                    const d = evt.data
                    // TODO: …Need to parse and add to queue…
                }
                dc.onclose = evt => this.dataChannel = null // When closed, reopen.
            }
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