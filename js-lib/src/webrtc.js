export default function init(sn) {
    class InternetSensor extends sn.Sensor {
        static docs() { return `Extends this network over the Internet.

Options:
- \`iceServers = []\`: the [list](https://gist.github.com/mondain/b0ec1cf5f60ae726202e) of [ICE servers](https://developer.mozilla.org/en-US/docs/Web/API/RTCIceServer/urls) (Interactive Connectivity Establishment).
- \`signaler = …\`: creates the channel over which negotiation of connections takes place. When called, constructs \`{ send(Uint8Array), close(), onopen, onmessage, onclose }\`, for example, [a \`WebSocket\`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket).
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
            //     TODO: this.dataChannel.onopen = ???
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
- \`signaler = …\`: creates the channel over which negotiation of connections takes place. When called, constructs \`{ send(Uint8Array), close(), onopen, onmessage, onclose }\`, for example, [a \`WebSocket\`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket).
    - TODO: ...Or should it be some returner-of-these-objects, one for each incoming connection?...
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
                // TODO: mc.onopen = ???
                // TODO: mc.onmessage = ???
                //   TODO: When signaled {icecandidate}, this.peer.addIceCandidate(that)
                //   TODO: When signaled {answer}, this.peer.setRemoteDescription({type:'answer', sdp})
                mc.onclose = evt => this.metaChannel = null // When closed, reopen.
            }
            if (this.peer == null) {
                this.peer = new RTCPeerConnection({iceServers:this.iceServers})
                this.peer.onicecandidate = evt => {
                    if (evt.candidate && this.metaChannel.readyState === 'open')
                        this.metaChannel.send({ icecandidate: evt.candidate })
                }
                // TODO: this.peer.onnegotiationneeded = this.peer.createOffer().then(offer => this.peer.setLocalDescription(offer)).then(() => sendToSensor(this.peer.localDescription)).catch(console.error)
                //   TODO: On error, this.peer.setRemoteDescription({type:'rollback'})
                // TODO: this.peer.onconnectionstatechange = this.peer.connectionState !== 'connected' && ??? TODO: What do we do, exactly? this.peer.close(), this.peer=null, so that `onValues` can attempt to re-open it?
            }
            if (this.dataChannel == null) {
                const dc = this.dataChannel = this.peer.createDataChannel('sn-internet', {
                    ordered: false, // Unordered
                    maxRetransmits: 0, // Unreliable
                })
                dc.binaryType = 'arraybuffer'
                // TODO: dc.onopen = ???
                //   TODO: ...Uh... do we do something special here, or?… Can't we just check this.dataChannel.readyState==='open'?
                // TODO: dc.onmessage = ???
                dc.onclose = evt => this.dataChannel = null // When closed, reopen.
            }
        }
        static onValues(then, {data, error, noData, noFeedback, cellShape, partSize}, feedback) {
            // TODO: How to send values across `this.dataChannel`, and get some back?
            //   TODO: How to accumulate out-of-order packets, waiting reasonably?
            //   TODO: When to drop packets?
        }
    }
    // TODO: The default signaling option: "the user will manually copy this".
    //   TODO: But also have a WebSocket option. (Once we have a Rust server to test it on, I mean.)
    return {
        sensor: InternetSensor,
        handler: InternetHandler,
    }
}