export default function init(sn) {
    class InternetSensor extends sn.Sensor {
        static docs() { return `Extends this network over the Internet.

Options:
- TODO: \`iceServers = []\`
- TODO: How do we allow signaling from anyone that connects?…
    - TODO: A function, that returns {send, onmessage, close}?
- TODO:
- TODO: (Also, how do we allow developers to implement an authentication mechanism?)
` }
        static options() {
            return {
                // TODO: iceServers:
                //   TODO: []
                //   TODO: ['stun.l.google.com:19302', 'stunserver.org:3478']
            }
        }
        resume(opts) {
            if (opts) {
                this.iceServers = opts.iceServers || []
                this.getPeer()
                opts.values = 0
                opts.name = ['internet'] // TODO: …No: should have funcs that defer to the received cells' names…
                //   TODO: …But isn't it much better to be able to pass raw names to sensor.sendCallback?… Much less memory wasting, at least. And, much easier to actually establish a correspondence between cells and their names, especially across timesteps… And aren't functions in names only called once anyway, meaning that without raw access, we will create LOTS of F32A garbage…
                // TODO: What else?
            }
            return super.resume(opts)
        }
        getPeer() {
            if (this.peer) return
            this.peer = new RTCPeerConnection({iceServers:this.iceServers})
            // TODO: …When we're signaled:
            //   TODO: this.peer.setRemoteDescription({type:'offer', sdp:offer})
            //   TODO: this.peer.createAnswer().then(answer => { this.peer.setLocalDescription(answer), SIGNAL(answer.sdp) }).catch(console.error)
            // TODO: this.peer.onicecandidate = evt => evt.candidate && SIGNAL(evt.candidate) TODO: How to signal, exactly?
            //   TODO: When signaled, this.peer.addIceCandidate(that)
            // TODO: How to listen to incoming connections, preparing to send data for each one?
            // TODO: On connection's data, remember it.
            //   ...What should we do on dropped, and on out-of-order messages?...
            //   TODO: this.peer.ondatachannel = evt => evt.channel??? TODO: What exactly do we do with the channel?
        }
        static onValues(sensor, data) { // TODO: …Oh yeah: no `sensor`, only `this`.
            // TODO: How to take all data from the queue?
            //   TODO: Also, sensor.resize(values) and give per-cell noData/noFeedback bool arrays to sensor.sendCallback. TODO: Actually, use `sensor.sendRawCallback` instead.
        }
        static onFeedback(feedback, sensor) {
            // TODO: How to send feedback to their appropriate connections?
        }
    }
    class InternetHandler extends sn.Handler {
        static docs() { return `Makes this environment a remote part of another sensor network.

Options:
- TODO: What options do we want?
- TODO: \`iceServers = []\`
- TODO: How do we allow both "signal another" and "accept data from there"? A function, that returns {send, onmessage, close}?
` }
        static options() {
            return {
                // TODO: iceServers:
                //   TODO: []
                //   TODO: ['stun.l.google.com:19302', 'stunserver.org:3478']
            }
        }
        resume(opts) {
            if (opts) {
                this.iceServers = opts.iceServers || []
                this.getPeer()
                // TODO: Also, this.peer.setConfiguration({iceServers}).
                // TODO: What else?
            }
            return super.resume(opts)
        }
        getPeer() {
            // Connects to another sensor network through WebRTC.
            if (this.peer) return
            // TODO: this.peer = new RTCPeerConnection({iceServers:this.iceServers})
            //   (The list should be an option. For example: https://gist.github.com/mondain/b0ec1cf5f60ae726202e)
            // TODO: this.dataChannel = this.peer.createDataChannel('sn-internet', {
            //   ordered: false,
            //   maxRetransmits: 0,
            // })
            //   TODO: this.dataChannel.onopen = ???
            //   TODO: this.dataChannel.onmessage = ???
            //   TODO: this.dataChannel.onclose = ???
            // TODO: this.peer.onicecandidate = evt => evt.candidate && SIGNAL(evt.candidate) TODO: How to signal, exactly?
            //   TODO: When signaled, this.peer.addIceCandidate(that)
            // TODO: this.peer.onnegotiationneeded = this.peer.createOffer().then(offer => this.peer.setLocalDescription(offer)).then(() => sendToSensor(this.peer.localDescription)).catch(console.error)
            //   TODO: When signaled back, this.peer.setRemoteDescription({type:'answer', sdp})
            //   TODO: On error, this.peer.setRemoteDescription({type:'rollback'})
            // TODO: this.peer.onconnectionstatechange = this.peer.connectionState !== 'connected' && ??? TODO: What do we do, exactly? this.peer.close(), this.peer=null, so that `onValues` can attempt to re-open it?
        }
        static onValues(then, {data, error, noData, noFeedback, cellShape, partSize}, feedback) {
            // TODO: How to send values across `this.dataChannel`, and get some back?
            //   TODO: How to accumulate out-of-order packets, waiting reasonably?
            //   TODO: When to drop packets?
        }
    }
    // TODO: What to use as signaling? Need options, which default to "the user will manually copy this".
    //   TODO: But also have a WebSocket option. (Once we have a Rust server to test it on, I mean.)
    return {
        sensor: InternetSensor,
        handler: InternetHandler,
    }
}