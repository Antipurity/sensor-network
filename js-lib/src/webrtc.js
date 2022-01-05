export default function init(sn) {
    class InternetSensor extends sn.Sensor {
        static docs() { return `Extends this network over the Internet.

Options:
- TODO:
- TODO: (Also, how do we allow developers to implement an authentication mechanism?)
` }
        // TODO: What `options` do we have?
        resume(opts) {
            // TODO: If we haven't yet, create a new RTCPeerConnection({iceServers:[…]}), right?
            //   TODO: How to listen to incoming connections, preparing to send data for each one?
            //   TODO: On connection's data, remember it.
            //     ...What should we do on dropped, and on out-of-order messages?...
            //     TODO: this.peer.ondatachannel = evt => evt.channel??? TODO: What exactly do we do with the channel?
            if (opts) {
                // TODO:
            }
            return super.resume(opts)
        }
        // TODO: Define `onValues(sensor, data)`.
        //   TODO: What does it do? Take all data from the queue?
        // TODO: Define `onFeedback(feedback, sensor)`.
        //   TODO: What does it do? Send feedback to their appropriate connections?
    }
    class InternetHandler extends sn.Handler {
        static docs() { return `Makes this environment a remote part of another sensor network.

Options:
- TODO: What options do we want?
` }
        resume(opts) {
            if (opts) {
                this.getPeer()
                // TODO:
            }
            return super.resume(opts)
        }
        getPeer() {
            // Connects to another sensor network through WebRTC.
            if (this.peer) return
            // TODO: Also, initiate a negotiation here, right? …But what if it's already done, and we only need to update our settings…
            //   TODO: this.peer = new RTCPeerConnection({iceServers:['stun.l.google.com:19302', 'stunserver.org:3478']})
            //     (The list should be an option. For example: https://gist.github.com/mondain/b0ec1cf5f60ae726202e)
            //   TODO: this.dataChannel = this.peer.createDataChannel('sn-internet', {
            //     ordered: false,
            //     maxRetransmits: 0,
            //   })
            //   TODO: this.peer.onicecandidate = evt => evt.candidate && SIGNAL(evt.candidate) TODO: How to signal, exactly?
            //   TODO: this.peer.onnegotiationneeded = this.peer.createOffer().then(offer => this.peer.setLocalDescription(offer)).then(() => sendToSensor(this.peer.localDescription)).catch(console.error)
            //   TODO: this.peer.onconnectionstatechange = this.peer.connectionState !== 'connected' && ??? TODO: What do we do, exactly? this.peer.close(), this.peer=null, so that `onValues` can attempt to re-open it?
        }
        // TODO: onValues(then, {data, error, noData, noFeedback, cellShape, partSize}, feedback)
    }
    // TODO: What to use as signaling? Need options, which default to "the user will manually copy this".
    //   TODO: But also have a WebSocket option. (Once we have a Rust server to test it on, I mean.)
    return {
        sensor: InternetSensor,
        handler: InternetHandler,
    }
}