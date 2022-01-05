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
                // TODO:
                // TODO: Also, initiate a negotiation here, right? …But what if it's already done, and we only need to update our settings…
                //   TODO: this.peer.createOffer().then(offer => this.peer.setLocalDescription(offer)).then(() => sendToSensor(this.peer.localDescription)).catch(err => ???)
                //     TODO: This is `onnegotiationneeded()`, right?
                //   TODO: ...Should this be done only once?... Never resetting the list of connections?
                //     But what if Handler options contain how to establish the connection?
            }
            return super.resume(opts)
        }
        // TODO: What else does this do? How do we actually handle data and turn it into feedback, again?
        //   onValues(then, {data, error, noData, noFeedback, cellShape, partSize}, feedback)
    }
    // Unreliable and unordered:
    // RTCPeerConnection.createDataChannel("label", {
    //     ordered: false,
    //     maxRetransmits: 0
    // })
    // TODO: How to get an RTCPeerConnection?
    //   TODO: ...Why do we need an ICE server if we need a separate signaling channel anyway?...
    // TODO: Call .createDataChannel in Handler and listen to ondatachannel(evt:{channel}) in Sensor.
    //   ...On handler's `resume`? Does the sensor just accept all channels always? I think so?…
    // TODO: How to negotiate an ICE connection?
    //   TODO: What channels do we negotiate over?

    // TODO: …Want to create a demo that can connect, right?
    //   …We need a signaling server though…
    return {
        sensor: InternetSensor,
        handler: InternetHandler,
    }
}