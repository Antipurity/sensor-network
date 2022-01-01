export default function init(sn) {
    return class Text extends sn.Sensor {
        resume(opts) {
            if (opts) {
                // TODO: Set `name`:
                //   ['text', (dataStart, dataEnd, dataLen) => dataLen > dataStart ? 2 * dataStart / (dataLen - dataEnd) - 1 : 1]
                //     (Also with `opts.name` added/unrolled in the middle if string/array.)
                // TODO: Set `values`. ...To what?
                //   Like, how do we know how many cells we could possibly have?
                //   ...Don't we need `tokenSize=64`...
                // TODO: Setting either `onValues` or `onFeedback` depending on opts.noFeedback, right?
                // TODO: Also handle `maxTokens=64`, `text`, `textToTokens`, `tokenToData`.
            }
            return super.resume(opts)
        }
        // TODO: `static onValues`; what does it do?
        // TODO: `static onFeedback`; what does it do?
        // TODO: `static readSelection(n=2048)`; what does it do, exactly?
        // TODO: `static readHover(n=2048)`; what does it do, exactly?
        // TODO: `readChanges(n=2048)`; what does it do, exactly?
        // TODO: `writeSelection()`
    }
}