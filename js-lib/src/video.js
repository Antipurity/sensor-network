export default function init(sn) {
    return class Video extends sn.Sensor {
        docs() { return `// TODO:

This sensor's output is composed of 1 or more tiles, which are square images.

Extra options:
- \`tileDimension = 8\`: each tile edge's length.
- \`source = Video.stitchCanvases\`: TODO:
` }
        // TODO: How do we do this?
        resume(opts) {
            if (opts) {
                const td = opts.tileDimension || 8
                const src = opts.source || Video.requestTab // TODO: Do Video.stitchCanvases by default, once we have that.
                sn._assertCounts('Non-integer tile side', td)
                sn._assert(typeof src == 'function' || src instanceof Promise || src instanceof MediaStream || src instanceof Element && (src.tagName === 'CANVAS' || src.tagName === 'VIDEO' || src.tagName === 'IMG'), "Bad source")
                opts.extraValues = 0
                opts.onValues = Video.onValues
                this.tileDimension = td
                opts.values = 1 * this.tileDimension // TODO: ...Wait: if we just set this, then we can't know where the naming would insert free space. ...Just clip our cells, who cares.
                // TODO: Set `this.source`, and if a promise, remember to replace.
                // TODO: Also set `name`.
                // TODO: What props do we want?
                //   TODO: `source`: `<canvas>`, `<video>`, `<img>`, `MediaStream`, function; promise (in which case, it returning will replace itself, before which `onValues` will simply return).
            }
            super.resume(opts)
        }
        // TODO: First, at least make it work for 1-tile no-zoomout no-targets.

        static onValues(sensor, data) {
            const targetShape = sensor.cellShape()
            if (!targetShape || this.source instanceof Promise) return
            // Make sure to limit to one tile per cell, so that there's no misalignment.
            const dataSize = targetShape[targetShape.length-1]
            const cells = data.length / dataSize | 0
            const valuesPerCell = Math.ceil(this.values / cells)
            Video._dataJS(sensor, data, valuesPerCell)
            sensor.sendCallback(Video.onFeedback, data)
        }
        static onFeedback(feedback, sensor) {
            if (!feedback || sensor.noFeedback) return
            // TODO: What do we do here?
        }

        static _dataJS(sensor, data, valuesPerTile) { // Fills data, through JS.
            // TODO: How to get the imageData?
            // TODO: How to sample from the imageData?
        }

        static requestTab() {
            // TODO: How to use `getDisplayMedia` here, to get a promise of a stream?
            //   promise = navigator.mediaDevices.getDisplayMedia({ audio:true, video:true })
            //   And how to make sound in the future be able to do this too? `sn._getDisplayMedia = promise`, self-replacing?
            // TODO: How to *verify* that this stream concerns the current tab?
            //   `.getVideoTracks[0].getCapabilities().displaySurface==='browser'` and `.logicalSurface===true`.
        }
    }
}