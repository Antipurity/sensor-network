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
                this.tileDimension = td
                // (Don't catch errors in `src`, so they'll be logged to console.)
                this.source = !(src instanceof Promise) ? src : (src.then(v => this.source=v), src)
                // TODO: Other props. (For example, handle tiling and zooming-in by forking ourselves, and making `pause` responsible for the forks too.)
                opts.extraValues = 0
                opts.onValues = Video.onValues
                opts.values = 1 * this.tileDimension
                opts.name = ['video', ''+td, ] // TODO: What funcs would return x & y & zoomOut & source of a cell?
                //   (zoomOut and source are the same per-`Video`-instance, I guess.)
                //   (zoomOut is -1 at level 2**0, 1 at levels 2**10+.)
                //   (source is -1 for tab (where feedback can happen), 1 for camera (where we can't control pixels).)
            }
            super.resume(opts)
        }
        // TODO: First, at least make it work for 1-tile no-zoomout no-targets.
        //   And test that it works with sound.

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

        static _sourceToDrawable(source) { // .drawImage and .texImage2D can use the result.
            if (!(source instanceof MediaStream)) return source
            const m = Video._streamToVideo || (Video._streamToVideo = new WeakMap)
            // TODO: For the efficiency of having 1 less copy, use https://developer.mozilla.org/en-US/docs/Web/API/VideoFrame and https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrackProcessor when available. (After we make <video> work, played into sound.)
            if (!m.has(source)) { // Go through <video>.
                const el = document.createElement('video')
                if ('srcObject' in el) el.srcObject = source
                else el.src = URL.createObjectURL(source)
                el.volume = 0
                el.play() // TODO: Is this required?
                m.set(source, el)
            }
            return m.get(source)
        }
        static _dataJS(sensor, data, valuesPerTile) { // Fills data, through JS.
            const frame = Video._sourceToDrawable(this.source)
            const width = frame.videoWidth || frame.displayWidth || frame.width
            const height = frame.videoHeight || frame.displayHeight || frame.height
            // Draw frame to canvas, get ImageData.
            if (!this._canvas)
                this._canvas = document.createElement('canvas'),
                this._ctx2d = this._canvas.getContext('2d')
            // TODO: Debug devicePixelRatio. Do we need to take it into account?
            this._canvas.width = width, this._canvas.height = height
            this._ctx2d.drawImage(frame, 0, 0)
            const imageData = this._ctx2d(0, 0, width, height)
            // Actually draw the data.
            const td = this.tileDimension
            // TODO: How to sample from the imageData?
            //   TODO: Right now, how to downsample it into td×td?
        }

        static requestTab() { // With the user's permission, gets this tab's contents.
            // (Make sure that users know to select the current tab.)
            // (May fail, stalling the whole `Video` object forever.)
            if (sn._getDisplayMedia) return sn._getDisplayMedia
            return sn._getDisplayMedia = navigator.mediaDevices.getDisplayMedia({ audio:true, video:true }).then(s => {
                const cap = s.getVideoTracks[0].getCapabilities()
                sn._assert(cap.displaySurface === 'browser', "Did not pick a tab")
                return s
            })
        }
    }
}