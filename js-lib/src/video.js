export default function init(sn) {
    return class Video extends sn.Sensor {
        docs() { return `// TODO:

This sensor's output is composed of 1 or more tiles, which are square images.

Extra options:
- \`tileDimension = 8\`: each tile edge's length.
- \`source = Video.stitchCanvases\`: where to fetch image data from. \`MediaStream\` or \`<canvas>\` or \`<video>\` or \`<img>\` or a function to one of these.
- \`monochrome = false\`: TODO:
` }
        resume(opts) {
            if (opts) {
                const td = opts.tileDimension || 8
                const src = opts.source || Video.requestTab // TODO: Do Video.stitchCanvases by default, once we have that.
                sn._assertCounts('Non-integer tile side', td)
                sn._assert(typeof src == 'function' || src instanceof Promise || src instanceof MediaStream || src instanceof Element && (src.tagName === 'CANVAS' || src.tagName === 'VIDEO' || src.tagName === 'IMG'), "Bad source")
                this.tileDimension = td
                // (Don't catch errors in `src`, so they'll be logged to console.)
                this.source = !(src instanceof Promise) ? src : (src.then(v => this.source=v), src)
                this._tiles = 1
                this.monochrome = !!opts.monochrome
                // TODO: Other props. (For example, handle tiling and zooming-in by forking ourselves, and making `pause` responsible for the forks too.)
                opts.extraValues = 0
                opts.onValues = Video.onValues
                opts.values = this._tiles * this.tileDimension * (this.monochrome ? 1 : 3)
                opts.name = ['video', ''+td, ] // TODO: What funcs would return x & y & zoomOut & source of a cell?
                //   (zoomOut and source are the same per-`Video`-instance, I guess.)
                //   (zoomOut is -1 at level 2**0, 1 at levels 2**10+.)
                //   (source is -1 for tab (where feedback can happen), 1 for camera (where we can't control pixels).)
            }
            super.resume(opts)
        }
        // TODO: First, at least make it work for 1-tile no-zoomout no-targets.
        //   And test that it works with sound.

        // TODO: ...Okay, I think this stub is (mostly) implemented, so, should actually run it.
        //   TODO: Why is there an infinite loop now?

        static onValues(sensor, data) {
            const targetShape = sensor.cellShape()
            // TODO: Why do no-data sensors cause nothing to ever return to the event loop? This is extremely poor UI.
            if (!targetShape) return
            // Make sure to limit to one tile per cell, so that there's no misalignment.
            const dataSize = targetShape[targetShape.length-1]
            const cells = data.length / dataSize | 0
            const valuesPerCell = Math.ceil(sensor.values / cells)
            if (sensor._dataContext2d(data, valuesPerCell))
                // console.log('z'), // TODO:
                sensor.sendCallback(Video.onFeedback, data)
        }
        static onFeedback(feedback, sensor) {
            if (!feedback || sensor.noFeedback) return
            // TODO: What do we do here?
            //   …Simply upscale from feedback, exactly reversing `_dataContext2d`?
        }

        static _sourceToDrawable(source) { // .drawImage and .texImage2D can use the result.
            if (typeof source == 'function') source = source()
            if (!(source instanceof MediaStream)) return source
            const m = Video._streamToVideo || (Video._streamToVideo = new WeakMap)
            // TODO: For the efficiency of having 1 less copy, use https://developer.mozilla.org/en-US/docs/Web/API/VideoFrame and https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrackProcessor when available. (After we make <video> work, played into sound.)
            if (!m.has(source)) { // Go through <video>.
                const el = document.createElement('video')
                if ('srcObject' in el) el.srcObject = source
                else el.src = URL.createObjectURL(source)
                el.volume = 0
                // el.play() // TODO: Is this required?
                m.set(source, el)
            }
            return m.get(source)
        }
        _dataContext2d(data, valuesPerCell) { // Fills `data`.
            const frame = Video._sourceToDrawable(this.source)
            if (frame instanceof Promise) return false
            const width = frame.videoWidth || frame.displayWidth || frame.width
            const height = frame.videoHeight || frame.displayHeight || frame.height
            // Draw frame to canvas, get ImageData.
            if (!this._canvas) {
                this._canvas = document.createElement('canvas')
                this._ctx2d = this._canvas.getContext('2d')
            }
            // TODO: Debug devicePixelRatio. Do we need to take it into account?
            const td = this.tileDimension, tiles = this._tiles
            this._canvas.width = td, this._canvas.height = tiles * td
            // Draw each tile and get its ImageData, and put that into `data`.
            for (let i = 0; i < this._tiles; ++i) {
                this._ctx2d.drawImage(frame,
                    0, 0, width, height,
                    0, 0, td, i * td,
                )
            }
            // Actually draw the data.
            const monochrome = this.monochrome
            const imageData = this._ctx2d.getImageData(0, 0, td, tiles * td).data
            for (let i = 0; i < this._tiles; ++i) {
                for (let j = 0; j < valuesPerCell; ++j) {
                    const R = imageData[4 * (td*td*i + j) + 0] / 255
                    const G = imageData[4 * (td*td*i + j) + 1] / 255
                    const B = imageData[4 * (td*td*i + j) + 2] / 255
                    if (!monochrome) { // Each tile is 3 successive R/G/B cells.
                        data[(i*3 + 0) * valuesPerCell + j] = R
                        data[(i*3 + 1) * valuesPerCell + j] = G
                        data[(i*3 + 2) * valuesPerCell + j] = B
                    } else // https://en.wikipedia.org/wiki/Relative_luminance
                        data[i * valuesPerCell + j] = (0.2126*R + 0.7152*G + 0.0722*B) * 2 - 1
                }
            }
            return true
        }

        // TODO: Make the main thing `stitchCanvases`, which will be renamed to `requestTab`, and request stream from extension if possible.
        //   (getDisplayMedia seems to be too unreliable. Not to mention, it has horrible UI. If API users really want it, they can implement it themselves.)
        static requestTab() { // With the user's permission, gets this tab's contents.
            // TODO: BIG PROBLEMS:
            //   TODO: Firefox throws if the user hasn't clicked on the page yet.
            //   TODO: ...Firefox doesn't allow the actual tab, only windows?... Did it change from what I knew?
            //   ...With these problems, is it even worth it?
            // (Make sure that users know to select the current tab.)
            // (May fail, stalling the whole `Video` object forever.)
            if (sn._getDisplayMedia) return sn._getDisplayMedia // TODO: ...Now, we can't really cache, do we? So, what, promises themselves should have `.result` and `.error`?
            return sn._getDisplayMedia = navigator.mediaDevices.getDisplayMedia({ audio:true, video:true }).then(s => {
                sn._getDisplayMedia = s
                return s
            }) // TODO: Maybe, on exception, pause this sensor, so that at least others can read `.paused`? ...But what's the sensor... Should set `.error` on the promise instead, and react to it near ` instanceof Promise`.
        }
        // TODO: Also, be able to request any getDisplayMedia but without a mouse (cause it's untethered), right? `requestDisplay`.
    }
}