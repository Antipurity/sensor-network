export default function init(sn) {
    const A = Object.assign
    return A(class Video extends sn.Sensor {
        docs() { return `// TODO:

This sensor's output is composed of 1 or more tiles, which are square images.

Extra options:
- \`tileDimension = 8\`: each tile edge's length.
- \`source = Video.stitchCanvases()\`: where to fetch image data from. \`MediaStream\` or \`<canvas>\` or \`<video>\` or \`<img>\` or a function to one of these.
- \`monochrome = false\`: make this \`true\` to only report [luminance](https://en.wikipedia.org/wiki/Relative_luminance) and use 3× less data.
` }
        resume(opts) {
            if (opts) {
                const td = opts.tileDimension || 8
                const src = opts.source || Video.requestDisplay() // TODO: Do Video.stitchTab() by default, once we have that.
                sn._assertCounts('Non-integer tile side', td)
                sn._assert(typeof src == 'function' || src instanceof Promise || src instanceof MediaStream || src instanceof Element && (src.tagName === 'CANVAS' || src.tagName === 'VIDEO' || src.tagName === 'IMG'), "Bad source")
                this.tileDimension = td
                // (Don't catch errors in `src`, so they'll be logged to console.)
                this.source = src
                this._tiles = 1
                this.monochrome = !!opts.monochrome
                this.noFeedback = true // TODO: If `source` includes feedback canvases, then set this to false.
                //   `typeof src == 'function' && typeof src.onFeedback == 'function'`? (`onFeedback` accepting the data-elem and the feedback-canvas. TODO: But how to synchronize frames?)
                // TODO: Other props. (For example, handle tiling and zooming-in by forking ourselves, and making `pause` responsible for the forks too.)
                opts.extraValues = 0
                opts.onValues = Video.onValues
                opts.values = this._tiles * td*td * (this.monochrome ? 1 : 3)
                const xy = this.noFeedback ? null : (dataStart, dataEnd, dataLen) => {
                    const cells = Math.ceil(dataLen / (dataEnd - dataStart))
                    const valuesPerCell = Math.ceil(this.values / cells)
                    const tile = dataStart / valuesPerCell / (this.monochrome ? 1 : 3) | 0
                    return {x:0, y:0} // TODO: Infer target x/y, and offset the tile.
                }
                const zoomOut = 1 // TODO: Set this.zoomOut to this. And make drawing respect this.
                opts.name = [
                    'video',
                    ''+td,
                    this.noFeedback ? 0 : (...args) => xy(...args).x,
                    this.noFeedback ? 0 : (...args) => xy(...args).y,
                    Math.log2(zoomOut) / 5 - 1,
                    this.noFeedback ? 1 : -1,
                ]
            }
            super.resume(opts)
        }
        // TODO: Allow many `targets`, self-duplicating to cover more than 1.
        //   By default, is Video.pointers(), and is called each frame.
        //   How exactly do we allow this? Make `onValues` check whether the next element exists, and pause/resume (and maybe create) `this._nextTarget` (with `opts._target = this._target+1`)?
        // TODO: Allow many zoom-out levels.
        //   (If no-target, zoom in on the center instead.)
        // TODO: Allow many tiles.

        static onValues(sensor, data) {
            const targetShape = sensor.cellShape()
            if (!targetShape) return
            // Make sure to limit to one tile per cell, so that there's no misalignment.
            const dataSize = targetShape[targetShape.length-1]
            const cells = data.length / dataSize | 0
            const valuesPerCell = Math.ceil(sensor.values / cells)
            if (sensor._dataContext2d(data, valuesPerCell))
                sensor.sendCallback(Video.onFeedback, data)
        }
        static onFeedback(feedback, sensor) {
            if (!feedback || sensor.noFeedback) return
            // TODO: What do we do here?
            //   …Simply upscale from feedback, exactly reversing `_dataContext2d`?
        }

        static _sourceToDrawable(source) { // .drawImage and .texImage2D can use the result.
            if (typeof source == 'function') source = source()
            if (source instanceof Promise) {
                if ('result' in source) source = source.result
                else if ('error' in source) return source
                else return null
            }
            if (!(source instanceof MediaStream)) return source
            const m = Video._streamToVideo || (Video._streamToVideo = new WeakMap)
            // TODO: For the efficiency of having 1 less copy, use https://developer.mozilla.org/en-US/docs/Web/API/VideoFrame and https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrackProcessor when available. (After we make <video> work, played into sound.)
            if (!m.has(source)) { // Go through <video>.
                const el = document.createElement('video')
                if ('srcObject' in el) el.srcObject = source
                else el.src = URL.createObjectURL(source)
                el.volume = 0
                el.play()
                m.set(source, el)
            }
            const el = m.get(source)
            return el
        }
        _dataContext2d(data, valuesPerCell) { // Fills `data`.
            let frame = Video._sourceToDrawable(this.source)
            if (frame instanceof Promise)
                return console.error(frame.error), this.pause(), false
            if (!frame) return null
            let width = frame.videoWidth || frame.displayWidth || frame.width
            let height = frame.videoHeight || frame.displayHeight || frame.height
            // Draw frame to canvas, get ImageData.
            if (!this._canvas) {
                this._canvas = document.createElement('canvas')
                this._ctx2d = this._canvas.getContext('2d')
                document.body.append(this._canvas) // TODO: Don't do this visualization after we're done.
            }
            const td = this.tileDimension, tiles = this._tiles
            this._canvas.width = td, this._canvas.height = tiles * td
            // Draw each tile and get its ImageData, and put that into `data`.
            for (let i = 0; i < tiles; ++i) {
                this._ctx2d.drawImage(frame,
                    0, 0, width, height,
                    0, 0, td, td,
                )
            }
            // Actually draw the data.
            const monochrome = this.monochrome
            const imageData = this._ctx2d.getImageData(0, 0, td, tiles * td).data
            for (let i = 0; i < tiles; ++i) {
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
    }, {
        pointers: A(function pointers() {
            const ps = []
            const inds = new Map
            const passive = {passive:true}
            let attached = false, lastRequest = performance.now()
            let id = setInterval(() => {
                // Auto-detach when not needed.
                if (performance.now() - lastRequest > 15000)
                    detachEvents(), clearInterval(id), id = null
            }, 10000)
            return attachEvents
            function onpointerdown(evt) { // Add/update.
                if (evt.pointerType === 'touch') return
                if (evt.touches) return Array.from(evt.touches).forEach(onpointerdown)
                const p = ps[indexOf(evt)]
                p.x = Math.max(0, Math.min(evt.clientX / innerWidth, 1))
                p.y = Math.max(0, Math.min(evt.clientY / innerHeight, 1))
            }
            function onpointerup(evt) { // Remove.
                if (evt.pointerType === 'touch') return
                if (!ps.length) return
                if (evt.touches) {
                    ps.length = 0, inds.clear()
                    return Array.from(evt.touches).forEach(onpointerdown)
                }
                if (ps.length <= 1) return
                const i = indexOf(evt), j = ps.length-1
                ;[ps[i], ps[j]] = [ps[j], ps[i]]
                ps.pop(), inds.delete(idOf(evt))
            }
            function idOf(evt) { return evt.pointerId !== undefined ? evt.pointerId : 'identifier' in evt ? evt.identifier : 'mouse' }
            function indexOf(evt) {
                const id = idOf(evt)
                if (!inds.has(id))
                    inds.set(id, ps.push({ x:.5, y:.5 })-1)
                return inds.get(id)
            }
            function attachEvents() {
                if (id == null) return null
                lastRequest = performance.now()
                if (attached) return ps
                addEventListener('touchstart', onpointerdown, passive)
                addEventListener('touchmove', onpointerdown, passive)
                addEventListener('touchcancel', onpointerup, passive)
                addEventListener('touchend', onpointerup, passive)
                if (typeof PointerEvent == 'undefined') {
                    addEventListener('mousedown', onpointerdown, passive)
                    addEventListener('mousemove', onpointerdown, passive)
                    addEventListener('mouseup', onpointerup, passive)
                } else {
                    addEventListener('pointerdown', onpointerdown, passive)
                    addEventListener('pointermove', onpointerdown, passive)
                    addEventListener('pointerup', onpointerup, passive)
                }
                attached = true
                return ps
            }
            function detachEvents() {
                if (!attached) return
                removeEventListener('touchstart', onpointerdown, passive)
                removeEventListener('touchmove', onpointerdown, passive)
                removeEventListener('touchcancel', onpointerup, passive)
                removeEventListener('touchend', onpointerup, passive)
                if (typeof PointerEvent == 'undefined') {
                    removeEventListener('mousedown', onpointerdown, passive)
                    removeEventListener('mousemove', onpointerdown, passive)
                    removeEventListener('mouseup', onpointerup, passive)
                } else {
                    removeEventListener('pointerdown', onpointerdown, passive)
                    removeEventListener('pointermove', onpointerdown, passive)
                    removeEventListener('pointerup', onpointerup, passive)
                }
                attached = false
            }
        }, {
            docs:`Returns a closure that returns the dynamic list of mouse/touch positions, \`[{x,y}]\`.

Usable as the \`source\` option for \`Video\`.`,
        }),

        // TODO: Make the main thing `stitchTab`:
        //   TODO: Request stream from extension if possible, through DOM events. Else:
        //   TODO: document.querySelectorAll('canvas, video, img')
        //   TODO: Clear the canvas, and draw each thing onto it.
        requestDisplay: A(function() { // With the user's permission, gets a screen/window/tab contents. // TODO: Document, via `Object.assign`ments.
            // Note that in Firefox, the user has to have clicked somewhere on the page first.
            const p = navigator.mediaDevices.getDisplayMedia({ audio:true, video:true }).then(s => {
                return p.result = s
            }).catch(e => p.error = e)
            return function() { return p }
        }, {
            docs:``, // TODO:
        }),
        requestCamera: A(function() { // With the user's permission, gets a screen/window/tab contents. // TODO: Document, via `Object.assign`ments.
            // Note that in Firefox, the user has to have clicked somewhere on the page first.
            const p = navigator.mediaDevices.getUserMedia({ audio:true, video:true }).then(s => {
                return p.result = s
            }).catch(e => p.error = e)
            return function() { return p }
        }, {
            docs:``, // TODO:
        })
    })
}