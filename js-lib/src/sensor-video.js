export default function init(sn) {
    const A = Object.assign
    const _tab = {}
    return A(class Video extends sn.Sensor {
        static docs() { return `A sequence of images.

Images are divided into [small patches, which has mostly been shown to work well in ML.](https://en.wikipedia.org/wiki/Vision_transformer)

This sensor's output is composed of 1 or more tiles, which are square images.    
It can target 0 or 1+ points, each shown in 1 or more tiles, and can include multiple zoom levels.

Extra options:
- \`tileDimension = 8\`: each tile edge's length.
- \`source = Video.stitchTab()\`: where to fetch image data from. \`MediaStream\` or \`<canvas>\` or \`<video>\` or \`<img>\` or a function to one of these.
    - Feedback is currently not implemented.
- \`monochrome = true\`: make this \`true\` to only report [luminance](https://en.wikipedia.org/wiki/Relative_luminance) and use 3× less data.
- \`targets = Video.pointers()\`: what to focus rectangles' centers on. This is a live array of \`{x,y}\` objects with 0…1 viewport coordinates, or a function to that, called every frame.
    - If empty, the whole \`source\` will be resized to fit, and zooming will zoom in on the center instead of zooming out; if not, the viewed rect will be centered on the target.
- \`tiling = 2\`: how many vertical/horizontal repetitions there are per target or screen.
- \`zoomSteps = 3\`: how many extra zoomed views to generate per target or screen.
- \`zoomStep = 2\`: the multiplier/divider of in-source tile dimension, per zoom step.
` }
        static options() {
            return {
                tileDimension: {
                    ['8×8']: () => 8,
                    ['16×16']: () => 16,
                    ['32×32']: () => 32,
                    ['4×4']: () => 4,
                },
                source: {
                    ["Stitch the tab's canvas/video/img elements"]: () => Video.stitchTab(),
                    ["Tab/window/screen"]: () => Video.requestDisplay(),
                    ["Camera"]: () => Video.requestCamera(),
                },
                monochrome: {
                    No: false,
                    Yes: true,
                },
                targets: {
                    ['Mouse/touch']: () => Video.pointers(),
                    ['None']: () => [],
                },
                tiling: {
                    ['2×2']: () => 2,
                    ['3×3']: () => 3,
                    ['4×4']: () => 4,
                    ['8×8']: () => 8,
                    ['1×1']: () => 1,
                },
                zoomSteps: {
                    ['3 ']: () => 3,
                    ['2 ']: () => 2,
                    ['1 ']: () => 1,
                    ['0 ']: () => 0,
                },
                zoomStep: {
                    ['2×']: () => 2,
                    ['3×']: () => 3,
                    ['4×']: () => 4,
                },
            }
        }
        pause() {
            this._nextTarget && this._nextTarget.pause()
            return super.pause()
        }
        resume(opts) {
            if (opts) {
                const td = opts.tileDimension || 8
                const src = opts.source || Video.stitchTab()
                const targ = opts.targets || Video.pointers()
                const zoomSteps = opts.zoomSteps !== undefined ? opts.zoomSteps : 3
                const zoomStep = opts.zoomStep !== undefined ? opts.zoomStep : 2
                const tiling = opts.tiling !== undefined ? opts.tiling : 2
                sn._assertCounts("Non-integer tile side", td), sn._assert(td > 0)
                sn._assert(typeof src == 'function' || src instanceof Promise || src instanceof MediaStream || src instanceof Element && (src.tagName === 'CANVAS' || src.tagName === 'VIDEO' || src.tagName === 'IMG'), "Bad source")
                sn._assert(typeof targ == 'function' || Array.isArray(targ), "Bad targets")
                sn._assertCounts("Non-integer zoom step count", zoomSteps)
                sn._assertCounts("Non-integer zoom step", zoomStep), sn._assert(zoomStep >= 2, "Pointless zoom step")
                sn._assertCounts("Non-integer tiling", tiling), sn._assert(tiling > 0)
                this.tileDimension = td
                // (Don't catch errors in `src`, so they'll be logged to console.)
                this.source = src
                this._tiles = (zoomSteps+1) * (tiling*tiling)
                this.monochrome = opts.monochrome === undefined ? true : !!opts.monochrome
                this.noFeedback = true
                this.targets = targ
                this._targetIndex = opts._targetIndex || 0
                this._opts = A(A(Object.create(null), opts), { source:src, targets:targ, _targetIndex: this._targetIndex+1 })
                if (!this._nextTarget)
                    this._nextTarget = null // Another `Video`, for multi-target support by forking.
                this.zoomSteps = zoomSteps
                this.zoomStep = zoomStep
                this.tiling = tiling
                opts.extraValues = 0
                opts.onValues = Video.onValues
                opts.values = this._tiles * td*td * (this.monochrome ? 1 : 3)
                const xyz = (dataStart, dataEnd, dataLen) => {
                    const cells = Math.ceil(dataLen / (dataEnd - dataStart))
                    const valuesPerCell = Math.ceil(this.values / cells)
                    const tile = dataStart / valuesPerCell / (this.monochrome ? 1 : 3) | 0
                    const td = this.tileDimension
                    const targets = this._targets()
                    const targ = targets[this._targetIndex]
                    const x = targ ? targ.x : 0, y = targ ? targ.y : 0
                    const zss = this.zoomSteps, zs = this.zoomStep
                    const tiling = this.tiling, t2 = tiling*tiling
                    const zoom = zs ** ((tile / t2 | 0) % (zss+1))
                    const dx = Video._tileMove(false, tile % t2, tiling)
                    const dy = Video._tileMove(true, tile % t2, tiling)
                    const color = this.monochrome ? -1 : ((dataStart / valuesPerCell | 0) % 3 - 1)
                    return {x: (x + dx*zoom*td)*2-1, y: (y + dy*zoom*td)*2-1, zoom, color}
                }
                opts.name = [
                    'video',
                    typeof opts.name == 'string' ? opts.name : String(td) + this.monochrome,
                    this.noFeedback ? 0 : (...args) => xyz(...args).x * 2 - 1,
                    this.noFeedback ? 0 : (...args) => xyz(...args).y * 2 - 1,
                    !zoomSteps ? -1 : (...args) => Math.min(Math.log2(xyz(...args).zoom) / 5 - 1, 1),
                ]
            }
            return super.resume(opts)
        }

        static bench() {
            const resolutions = new Array(3).fill().map((_,i) => 2 ** (i+9))
            return resolutions.map(river)
            function river(resolution) { // Read a MediaStream.
                const dataSize = 64
                return function start() {
                    const canvas = document.createElement('canvas')
                    const ctx = canvas.getContext('2d')
                    canvas.width = canvas.height = resolution
                    let ended = false
                    function draw() { // Make new data each animation frame.
                        if (ended) return
                        requestAnimationFrame(draw)
                        ctx.fillStyle = ['red', 'green', 'blue'][Math.random()*3 | 0]
                        ctx.fillRect(Math.random() * resolution | 0, Math.random() * resolution | 0, 51, 51)
                    }
                    draw()
                    const from = new Video({
                        // (If `source` is just `canvas`, it's super fast. Streams are slow.)
                        source: canvas.captureStream(),
                        targets: [{x:.5, y:.5}],
                        monochrome: false,
                        tileDimension: 8,
                        zoomSteps: 3,
                        zoomStep: 2,
                        tiling: 2,
                    })
                    const to = new sn.Handler({
                        dataSize,
                        noFeedback: true,
                        onValues(then, {data, error, cellShape}) { then() },
                    })
                    setTimeout(() => sn.meta.metric('resolution', resolution), 500)
                    return function stop() { ended = true, from.pause(), to.pause() }
                }
            }
        }

        static onValues(sensor, data) {
            const targetShape = sensor.cellShape()
            if (!targetShape) return
            // Make sure to limit to one tile per cell, so that there's no misalignment.
            const dataSize = targetShape[targetShape.length-1]
            const cells = data.length / dataSize | 0
            const valuesPerCell = Math.ceil(sensor.values / cells)

            // Targeting stuff.
            const targets = this._targets()
            const target = targets[this._targetIndex]
            const nextTarget = targets[this._targetIndex+1]
            if (nextTarget && !this._nextTarget) // Create.
                this._opts._targetIndex = this._targetIndex + 1,
                this._nextTarget = new Video(this._opts)
            if (nextTarget && this._nextTarget && this._nextTarget.paused) // Resume.
            this._opts._targetIndex = this._targetIndex + 1,
                this._nextTarget.resume(this._opts)
            if (!nextTarget && this._nextTarget) // Destroy.
                this._nextTarget.pause(), this._nextTarget = null

            if (sensor._dataContext2d(data, valuesPerCell, target))
                sensor.sendCallback(Video.onFeedback, data)
        }
        static onFeedback(feedback, sensor) {
            if (!feedback || sensor.noFeedback) return
            // Yep. Handle feedback here. Handle it good.
        }

        _targets() {
            const targets = typeof this.targets == 'function' ? this.targets() : this.targets
            sn._assert(Array.isArray(targets), "Bad targets")
            return targets
        }
        static _tileMove(needY=false, tile, tilesSqrt) {
            // Returns how many tiles we need to move by.
            if (tilesSqrt === 1) return 0
            const x = tile % tilesSqrt, y = tile / tilesSqrt | 0
            return needY ? y + .5 - .5*tilesSqrt : x + .5 - .5*tilesSqrt
            // Min: .5 - .5*tilesSqrt;   max: .5*tilesSqrt-.5
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
        _dataContext2d(data, valuesPerCell, target) { // Fills `data`.
            let frame = Video._sourceToDrawable(this.source)
            if (frame instanceof Promise)
                return console.error(frame.error), this.pause(), false
            if (!frame) return null
            let width = frame.videoWidth || frame.width
            let height = frame.videoHeight || frame.height
            // Draw frame to canvas, get ImageData.
            const N = 2 // Delay read-back if slow.
            if (!this._canvas) {
                this._canvas = many(() => document.createElement('canvas'))
                this._ctx2d = many((_,i) => this._canvas[i].getContext('2d', {alpha:false}))
                this._i = 0, this._slow = 0
                function many(f) { return new Array(N).fill().map(f) }
            }
            let i = ++this._i, iR = this._slow>.5 ? (i+1) % N : i % N, iW = i % N
            const canvas = this._canvas[iW], ctxWrite = this._ctx2d[iW], ctxRead = this._ctx2d[iR]
            const td = this.tileDimension, tiles = this._tiles
            const zss = this.zoomSteps, zs = this.zoomStep
            const tiling = this.tiling, t2 = tiling*tiling
            canvas.width = tiling * td, canvas.height = tiling * (zss+1) * td
            // Draw each tiling, one draw call per zoom level.
            for (let i = 0; i <= zss; ++i) {
                const zoom = zs ** i
                if (!target) { // Fullscreen.
                    const x = (width * .5 * (1-1/zoom)) | 0
                    const y = (height * .5 * (1-1/zoom)) | 0
                    ctxWrite.drawImage(frame,
                        x, y, width/zoom, height/zoom,
                        0, tiling * i * td, tiling * td, tiling * td,
                    )
                } else { // Around a target.
                    const x = (target.x * width + zoom*td*.5*(1-tiling)) | 0
                    const y = (target.y * height + zoom*td*.5*(1-tiling)) | 0
                    ctxWrite.drawImage(frame,
                        x, y, zoom*td, zoom*td,
                        0, tiling * i * td, tiling * td, tiling * td,
                    )
                }
            }
            // Actually draw the data.
            const monochrome = this.monochrome
            const readStart = performance.now()
            const imageData = ctxRead.getImageData(0, 0, tiling * td, tiling * (zss+1) * td).data
            const toRead = performance.now() - readStart
            this._slow = .9*this._slow + .1 * (toRead > 10)
            for (let i = 0; i < tiles; ++i) {
                // Only read our tile, not the ones horizontally adjacent to it.
                //   And get our tile's upper-left corner right.
                const skipPerRow = td * (tiling-1)
                const tileX = i % tiling, tileY = i / tiling | 0
                const start = tileY * tiling * td*td + tileX * td
                for (let j = 0; j < valuesPerCell; ++j) {
                    const row = j / td | 0
                    const R = imageData[4 * (start + row*skipPerRow + j) + 0] / 255
                    const G = imageData[4 * (start + row*skipPerRow + j) + 1] / 255
                    const B = imageData[4 * (start + row*skipPerRow + j) + 2] / 255
                    if (!monochrome) { // Each tile is 3 successive R/G/B data cells.
                        data[(i*3 + 0) * valuesPerCell + j] = R * 2 - 1
                        data[(i*3 + 1) * valuesPerCell + j] = G * 2 - 1
                        data[(i*3 + 2) * valuesPerCell + j] = B * 2 - 1
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
            let id = setInterval(autodetach, 10000)
            return attachEvents
            function autodetach() {
                if (performance.now() - lastRequest > 15000)
                    detachEvents(), clearInterval(id), id = null
            }
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
                if (id == null) setInterval(autodetach, 10000)
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

The result is usable as the \`targets\` option for \`Video\`.`,
        }),

        stitchTab: A(function stitchTab() {
            if (!_tab.canvas) {
                _tab.canvas = document.createElement('canvas')
                _tab.ctx = _tab.canvas.getContext('2d', {alpha:false})
                _tab.lastStitch = performance.now()
            }
            const canvas = _tab.canvas, ctx = _tab.ctx
            return function tab() {
                if (performance.now() - _tab.lastStitch < 15) return canvas
                _tab.lastStitch = performance.now()
                const w = innerWidth, h = innerHeight
                canvas.width = w, canvas.height = h
                ctx.clearRect(0, 0, w, h)
                Array.from(document.getElementsByTagName('canvas')).forEach(draw)
                Array.from(document.getElementsByTagName('video')).forEach(draw)
                Array.from(document.getElementsByTagName('img')).forEach(draw)
                return canvas
            }
            function draw(elem) {
                const r = elem.getBoundingClientRect()
                ctx.drawImage(elem, r.x | 0, r.y | 0, r.width, r.height)
            }
        }, {
            docs:`Views on-page \`<canvas>\`/\`<video>\`/\`<img>\` elements. The rest of the page is black.

The result is usable as the \`source\` option for \`Video\`.`,
        }),
        requestDisplay: A(function requestDisplay(maxWidth = 1024) { // With the user's permission, gets a screen/window/tab contents.
            // Note that in Firefox, the user has to have clicked somewhere on the page first.
            const opts = { audio:true, video: maxWidth ? { width:{max:maxWidth} } : true }
            const p = navigator.mediaDevices.getDisplayMedia(opts).then(s => {
                return p.result = s
            }).catch(e => p.error = e)
            return function() { return p }
        }, {
            docs:`[Requests a screen/window/tab stream.](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getDisplayMedia)

For performance, max width is 1024 by default; pass in 0 or something else if needed.

The result is usable as the \`source\` option for \`Video\`.`,
        }),

        requestCamera: A(function requestDisplay(maxWidth = 1024) { // With the user's permission, gets a screen/window/tab contents.
            // Note that in Firefox, the user has to have clicked somewhere on the page first.
            const opts = { audio:true, video: maxWidth ? { width:{max:maxWidth} } : true }
            const p = navigator.mediaDevices.getUserMedia(opts).then(s => {
                return p.result = s
            }).catch(e => p.error = e)
            return function() { return p }
        }, {
            docs:`[Requests a camera stream.](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)

For performance, max width is 1024 by default; pass in 0 or something else if needed.

The result is usable as the \`source\` option for \`Video\`.`,
        })
    })
}