export default function init(sn) {
    const A = Object.assign
    return A(class Video extends sn.Sensor {
        static docs() { return `A sequence of images.

Images are divided into [small patches, which has mostly been shown to work well in ML.](https://en.wikipedia.org/wiki/Vision_transformer)

This sensor's output is composed of 1 or more tiles, which are square images.    
It can target 0 or 1+ points, each shown in 1 or more tiles, and can include multiple zoom levels.

Extra options:
- \`name\`: heeded, augmented.
- \`tileDimension = 8\`: each tile edge's length.
- \`source = Video.stitchTab()\`: where to fetch image data from. \`MediaStream\` or \`<canvas>\` or \`<video>\` or \`<img>\` or a function to one of these.
    - Feedback is currently not implemented.
- \`monochrome = true\`: make this \`true\` to only report [luminance](https://en.wikipedia.org/wiki/Relative_luminance) and use 3× less data.
- \`targets = Pointer.tab()\`: what to focus rectangles' centers on. This is a live array of \`{x,y}\` objects with 0…1 viewport coordinates, or a function to that, called every frame.
    - If empty, the whole \`source\` will be resized to fit, and zooming will zoom in on the center instead of zooming out; if not, the viewed rect will be centered on the target.
- \`tiling = 2\`: how many vertical/horizontal repetitions there are per target or screen.
- \`zoomSteps = 3\`: how many extra zoomed views to generate per target or screen.
- \`zoomStepStart = 0\`: the least-zoomed zoom level.
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
                    ["Stitch the tab's canvas/video/img elements"]: () => sn.Sensor.Video.stitchTab(),
                    ["Tab/window/screen"]: () => sn.Sensor.Video.requestDisplay(),
                    ["Camera"]: () => sn.Sensor.Video.requestCamera(),
                },
                monochrome: {
                    Yes: true,
                    No: false,
                },
                targets: {
                    ["Tab's pointers"]: () => sn.Sensor.Pointer.tab(),
                    ["Virtual pointer 1"]: () => sn.Sensor.Pointer.pointer1(),
                    ["Virtual pointer 2"]: () => sn.Sensor.Pointer.pointer2(),
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
                zoomStepStart: {
                    ['0 ']: () => 0,
                    ['1 ']: () => 1,
                    ['2 ']: () => 2,
                    ['3 ']: () => 3,
                },
                zoomStep: {
                    ['2×']: () => 2,
                    ['3×']: () => 3,
                    ['4×']: () => 4,
                    ['8×']: () => 8,
                    ['16×']: () => 16,
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
                const targ = opts.targets || sn.Sensor.Pointer.tab()
                const zoomSteps = opts.zoomSteps !== undefined ? opts.zoomSteps : 3
                const zoomStepStart = opts.zoomStepStart || 0
                const zoomStep = opts.zoomStep !== undefined ? opts.zoomStep : 2
                const tiling = opts.tiling !== undefined ? opts.tiling : 2
                sn._assertCounts("Non-integer tile side", td), sn._assert(td > 0)
                sn._assert(typeof src == 'function' || src instanceof Promise || src instanceof MediaStream || src instanceof Element && (src.tagName === 'CANVAS' || src.tagName === 'VIDEO' || src.tagName === 'IMG'), "Bad source")
                sn._assert(typeof targ == 'function' || Array.isArray(targ), "Bad targets")
                sn._assertCounts("Non-integer zoom step count", zoomSteps, zoomStepStart)
                sn._assert(zoomStepStart < zoomSteps, "Too zoomed-out at the start")
                sn._assertCounts("Non-integer zoom step", zoomStep), sn._assert(zoomStep >= 2, "Pointless zoom step")
                sn._assertCounts("Non-integer tiling", tiling), sn._assert(tiling > 0)
                this.tileDimension = td
                // (Don't catch errors in `src`, so they'll be logged to console.)
                this.source = src
                this._tiles = (zoomSteps - zoomStepStart + 1) * (tiling*tiling)
                this.monochrome = opts.monochrome === undefined ? true : !!opts.monochrome
                this.targets = targ
                this._targetIndex = opts._targetIndex || 0
                const A = Object.assign, C = Object.create
                this._opts = A(A(C(null), opts), { source:src, targets:targ, _targetIndex: this._targetIndex+1 })
                if (!this._nextTarget)
                    this._nextTarget = null // Another `Video`, for multi-target support by forking.
                this.zoomSteps = zoomSteps
                this.zoomStepStart = zoomStepStart
                this.zoomStep = zoomStep
                this.tiling = tiling
                this._width = 1, this._height = 1
                opts.emptyValues = 0
                opts.onValues = this.onValues
                opts.noFeedback = true
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
                    const zoom = zs ** ((tile / t2 | 0) % (zss - this.zoomStepStart + 1) + this.zoomStepStart)
                    const dx = Video._tileMove(false, tile % t2, tiling)
                    const dy = Video._tileMove(true, tile % t2, tiling)
                    const color = this.monochrome ? -1 : ((dataStart / valuesPerCell | 0) % 3 - 1)
                    const x2 = clamp(x + dx*zoom*td/this._width)
                    const y2 = clamp(y + dy*zoom*td/this._height)
                    return {x: x2, y: y2, zoom, color}
                    function clamp(x) { return Math.max(0, Math.min(x, 1)) }
                }
                const name = Array.isArray(opts.name) ? opts.name : typeof opts.name == 'string' ? [opts.name] : [String(td) + this.monochrome]
                opts.name = [
                    'video',
                    ...name,
                    (...args) => xyz(...args).x * 2 - 1,
                    (...args) => xyz(...args).y * 2 - 1,
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

        onValues(data) {
            // Make sure to limit to one tile per cell, so that there's no misalignment.
            const cellShape = this.cellShape()
            if (!cellShape) return
            const dataSize = cellShape[cellShape.length-1]
            const valuesPerCell = dataSize // Since this.emptyValues === 0.

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

            if (this._dataContext2d(data, valuesPerCell, target))
                this.sendCallback(this.onFeedback, data)
        }
        onFeedback(feedback) {
            if (!feedback || this.noFeedback) return
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
            let width = this._width = frame.videoWidth || frame.width || 1
            let height = this._height = frame.videoHeight || frame.height || 1
            // Draw frame to canvas, get ImageData.
            const N = 2 // Delay read-back if slow.
            if (!this._canvas) {
                this._canvas = many(() => document.createElement('canvas'))
                this._canvas.forEach(c => {c.style.width='64px', c.style.imageRendering = 'pixelated'}) // TODO:
                this._canvas.forEach(c => document.body.append(c)) // TODO:
                this._ctx2d = many((_,i) => this._canvas[i].getContext('2d', {alpha:false}))
                this._i = 0, this._slow = 0
                function many(f) { return new Array(N).fill().map(f) }
            }
            let i = ++this._i, iR = this._slow>.5 ? (i+1) % N : i % N, iW = i % N
            const canvas = this._canvas[iW], ctxWrite = this._ctx2d[iW], ctxRead = this._ctx2d[iR]
            const td = this.tileDimension, tiles = this._tiles
            const zss = this.zoomSteps, zs = this.zoomStep
            const tiling = this.tiling
            canvas.width = tiling * td, canvas.height = tiling * (zss+1) * td
            // Draw each tiling, one draw call per zoom level.
            for (let i = this.zoomStepStart, j = 0; i <= zss; ++i, ++j) {
                const zoom = zs ** i
                if (!target) { // Fullscreen.
                    const x = (width * .5 * (1-1/zoom)) | 0
                    const y = (height * .5 * (1-1/zoom)) | 0
                    ctxWrite.drawImage(frame,
                        x, y, width/zoom, height/zoom,
                        0, tiling * j * td, tiling * td, tiling * td,
                    )
                } else { // Around a target.
                    const x = (target.x * width + zoom*td*.5*(1-tiling)) | 0
                    const y = (target.y * height + zoom*td*.5*(1-tiling)) | 0
                    ctxWrite.drawImage(frame,
                        x, y, zoom*td, zoom*td,
                        0, tiling * j * td, tiling * td, tiling * td,
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
        stitchTab: A(function stitchTab() {
            const _tab = sn.Sensor.Video._tab || (sn.Sensor.Video._tab = {})
            if (!_tab.canvas) {
                _tab.canvas = document.createElement('canvas')
                _tab.ctx = _tab.canvas.getContext('2d', {alpha:false})
                _tab.lastStitch = performance.now()
            }
            const canvas = _tab.canvas, ctx = _tab.ctx
            let w, h
            return function tab() {
                if (performance.now() - _tab.lastStitch < 15) return canvas
                _tab.lastStitch = performance.now()
                w = innerWidth, h = innerHeight
                canvas.width = w, canvas.height = h
                ctx.clearRect(0, 0, w, h)
                Array.from(document.getElementsByTagName('canvas')).forEach(draw)
                Array.from(document.getElementsByTagName('video')).forEach(draw)
                Array.from(document.getElementsByTagName('img')).forEach(draw)
                return canvas
            }
            function draw(elem) {
                const st = getComputedStyle(elem)
                if (st.visibility !== 'visible') return
                const r = elem.getBoundingClientRect()
                if (r.x + r.width < 0 || r.y + r.height < 0 || r.x > w || r.y > h) return
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
        }),
    })
}