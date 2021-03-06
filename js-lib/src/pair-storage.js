export default function init(sn) {
    const A = Object.assign
    const arrayCache = []

    // Storage is split into equal-sized chunks, 1MiB each.
    //   The first chunk is special:
    //     2b: bpv (0|1|2)
    //     4b: partSize
    //     2b: cellShape.length
    //       4b each: cellShape[i]
    //   All other chunks simply contain as many cells as can fit:
    //       cells = floor(8*chunkSize / (8 * (bpv||4) * cellSize + 2))
    //       Cells bytes: (bpv||4) * cells * cellSize
    //       noData & noFeedback bytes: 2*Math.ceil(cells / 8)
    //   (There's no separation between different sessions or even different time steps.)
    //     (Reading happens in-order.)
    //     (Use `sn.Transform.Time` to be able to distinguish time steps.)
    const chunkSize = 1024*1024
    const chunkCache = []



    class StorageSensor extends sn.Sensor {
        static docs() { return `Loads data from storage.

Note that this does not separate steps but just outputs data in ~1MiB chunks, so it is likely to sound different when visualized with \`sn.Sensor.Sound\`.

Uses [\`indexedDB\`.](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API)

Options:
- \`filename = 'sn'\`: which file was saved.
- \`pauseOnEnd = false\`: to only process the file once, use this.
- \`randomEveryNCells = null\`: if an integer, \`.random()\` will be called periodically and at the start.

Properties, to get/set at will:
- \`1 <= .nextChunk < .maxChunks\`. Note that changing this will not clear the chunk queue, so the change may take a while to become visible unless you \`.flush()\`.
- \`.maxChunks\`

Methods:
- \`.random()\`: sets \`.nextChunk\` to a random point.
- \`.flush()\`: makes the transition to another chunk instant, at the cost of a fetching delay.
` }
        static options() {
            return {
                filename: {
                    ['sn']: () => 'sn',
                    ['1 ']: () => '1',
                    ['2 ']: () => '2',
                    ['3 ']: () => '3',
                    ['4 ']: () => '4',
                },
                pauseOnEnd: {
                    No: false,
                    Yes: true,
                },
                randomEveryNCells: {
                    ['No']: () => null,
                    ['10000 ']: () => 10000,
                    ['1000000 ']: () => 1000000,
                    ['100000000 ']: () => 100000000,
                },
            }
        }
        random() { this.nextChunk = 1 + (Math.random() * (this.maxChunks-1) | 0) }
        flush() { this._chunks = [] }
        pause(inResume = false) {
            if (!inResume) {
                this.file && Promise.resolve(this.file).then(f => f.close())
                this.file = this.filename = null
            }
            return super.pause()
        }
        resume(opts) {
            const filename = (opts ? opts.filename : this.filename) || 'sn'
            if (opts) {
                sn._assert(typeof filename == 'string')
                opts.values = 0, opts.emptyValues = 0, opts.name = []
                opts.onValues = this.onValues
                opts.noFeedback = true
                this.pauseOnEnd = !!opts.pauseOnEnd
                this.randomEveryNCells = opts.randomEveryNCells
                if (!this._chunks) // Only once.
                    this._chunks = [], this._chunksToGet = 0
            }
            try {
                return super.resume(opts)
            } finally {
                // Connect to the indexedDB 'file'.
                if (filename !== this.filename) {
                    this._warned = false
                    this.file && Promise.resolve(this.file).then(f => f.close())
                    this.filename = filename
                    const p = this.file = openFile(filename).then(async f => {
                        if (p !== this.file) return f
                        // Read metadata, and/or create it if needed.
                        let ch = await loadChunk(f, 0)
                        if (p !== this.file) return f
                        if (!ch) return this.pause(), f
                        const dv = new DataView(ch.buffer, ch.byteOffset, ch.byteLength)
                        let offset = 0
                        const bpv = dv.getUint16(offset);  offset += 2
                        const partSize = dv.getUint32(offset);  offset += 4
                        const cellShape = new Array(dv.getUint16(offset));  offset += 2
                        for (let i = 0; i < cellShape.length; ++i)
                            cellShape[i] = dv.getUint32(offset), offset += 4
                        sn._assert(bpv === 0 || bpv === 1 || bpv === 2)
                        this._bytesPerValue = bpv
                        this._partSize = partSize
                        this._cellShape = cellShape
                        const cellSize = cellShape.reduce((a,b)=>a+b)
                        this._chunkCells = Math.floor(8*chunkSize / (8 * (bpv||4) * cellSize + 2))
                        this.nextChunk = 1
                        this.maxChunks = await countChunks(f)
                        if (p !== this.file) return f
                        this.cellsUntilReset = this.randomEveryNCells || 0
                        if (this.cellsUntilReset) this.random()
                        return this.file = f
                    })
                }
            }
        }
        onValues(data) {
            sn._deallocF32(data)
            if (!this.file || this.file instanceof Promise || !this.filename) return
            // Fill up the buffer.
            while (this._chunks.length + this._chunksToGet < 8) {
                const ch = this._chunks
                loadChunk(this.file, this.nextChunk++).then(chunk => { chunk && ch.push(chunk), --this._chunksToGet })
                ++this._chunksToGet
                if (this.nextChunk >= this.maxChunks) {
                    this.nextChunk = 1
                    if (this.pauseOnEnd) return this.pause()
                }
                if (Math.random()<.01) {
                    const f = this.file
                    countChunks(f).then(ch => this.file === f && (this.maxChunks = ch))
                }
            }
            // Send off a chunk from the buffer. (A whole chunk each time, no regards for step boundaries.)
            if (!this._chunks.length) return
            const chunk = this._chunks.shift()
            this.sendRawCallback(null, function name({cellShape, partSize, summary}, namer, packet, then, unname) {
                const cellSize = cellShape.reduce((a,b)=>a+b)
                const bpv = this._bytesPerValue || 4
                const fileCellSize = this._cellShape.reduce((a,b)=>a+b)
                const values = this._chunkCells * fileCellSize
                const bitLen = Math.ceil(this._chunkCells / 8)
                const ndOffset = values*bpv
                const nfOffset = ndOffset + bitLen

                const noData = fromBits(chunk.subarray(ndOffset, nfOffset))
                const noFeedback = fromBits(chunk.subarray(nfOffset, nfOffset + bitLen))
                let cells = values / fileCellSize | 0
                noData.length = cells, noFeedback.length = cells
                while (cells && noData[cells-1] && noFeedback[cells-1])
                    --cells, noData.pop(), noFeedback.pop()
                if (!cells) return

                const data = sn._allocF32(cells * cellSize)
                const dv = new DataView(chunk.buffer, chunk.byteOffset, chunk.byteLength)
                for (let c = 0; c < cells; ++c) {
                    const dataStart = c * cellSize, dataEnd = dataStart + cellSize
                    const chStart = c * fileCellSize, chEnd = chStart + fileCellSize
                    const dataNameEnd = dataStart + (cellSize - cellShape[cellShape.length-1])
                    const chNameEnd = chStart + (fileCellSize - this._cellShape[this._cellShape.length-1])
                    for (let i = chStart, j = dataStart; i < chNameEnd && j < dataNameEnd; ++i, ++j)
                        data[j] = bpv === 1 ? dv.getUint8(i) : bpv === 2 ? dv.getUint16(i*2) : dv.getFloat32(i*4) // Copy the name. Don't resize parts.
                    for (let i = chNameEnd, j = dataNameEnd; i < chEnd && j < dataEnd; ++i, ++j)
                        data[j] = bpv === 1 ? dv.getUint8(i) : bpv === 2 ? dv.getUint16(i*2) : dv.getFloat32(i*4) // Copy data.
                    if (this.cellsUntilReset)
                        if (!--this.cellsUntilReset)
                            this.random()
                }

                const error = unquantizeError(cells * cellSize, this._bytesPerValue)
                packet.send(this, then, unname, data, error, noData, noFeedback)
            })
        }
    }
    class StorageHandler extends sn.Handler {
        static docs() { return `Saves data to storage.

Data of no-data cells is already replaced with feedback, if the main handler is present to give it.

Uses [\`indexedDB\`.](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API)

Options:
- \`filename = 'sn'\`: which file to append to.
- \`bytesPerValue = 0\`: 1 to store as uint8, 2 to store as uint16, 0 to store as float32. Only relevant when first creating the file.

Functions:
- \`sn.Handler.Storage.download(filename)\`: downloads the file to the OS, importing the [StreamSaver](https://github.com/jimmywarting/StreamSaver.js?) library.
- \`sn.Handler.Storage.upload(filename, file)\`: gets the [file](https://developer.mozilla.org/en-US/docs/Web/API/File) from the OS, overwriting what is already in \`filename\`; to append, upload to a temporary location, then sense from there and handle to elsewhere.
- [\`navigator.storage.persist()\`](https://developer.mozilla.org/en-US/docs/Web/API/StorageManager/persist)
- [\`indexedDB.deleteDatabase(filename)\`](https://developer.mozilla.org/en-US/docs/Web/API/IDBFactory/deleteDatabase)
` }
        static options(opts) {
            const el = document.createElement.bind(document)
            const fileSize = el('div')
            const int1 = setInterval(async () => {
                if (document.visibilityState !== 'visible') return
                if (getComputedStyle(fileSize).visibility === 'hidden') return
                const file = await openFile(opts.filename)
                const bytes = (await countChunks(file)) * chunkSize
                const kb = bytes / 1024, mb = kb / 1024, gb = mb / 1024, tb = gb / 1024
                fileSize.textContent = tb>=1 ? '???? '+tb.toFixed(2)+' TiB' : gb>=1 ? '???? '+gb.toFixed(2)+' GiB' : mb>=1 ? '???? '+mb.toFixed(2)+' MiB' : kb>=1 ? '???? '+kb.toFixed(2)+' KiB' : '???? '+bytes.toFixed(2)+' bytes'
                file.close()
                if (!fileSize.isConnected) clearInterval(int1)
            }, 200)
            const persist = A(el('button'), { onclick() { navigator.storage.persist() } })
            persist.append('??? Request persistence')
            const download = A(el('button'), { onclick() { sn.Handler.Storage.download(opts.filename) } })
            download.append('??? To file')
            const upload = A(el('input'), { type:'file', onchange() { this.files && this.files.length && sn.Handler.Storage.upload(opts.filename, this.files[0]) } })
            upload.append('??? Replace from file')
            const deletion = A(el('button'), { onclick() { confirm('Delete '+opts.filename+'?') && deleteFile(opts.filename) } })
            deletion.append('???? No more data')
            return {
                filename: {
                    ['sn']: () => 'sn',
                    ['1 ']: () => '1',
                    ['2 ']: () => '2',
                    ['3 ']: () => '3',
                    ['4 ']: () => '4',
                },
                bytesPerValue: {
                    ['float32 (4?? size)']: () => 0,
                    ['uint16 (2?? size)']: () => 2,
                    ['uint8 (1?? size)']: () => 1,
                },
                fileSize,
                persist,
                download,
                upload,
                delete: deletion,
            }
        }
        static async download(filename, extension = '.num') {
            const file = await openFile(filename)
            try {
                const chunks = await countChunks(file)
                const streamSaver = await importStreamSaver()
                streamSaver.WritableStream = streamSaver.WritableStream
                streamSaver.TransformStream = streamSaver.TransformStream
                // `.createWriteStream` should be called on user interaction to get around popup blockers. But, eh.
                const stream = streamSaver.createWriteStream(filename+extension, {
                    size: chunks * chunkSize,
                })
                const writer = stream.getWriter()
                for (let c = 0; c < chunks; ++c)
                    await writer.write(await loadChunk(file, c))
                // Should also handle leaving:
                //   `onunload = ()=>{ writer.abort(), stream.abort() }`
                //   `onbeforeunload = evt=>{ evt.returnValue = 'Stop the download?' }`
                writer.close()
            } finally { file.close() }
        }
        static async upload(filename, file) {
            // `file`: https://developer.mozilla.org/en-US/docs/Web/API/File
            sn._assert(typeof filename == 'string')
            const chunks = Math.ceil(file.size / chunkSize)
            await deleteFile(filename)
            const ourFile = await openFile(filename)
            try {
                for (let c = 0; c < chunks; ++c)
                    saveChunk(ourFile, c, new Uint8Array(await file.slice(c*chunkSize, (c+1)*chunkSize).arrayBuffer()))
            } finally { ourFile.close() }
        }
        pause(inResume = false) {
            if (!inResume) {
                if (Array.isArray(this._chunks))
                    while (this._chunks.length > 0)
                        this.saveChunk(this._chunks.shift())
                this.file && Promise.resolve(this.file).then(f => f.close())
                this.file = this.filename = null
            }
            return super.pause()
        }
        resume(opts) {
            const filename = (opts ? opts.filename : this.filename) || 'sn'
            if (opts) {
                const bpv = opts.bytesPerValue || 0
                sn._assert(bpv === 0 || bpv === 1 || bpv === 2)
                sn._assert(typeof filename == 'string')
                opts.onValues = this.onValues
                opts.noFeedback = true
                this.bytesPerValue = bpv
                if (!this._chunks)
                    this._chunks = []
            }
            try {
                return super.resume(opts)
            } finally {
                // Connect to the indexedDB 'file'.
                if (filename !== this.filename) {
                    this._warned = false
                    this._initResolve && this._initResolve()
                    this._initPromise = new Promise(resolve => this._initResolve = resolve)
                    this.file && Promise.resolve(this.file).then(f => f.close())
                    this.filename = filename
                    const p = this.file = openFile(filename).then(async f => {
                        if (p !== this.file) return f
                        // Read metadata, and/or create it if needed.
                        let ch = await loadChunk(f, 0), dv
                        if (p !== this.file) return f
                        if (!ch) {
                            ch = allocChunk().fill(0)
                            dv = new DataView(ch.buffer, ch.byteOffset, ch.byteLength)
                            let offset = 0
                            dv.setUint16(offset, this.bytesPerValue), offset += 2
                            dv.setUint32(offset, this.partSize), offset += 4
                            dv.setUint16(offset, this.cellShape.length), offset += 2
                            for (let i = 0; i < this.cellShape.length; ++i)
                                dv.setUint32(offset, this.cellShape[i]), offset += 4
                            await saveChunk(f, 0, ch)
                            if (p !== this.file) return f
                        } else dv = new DataView(ch.buffer, ch.byteOffset, ch.byteLength)
                        let offset = 0
                        const bpv = dv.getUint16(offset);  offset += 2
                        const partSize = dv.getUint32(offset);  offset += 4
                        const cellShape = new Array(dv.getUint16(offset));  offset += 2
                        for (let i = 0; i < cellShape.length; ++i)
                            cellShape[i] = dv.getUint32(offset), offset += 4
                        sn._assert(bpv === 0 || bpv === 1 || bpv === 2)
                        this._bytesPerValue = bpv
                        this._partSize = partSize
                        this._cellShape = cellShape
                        const cellSize = cellShape.reduce((a,b)=>a+b)
                        this._chunkCells = Math.floor(8*chunkSize / (8 * (bpv||4) * cellSize + 2))
                        this.nextChunk = await countChunks(f)
                        if (p !== this.file) return f
                        this.nextCell = 999999999
                        this._initPromise = null
                        this._initResolve()
                        return this.file = f
                    })
                }
            }
        }
        async onValues(then, {data, noData, noFeedback, cellShape, partSize}) {
            if (this._initPromise) await this._initPromise
            if (!this.filename) return then()
            const mustReshape = partSize !== this._partSize || !arrayEqual(cellShape, this._cellShape)
            if (!this._warned && mustReshape) {
                console.warn("Cell shape differs from what it is in the file: got", partSize, cellShape, "but had", this._partSize, this._cellShape)
                this._warned = true
            }
            const cellSize = cellShape.reduce((a,b)=>a+b), cells = data.length / cellSize | 0
            const bpv = this._bytesPerValue || 4
            const fileCellSize = this._cellShape.reduce((a,b)=>a+b)
            const ndOffset = this._chunkCells * fileCellSize*bpv
            const nfOffset = ndOffset + Math.ceil(this._chunkCells / 8)
            const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
            for (let c = 0; c < cells; ++c) {
                if (this.nextCell >= this._chunkCells || !this._chunks.length)
                    this._chunks.push(allocChunk().fill(0)), this.nextCell = 0
                // Save name and data.
                const chunk = this._chunks[this._chunks.length-1]
                const dataStart = c * cellSize, dataEnd = (c+1) * cellSize
                const chStart = this.nextCell * fileCellSize, chEnd = ++this.nextCell * fileCellSize
                if (!mustReshape)
                    for (let i = chStart*bpv, j = dataStart*bpv; i < chEnd*bpv && j < dataEnd*bpv; ++i, ++j)
                        chunk[i] = bytes[j]
                else { // Reshape if needed. (Make some effort, at least.)
                    const dataNameEnd = dataStart + (cellSize - cellShape[cellShape.length-1])
                    const chNameEnd = chStart + (fileCellSize - this._cellShape[this._cellShape.length-1])
                    for (let i = chStart*bpv, j = dataStart*bpv; i < chNameEnd*bpv && j < dataNameEnd*bpv; ++i, ++j)
                        chunk[i] = bytes[j] // Copy the name. Don't resize parts.
                    for (let i = chNameEnd*bpv, j = dataNameEnd*bpv; i < chEnd*bpv && j < dataEnd*bpv; ++i, ++j)
                        chunk[i] = bytes[j] // Copy data.
                }
                // Save noData and noFeedback.
                if (this.nextCell === 1) chunk.fill(255, ndOffset)
                const C = this.nextCell-1 // Imagine confusing `c` for `C`. Imagine using an editor that can't highlight variables.
                const nd = noData[c], nf = noFeedback[c]
                const byte = C >>> 3, bit = C & 7
                if (nd === false) chunk[ndOffset + byte] = chunk[ndOffset + byte] & ~(1 << bit)
                if (nf === false) chunk[nfOffset + byte] = chunk[nfOffset + byte] & ~(1 << bit)
            }
            while (this._chunks.length > 1)
                this.saveChunk(this._chunks.shift())
            then()
        }
        saveChunk(chunk) { // Saves a Uint8Array chunk (consuming it) to the file.
            if (!this.file) return
            sn._assert(!(this.file instanceof Promise))
            const bpv = this._bytesPerValue, cellSize = this._cellShape.reduce((a,b)=>a+b), cells = this._chunkCells
            bigEndian(chunk.subarray(0, cells * cellSize * (bpv||4)), bpv, true)
            saveChunk(this.file, this.nextChunk++, chunk)
        }
    }
    Object.defineProperty(StorageSensor, 'name', {value:'Storage', configurable:true, writable:true})
    Object.defineProperty(StorageHandler, 'name', {value:'Storage', configurable:true, writable:true})
    return {
        sensor: StorageSensor,
        handler: StorageHandler,
    }
    function allocArray(n) { return arrayCache.length ? (arrayCache[arrayCache.length-1].length = n, arrayCache.pop()) : new Array(n) }
    function deallocArray(a) { Array.isArray(a) && arrayCache.length < 16 && (a.length = 0, arrayCache.push(a)) }
    function quantize(f32a, bpv = 0) {
        // From floats to an array of bytes, quantized to lower-but-still-`-1???1` resolutions.
        sn._assert(f32a instanceof Float32Array)
        if (!bpv) return bigEndian(new Uint8Array(f32a.buffer, f32a.byteOffset, f32a.byteLength), bpv)
        sn._assert(bpv === 1 || bpv === 2)
        const r = bpv === 1 ? new Uint8Array(f32a.length) : new Uint16Array(f32a.length)
        const scale = bpv === 1 ? 255 : 65535
        for (let i = 0; i < r.length; ++i)
            r[i] = Math.max(0, Math.min(Math.round((f32a[i]+1)/2 * scale), scale))
        return bpv === 1 ? r : bigEndian(new Uint8Array(r.buffer, r.byteOffset, r.byteLength), bpv, true)
    }
    function unquantize(a, bpv = 0) {
        sn._assert(a instanceof Uint8Array)
        if (!bpv) {
            a = bigEndian(a, bpv)
            return new Float32Array(a.buffer, a.byteOffset, a.byteLength / 4 | 0)
        }
        sn._assert(bpv === 1 || bpv === 2, bpv)
        if (bpv === 2) a = new Uint16Array(bigEndian(a, bpv).buffer)
        const r = new Float32Array(a.length)
        const scale = bpv === 1 ? 255 : 65535
        for (let i = 0; i < r.length; ++i)
            r[i] = a[i]/scale * 2 - 1
        return r
    }
    function unquantizeError(a, bpv = 0) {
        // `a`: pre-existing error; un/quantization error is added to that.
        if (!bpv) return typeof a == 'number' ? new Float32Array(a).fill(-1) : a
        const scale = bpv === 1 ? 255 : 65535
        if (typeof a == 'number') a = new Float32Array(a), a.fill(1 / scale - 1)
        else for (let i = 0; i < a.length; ++i) a[i] += 1 / scale
        return a
    }
    function checkError(a, bpv = 0) {
        if (typeof a == 'number') {
            a = new Float32Array(a)
            for (let i = 0; i < a.length; ++i) a[i] = Math.random()*2-1
        }
        const b = unquantize(quantize(a, bpv), bpv)
        const e = unquantizeError(a.length, bpv)
        sn._assert(a.length === b.length, "Unequal lengths")
        for (let i = 0; i < a.length; ++i)
            if (!(Math.abs(a[i] - b[i]) <= e[i]+1))
                return a[i] + " ??? " + b[i]
        return true
    }
    function bigEndian(a, bpv, inPlace = false) {
        // `a` is copied unless `inPlace`.
        if (bigEndian.bigEnd === undefined) {
            const x = new ArrayBuffer(2), y = new Uint16Array(x), z = new Uint8Array(x)
            y[0] = 0x0102
            bigEndian.bigEnd = z[0] === 0x01
        }
        sn._assert(a instanceof Uint8Array, "Bad byte-array")
        if (bigEndian.bigEnd || bpv === 1) return a
        if (!inPlace) a = new Uint8Array(a)
        if (bpv === 2)
            for (let i = 0; i < a.length; i += 2)
                [a[i+0], a[i+1]] = [a[i+1], a[i+0]]
        else if (bpv === 0 || bpv === 4)
            for (let i = 0; i < a.length; i += 4)
                [a[i+0], a[i+1], a[i+2], a[i+3]] = [a[i+3], a[i+2], a[i+1], a[i+0]]
        return a
    }
    function fromBits(b) { // Uint8Array ??? Array<bool>
        const a = new Array(b.length * 8) // The length is a bit inexact, which is not important for us.
        for (let i = 0; i < b.length; ++i) {
            const j = 8*i
            a[j+0] = !!(b[i] & (1<<0))
            a[j+1] = !!(b[i] & (1<<1))
            a[j+2] = !!(b[i] & (1<<2))
            a[j+3] = !!(b[i] & (1<<3))
            a[j+4] = !!(b[i] & (1<<4))
            a[j+5] = !!(b[i] & (1<<5))
            a[j+6] = !!(b[i] & (1<<6))
            a[j+7] = !!(b[i] & (1<<7))
        }
        return a
    }
    function arrayEqual(a,b) {
        if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false
        for (let i = 0; i < a.length; ++i) if (a[i] !== b[i]) return false
        return true
    }

    function allocChunk() { return chunkCache.length ? chunkCache.pop() : new Uint8Array(chunkSize) }
    function deallocChunk(a) { if (chunkCache.length < 16) chunkCache.push(a) }

    function openFile(filename) { // ??? Promise<file>
        sn._assert(typeof filename == 'string')
        const req = indexedDB.open(filename)
        return new Promise((resolve, reject) => {
            req.onerror = evt => reject(evt.target.error)
            req.onupgradeneeded = evt => evt.target.result.createObjectStore('sn-storage')
            req.onsuccess = evt => resolve(evt.target.result)
        })
    }
    function deleteFile(filename) { // ??? Promise<file>
        sn._assert(typeof filename == 'string')
        const req = indexedDB.deleteDatabase(filename)
        return new Promise((resolve, reject) => {
            req.onerror = evt => reject(evt.target.error)
            req.onsuccess = evt => {
                resolve(evt.target.result)
            }
        })
    }
    function countChunks(file) {
        const transaction = file.transaction('sn-storage', 'readonly')
        const store = transaction.objectStore('sn-storage')
        return new Promise((resolve, reject) => {
            const req = store.count()
            req.onsuccess = () => resolve(req.result)
            req.onerror = reject
        })
    }
    function loadChunk(file, index) { // ??? Promise<chunk> (Uint8Array)
        const transaction = file.transaction('sn-storage', 'readonly')
        const store = transaction.objectStore('sn-storage')
        return new Promise((resolve, reject) => {
            const req = store.get(index)
            req.onsuccess = () => resolve(req.result)
            req.onerror = reject
        })
    }
    function saveChunk(file, index, chunk) {
        // `chunk` is owned here.
        if (chunk.length !== chunkSize) {
            const old = chunk;  chunk = allocChunk().fill(255)
            for (let i = 0; i < chunkSize; ++i)
                chunk[i] = old[i] || 0
        }
        const transaction = file.transaction('sn-storage', 'readwrite', {durability:'relaxed'})
        const store = transaction.objectStore('sn-storage')
        transaction.oncomplete = () => deallocChunk(chunk)
        store.put(chunk, index)
        transaction.commit()
    }
    function importStreamSaver() { // For saving >2GiB files.
        if (importStreamSaver.did) return importStreamSaver.did
        return importStreamSaver.did = new Promise(resolve => {
            const el1 = document.createElement('script')
            const el2 = document.createElement('script')
            el1.src = 'https://cdn.jsdelivr.net/npm/web-streams-polyfill@2.0.2/dist/ponyfill.min.js'
            el2.src = 'https://cdn.jsdelivr.net/npm/streamsaver@2.0.3/StreamSaver.min.js'
            document.head.append(el1, el2)
            const id = setInterval(() => {
                if (self.streamSaver && self.WebStreamsPolyfill)
                    clearInterval(id), resolve(self.streamSaver)
            }, 50)
        })
    }
}