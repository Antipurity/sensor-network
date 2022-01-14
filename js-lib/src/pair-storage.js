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
        static docs() { return `TODO:

Options:
- TODO: \`filename = 'sn'\`
- TODO: What else do we want?
` }
        static options() {
            return {
                // TODO: filename. Like 5 options.
            }
        }
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
                        return this.file = f
                    })
                }
            }
        }
        // TODO: How would we decide whether to go from the start, or from a random position, and when to switch to a random position? Maybe just start at 1, and allow users to reset position to random?
        //   TODO: Maybe, have an option for periodic auto-reset?
        // TODO: Maybe, a method to pick `this.nextChunk` randomly (or set it to an index, or even get it)?
        onValues(data) {
            sn._deallocF32(data)
            if (this.file instanceof Promise || !this.filename) return
            // Fill up the buffer.
            while (this._chunks.length + this._chunksToGet < 8) {
                loadChunks.then(chunk => { this._chunks.push(chunk), --this._chunksToGet })
                ++this._chunksToGet
            }
            // Send off a chunk from the buffer. (A whole chunk each time, no regards for step boundaries.)
            if (!this._chunks.length) return
            const chunk = this._chunks.shift()
            this.sendRawCallback(null, function name({cellShape, partSize, summary}, namer, packet, then, unname) {
                const cellSize = cellShape.reduce((a,b)=>a+b), cells = data.length / cellSize | 0
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
                const dataStart = 0, dataEnd = cells * cellSize
                const chStart = 0, chEnd = cells * fileCellSize
                const dataNameEnd = dataStart + (cellSize - cellShape[cellShape.length-1])
                const chNameEnd = chStart + (fileCellSize - this._cellShape[this._cellShape.length-1])
                for (let i = chStart, j = dataStart; i < chNameEnd && j < dataNameEnd; ++i, ++j)
                    data[j] = dv[i] // Copy the name. Don't resize parts.
                for (let i = chNameEnd, j = dataNameEnd; i < chEnd && j < dataEnd; ++i, ++j)
                    data[j] = dv[i] // Copy data.

                const error = unquantizeError(cells * cellSize, this._bytesPerValue)
                packet.send(this, then, unname, data, error, noData, noFeedback)
            })
        }
    }
    class StorageHandler extends sn.Handler {
        static docs() { return `TODO:

Options:
- \`filename = 'sn'\`: which file to append to.
- \`bytesPerValue = 0\`: 1 to store as uint8, 2 to store as uint16, 0 to store as float32. Only relevant when first creating the file.
- TODO: What else do we want? ...Nothing?
` }
        static options() {
            // TODO: Make `options`, when it encounters `x instanceof Node`, just put it into the result. (So that we can have buttons.)
            return {
                // TODO: filename. Provide like, 5 options.
                // TODO: bytesPerValue.
                // TODO: A button that does navigator.storage.persist()
                // TODO: Also a button that deletes the file. (And maybe also show its current size, based on the chunk-count, updating every second?)
            }
        }
        pause(inResume = false) {
            if (!inResume) {
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
            for (let c = 0; c < cells; ++c) {
                if (this.nextCell >= this._chunkCells || !this._chunks.length)
                    this._chunks.push(allocChunk().fill(0)), this.nextCell = 0
                // Save name and data.
                const chunk = this._chunks[this._chunks.length-1]
                const dataStart = c * cellSize, dataEnd = (c+1) * cellSize
                const chStart = this.nextCell * fileCellSize, chEnd = ++this.nextCell * fileCellSize
                if (!mustReshape)
                    for (let i = chStart*bpv, j = dataStart*bpv; i < chEnd*bpv && j < dataEnd*bpv; ++i, ++j)
                        chunk[i] = data[j]
                else { // Reshape if needed. (Make some effort, at least.)
                    const dataNameEnd = dataStart + (cellSize - cellShape[cellShape.length-1])
                    const chNameEnd = chStart + (fileCellSize - this._cellShape[this._cellShape.length-1])
                    for (let i = chStart*bpv, j = dataStart*bpv; i < chNameEnd*bpv && j < dataNameEnd*bpv; ++i, ++j)
                        chunk[i] = data[j] // Copy the name. Don't resize parts.
                    for (let i = chNameEnd*bpv, j = dataNameEnd*bpv; i < chEnd*bpv && j < dataEnd*bpv; ++i, ++j)
                        chunk[i] = data[j] // Copy data.
                }
                // Save noData and noFeedback.
                if (this.nextCell === 1) chunk.fill(255, ndOffset)
                const nd = noData[c], nf = noFeedback[c]
                const byte = c >>> 3, bit = c & 7
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
            bigEndian(chunk.subarray(0, cells * cellSize), bpv, true)
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
        // From floats to an array of bytes, quantized to lower-but-still-`-1…1` resolutions.
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
                return a[i] + " ≠ " + b[i]
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
    function toBits(a) { // Array<bool> → Uint8Array
        const b = new Uint8Array(Math.ceil(a.length / 8))
        for (let i = 0; i < b.length; ++i) {
            const j = 8*i
            b[i] = (a[j+0]<<7) | (a[j+1]<<6) | (a[j+2]<<5) | (a[j+3]<<4) | (a[j+4]<<3) | (a[j+5]<<2) | (a[j+6]<<1) | (a[j+7]<<0)
        }
        return b
    }
    function fromBits(b) { // Uint8Array → Array<bool>
        const a = allocArray(b.length * 8) // The length is a bit inexact, which is not important for us.
        for (let i = 0; i < b.length; ++i) {
            const j = 8*i
            a[j+0] = !!(b[i] & (1<<7))
            a[j+1] = !!(b[i] & (1<<6))
            a[j+2] = !!(b[i] & (1<<5))
            a[j+3] = !!(b[i] & (1<<4))
            a[j+4] = !!(b[i] & (1<<3))
            a[j+5] = !!(b[i] & (1<<2))
            a[j+6] = !!(b[i] & (1<<1))
            a[j+7] = !!(b[i] & (1<<0))
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

    function openFile(filename) { // → Promise<file>
        sn._assert(typeof filename == 'string')
        const req = indexedDB.open(filename)
        return new Promise((resolve, reject) => {
            req.onerror = evt => reject(evt.target.error)
            req.onupgradeneeded = evt => {
                const db = evt.target.result
                db.createObjectStore('sn-storage')
            }
            req.onsuccess = evt => {
                const db = evt.target.result
                resolve(db)
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
    function loadChunk(file, index) { // → Promise<chunk> (Uint8Array)
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
        const transaction = file.transaction('sn-storage', 'readwrite', {durability:'relaxed'})
        const store = transaction.objectStore('sn-storage')
        transaction.oncomplete = () => deallocChunk(chunk)
        store.put(chunk, index)
        transaction.commit()
    }
}