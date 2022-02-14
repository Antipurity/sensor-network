export default function init(sn) {
    const arrayCache = []
    return class Shuffle extends sn.Transform {
        static docs() { return `Shuffles cells randomly each step.

\`.Handler.Storage\` erases borders between steps, so this can help with autoregressive modeling. Also useful for learning position-invariance for models that do not have that enforced.` }
        static options() { return {
        } }
        static tests() {
            const cellSize = 2
            const n = 32, data = new Array(n).fill().map((_,i) => i/cellSize|0)
            const A = shuffledIndices(n / cellSize | 0)
            const B = new Array(n).fill().map((_,i) => A[i/cellSize|0])
            return [
                ["Shuffle works", B, shuffleArray(data.slice(), A, cellSize)],
                ["Shuffle+unshuffle works", data, shuffleArray(B.slice(), reverseIndices(A.slice()), cellSize)],
            ]
        }
        resume(opts) {
            if (opts)
                opts.onValues = Shuffle.onValues
            return super.resume(opts)
        }
        static onValues(then, {data, error, noData, noFeedback, cellShape, partSize}) {
            if (!data || !data.length || data.length == 1) return then()
            const cellSize = cellShape.reduce((a,b)=>a+b), cells = data.length / cellSize | 0
            const inds = shuffledIndices(cells)
            shuffleArray(data, inds, cellSize)
            shuffleArray(error, inds, cellSize)
            shuffleArray(noData, inds, 1)
            shuffleArray(noFeedback, inds, 1)
            then(inds)
        }
        static onFeedback(then, {data, error, noData, noFeedback, cellShape, partSize}, feedback, extra) {
            if (!extra) return then()
            const inds = reverseIndices(extra)
            shuffleArray(feedback, inds, cellSize)
            shuffleArray(data, inds, cellSize)
            shuffleArray(error, inds, cellSize)
            shuffleArray(noData, inds, 1)
            shuffleArray(noFeedback, inds, 1)
            deallocArray(inds)
            then()
        }
    }
    function allocArray(n) { return arrayCache.length ? (arrayCache[arrayCache.length-1].length = n, arrayCache.pop()) : new Array(n) }
    function deallocArray(a) { Array.isArray(a) && arrayCache.length < 32 && (a.length = 0, arrayCache.push(a)) }
    function shuffledIndices(n) {
        // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        const a = allocArray(n)
        for (let i=0; i<n; ++i) {
            const j = Math.random() * (i+1) | 0
            if (i !== j) a[i] = a[j]
            a[j] = i
        }
        return a
    }
    function reverseIndices(inds) { // Consumes `inds`.
        const inds2 = shuffledIndices(inds.length)
        for (let i=0; i < inds.length; ++i) inds2[inds[i]] = i
        deallocArray(inds)
        return inds2
    }
    function shuffleArray(data, inds, cellSize) {
        // data2[i] = data[inds[i]] but in-place.
        // https://stackoverflow.com/questions/7365814/in-place-array-reordering
        if (!data) return data
        const n = inds.length, inds2 = allocArray(n), x = allocArray(cellSize).fill(0)
        for (let i=0; i<n; ++i) inds2[i] = inds[i]
        for (let i=0; i<n; ++i) {
            for (let z = 0; z < cellSize; ++z)
                x[z] = data[i*cellSize + z]
            let j = i, k
            while (true) {
                k = inds2[j], inds2[j] = j
                if (k === i) break
                for (let z = 0; z < cellSize; ++z)
                    data[j*cellSize + z] = data[k*cellSize + z]
                j = k
            }
            for (let z = 0; z < cellSize; ++z)
                data[j*cellSize + z] = x[z]
        }
        deallocArray(x), deallocArray(inds2)
        return data
    }
}