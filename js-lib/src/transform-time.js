export default function init(sn) {
    return class Time extends sn.Transform {
        static docs() { return `Reports start & end times of when each data cell was collected.

[Depends on the user's system time, not synchronized explicitly.](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/Date)

Reserve at least 1 of \`userParts\` for this.    
With at least 24 \`partSize\`, the augmentation looks like this (down is time):    
![Sine-waves of exponentially-decreasing frequency.](data:image/png;base64,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)    
With less, it has to improvise, with max resolution being 1/8sec, and min resolution at least 8sec.

Options:
- \`timing = () => +new Date()/1000\`: provides the seconds used to annotate data.
` }
        static options() { return {
            // TODO: An option for providing another timing function, such as `performance.now()` or relative-performance-now or step-counting. (Because synthetic data would really like to have control, this isn't just for humans.)
        } }
        resume(opts) {
            if (opts)
                this.timing = opts.timing || (() => +new Date()/1000),
                opts.onValues = Time.onValues, this.last = this.timing()
            return super.resume(opts)
        }
        static onValues(then, {data, cellShape, partSize}) {
            if (!data.length) return then()
            const user = cellShape[0], userParts = user / partSize | 0
            if (!userParts) return then()
            const now = this.timing()
            let start, mid
            if (userParts >= 3) // 2 last parts.
                start = partSize * (userParts-2) || 1, mid = start + partSize
            else // 1 last part, split.
                start = partSize * (userParts-1) || 1, mid = start + (partSize / 2 | 0)
            Time.fill(data, start, mid, this.last)
            Time.fill(data, mid, user, now)
            const cellSize = cellShape.reduce((a,b) => a+b), cells = data.length / cellSize | 0
            for (let c = 1; c < cells; ++c)
                data.copyWithin(start + c * cellSize, start, user)
            this.last = now
            then()
        }
        static fill(data, start, end, sec) {
            if (end-start >= 24) { // Ensure at least 8 seconds as the most-coarse period.
                const tpi = sec*1000 * 2*Math.PI / 10
                for (let i = start, div = 1; i < end; ++i, div *= 1.4)
                    data[i] = Math.sin(tpi / div)
            } else { // 1/8s, 1/2s, 2s, 8s…
                const rate = end-start >= 8 ? 2 : end-start >= 4 ? 4 : 16
                const tpi = sec*8 * 2*Math.PI
                for (let i = start, div = 1; i < end; ++i, div *= rate)
                    data[i] = Math.sin(tpi / div)
            }
        }
    }
}