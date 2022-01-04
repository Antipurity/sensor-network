export default function init(sn) {
    return class Scroll extends sn.Sensor {
        static docs() { return `DOM scroll position and its change.

Options:
- \`name\`: heeded, augmented.
- \`mode: 'read'|'write'|'move' = 'read'\`: can read, and either set or add-to scroll position.
- \`target:{x,y} = Pointer.tab()\`: where to get target-coordinates from (or a function, or an array). Many DOM elements can be scrollable, and \`Scroll\` tries to return the most relevant ones.
`}
        static options() {
            return {
                mode: {
                    ['Read']: () => 'read',
                    ['Read + write']: () => 'write',
                    ['Read + move']: () => 'move',
                },
                target: {
                    ["Tab's pointers"]: () => sn.Sensor.Pointer.tab(),
                    ["Virtual pointer 1"]: () => sn.Sensor.Pointer.pointer1()[0],
                    ["Virtual pointer 2"]: () => sn.Sensor.Pointer.pointer2()[0],
                },
            }
        }
        resume(opts) {
            if (opts) {
                const name = Array.isArray(opts.name) ? opts.name : typeof opts.name == 'string' ? [opts.name] : []
                const target = opts.target || sn.Sensor.Pointer.tab()
                const mode = opts.mode || 'read'
                sn._assert(target && (typeof target == 'object' || typeof target == 'function'))
                sn._assert(['read', 'write', 'move'].includes(mode))
                opts.values = 16
                opts.emptyValues = 0
                opts.name = ['scroll', ...name]
                opts.onValues = Scroll.onValues
                opts.noFeedback = mode === 'read'
                this.target = target, this.mode = mode
                this.prevs = new WeakMap // el → [curX, curY]
                this.els = []
            }
            return super.resume(opts)
        }
        static onValues(sensor, data) {
            const els = getScrollableElems(sensor.target), m = sensor.prevs
            for (let i = 0; i*4 < data.length; ++i)
                if (els[i]) {
                    const el = els[i]
                    const curX = el.scrollLeft, maxX = el.scrollWidth - el.clientWidth || 1
                    const curY = el.scrollTop, maxY = el.scrollHeight - el.clientHeight || 1
                    if (!m.has(el)) m.set(el, [curX, curY]);
                    const prev = m.get(el)
                    data[i*4+0] = (curX / maxX)*2-1
                    data[i*4+1] = (curY / maxY)*2-1
                    data[i*4+2] = Math.max(-1, Math.min((curX - prev[0]) / innerWidth, 1))
                    data[i*4+3] = Math.max(-1, Math.min((curY - prev[1]) / innerHeight, 1))
                    prev[0] = curX, prev[1] = curY
                }
            sensor.els.push(els)
            sensor.sendCallback(Scroll.onFeedback, data)
        }
        static onFeedback(feedback, sensor) {
            const els = sensor.els.shift()
            if (!feedback || sensor.mode === 'read') return
            const move = sensor.mode === 'move'
            for (let i = 0; i*4 < feedback.length; ++i)
                if (els[i]) {
                    const el = els[i]
                    if (!move) { // Set.
                        const maxX = el.scrollWidth - el.clientWidth
                        const maxY = el.scrollHeight - el.clientHeight
                        el.scrollLeft = (feedback[i*4+0]+1)/2 * maxX
                        el.scrollTop = (feedback[i*4+1]+1)/2 * maxY
                    } else { // Add.
                        el.scrollLeft += feedback[i*4+2] * innerWidth
                        el.scrollTop += feedback[i*4+3] * innerHeight
                    }
                }
        }
    }
    function getScrollableElems(pos, n=4) {
        const main = document.scrollingElement || document.documentElement
        if (!document.elementsFromPoint || n === 1) return [main]

        let p = pos
        if (typeof p == 'function') p = p()
        if (Array.isArray(p) && !p.length) return ''
        if (Array.isArray(p)) p = p[0]

        const els = document.elementsFromPoint(p.x * innerWidth, p.y * innerHeight)
        return [main, ...els.filter(isScrollable).slice(-(n-1))]
    }
    function isScrollable(el) {
        if (el === document.scrollingElement) return false // Already accounted for.
        if (el.scrollTopMax !== undefined) return el.scrollTopMax > 0
        return el.scrollHeight > el.clientHeight && ['auto', 'scroll'].includes(getComputedStyle(el).overflowY)
    }
}