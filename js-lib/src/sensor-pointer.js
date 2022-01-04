export default function init(sn) {
    const A = Object.assign
    return A(class Pointer extends sn.Sensor {
        static docs() { return `Tracks or controls mouse/touch state.

Options:
- \`name\`: heeded, augmented.
- \`pointers = 1\`: how many pointers to track/control. [Also see \`navigator.maxTouchPoints\`.](https://developer.mozilla.org/en-US/docs/Web/API/Navigator/maxTouchPoints)
- \`pointerSize = 16\`: how many numbers each pointer takes up.
- \`targets = Pointer.tab()\`: the array of \`{x,y, data:[…], set()}\` objects to track or control.
- TODO: \`noFeedback=true\`
` }
        static options() {
            return {
                // TODO: What do we want?
                //   TODO: pointers=1
                //   TODO: pointerSize=16
                //   TODO: targets=Pointer.tab()
                //      …Should also allow 2 extra virtual groups, which can be used by `Video` and `Text.readHover`, right?
            }
        }
        resume(opts) {
            if (opts) {
                const pointers = opts.pointers || 1
                const pointerSize = opts.pointerSize || 16
                const targ = opts.targets || Pointer.tab()
                sn._assertCounts('', pointers, pointerSize)
                const name = Array.isArray(opts.name) ? opts.name : typeof opts.name == 'string' ? [opts.name] : []
                sn._assert(typeof targ == 'function' || Array.isArray(targ), "Bad targets")
                // TODO: What about `noFeedback`?
                opts.onValues = Pointer.onValues
                opts.values = pointers * pointerSize
                opts.emptyValues = 0
                opts.name = [
                    'pointer',
                    ...name,
                ]
                opts.noFeedback = true
                this.pointers = pointers, this.pointerSize = pointerSize
                this.targets = targ
            }
            return super.resume(opts)
        }

        static onValues(sensor, data) {
            // Make sure to limit to one tile per cell, so that there's no misalignment.
            const cellShape = sensor.cellShape()
            if (!cellShape) return
            const dataSize = cellShape[cellShape.length-1]
            const valuesPerCell = dataSize // Since sensor.emptyValues === 0.

            const targ = sensor._targets()
            // TODO: How to read {x,y}-object props?
            //   ...Do we just iterate over all `targ`s...
            //     I think we really do just do that.
        }
        static onFeedback(feedback, sensor) {
            if (!feedback || sensor.noFeedback) return
            // Yep. Handle feedback here. Handle it good.
            // TODO: Actually update those objects.
            //   Iterate over `targ`s, set their x/y/data, and .set().
        }

        _targets() {
            const targets = typeof this.targets == 'function' ? this.targets() : this.targets
            sn._assert(Array.isArray(targets), "Bad targets")
            return targets
        }
    }, {
        tab: A(function tab(n=0) { // TODO: Have all-props and `.set()`…
            // TODO: How do we handle `n`? How to not remove if we're at `n` objects, and pre-fill up to `n` objects, and re-use the inert objects?
            if (tab.result) return tab.result
            const ps = []
            const inds = new Map
            const passive = {passive:true}
            let attached = false, lastRequest = performance.now()
            let id = setInterval(autodetach, 10000)
            return tab.result = attachEvents
            function autodetach() {
                if (performance.now() - lastRequest > 15000)
                    detachEvents(), clearInterval(id), id = null
            }
            function clamp(x) { return Math.max(0, Math.min(x, 1)) }
            function onpointerdown(evt) { // Add/update.
                if (evt.pointerType === 'touch') return
                if (evt.touches) return Array.from(evt.touches).forEach(onpointerdown)
                const p = ps[indexOf(evt)]
                p.x = clamp(evt.clientX / innerWidth)
                p.y = clamp(evt.clientY / innerHeight)
                p.data[0] = 1
                p.data[1] = evt.isPrimary ? 1 : 0
                p.data[2] = evt.pointerType === 'mouse' ? 0 : evt.pointerType === 'pen' ? .5 : 1
                p.data[3] = clamp(evt.width / innerWidth)
                p.data[4] = clamp(evt.height / innerHeight)
                p.data[5] = clamp(evt.pressure)
                p.data[6] = clamp((evt.tangentialPressure+1) / 2)
                p.data[7] = clamp((evt.tiltX+90) / 180)
                p.data[8] = clamp((evt.tiltY+90) / 180)
                p.data[9] = clamp(evt.twist / 360)
            }
            function onpointerup(evt) { // Remove.
                if (evt.pointerType === 'touch') return
                if (!ps.length) return
                if (evt.touches) {
                    ps.length = 0, inds.clear()
                    return Array.from(evt.touches).forEach(onpointerdown)
                }
                if (ps.length <= 1) return
                const i = indexOf(evt), p = ps[i], j = ps.length-1
                p.x = .5, p.y = .5, p.data.fill(0)
                ;[ps[i], ps[j]] = [ps[j], ps[i]]
                ps.pop(), inds.delete(idOf(evt))
            }
            function pointerSet() {
                const p = this
                // TODO: How to update the '`this`-virtual-pointer' position via firing pointer/mouse/touch events?
                //   TODO: Reverse `onpointerdown`'s setting.
            }
            function idOf(evt) { return evt.pointerId !== undefined ? evt.pointerId : 'identifier' in evt ? evt.identifier : 'mouse' }
            function indexOf(evt) {
                const id = idOf(evt)
                if (!inds.has(id))
                    inds.set(id, ps.push({ x:.5, y:.5, data:[] })-1)
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
            docs:`Returns a closure that returns the dynamic list of mouse/touch positions, \`[{x,y,data}]\` (all are 0…1; \`data\` is an optional array of extra data).

The result is usable as the \`targets\` option for \`Video\` and \`Pointer\`, and as an arg of \`Text.readHover\`.

Can pass it \`n=0\`: how many pointer objects are guaranteed to be kept alive (though inert).`,
        }),
    })
}