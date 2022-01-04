export default function init(sn) { // TODO: Assign this in `sn`.
    const A = Object.assign
    return A(class Pointer extends sn.Sensor {
        static docs() { return `// TODO:

Options:
- \`name\`: heeded, augmented.
- \`pointers=1\`: how many pointers to track/control. [Also see \`navigator.maxTouchPoints\`.](https://developer.mozilla.org/en-US/docs/Web/API/Navigator/maxTouchPoints)
    - TODO: ...Also want the pointerSize, don't we…
- TODO: \`targets = Pointer.tab()\`
- TODO: \`noFeedback=true\`
` }
        static options() {
            return {
                // TODO: What do we want?
            }
        }
        resume(opts) {
            if (opts) {
                const pointers = opts.pointers || 1
                const targ = opts.targets || Pointer.tab()
                sn._assertCounts('', pointers)
                const name = Array.isArray(opts.name) ? opts.name : typeof opts.name == 'string' ? [opts.name] : []
                sn._assert(typeof targ == 'function' || Array.isArray(targ), "Bad targets")
                // TODO: What about `targets` and `noFeedback`?
                opts.onValues = Pointer.onValues
                opts.values = pointers * 123 // TODO: What's our value-count? ...Do we need to pass it in *again*…
                opts.emptyValues = 0
                opts.name = [
                    'pointer',
                    ...name,
                ]
                opts.noFeedback = true
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

            // TODO: How to read {x,y}-object props?
        }
        static onFeedback(feedback, sensor) {
            if (!feedback || sensor.noFeedback) return
            // Yep. Handle feedback here. Handle it good.
            // TODO: Actually update those objects.
        }

        _targets() { // TODO: No, right? Or maybe yes…
            const targets = typeof this.targets == 'function' ? this.targets() : this.targets
            sn._assert(Array.isArray(targets), "Bad targets")
            return targets
        }
    }, {
        tab: A(function tab() { // TODO: Have all-props and `.set()` and `.get()`…
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
            function onpointerdown(evt) { // Add/update.
                if (evt.pointerType === 'touch') return
                if (evt.touches) return Array.from(evt.touches).forEach(onpointerdown)
                const p = ps[indexOf(evt)]
                p.x = Math.max(0, Math.min(evt.clientX / innerWidth, 1))
                p.y = Math.max(0, Math.min(evt.clientY / innerHeight, 1))
                // TODO: Also have `p.data=[…]` (with every single pointer-event's prop) and `p.set()` (updating the 'virtual-pointer' position, firing events as needed)…
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
                    inds.set(id, ps.push({ x:.5, y:.5 })-1) // TODO: Also `.data:[]`.
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

The result is usable as the \`targets\` option for \`Video\`.`, // TODO: Also for Pointer and for Text.readHover.
        }),
    })
}