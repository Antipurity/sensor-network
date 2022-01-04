export default function init(sn) {
    const A = Object.assign
    return A(class Pointer extends sn.Sensor {
        static docs() { return `Tracks or controls mouse/touch state.

Options:
- \`name\`: heeded, augmented.
- \`noFeedback = true\`: \`true\` to track, \`false\` to control.
- \`pointers = 1\`: how many pointers to track/control. [Also see \`navigator.maxTouchPoints\`.](https://developer.mozilla.org/en-US/docs/Web/API/Navigator/maxTouchPoints)
- \`pointerSize = 16\`: how many numbers each pointer takes up.
- \`targets = Pointer.tab(noFeedback ? 0 : pointers)\`: the array of \`{x,y, data:[…], set()}\` objects to track or control.
` }
        static options() {
            return {
                noFeedback: {
                    Yes: true,
                    No: false,
                },
                pointers: {
                    ['1×']: () => 1,
                    ['2×']: () => 2,
                    ['4×']: () => 4,
                    ['8×']: () => 8,
                },
                pointerSize: {
                    ['16 ']: () => 16,
                    ['64 ']: () => 64,
                    ['256 ']: () => 256,
                },
                targets: {
                    ["Tab's pointers"]: () => sn.Sensor.Pointer.tab(),
                    // TODO: Also allow 2 extra virtual groups, which can be used by `Video` and `Text.readHover`.
                },
            }
        }
        resume(opts) {
            if (opts) {
                const pointers = opts.pointers || 1
                const pointerSize = opts.pointerSize || 16
                sn._assertCounts('', pointers, pointerSize)
                const targ = opts.targets || Pointer.tab(opts.noFeedback ? 0 : pointers)
                const name = Array.isArray(opts.name) ? opts.name : typeof opts.name == 'string' ? [opts.name] : []
                sn._assert(typeof targ == 'function' || Array.isArray(targ), "Bad targets")
                opts.onValues = Pointer.onValues
                opts.values = pointers * pointerSize
                opts.name = [
                    'pointer',
                    ...name,
                ]
                this.pointers = pointers, this.pointerSize = pointerSize
                this.targets = targ
            }
            return super.resume(opts)
        }

        static onValues(sensor, data) {
            const targ = sensor._targets()
            for (let i = 0; i < sensor.pointers; ++i) {
                const p = targ[i], d = p && p.data, sz = sensor.pointerSize, k = i * sz
                if (!p) { data.fill(0, k, k+sz);  continue }
                data[k+0] = p.x*2-1
                data[k+1] = p.y*2-1
                if (d) for (let i = 0; i < d.length; ++i) data[k+2+i] = d[i]*2-1
                sn._dataNamer.fill(data, k, 2+d.length, sz)
            }
            sensor.sendCallback(Pointer.onFeedback, data)
        }
        static onFeedback(feedback, sensor) {
            if (!feedback || sensor.noFeedback) return
            const targ = sensor._targets()
            for (let i = 0; i < sensor.pointers; ++i) {
                const p = targ[i], d = p && p.data, sz = sensor.pointerSize, k = i * sz
                if (!p) continue
                sn._dataNamer.unfill(feedback, k, 2+d.length, sz)
                p.x = (feedback[k+0]+1)/2
                p.y = (feedback[k+1]+1)/2
                if (d) for (let i = 0; i < d.length; ++i) d[i] = (feedback[k+2+i]+1)/2
                typeof p.set == 'function' && p.set()
            }
        }

        _targets() {
            const targets = typeof this.targets == 'function' ? this.targets() : this.targets
            sn._assert(Array.isArray(targets), "Bad targets")
            return targets
        }
    }, {
        tab: A(function tab(n=0) {
            if (!n && tab.result) return tab.result
            const ps = [] // i → {x,y, data:[…], set()}
            const inds = new Map // .pointerId → i
            const passive = {passive:true}
            let attached = false, lastRequest = performance.now(), lastEvent = null
            let id = setInterval(autodetach, 10000)
            if (n) // Need pointer objects to write to.
                for (let i = 0; i < n; ++i)
                    indexOf({pointerId: 'n'+i})
            return !n ? (tab.result = attachEvents) : attachEvents
            function autodetach() {
                if (performance.now() - lastRequest > 15000)
                    detachEvents(), clearInterval(id), id = null
            }
            function clamp(x) { return Math.max(0, Math.min(x, 1)) }
            function pointerGet(evt, p) {
                p.x = clamp(evt.clientX / innerWidth)
                p.y = clamp(evt.clientY / innerHeight)
                p.data[0] = evt.buttons&1
                p.data[1] = evt.buttons&2
                p.data[2] = evt.buttons&4
                p.data[3] = evt.isPrimary ? 1 : 0
                p.data[4] = evt.pointerType === 'mouse' ? 0 : evt.pointerType === 'pen' ? .5 : 1
                p.data[5] = clamp(evt.width / innerWidth)
                p.data[6] = clamp(evt.height / innerHeight)
                p.data[7] = clamp(evt.pressure)
                p.data[8] = clamp((evt.tangentialPressure+1) / 2)
                p.data[9] = clamp((evt.tiltX+90) / 180)
                p.data[10] = clamp((evt.tiltY+90) / 180)
                p.data[11] = clamp(evt.twist / 360)
            }
            function onpointerdown(evt) { // Add/update.
                lastEvent = evt
                if (evt.pointerType === 'touch') return
                if (evt.touches) return Array.from(evt.touches).forEach(onpointerdown)
                pointerGet(evt, ps[indexOf(evt)])
            }
            function onpointerup(evt) { // Remove.
                lastEvent = evt
                if (evt.pointerType === 'touch') return
                if (!ps.length) return
                if (evt.touches) {
                    ps.length = 0, inds.clear()
                    return Array.from(evt.touches).forEach(onpointerdown)
                }
                const i = indexOf(evt), p = ps[i], j = ps.length-1
                pointerGet(evt, p)
                if (ps.length <= 1) return
                ;[ps[i], ps[j]] = [ps[j], ps[i]]
                ps.pop(), inds.delete(idOf(evt))
            }
            function pointerSet() {
                // Reverse `onpointerdown`.
                // This does not replicate browser behavior perfectly, but is usually good enough.
                //   (For example, touch events are not dispatched: Edge & Safari hardly support creating them.)
                //   (There are enough browser implementations in the world.)
                const p = this, d = p.data
                const clientX = p.x * innerWidth, clientY = p.y * innerHeight
                const button1 = d[0]>=.5
                const button2 = d[1]>=.5
                const button3 = d[2]>=.5
                const isPrimary = d[3]>=.5
                const pointerType = d[4]<1/3 ? 'mouse' : d[4]<2/3 ? 'pen' : 'touch'
                const width = d[5] * innerWidth
                const height = d[6] * innerHeight
                const pressure = d[7]
                const tangentialPressure = d[8]*2 - 1
                const tiltX = d[9]*180 - 90
                const tiltY = d[10]*180 - 90
                const twist = d[11]*360
                const el = document.elementFromPoint(clientX, clientY)
                const buttons = button1 | (button2<<1) | (button3<<2), prev = p._prevButtons
                const ptrOpts = {
                    bubbles:true,
                    // Mouse
                    screenX: clientX, screenY: clientY,
                    clientX, clientY,
                    ctrlKey: lastEvent && lastEvent.ctrlKey,
                    shiftKey: lastEvent && lastEvent.shiftKey,
                    altKey: lastEvent && lastEvent.altKey,
                    metaKey: lastEvent && lastEvent.metaKey,
                    // (Note: ctrlKey, shiftKey, altKey, metaKey are not set.)
                    button: prev & ~buttons ? 0 : buttons&4 ? 2 : buttons&2 ? 1 : 0,
                    buttons,
                    relatedTarget: null,
                    // Pointer
                    pointerId: typeof p.id == 'string' ? parseFloat(p.id.slice(1)) : p.id,
                    width, height,
                    pressure, tangentialPressure,
                    tiltX, tiltY, twist,
                    pointerType,
                    isPrimary,
                }
                // Dispatch pointer/mouse events.
                const ms = pointerType === 'mouse' || isPrimary
                Do(el, self.PointerEvent, ~prev & buttons ? 'pointerdown' : prev & ~buttons ? 'pointerup' : 'pointermove', ptrOpts)
                ms && Do(el, MouseEvent, ~prev & buttons ? 'mousedown' : prev & ~buttons ? 'mouseup' : 'mousemove', ptrOpts)
                const prevEl = p._target
                if (prevEl !== el) {
                    ptrOpts.relatedTarget = el
                    Do(prevEl, self.PointerEvent, 'pointerout', ptrOpts)
                    ms && Do(prevEl, MouseEvent, 'mouseout', ptrOpts)
                    ptrOpts.relatedTarget = prevEl
                    Do(el, self.PointerEvent, 'pointerover', ptrOpts)
                    ms && Do(el, MouseEvent, 'mouseover', ptrOpts)
                    ptrOpts.bubbles = false
                    ptrOpts.relatedTarget = el
                    Do(prevEl, self.PointerEvent, 'pointerleave', ptrOpts)
                    ms && Do(prevEl, MouseEvent, 'mouseleave', ptrOpts)
                    ptrOpts.relatedTarget = prevEl
                    Do(el, self.PointerEvent, 'pointerenter', ptrOpts)
                    ms && Do(el, MouseEvent, 'mouseenter', ptrOpts)
                    ptrOpts.bubbles = true
                }
                if (isPrimary) { // 'click'
                    button1 && !(prev&1) && (p._startTarget = el)
                    !button1 && prev&1 && el === p._startTarget && el && el.click()
                }
                p._prevButtons = buttons, p._target = el
                function Do(target, cons, type, opts) {
                    target && cons && target.dispatchEvent(new cons(type, opts))
                }
            }
            function idOf(evt) { return evt.pointerId !== undefined ? evt.pointerId : 'identifier' in evt ? evt.identifier : 'mouse' }
            function indexOf(evt) {
                const id = idOf(evt)
                if (!inds.has(id))
                    inds.set(id, ps.push({ x:.5, y:.5, data:[], set:pointerSet, id, _prevButtons:0, _startTarget:null, _target:null })-1)
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

Can pass it \`n=0\`: how many writable pointer objects are guaranteed to be kept alive (but not updated).`,
        }),
    })
}