export default function init(sn) {
    const A = Object.assign
    const UI = {
        options(x, selected = {}, parentOpts = null) {
            // Given an object, returns the DOM tree that allows the user to select among options.
            // The object should define `.options() → { option:{ valueName: getJSValue() } }`.
            // The result has `.selected` (JSON-serializable) and `.opts` (passable as `parentOpts` here) and `.pause()` and `.resume()`.
            if (typeof x.options != 'function' || x === UI) return
            const proto = Object.getPrototypeOf(x)
            const isClass = proto === sn.Sensor || proto === sn.Transform || proto === sn.Handler
            const variants = x.options() // {opt:{valueName:jsValue}}
            sn._assert(variants && typeof variants == 'object', "Invalid options format")
            selected = getDefaults(variants, selected)
            const opts = Object.create(parentOpts) // TODO: Wait, how to *react* to `parentOpts` changing?
            const instance = isClass ? new x(optsFor(variants, selected)).pause() : null
            const running = isClass && dom([{
                tag:'input',
                type:'checkbox',
                title:'Currently active',
                onchange() { if (instance) instance.pause(), this.checked && instance.resume(optsFor(variants, selected)) },
            }])
            const arr = running ? [running] : []
            putElems(arr, instance, variants, selected)
            const el = dom(arr)
            const disconnectListener = instance && setInterval(() => {
                // Autodie when disconnected. (But never revive.)
                if (!el.isConnected)
                    instance.pause(), clearInterval(disconnectListener)
            }, 10000)
            return A(el, {
                selected,
                opts,
                pause() { running && running.checked && running.click() },
                resume() { running && !running.checked && running.click() },
            })

            function getDefaults(vars, selected = {}) {
                for (let k of Object.keys(vars))
                    if (!selected[k]) {
                        sn._assert(Object.keys(vars[k]).length, "Can't just be an empty object")
                        selected[k] = Object.keys(vars[k])[0]
                    }
                return selected
            }
            function putElems(into, instance, vars, selected) {
                const table = [{tag:'table'}]
                for (let k of Object.keys(vars)) {
                    let opt
                    if (!isCheckboxy(vars[k])) {
                        const optId = ''+Math.random()
                        opt = [
                            {tag:'select', id:optId, style:'text-align:right', onchange},
                        ]
                        for (let variant of Object.keys(vars[k]))
                            opt.push([
                                {tag:'option', value:variant},
                                selected[k] === variant ? {selected:''} : null,
                                [{tag:'code'}, variant],
                            ])
                    } else
                        opt = [{
                            tag:'input',
                            type:'checkbox',
                            onchange,
                        }]
                    table.push([{tag:'tr'}, [{tag:'td'}, opt], [{tag:'td'}, prettifyCamelCase(k)]])
                }
                into.push(table)
                function onchange() {
                    selected[k] = typeof this.checked == 'boolean' ? this.checked : this.value
                    if (instance) !instance.paused && (instance.pause(), instance.resume(optsFor(vars, selected)))
                }
                function isCheckboxy(o) { return Object.values(o).every(v => typeof v == 'boolean') }
                function prettifyCamelCase(s) {
                    return s[0].toUpperCase() + s.slice(1).replace(/[A-Z]/g, s => ' '+s.toLowerCase())
                }
            }
            function optsFor(vars, selected) {
                for (let k of Object.keys(vars)) {
                    const f = vars[k][selected[k]]
                    opts[k] = typeof f == 'function' ? f() : f
                }
                return opts
            }
        },
        collapsed(summary, content, byDefault = true) {
            // Made specifically for `test.html`. Wraps a DOM element in a collapsible container.
            return dom([
                {tag:'div', class: byDefault ? 'hiding isHiding' : 'hiding'},
                [{tag:'div', class:'hidingSurface'}],
                [{class:'hidingMarker'}, '▶'],
                summary,
                [
                    {tag:'div', class:'hidable'},
                    byDefault && {style:'height:0px'},
                    content,
                ],
            ])
        },
        docsTransformer(docs) { 'Override this: `sn.UI.docsTransformer = docs => …`'
            return docs.split('\n')[0]
        },
        oneOrMore(fn) {
            // Given a DOM-elem-returning func, this returns a spot that can replicate its result.
            const first = fn(true)
            const others = []
            const container = dom([
                [
                    [{tag:'button', onclick() {
                        const another = fn(false)
                        others.push(another)
                        const wrapper = dom([
                            [{tag:'button', onclick() {
                                wrapper.remove(), others.splice(others.lastIndexOf(another), 1), another.pause()
                            }}, '-'],
                            another,
                        ])
                        container.append(wrapper)
                    }}, '+'],
                    first,
                ],
            ])
            return A(container, {
                pause() { first.pause && first.pause(), others.forEach(e => e.pause && e.pause()), others.length = 0 },
                resume() { first.resume && first.resume() },
            })
        },
        describe(x, selected = {}, parentOpts = null) {
            // Describes an object: name, options, docs.
            const docs = typeof x.docs == 'string' ? x.docs : typeof x.docs == 'function' ? x.docs() : null
            const proto = Object.getPrototypeOf(x)
            const group = proto === sn.Sensor ? 'sensor' : proto === sn.Transform ? 'transform' : proto === sn.Handler ? 'handler' : 'object'
            return dom([
                ' ' + (x.name || (x === sn ? 'Sensor network' : `(Unnamed ${group})`)),
                group !== 'object' ? UI.oneOrMore(() => UI.options(x, selected, parentOpts)) : UI.options(x, selected, parentOpts),
                docs && UI.collapsed('Documentation', UI.docsTransformer(docs), true), // TODO: The title should be the first line, not the generic "Documentation", to maximize content-per-view.
            ])
        },
        channel() {
            // Creates a UI for easy setup of single-channel sensors/transforms/handlers.
            return walk(sn)
            function walk(x, selected = {}, parentOpts = null) {
                if (!x || typeof x != 'object' && typeof x != 'function') return
                const children = Object.values(x).map(v => walk(v)).filter(x => x)
                if (typeof x.options == 'function' && x !== UI || children.length) {
                    const us = UI.describe(x, selected, parentOpts)
                    const container = children.length ? UI.collapsed(us, children, true) : us
                    //   (TODO: How to allow clicking in summary-`us`?)
                    // TODO: But what about a parent's "Running" checkbox? …Or maybe a counter, or at least a color-based indicator?…
                    return A(container, {
                        pause() { us.pause && us.pause(), children.forEach(c => c.pause && c.pause()) },
                        resume() { us.resume && us.resume(), children.forEach(c => c.resume && c.resume()) },
                    })
                }
            }
        },
        //   TODO: Maintain & pass parent .opts.
        //   TODO: (And a hierarchy of "Running" checkboxes, which force children to their state when flicked.)
        //   TODO: (And a hierarchy or store of `options().selected`, which are synced to extension places or localStorage.)

        // TODO: Have `Sensor`, `Transform`, `Handler`.
        //   (Don't be boring, come on.)

        // TODO: Make `UI` itself return one-or-more channels.
        //   (TODO: Also, maybe, a collapsed area for JS code that creates everything currently-active?)
        // (TODO: Also make `test.html` put the full UI compiler there. Possibly instead of docs.)
        //   (TODO: And make it all look good.)
    }
    return UI
    function dom(x) { // Ex: [{ tag:'div', style:'color:red', onclick() { api.levelLoad() } }, 'Click to reload the level']
        if (x instanceof Promise) {
            const el = document.createElement('div')
            x.then(x => el.replaceWith(dom(x)), err => el.replaceWith(err instanceof Error ? '<Error: '+err.message+'>' : '<Error>'))
            el.classList.add('promise')
            return el
        } else if (Array.isArray(x)) {
            let tag = 'span'
            for (let i = 0; i < x.length; ++i) if (x[i] && !(x[i] instanceof Node) && typeof x[i].tag == 'string') tag = x[i].tag
            const el = document.createElement(tag)
            for (let i = 0; i < x.length; ++i)
                if (x[i] && !Array.isArray(x[i]) && typeof x[i] == 'object' && !(x[i] instanceof Promise) && !(x[i] instanceof Node))
                    for (let k of Object.keys(x[i])) {
                        const v = el[k] = x[i][k]
                        if (typeof v == 'string' || typeof v == 'number' || typeof v == 'boolean')
                            el.setAttribute(k, v)
                    }
                else if (x[i] != null) el.append(dom(x[i]))
            return el
        } else if (x instanceof Node) return x
        else return document.createTextNode(''+x)
    }
}