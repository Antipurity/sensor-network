export default function init(sn) {
    const A = Object.assign
    const UI = {
        options(x, selected = {}, parentOpts = null) {
            // Given an object, returns the DOM tree that allows the user to select among options.
            // The object should define `.options() → { option:{ valueName: getJSValue() } }`.
            // The result has `.selected` (JSON-serializable) and `.opts` (passable as `parentOpts` here) and `.pause()` and `.resume()`.
            if (typeof x.options != 'function') return
            const proto = Object.getPrototypeOf(x)
            sn._assert(proto === sn.Sensor || proto === sn.Transform || proto === sn.Handler, "Must be a sensor/transform/handler")
            const variants = x.options() // {opt:{valueName:jsValue}}
            sn._assert(variants && typeof variants == 'object', "Invalid options format")
            selected = getDefaults(variants, selected)
            const opts = Object.create(parentOpts) // TODO: Wait, how to *react* to `parentOpts` changing?
            const instance = new x(optsFor(variants, selected)).pause()
            const arr = []
            putElems(arr, instance, variants, selected)
            const el = dom(arr)
            const disconnectListener = setInterval(() => {
                // Autodie when disconnected. (But never revive.)
                if (!el.isConnected)
                    instance.pause(), clearInterval(disconnectListener)
            }, 10000)
            return A(el, {
                selected,
                opts,
                pause() { instance.pause() },
                resume() { instance.resume() },
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
                const runningId = ''+Math.random()
                into.push([
                    {tag:'div'},
                    [{ // TODO: Don't have the checkbox.
                        tag:'input',
                        type:'checkbox',
                        runningCheckbox:true,
                        id:runningId,
                        onchange() { instance.pause(), this.checked && instance.resume(optsFor(vars, selected)) },
                    }],
                    [{tag:'label', htmlFor:runningId}, 'Running'],
                ])
                for (let k of Object.keys(vars)) {
                    const optId = ''+Math.random()
                    const opt = [
                        {tag:'select', id:optId, onchange() {
                            selected[k] = this.value
                            !instance.paused && (instance.pause(), instance.resume(optsFor(vars, selected)))
                        }},
                    ]
                    for (let variant of Object.keys(vars[k]))
                        opt.push([
                            {tag:'option', value:variant},
                            selected[k] === variant ? {selected:''} : null,
                            [{tag:'code'}, variant],
                        ])
                    into.push([opt, ' ', k])
                }
            }
            function optsFor(vars, selected) {
                for (let k of Object.keys(vars))
                    opts[k] = vars[k][selected[k]]()
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
        describe(x, selected = {}, parentOpts = null) {
            // Describes an object: name, options, docs.
            const proto = Object.getPrototypeOf(x)
            const docs = typeof x.docs == 'string' ? x.docs : typeof x.docs == 'function' ? x.docs() : null
            return dom([
                x.name || `(Unnamed ${proto === sn.Sensor ? 'sensor' : proto === sn.Transform ? 'transform' : proto === sn.Handler ? 'handler' : 'object'})`,
                UI.options(x, selected, parentOpts),
                docs && UI.collapsed('Documentation', UI.docsTransformer(docs), true),
            ])
        },
        oneOrMore(fn) { // TODO: ...In its use, when a non-class is used, there should only be one instance, not many (currently, assertion-failing instead)...
            // Given a DOM-elem-returning func, this returns a spot that can replicate its result.
            const first = fn(true)
            const container = dom([
                [
                    [{tag:'button', onclick() {
                        const another = fn(false)
                        const wrapper = dom([
                            [{tag:'button', onclick() {
                                another.pause(), wrapper.remove()
                            }}, '-'],
                            another,
                        ])
                        container.append(wrapper)
                    }}, '+'],
                    first,
                ],
            ])
            return A(container, {
                pause() { first.pause && first.pause() },
                resume() { first.resume && first.resume() },
            })
        },
        channel() {
            // Creates a UI for easy setup of single-channel sensors/transforms/handlers.
            return walk(sn)
            function walk(x, selected = {}, parentOpts = null) {
                if (!x || typeof x != 'object' && typeof x != 'function') return
                const children = Object.values(x).map(v => walk(v)).filter(x => x)
                if (typeof x.options == 'function' || children.length) {
                    // TODO: Return a DOM elem with UI.describe(x, selected, parentOpts) and UI.collapsed(description, childrenIfAny, true).
                }
            }
        },
        //   TODO: Maintain & pass parent .opts.
        //   TODO: (And a hierarchy of "Running" checkboxes, which force children to their state when flicked.)
        //   TODO: (And a hierarchy or store of `options().selected`, which are synced to extension places or localStorage.)

        // TODO: Make `UI` itself return one-or-more channels.
        //   (TODO: Also, maybe, a collapsed area for JS code that creates everything currently-active?)
        // (TODO: Also make `test.html` put the full UI compiler there. Possibly instead of docs.)
        //   (TODO: And make it all look good.)
    }
    return UI // TODO: ...Wait, who did the infinite loop...
    function dom(x) { // Ex: [{ tag:'div', style:'color:red', onclick() { api.levelLoad() } }, 'Click to reload the level']
        if (x instanceof Promise) {
            const el = document.createElement('div')
            x.then(x => el.replaceWith(dom(x)), err => el.replaceWith(err instanceof Error ? '<Error: '+err.message+'>' : '<Error>'))
            el.classList.add('promise')
            return el
        } else if (Array.isArray(x)) {
            let tag = 'span'
            for (let i = 0; i < x.length; ++i) if (x[i] && typeof x[i].tag == 'string') tag = x[i].tag
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