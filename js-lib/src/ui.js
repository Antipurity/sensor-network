export default function init(sn) {
    const A = Object.assign
    const active = '_uiActive'
    const UI = A(function UI() {
        // Just one channel for now.
        return UI.channel()
    }, {
        docs:`Creates UI for convenient configuration of sensors. Append it to \`document.body\` or something.

(It doesn't affect code size that much, so why not.)`,
        groupOf(x) {
            const p = Object.getPrototypeOf(x)
            return p === sn.Sensor ? 'sensor' : p === sn.Transform ? 'transform' : p === sn.Handler ? 'handler' : 'object'
        },
        nameOf(x) {
            return x.name || (x === sn ? 'Sensor network' : `(Unnamed ${UI.groupOf(x)})`)
        },
        options(x, selected = {}, parentOpts = null) {
            // Given an object, returns the DOM tree that allows the user to select among options.
            // The object should define `.options() → { option:{ valueName: getJSValue() } }`.
            // The result has `.selected` (JSON-serializable) and `.opts` (passable as `parentOpts` here) and `.pause()` and `.resume()`.
            if (typeof x.options != 'function' || x === UI) return
            const isClass = UI.groupOf(x) !== 'object'
            const variants = x.options() // {opt:{valueName:jsValue}}
            sn._assert(variants && typeof variants == 'object', "Invalid options format")
            selected = getDefaults(variants, selected)
            const opts = Object.create(parentOpts) // TODO: Wait, how to *react* to `parentOpts` changing?
            const instance = isClass ? new x() : null
            const arr = []
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
                pause() { instance && instance.pause() },
                resume() { instance && instance.resume(optsFor(variants, selected)) },
            })

            function getDefaults(vars, selected = {}) {
                for (let k of Object.keys(vars))
                    if (k !== active && !selected[k]) {
                        sn._assert(Object.keys(vars[k]).length, "Can't just be an empty object")
                        selected[k] = Object.keys(vars[k])[0]
                    }
                return selected
            }
            function putElems(into, instance, vars, selected) {
                const table = [{tag:'table'}]
                for (let k of Object.keys(vars)) {
                    if (k === active) continue
                    let opt
                    if (!isCheckboxy(vars[k])) {
                        const optId = ''+Math.random()
                        opt = [
                            {tag:'select', id:optId, onchange},
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
                    table.push([{tag:'tr'}, [{tag:'td', style:'text-align:right'}, prettifyCamelCase(k) + ':'], [{tag:'td'}, opt]])
                    function onchange() {
                        selected[k] = typeof this.checked == 'boolean' ? this.checked : this.value
                        optsFor(vars, selected)
                        if (instance) !instance.paused && (instance.pause(), instance.resume(opts))
                        // TODO: Why is there a "Data must be divided into cells" error?
                    }
                }
                into.push(table)
                function isCheckboxy(o) { return Object.values(o).every(v => typeof v == 'boolean') }
                function prettifyCamelCase(s) {
                    return s[0].toUpperCase() + s.slice(1).replace(/[A-Z]/g, s => ' '+s.toLowerCase())
                }
            }
            function optsFor(vars, selected) {
                for (let k of Object.keys(vars)) {
                    if (k === active) continue
                    const f = typeof selected[k] == 'boolean' ? selected[k] : vars[k][selected[k]]
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
            // For example:
            // import sn from '../main.js'
            // import 'https://cdn.jsdelivr.net/npm/marked/marked.min.js'
            // sn.UI.docsTransformer = docs => {
            //   const html = document.createElement('div')
            //   html.innerHTML = marked.parse(docs)
            //   return html
            // }
            return docs.split('\n')[0]
        },
        oneOrMore(fn) {
            // Given a DOM-elem-returning func, this returns a spot that can replicate its result.
            //   The function accepts a DOM element that adds/removes an item, and should put it.
            const first = fn(dom([{tag:'button', onclick() {
                const another = fn(dom([{tag:'button', onclick() {
                    another.remove(), others.splice(others.lastIndexOf(another), 1), another.pause()
                }}, '-']))
                others.push(another)
                container.append(another)
            }}, '+']))
            const others = []
            const container = dom(first)
            return A(container, {
                pause() { first.pause && first.pause(), others.forEach(e => e.pause && e.pause()), others.length = 0 },
                resume() { first.resume && first.resume() },
            })
        },
        describe(x, selected = {}, parentOpts = null, extraDOM = null) {
            // Describes an object: name, options, docs.
            let docs = typeof x.docs == 'string' ? x.docs : typeof x.docs == 'function' ? x.docs() : null
            docs = docs && docs.split('\n\n')
            docs = docs && (docs.length > 1 ? UI.collapsed([{style:'display:inline-block'}, UI.docsTransformer(docs[0])], [UI.docsTransformer(docs.slice(1).join('\n\n'))]) : dom(docs))
            const isClass = UI.groupOf(x) !== 'object'
            const name = ' ' + UI.nameOf(x)
            return isClass ? UI.oneOrMore(anItem) : anItem()
            function anItem(btn) {
                const el = UI.options(x, selected, parentOpts)
                if (!el) {
                    const header = dom([btn || null, name])
                    return UI.collapsed(header, [docs && docs.cloneNode(true), extraDOM], true)
                }
                const id = ''+Math.random()
                const running = isClass && dom([{
                    id,
                    tag:'input',
                    type:'checkbox',
                    class:'checkboxRunning',
                    title:'Currently active',
                    onchange() { if (el) selected[active] = this.checked, this.checked ? el.resume() : el.pause() },
                }])
                if (running && selected[active]) running.click()
                return A(UI.collapsed(
                    [running ? {
                        style:'position:relative; z-index:2'
                    } : null, btn || null, running || null, running ? [
                        {
                            tag: 'label',
                            htmlFor: id,
                        },
                        name,
                    ] : name],
                    [docs && docs.cloneNode(true), el, extraDOM],
                    true,
                ), {
                    opts: el && el.opts,
                    pause() { el.pause && el.pause() },
                    resume() { el.resume && el.resume() },
                })
            }
        },
        channel(x = sn) {
            // Creates a UI for easy setup of single-channel sensors/transforms/handlers.
            return walk(x)
            function walk(x, selected = {}, parentOpts = null) {
                if (!x || typeof x != 'object' && typeof x != 'function') return
                const chElem = dom([])
                const us = UI.describe(x, selected, parentOpts, chElem)
                const children = Object.values(x).map(v => walk(v, {}, us.opts || parentOpts)).filter(x => x)
                if (x === sn) return dom(children)
                if (typeof x.options == 'function' && x !== UI || children.length) {
                    chElem.replaceWith(dom(children))
                    return A(onchange.call(dom([
                        { onchange },
                        us,
                    ])), {
                        pause() { us.pause && us.pause(), children.forEach(c => c.pause && c.pause()) },
                        resume() { us.resume && us.resume(), children.forEach(c => c.resume && c.resume()) },
                    })
                    function onchange() {
                        this.classList.toggle('anyRunningInside', !!this.querySelectorAll('.checkboxRunning:checked').length)
                        return this
                    }
                }
            }
            // TODO: (And a hierarchy or store of `options().selected`, which are synced to extension places or localStorage.)
        },

        // TODO: Have `UI.toJS(selected)`.
        //   TODO: Use it in `UI`, in a <textarea> that allows users to quickly get started (and update `onchange`).
    })
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
                        if (k !== 'tag' && (typeof v == 'string' || typeof v == 'number' || typeof v == 'boolean'))
                            el.setAttribute(k, v)
                    }
                else if (x[i] != null) el.append(dom(x[i]))
            return el
        } else if (x instanceof Node) return x
        else return document.createTextNode(''+x)
    }
}