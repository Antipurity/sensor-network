export default function init(sn) {
    const A = Object.assign
    const active = '_uiActive'
    const selectedJS = new WeakMap
    const UI = A(function UI() {
        // Just one channel for now.
        const state = load() || [{}]
        const js = dom([{tag:'textarea', readonly:true, style:'width:100%; resize:vertical'}])
        const r = dom([
            // (This double-saves each time. Might want to throttle saving.)
            { onchange: stateMaybeChanged, onclick: stateMaybeChanged },
            UI.collapsed(
                ' As JS',
                js,
                true,
            ),
            UI.channel(sn, state[0]),
        ])
        js.value = UI.toJS(state), js.rows = js.value.split('\n').length
        return r
        function stateMaybeChanged(evt) {
            if (evt.type === 'click' && (!evt.target || evt.target.tagName !== 'BUTTON')) return
            js.value = UI.toJS(state), js.rows = js.value.split('\n').length
            save(state)
        }
        function load() {
            const str = localStorage.snOptions
            return str && JSON.parse(str)
        }
        function save(s) {
            localStorage.snOptions = JSON.stringify(s)
        }
    }, {
        docs:`Creates UI for convenient configuration of sensors. Append it to \`document.body\` or something.

CSS not included. Markdown parsing not included.

(It doesn't affect code size that much, was quick to develop, and made apparent a few bugs, so why not?)`,
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
            const opts = Object.create(parentOpts)
            const variants = x.options(opts) // {opt:{valueName:jsValue}}
            sn._assert(variants && typeof variants == 'object', "Invalid options format")
            selected = getDefaults(variants, selected)
            optsFor(variants, selected)
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
                    if (k !== active && selected[k] === undefined) {
                        sn._assert(Object.keys(vars[k]).length, "Can't just be an empty object")
                        selected[k] = Object.keys(vars[k])[0]
                        if (isCheckboxy(vars[k])) selected[k] = vars[k][selected[k]]
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
                                selected[k] === variant ? {selected:'selected'} : null,
                                [{tag:'code'}, variant],
                            ])
                    } else
                        opt = [{
                            tag:'input',
                            type:'checkbox',
                            onchange,
                        }, selected[k] ? {checked:true} : null]
                    table.push([{tag:'tr'}, [{tag:'td', style:'text-align:right'}, prettifyCamelCase(k) + ':'], [{tag:'td'}, opt]])
                    function onchange() {
                        selected[k] = typeof this.checked == 'boolean' ? this.checked : this.value
                        optsFor(vars, selected)
                        if (instance) instance.paused === false && (instance.pause(true), instance.resume(opts))
                    }
                }
                into.push(table)
                function prettifyCamelCase(s) {
                    return s[0].toUpperCase() + s.slice(1).replace(/[A-Z]/g, s => ' '+s.toLowerCase())
                }
            }
            function isCheckboxy(o) { return Object.values(o).every(v => typeof v == 'boolean') }
            function optsFor(vars, selected) {
                const js = {}
                for (let k of Object.keys(opts)) delete opts[k]
                for (let k of Object.keys(vars)) {
                    if (k === active) continue
                    const f = typeof selected[k] == 'boolean' ? selected[k] : vars[k][selected[k]]
                    opts[k] = typeof f == 'function' ? f() : f
                    const Default = isCheckboxy(vars[k]) ? Object.values(vars[k])[0] : Object.keys(vars[k])[0]
                    if (selected[k] !== Default)
                        js[k] = ''+f // Only non-defaults are saved.
                }
                selectedJS.set(selected, js)
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
        docsTransformer(docs) { 'Override this: `sn.meta.UI.docsTransformer = docs => …`'
            // For example:
            // import sn from '../main.js'
            // import 'https://cdn.jsdelivr.net/npm/marked/marked.min.js'
            // sn.meta.UI.docsTransformer = docs => {
            //   const html = document.createElement('div')
            //   html.innerHTML = marked.parse(docs)
            //   return html
            // }
            return docs.split('\n')[0]
        },
        oneOrMore(fn, selected = []) {
            // Given a DOM-elem-returning func, this returns a spot that can replicate its result.
            //   The function accepts a DOM element that adds/removes an item, and should put it.
            let i = 0
            const subselected = selected[i] || (selected[i] = {});  ++i
            const first = fn(dom([{tag:'button', onclick:addNewItem}, '+']), subselected)
            const others = []
            const container = dom(first)
            while (i < selected.length) addNewItem()
            return A(container, {
                pause() { first.pause && first.pause(), others.forEach(e => e.pause && e.pause()), others.length = 0 },
                resume() { first.resume && first.resume() },
            })
            function addNewItem() {
                const subselected = selected[i] || (selected[i] = {});  ++i
                const another = fn(dom([{tag:'button', onclick() {
                    sn._assert(selected.indexOf(subselected) >= 0)
                    selected.splice(selected.indexOf(subselected), 1), --i
                    another.remove(), others.splice(others.lastIndexOf(another), 1), another.pause()
                }}, '-']), subselected)
                others.push(another)
                container.append(another)
            }
        },
        describe(x, selected = [], parentOpts = null, extraDOM = null) {
            // Describes an object: name, options, docs.
            let docs = typeof x.docs == 'string' ? x.docs : typeof x.docs == 'function' ? x.docs() : null
            docs = docs && docs.split('\n\n')
            docs = docs && (docs.length > 1 ? UI.collapsed([{style:'display:inline-block'}, UI.docsTransformer(docs[0])], [UI.docsTransformer(docs.slice(1).join('\n\n'))]) : dom(docs))
            const isClass = UI.groupOf(x) !== 'object'
            let name = ' ' + UI.nameOf(x)
            if (name === ' Sensor') name = ' Sensor →'
            if (name === ' Transform') name = ' → Transform →'
            if (name === ' Handler') name = ' → Handler'
            return isClass ? UI.oneOrMore(anItem, selected) : anItem(null, selected[0] || (selected[0] = {}))
            function anItem(btn, selected) {
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
                    onchange(evt) { if (el) selected[active] = this.checked, this.checked ? el.resume() : el.pause() },
                }])
                if (running && selected[active]) setTimeout(() => running.click(), 0)
                if (!selected[active]) selected[active] = false
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
        channel(x = sn, selected = {}) {
            // Creates a UI for easy setup of single-channel sensors/transforms/handlers.
            return walk(x, selected)
            function walk(x, selected = {}, parentOpts = null) {
                if (!x || typeof x != 'object' && typeof x != 'function') return
                const chElem = dom([])
                const us = UI.describe(x, selected._ || (selected._ = []), parentOpts, chElem)
                if (selected._.length === 1 && !Object.keys(selected._[0]).length)
                    delete selected._
                const children = Object.keys(x).map(k => {
                    const r = walk(x[k], selected[k] || (selected[k] = {}), us.opts || parentOpts)
                    if (!Object.keys(selected[k]).length) delete selected[k]
                    return r
                }).filter(x => x)
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
                    function onchange(evt) {
                        const checks = this.querySelectorAll('.checkboxRunning:checked')
                        this.classList.toggle('anyRunningInside', !!checks.length)
                        if (evt && evt.target) { // Re-init children when their parent changes.
                            if (!children.some(c => c.contains(evt.target))) { // (Only re-init the changed subtree.)
                                for (let c of Array.from(checks))
                                    c.click(), c.click()
                            }
                        }
                        return this
                    }
                }
            }
        },

        toJS(selected = []) {
            // Collects active items in `selected` into JS code that recreates those items.
            let js = [`import sn from 'sensor-network'\n`]
            selected.forEach(s => walk(s, 'sn', {}))
            return js.join('\n')
            function selectedToJS(o) { return selectedJS.get(o) || o }
            function walk(x, path, parentOpts) {
                if (Array.isArray(x._))
                    x._.forEach(item => {
                        if (!item[active]) return
                        const opts = A({}, parentOpts)
                        A(opts, selectedToJS(item))
                        const prettier = k => {
                            let js = opts[k]
                            if (typeof js == 'boolean' || js === 'false' || js === 'true') return `${k}:${js}`
                            if (js.slice(0,4) === '()=>') js = '() ' + js.slice(2)
                            if (js.slice(0,5) === '() =>') {
                                js = js.slice(5).trim()
                                if (js[0] !== '{') return `${k}:${js}`
                            }
                            return `${k}:(${js})()`
                        }
                        js.push(`new ${path}({${Object.keys(opts).map(prettier)}})`)
                    })
                if (!x || typeof x != 'object') return
                const opts = A({}, parentOpts)
                if (Array.isArray(x._) && x._[0]) A(opts, selectedToJS(x._[0]))
                for (let k of Object.keys(x))
                    if (k !== '_')
                        walk(x[k], path + '.' + k, opts)
            }
        },
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