export default function init(sn) {
    const A = Object.assign
    const O = {
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
            const opts = Object.create(parentOpts)
            const instance = new x(optsFor(variants, selected)).pause()
            const arr = []
            putElems(arr, instance, variants, selected)
            return A(dom(arr), {
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
                    [{
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
        collapsed(el, byDefault = true) {
            // Made specifically for `test.html`.
            // TODO: How do we do this, exactly?
            /* TODO:
                <div class="hiding isHiding">
                    <div class=hidingSurface></div>
                    <span class=hidingMarker>▶</span>
                    Documentation
                    <div class="hidable" style="height:0px" id=docs>···</div>
                </div>
            */
        },
        // TODO: Have "collapsed DOM element" (in `test.html`, of course).
        // TODO: Have "describe this object": name, options (given saved `selected` and parent opts), and collapsed docs (Markdown support only if a function is passed in, else just the first line).
        // TODO: Have "one or more of this function call's invocations".
        // TODO: Have "describe this channel": walk `sn`, and for each object-with-options and its parent, add one-or-more: object-descriptions and collapsed children.
        //   TODO: (And a hierarchy of "Running" checkboxes, which force children to their state when flicked.)
        //   TODO: (And a hierarchy or store of `options().selected`, which are synced to extension places or localStorage.)
        // TODO: Make `UI` return one-or-more channels.
        // (TODO: Also make `test.html` put the full UI compiler there. Possibly instead of docs.)
        //   (TODO: And make it look good.)
    }
    return O
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