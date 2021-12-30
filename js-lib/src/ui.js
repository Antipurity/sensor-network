export default function init(sn) {
    const A = Object.assign
    const O = {
        options: A(function options(x) {
            // TODO: How to be able to accept parent paused-ness checkboxes?…
            //   (Maybe even don't make the actual checkbox ourselves?)
            // TODO: ...How to have global opts, such as cell shape, which would be too error-prone to specify everywhere...
            //   Accept `selected` objects of parents, and make `optsFor` also add those...
            //     For this, need to make `optsFor` always overwrite just one object, which is exposed on the result instead of `selected`.
            if (typeof x.options != 'function') return
            const proto = Object.getPrototypeOf(x)
            sn._assert(proto === sn.Sensor || proto === sn.Transform || proto === sn.Handler, "Must be a sensor/transform/handler")
            const variants = x.options() // {opt:{valueName:jsValue}}
            sn._assert(variants && typeof variants == 'object', "Invalid options format")
            const selected = getDefaults(variants)
            const instance = new x(optsFor(variants, selected)).pause()
            const arr = []
            putElems(arr, instance, variants, selected)
            return A(dom(arr), {selected}) // TODO: For symmetry here, should allow passing in `selected` too, for state-preservation, right?

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
                const opts = {}
                for (let k of Object.keys(vars))
                    opts[k] = vars[k][selected[k]] // TODO: Actually, should always call this, so that we don't instantly request screen/camera access that we don't need, right?
                return opts
            }
        }, {
            docs:`Given an object, returns the DOM tree that allows the user to select among options.

The object should define \`.options() → { option:{ valueName: valueJSValue } }\`.`,
        }),
        // TODO: Have "collapsed DOM element" (in `test.html`, of course).
        // TODO: Have "describe this object": name, options, and collapsed docs (Markdown support only if a function is passed in, else just the first line).
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