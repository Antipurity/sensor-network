export default function init(sn) {
    return class Reward extends sn.Transform {
        static docs() { return `// TODO: 






` }
        resume(opts) {
            if (opts) {
                const rf = opts.reward || Reward.keybindings()
                sn._assert(typeof rf == 'function')
                this.reward = rf
                opts.onValues = Reward.onValues
            }
            return super.resume(opts)
        }
        static onValues(then, {data, error, noData, noFeedback, cellShape, partSize}) {
            try {
                const R = this.reward()
                sn._assert(typeof R == 'number' && R >= -1 && R <= 1)
                if (R === 0) return
                const cellSize = cellShape.reduce((a,b) => a+b)
                for (let i = 0; i < data.length; i += cellSize)
                    if (data[i] === 0)
                        data[i] = R
            } finally { then() }
        }
        // TODO: Test.
        //   TODO: Does it work?
        // TODO: Fill `0`s of 0th numbers of cells with the numeric result of calling a function unless it's `0` too.
        // TODO: By default, make Ctrl+ArrowUp/Ctrl+ArrowDown give +1/-1 reward.
        //   `static keybindings(upKey='Ctrl+Up', downKey='Ctrl+Up')`
        //   https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/key/Key_Values
        static keybindings(upKey = 'Ctrl+ArrowUp', downKey = 'Ctrl+ArrowUp') {
            // Reward: +1 while `upKey` is held, -1 while `downKey` is held (0 when both).
            upKey = parseKey(upKey), downKey = parseKey(downKey)
            const passive = {passive:true}
            let lastListen = 0, attached = false, up = false, down = false
            let unlistener = null
            listen()
            return function reward() { return listen(), up && !down ? 1 : !up && down ? -1 : 0 }
            function parseKey(s) { 'Ctrl+ArrowUp → (ArrowUp ctrlKey)'
                sn._assert(typeof s == 'string')
                const r = [''], spl = s.split('+')
                for (let k of spl.slice(0,-1))
                    if (['Alt', 'Ctrl', 'Meta', 'Shift'].includes(k))
                        r.push(k.toLowerCase() + 'Key')
                    else if (k) sn._assert(false, "Bad key: " + k)
                r[0] = spl[spl.length-1] || '+'
                return r
            }
            function onKeyDown(evt) {
                if (evt.key === upKey[0]) {
                    let ok = true
                    for (let i = 1; i < upKey.length; ++i) if (!evt[upKey[i]]) ok = false
                    if (ok) up = true
                }
                if (evt.key === downKey[0]) {
                    let ok = true
                    for (let i = 1; i < downKey.length; ++i) if (!evt[downKey[i]]) ok = false
                    if (ok) down = true
                }
            }
            function onKeyUp(evt) {
                if (evt.key === upKey[0]) up = false
                if (evt.key === downKey[0]) down = false
            }
            function listen() {
                lastListen = performance.now()
                if (attached) return;  else attached = true
                addEventListener('keydown', onKeyDown, passive)
                addEventListener('keyup', onKeyUp, passive)
                unlistener = setInterval(() => {
                    // Auto-detach when too much time has passed.
                    if (performance.now() - lastListen > 15000) unlisten()
                }, 10000)
            }
            function unlisten() {
                if (!attached) return;  else attached = false
                removeEventListener('keydown', onKeyDown, passive)
                removeEventListener('keyup', onKeyUp, passive)
                clearInterval(unlistener), unlistener = null
            }
        }
    }
}