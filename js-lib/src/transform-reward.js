export default function init(sn) {
    const A = Object.assign
    return A(class Reward extends sn.Transform {
        static docs() { return `Sets reward for all cells unless already set.

Options:
- \`reward = Reward.keybindings('Ctrl+ArrowUp', 'Ctrl+ArrowDown')\`: the function that, given nothing, will return the reward each frame, -1…1.
` }
        static options() {
            return {
                reward: {
                    ['+1 Ctrl+Up / -1 Ctrl+Down']: Reward.keybindings('Ctrl+ArrowUp', 'Ctrl+ArrowDown'),
                    ['+1 Shift+Up / -1 Shift+Down']: Reward.keybindings('Shift+ArrowUp', 'Shift+ArrowDown'),
                    ['+1 F8 / -1 F9']: Reward.keybindings('F8', 'F9'),
                },
            }
        }
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
    }, {
        keybindings: A(function keybindings(upKey = 'Ctrl+ArrowUp', downKey = 'Ctrl+ArrowDown') {
            // Reward: +1 while `upKey` is held, -1 while `downKey` is held (0 when both).
            sn._assert(upKey !== downKey)
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
                if (evt.key === upKey[0] && !evt.repeat) {
                    let ok = true
                    for (let i = 1; i < upKey.length; ++i) if (!evt[upKey[i]]) ok = false
                    if (ok) up = true
                }
                if (evt.key === downKey[0] && !evt.repeat) {
                    let ok = true
                    for (let i = 1; i < downKey.length; ++i) if (!evt[downKey[i]]) ok = false
                    if (ok) down = true
                }
            }
            function onKeyUp(evt) {
                // (Modifiers are only important when the key starts being held.)
                //   (Release the modifier but hold the key, and nothing will change.)
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
        }, {
            docs:`The human has access to 2 buttons: +1 reward and -1 reward.

By default, 'Ctrl+ArrowUp' is +1, 'Ctrl+ArrowDown' is -1. [Can use other keybindings.](https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/key/Key_Values)
`,
        }),
    })
}