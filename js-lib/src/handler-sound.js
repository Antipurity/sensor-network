import sn from '../main.js'

export default class Sound extends sn.Handler {
    // TODO: How do we do a sound handler?
    //   Need `onValues`, passed to the constructor, right? ...Or, to `resume`, I guess.
    // TODO: Override `resume` to assert no-`onValues`, passing in its own `onValues`.
}

console.log(sn, Sound)