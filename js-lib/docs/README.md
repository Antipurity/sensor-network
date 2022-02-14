<a id=toc></a>
# Table of contents

Sensor network:

- [sn.Sensor](#sn-sensor)

    - [sn.Sensor.Text](#sn-sensor-text)

        - [sn.Sensor.Text.readSelection](#sn-sensor-text-readselection)

        - [sn.Sensor.Text.readHover](#sn-sensor-text-readhover)

        - [sn.Sensor.Text.writeSelection](#sn-sensor-text-writeselection)

    - [sn.Sensor.Video](#sn-sensor-video)

        - [sn.Sensor.Video.stitchTab](#sn-sensor-video-stitchtab)

        - [sn.Sensor.Video.requestDisplay](#sn-sensor-video-requestdisplay)

        - [sn.Sensor.Video.requestCamera](#sn-sensor-video-requestcamera)

    - [sn.Sensor.Audio](#sn-sensor-audio)

    - [sn.Sensor.Keyboard](#sn-sensor-keyboard)

    - [sn.Sensor.Pointer](#sn-sensor-pointer)

        - [sn.Sensor.Pointer.tab](#sn-sensor-pointer-tab)

    - [sn.Sensor.Scroll](#sn-sensor-scroll)

    - [sn.Sensor.Storage](#sn-sensor-storage)

    - [sn.Sensor.Internet](#sn-sensor-internet)

- [sn.Transform](#sn-transform)

    - [sn.Transform.Time](#sn-transform-time)

    - [sn.Transform.Reward](#sn-transform-reward)

        - [sn.Transform.Reward.keybindings](#sn-transform-reward-keybindings)

    - [sn.Transform.Shuffle](#sn-transform-shuffle)

    - [sn.Transform.LimitFPS](#sn-transform-limitfps)

- [sn.Handler](#sn-handler)

    - [sn.Handler.Sound](#sn-handler-sound)

    - [sn.Handler.Storage](#sn-handler-storage)

    - [sn.Handler.Internet](#sn-handler-internet)

        - [sn.Handler.Internet.broadcastChannel](#sn-handler-internet-broadcastchannel)

        - [sn.Handler.Internet.consoleLog](#sn-handler-internet-consolelog)

        - [sn.Handler.Internet.webSocket](#sn-handler-internet-websocket)

    - [sn.Handler.Random](#sn-handler-random)

- [sn.meta](#sn-meta)

    - [sn.meta.docs](#sn-meta-docs)

    - [sn.meta.tests](#sn-meta-tests)

    - [sn.meta.metric](#sn-meta-metric)

    - [sn.meta.bench](#sn-meta-bench)

    - [sn.meta.UI](#sn-meta-ui)

<a id="sn"></a>
# `sn`[ ↑](#toc)

<a id="sn-sensor"></a>
## `sn.Sensor`[ ↑](#toc)

Generalization of eyes and ears and hands, hotswappable and adjustable.

- `constructor({ name, values, onValues=null, channel='', noFeedback=false, userName=[], emptyValues=0, hasher=… })`
    - `name`: a human-readable string, or an array of that or a -1…1 number or a function from `dataStart, dataEnd, dataLen` to a -1…1 number.
    - `values`: how many -1…1 numbers this sensor exposes.
        - Usually a good idea to keep this to powers-of-2, and squares. Such as 64.
    - `onValues.call(sensor, data)`: the regularly-executed function that reports data, by calling `sensor.send(data, …)` inside once.
        - To not allocate garbage, use `sensor.sendCallback(then, data, …)` with a static function.
        - `data` is owned; `sensor.send` it only once, or `sn._deallocF32(data)` if unused.
    - Extra flexibility:
        - `channel`: the human-readable name of the channel. Communication only happens within the same channel.
        - `noFeedback`: set to `true` if applicable to avoid some processing. Otherwise, feedback is the data that should have been.
        - `userName`: the name of the machine that sources data. Makes it possible to reliably distinguish sources.
        - `emptyValues`: the guaranteed extra padding, for fractal folding. See `sn._dataNamer.fill`.
        - `hasher(…)(…)(…)`: see `sn._dataNamer.hasher`. The default mainly hashes strings in `userName`/`name` with MD5 and rescales bytes into -1…1.
    - To change any of this, `resume({…})`.

- `cellShape() → [user, name, data] | null`: returns the target's cell shape. Note that this may change rarely.

- `send(values = null, error = null, reward = 0) → Promise<null|feedback>`
    - (Do not override in child classes, only call.)
    - `values`: `null` or owned flat data, -1…1 `Float32Array`. Do not perform ANY operations on it once called.
        - (`null` means "feedback only please", meaning, actions.)
        - To mitigate misalignment, try to stick to powers of 2 in all sizes.
        - (Can use `sn._allocF32(len)` for efficient reuse.)
    - `error`: `null` or owned flat data, -1…1 `Float32Array` of length `values.length`: `max abs(truth - observation) - 1`. Do not perform ANY operations on it once called.
    - `reward`: every sensor can tell handlers what to maximize, -1…1. (What is closest in meaning for you? Localized pain and pleasure? Satisfying everyone's needs rather than the handler's? …Money? Either way, it sounds like a proper body.)
        - Can be a number or a per-cell array or a function from `valueStart, valueEnd, valuesTotal` to that.
    - (Result: `feedback` is owned by you. Can use `feedback && sn._deallocF32(feedback)` once you are done with it, or simply ignore it and let GC collect it.)

- `sendCallback(then.call(sensor, null|feedback, cellShape, partSize), values, error = null, reward = 0)`: exactly like `send` but does not have to allocate a promise, which is more efficient.

- `resize(newValues, newEmptyValues = emptyValues)`: fast changing of flat-data length.
    - (If used in `onValues`, do `sn._deallocF32(data), data = sn._allocF32(newValues)` for efficiency.)

- `pause()`, `resume(opts?)`: for convenience, these return the object.

- `needsExtensionAPI() → null|String`: overridable in child classes. By default, the sensor is entirely in-page in a [content script](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Content_scripts) if injected by an extension. For example, make this return `'tabs'` to get access to [`chrome.tabs`](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/tabs) in an extension.

<a id="sn-sensor-text"></a>
### `sn.Sensor.Text`[ ↑](#toc)

Observe text, or suggest completions.

Text: abstract, compressed, easy for humans to create. You are reading it.    
Split into tokens, which are presumably interconnected, and defined by each other.

Options:
- `name`: heeded, augmented.
- `tokens = 64`: max token count, in characters by default.
- `tokenSize = 64`: how many numbers each token takes up. Ideally, should match the handler's `dataSize`.
- `text = Text.readSelection()`: the actual text observation.
    - A string, or `<input>` or `<textarea>`, or a function that returns a string.
    - Optional `.feedback(string)`.
- `textToTokens = Text.textToTokens`: splits text into tokens, characters by default.
    - `function(string, tokens) → [...token]`
    - `.feedback([...token]) → string`
- `tokenToData = Text.tokenToDataMD5`: converts a token to actual numbers.
    - `function(token, data, start, end)`
    - `.feedback(feedback, start, end) → token`


<a id="sn-sensor-text-readselection"></a>
#### `sn.Sensor.Text.readSelection`[ ↑](#toc)

```js
function readSelection(n=2048)
```

Reads the selection, in document and `<input>`s and `<textarea>`s.

Can pass the maximum returned string length, 2048 by default.

<a id="sn-sensor-text-readhover"></a>
#### `sn.Sensor.Text.readHover`[ ↑](#toc)

```js
function readHover(pos = sn.Sensor.Pointer.tab(), n=2048)
```

Reads the text under the pointer.

Can pass the `{x,y}` object or array/function to that object (`Pointer.tab()` by default), and maximum returned string length, 2048 by default.

<a id="sn-sensor-text-writeselection"></a>
#### `sn.Sensor.Text.writeSelection`[ ↑](#toc)

```js
function writeSelection()
```

Modifies the selection to be the feedback, if possible.

The new text will still be selected, so it can function as autocomplete or autocorrect.

<a id="sn-sensor-video"></a>
### `sn.Sensor.Video`[ ↑](#toc)

A sequence of images.

Images are divided into [small patches, which has mostly been ](https://en.wikipedia.org/wiki/Vision_transformer)[shown to work well in ML.](https://arxiv.org/abs/2006.09882v5)

This sensor's output is composed of 1 or more tiles, which are square images.    
It can target 0 or 1+ points, each shown in 1 or more tiles, and can include multiple zoom levels.

Extra options:
- `name`: heeded, augmented.
- `tileDimension = 8`: each tile edge's length.
- `source = Video.stitchTab()`: where to fetch image data from. `MediaStream` or `<canvas>` or `<video>` or `<img>` or a function to one of these.
    - Feedback is currently not implemented.
- `monochrome = true`: make this `true` to only report [luminance](https://en.wikipedia.org/wiki/Relative_luminance) and use 3× less data.
- `targets = Pointer.tab()`: what to focus rectangles' centers on. This is a live array of `{x,y}` objects with 0…1 viewport coordinates, or a function to that, called every frame.
    - If empty, the whole `source` will be resized to fit, and zooming will zoom in on the center instead of zooming out; if not, the viewed rect will be centered on the target.
- `tiling = 2`: how many vertical/horizontal repetitions there are per target or screen.
- `zoomSteps = 3`: how many extra zoomed views to generate per target or screen.
- `zoomStepStart = 0`: the least-zoomed zoom level.
- `zoomStep = 2`: the multiplier/divider of in-source tile dimension, per zoom step.


<a id="sn-sensor-video-stitchtab"></a>
#### `sn.Sensor.Video.stitchTab`[ ↑](#toc)

```js
function stitchTab()
```

Views on-page `<canvas>`/`<video>`/`<img>` elements. The rest of the page is black.

The result is usable as the `source` option for `Video`.

<a id="sn-sensor-video-requestdisplay"></a>
#### `sn.Sensor.Video.requestDisplay`[ ↑](#toc)

```js
function requestDisplay(maxWidth = 1024)
```

[Requests a screen/window/tab stream.](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getDisplayMedia)

For performance, max width is 1024 by default; pass in 0 or something else if needed.

The result is usable as the `source` option for `Video`.

<a id="sn-sensor-video-requestcamera"></a>
#### `sn.Sensor.Video.requestCamera`[ ↑](#toc)

```js
function requestDisplay(maxWidth = 1024)
```

[Requests a camera stream.](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)

For performance, max width is 1024 by default; pass in 0 or something else if needed.

The result is usable as the `source` option for `Video`.

<a id="sn-sensor-audio"></a>
### `sn.Sensor.Audio`[ ↑](#toc)

Sound that's already playing.

If you need to test what this sensor records, here:

<audio controls crossorigin src="https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3"></audio>

Options:
- `source = Audio.DOM(Audio)`: `<video>` or `<audio>` (with the `crossorigin` attribute if crossorigin), `MediaStream` `function(Audio)(audioContext)`, or the `AudioContext` whose `.destination` is augmented.
- `fftSize = 2048`: the window size: how many values are exposed per packet (or twice that if `frequency`). [Must be a power-of-2.](https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode/fftSize)
- `frequency = {minDecibels:-100, maxDecibels:-30}`: `null` to expose `fftSize` time-domain numbers, or an object to expose `fftSize/2` frequency-domain numbers.

<a id="sn-sensor-keyboard"></a>
### `sn.Sensor.Keyboard`[ ↑](#toc)

Tracks or controls keyboard state.

Options:
- `name`: heeded, augmented.
- `noFeedback = true`: `true` to track, `false` to control.
- `keys = 4`: max simultaneously-pressed [keys](https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/key/Key_Values) to report.
- `keySize = 16`: numbers per key.
- `tokenToData = sn.Sensor.Text.tokenToDataMD5`: converts a token to actual numbers.
    - `function(token, data, start, end)`
    - `.feedback(feedback, start, end) → token`


<a id="sn-sensor-pointer"></a>
### `sn.Sensor.Pointer`[ ↑](#toc)

Tracks or controls mouse/touch state.

Options:
- `name`: heeded, augmented.
- `noFeedback = true`: `true` to track, `false` to control.
- `pointers = 1`: how many pointers to track/control. [Also see `navigator.maxTouchPoints`.](https://developer.mozilla.org/en-US/docs/Web/API/Navigator/maxTouchPoints)
- `pointerSize = 16`: how many numbers each pointer takes up.
- `targets = Pointer.tab(noFeedback ? 0 : pointers)`: the array of `{x,y, data:[…], set()}` objects to track or control.


<a id="sn-sensor-pointer-tab"></a>
#### `sn.Sensor.Pointer.tab`[ ↑](#toc)

```js
function tab(n=0)
```

Returns a closure that returns the dynamic list of mouse/touch positions, `[{x,y,data}]` (all are 0…1; `data` is an optional array of extra data).

The result is usable as the `targets` option for `Video` and `Pointer`, and as an arg of `Text.readHover`.

Can pass it `n=0`: how many writable pointer objects are guaranteed to be kept alive (but not updated).

<a id="sn-sensor-scroll"></a>
### `sn.Sensor.Scroll`[ ↑](#toc)

DOM scroll position and its change.

Options:
- `name`: heeded, augmented.
- `mode: 'read'|'write'|'move' = 'read'`: can read, and either set or add-to scroll position.
- `target:{x,y} = Pointer.tab()`: where to get target-coordinates from (or a function, or an array). Many DOM elements can be scrollable, and `Scroll` tries to return the most relevant ones.


<a id="sn-sensor-storage"></a>
### `sn.Sensor.Storage`[ ↑](#toc)

Loads data from storage.

Note that this does not separate steps but just outputs data in ~1MiB chunks, so it is likely to sound different when visualized with `sn.Sensor.Sound`.

Uses [`indexedDB`.](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API)

Options:
- `filename = 'sn'`: which file was saved.
- `pauseOnEnd = false`: to only process the file once, use this.
- `randomEveryNCells = null`: if an integer, `.random()` will be called periodically and at the start.

Properties, to get/set at will:
- `1 <= .nextChunk < .maxChunks`. Note that changing this will not clear the chunk queue, so the change may take a while to become visible unless you `.flush()`.
- `.maxChunks`

Methods:
- `.random()`: sets `.nextChunk` to a random point.
- `.flush()`: makes the transition to another chunk instant, at the cost of a fetching delay.


<a id="sn-sensor-internet"></a>
### `sn.Sensor.Internet`[ ↑](#toc)

Extends this sensor network over the Internet, to control others.

Options:
- `iceServers = []`: the [list](https://gist.github.com/mondain/b0ec1cf5f60ae726202e) of [ICE servers](https://developer.mozilla.org/en-US/docs/Web/API/RTCIceServer/urls) (Interactive Connectivity Establishment).
- `signaler`: a convenience: on construction, does `sensor.signal(signaler(sensor))` for you. Props of `sn.Handler.Internet` go well here.

Methods:
- `signal(metaChannel: { send(string), close(), onopen, onmessage, onclose }, maxCells=65536)` for manually establishing a connection: on an incoming connection, someone must notify us of it so that negotiation of a connection can take place, for example, [of a `WebSocket` which can be passed directly](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket).

Browser compatibility: [Edge 79.](https://developer.mozilla.org/en-US/docs/Web/API/RTCPeerConnection/createDataChannel)

Imports [100 KiB](https://github.com/feross/simple-peer) on use.


<a id="sn-transform"></a>
## `sn.Transform`[ ↑](#toc)

Modifies data/feedback, after sensors and before handlers.

- `constructor({ onValues=null, onFeedback=null, priority=0, channel='' })`
    - Needs one or both:
        - `onValues(then, {data, error, noData, noFeedback, cellShape, partSize})`: can modify `data` and the optional `error` in-place.
            - ALWAYS do `then(extra, …)`, at the end, even on errors. `extra` will be seen by `onFeedback` if specified.
                - To resize `data` and possibly `error`, pass the next version (`._allocF32(len)`) to `then(extra, data2)` or `then(extra, data2, error2)`; also resize `noData` and `noFeedback`; do not deallocate arguments.
            - `cellShape: [user, name, data]`
            - Data is split into cells, each made up of `cellShape.reduce((a,b)=>a+b)` -1…1 numbers.
            - `noData` and `noFeedback` are JS arrays, from cell index to boolean.
        - `onFeedback(then, {data, error, noData, noFeedback, cellShape, partSize}, feedback, extra)`: can modify `feedback` in-place.
            - ALWAYS do `then()`, at the end, even on errors.
                - If `data` was resized and `feedback` was given, must resize it back, by passing the next version (`._allocF32(len)`) to `then(feedback)`; do not deallocate arguments.
    - Extra flexibility:
        - `priority`: transforms run in order, highest priority first.
        - `channel`: the human-readable name of the channel. Communication only happens within the same channel.
    - To change any of this, `resume({…})`.

- `pause()`, `resume(opts?)`: for convenience, these return the object.

<a id="sn-transform-time"></a>
### `sn.Transform.Time`[ ↑](#toc)

Reports start & end times of when each data cell was collected.

[Depends on the user's system time, not synchronized explicitly.](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/Date)

Reserve at least 1 of `userParts` for this.    
With at least 24 `partSize`, the augmentation looks like this (down is time):    
![Sine-waves of exponentially-decreasing frequency.](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAAXNSR0IArs4c6QAABz1JREFUeF7tW3tIl1cYPnmpaFkha42mZDFqEcJGa9moqFgWrVkNosVqFBZNZV2M1WhYq9aSREbQrNgKKhCrkeaC0mDViiJiYBfLS/fykt3VLmbm4Dnvo/N8HD613IJj/zznfL+f8eM97/e+z/u87+mglKpXDv/r0G4A8YATJ07AD6KiooC3bt0CnjlzBnj27FlgUVER8ObNm8Dbt28DHz58CHz06BHw6dOnwGfPngGfP38OrKurA7548QJYX68d0Ib48F//+D3zeWv3DR7gvAFqxIS9BLMOH8Zq5MiRwLt37wLpEUR6xI0bN/A5PeLBgwfYv+4e0eABzhtgwYIFOLGf168HThRP+GL7dqymT58OfPLkCTA/Px94+vRpYEFBAfD69etAxpD79+9jX11dDWRsqKnRJrfFBr8YIT/PAy2NEQ0e0G4A8YDs7GxY9eKVK8B3xcYbDxzAKjo6GsiTZXZgTCgsLMTnjAkVFRXYtzQmMFu0NEvQJZrrCR4PcNYADIKBkq9TgoJgzPFi0jjhB+np6XgSEREBvHbtGpCx4Ny5c9qDLl4ElpaWAu/cuQOsrKwEPn78GMiYUFtbiz1jAmOAiebJ2k66xR7gvAFIhOKHDcNJzEhNBX6zeDFwtnjCiE2b9H62fsKTO3/+fBNPuHDhQhMPsWUFZhUyRv5/JmNsq6zgYYLOGkAOWLn2KniqQWcNoGtBpZaNHg0MCAgA8t3789Ah7AfL95IyM7GKiYkBMrozCzArkBeQIZq8wGSItuqxrXhBYwxw3QDk5ozCeV27wiThkudr+vTBPkcMdXnRIqxWrFgB7NKlC5AnTg9gdrgizLKsrAzfu3fvHrCqqgrIbGD+Dp68n44gP6tBV+De73ljNSjFibMG0LWgUprXKfWLYIgwt++7dcOT+fI8ZuBArDIlFgwYMAB71gD0ANYKZIYlJSX4nskMqRvQAxgLbB5g8gK/k7Yxw8ZaoN0A2gI/iCHe7tQJq9jYWGBubi6wUDh+f/nehv37sRo3bhzQViVSObJlg+YqR686G3g8wFkD8N2bN28eTjIlJQW4pGdP4PidO4GfTpsGXCoe0E9qhvj4eDzhu8ZaIC8vD8+pGF29ehX78vLyJh5jZoPW1gbys5qdDTxZwFkDkAn2M/oBg8eOhVE7S/1+SvK91nmUyp45E5gqnhAaGoo9oz49gBoi+QB1AvIBUyfw4wP0tJfNBh4m6KwB+O64xgc81aCzBqAi9I4oQmHiEp8Jfr5lC1Zxwgu0sqeUzhFK7TtyBMhOEjtEtt7i69JJ8ihCzhqAQsiPSUk4yaDVq4F1y5cD09LSgKz/t2zdiv0n4gGxohZPnToVT6j6NreDxL4B9QFTK2yrDpKnN+isAdgaS0hIwAnm5OjKf+7cucDVnTvrd1yefyTc/yfxgNC1a7GaP1/Xi1SUyAhZHXJvMkLyAdMDzB6irW9gdpDkZ/kyQk9nyHkDhEl3eKF0aljFhQwaBKN2k75/cY8e2Gv1X6m/5swBJicnA7t37w4sLi4G0gP8GCFrAsYQP0Zo6xe02gOcNQCzgO7yK5UwfDiwob6XeQAlsz1pohp/KN//btQorLZt2wYMDw8H8l1nTUDV+NKlS/icNQEnUFgTmApRW/UOPVnAWQOQCZK5vS+6gJ7/UEpHAKXCRCVeKCrxBnk+pHdvrA7IHEFkZCT21ADN7jFjAz83p838usd+KrEtK5ixwcMEnTUALeNaTeCpBp01ABWhy+IKcdIH+EP6AmvH61kRvpudRDXOPXgQz4fI3y3bswerSZMmAc2eoTlLZE6VmTWB38Tpy6rEHkXIWQOQcR2Tk/1AendhovHlHD2KE40bMQL41bp1wEVLlui9eMCozZuxmjVrFpD521YVcsbo/5og8ajCzhqAUtiYvXtxcsHBwUC+YxMmTMB+fWCgfufFI94Tj9DzpUp1XLUKmJiYCAySaTO/qpBdYzJPW9eY1eCrqgo9naF2A4gHpEsUz5B5fjK2qjCtFr4p9wRKpXN0Sjzg77g4rNasWQMMCQkBsqq0VYXm3ACzR1vPDVg9wDkDmFlATwop9fuuXUBG8aSVK7EPkKowS6rCN+T7GydPxooaYq9e+gYCqz/TA2xVoZ8uYGqEre0UWbOAcwYwmaCe+1TqN0HqBXryV6kguQuU3LEj9pr3KTVniOaEGRkZwL59+wLJ+Gy6gG1yxK8qfNmZYisTdM4AcoDKNV3AWg26ogtYBRFy+UCpDfSNH6V0v0ip0fv2Ab+eqG8XUR3WWrBSx44fBw6TXiMnRM3pMSpDZq/wv7qH2G4AXp01VWFG8y9PaY6nJ4eUekswfqmeEtqxY4eO8nIzpJ98/qt0kMbKhAk7P+YdI/Peoe2OERkhq0vbvYKW3jGyqsLOGYC9QTZGxvCkhw7F6tuTJ4FZM2YAeZK8O5QlNcTH8neJu3djNWXKFCB7fs2dJmdVyP5AW3WLPb1B1wzwD7XJyQ9o5gXsAAAAAElFTkSuQmCC)    
With less, it has to improvise, with max resolution being 1/8sec, and min resolution at least 8sec.

Options:
- `timing = () => +new Date()/1000`: provides the seconds used to annotate data.


<a id="sn-transform-reward"></a>
### `sn.Transform.Reward`[ ↑](#toc)

Sets reward for all cells unless already set.

Options:
- `reward = Reward.keybindings('Ctrl+ArrowUp', 'Ctrl+ArrowDown')`: the function that, given nothing, will return the reward each frame, -1…1.


<a id="sn-transform-reward-keybindings"></a>
#### `sn.Transform.Reward.keybindings`[ ↑](#toc)

```js
function keybindings(upKey = 'Ctrl+ArrowUp', downKey = 'Ctrl+ArrowDown')
```

The human has access to 2 buttons: +1 reward and -1 reward.

By default, 'Ctrl+ArrowUp' is +1, 'Ctrl+ArrowDown' is -1. [Can use other keybindings.](https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/key/Key_Values)


<a id="sn-transform-shuffle"></a>
### `sn.Transform.Shuffle`[ ↑](#toc)

Shuffles cells randomly each step.

`.Handler.Storage` erases borders between steps, so this can help with autoregressive modeling. Also useful for learning position-invariance for models that do not have that enforced.

<a id="sn-transform-limitfps"></a>
### `sn.Transform.LimitFPS`[ ↑](#toc)

Limits steps-per-second.

Useful if you don't want to record ten thousand steps per second, almost none of which convey any useful information.

Options:
- `hz = 60`: max frequency.

<a id="sn-handler"></a>
## `sn.Handler`[ ↑](#toc)

Given data, gives feedback: is a human or AI model.

- `constructor({ onValues, partSize=8, userParts=1, nameParts=3, dataSize=64, noFeedback=false, priority=0, channel='' })`
    - `onValues(then, {data, error, noData, noFeedback, cellShape, partSize}, feedback)`: process.
        - ALWAYS do `then()` when done, even on errors.
            - If you stall artificially to maintain tempo (as `sn.Handler.Sound` does), pass in how many milliseconds you've stalled for.
        - `feedback` is available in the one main handler, which should write to it in-place.
            - In other handlers, data of `noData` cells will be replaced by feedback.
        - `noData` and `noFeedback` are JS arrays, from cell index to boolean.
        - `data` and `error` are not owned; do not write. `error` and `feedback` can be `null`s.
    - Cell sizes:
        - `partSize`: how many numbers each part in the cell ID takes up, where each string in a name takes up a whole part:
            - `userParts`
            - `nameParts`
        - `dataSize`: numbers in the data segment.
    - Extra flexibility:
        - `noFeedback`: can't provide feedback if `true`, only observe it.
        - `priority`: the highest-priority handler without `noFeedback` will be the *main* handler, and give feedback.
        - `channel`: the human-readable name of the channel. Communication only happens within the same channel.
    - To change any of this, `resume({…})`.

- `pause()`, `resume(opts?)`: for convenience, these return the object.

<a id="sn-handler-sound"></a>
### `sn.Handler.Sound`[ ↑](#toc)

Exposes data as sound, for humans to listen.

In Chrome, users might have to first click on the page for sound to play.

(This handler normalizes output frequencies to be zero-mean, so provide a diverse picture rather than a single number with a varying magnitude.)

- Extra options, for `constructor` and `resume`:
    - `volume = .3`: amplitude of sound output.
    - `minFrequency = 1000`, `maxFrequency = 13000`: how well you can hear. [From 20 or 50, to 16000 or 20000 is reasonable.](https://en.wikipedia.org/wiki/Hearing_range) The wider the range, the higher the bandwidth.
    - `nameImportance = .5`: multiplier of cell names. Non-1 to make it easier on your ears, and emphasize data.
    - `foregroundOnly = false`: if set, switching away from the tab will stop the sound.
    - `debug = false`: if set, visualizes frequency data in a `<canvas>`. (Usable for quickly testing `.Sensor.Video`.)


<a id="sn-handler-storage"></a>
### `sn.Handler.Storage`[ ↑](#toc)

Saves data to storage.

Data of no-data cells is already replaced with feedback, if the main handler is present to give it.

Uses [`indexedDB`.](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API)

Options:
- `filename = 'sn'`: which file to append to.
- `bytesPerValue = 0`: 1 to store as uint8, 2 to store as uint16, 0 to store as float32. Only relevant when first creating the file.

Functions:
- `sn.Handler.Storage.download(filename)`: downloads the file to the OS, importing the [StreamSaver](https://github.com/jimmywarting/StreamSaver.js?) library.
- `sn.Handler.Storage.upload(filename, file)`: gets the [file](https://developer.mozilla.org/en-US/docs/Web/API/File) from the OS, overwriting what is already in `filename`; to append, upload to a temporary location, then sense from there and handle to elsewhere.
- [`navigator.storage.persist()`](https://developer.mozilla.org/en-US/docs/Web/API/StorageManager/persist)
- [`indexedDB.deleteDatabase(filename)`](https://developer.mozilla.org/en-US/docs/Web/API/IDBFactory/deleteDatabase)


<a id="sn-handler-internet"></a>
### `sn.Handler.Internet`[ ↑](#toc)

Makes this environment a remote part of another sensor network, to be controlled.

Options:
- `iceServers = []`: the [list](https://gist.github.com/mondain/b0ec1cf5f60ae726202e) of [ICE servers](https://developer.mozilla.org/en-US/docs/Web/API/RTCIceServer/urls) (Interactive Connectivity Establishment).
- `signaler = sn.Handler.Internet.broadcastChannel`: creates the channel over which negotiation of connections takes place. When called, constructs `{ send(Uint8Array), close(), onopen, onmessage, onclose }`, for example, [`new WebSocket(url)`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket).
- `bytesPerValue=0`: 0 to transmit each value as float32, 1 to quantize as uint8, 2 to quantize as uint16. 1 is max-compression min-precision; 0 is the opposite.
- `autoresume = true`: whether the connection closing will trigger an attempt to re-establish it.
- `untrustedWorkaround = false`: if set, will request a microphone stream and do nothing with it, so that a WebRTC connection can connect. The need for this was determined via alchemy, so its exact need-to-use is unknown.

Imports [100 KiB](https://github.com/feross/simple-peer) on use.


<a id="sn-handler-internet-broadcastchannel"></a>
#### `sn.Handler.Internet.broadcastChannel`[ ↑](#toc)

```js
function signalViaBC(sensor=null)
```

Connects to all browser tabs, one connection per handler. Signals via a [`BroadcastChannel`](https://developer.mozilla.org/en-US/docs/Web/API/Broadcast_Channel_API). (Not in Safari.)

<a id="sn-handler-internet-consolelog"></a>
#### `sn.Handler.Internet.consoleLog`[ ↑](#toc)

```js
function signalViaConsole(sensor=null)
```

[`console.log`](https://developer.mozilla.org/en-US/docs/Web/API/Console/log) and `self.internetSensor(messageString)` and `self.internetHandler(messageString)` is used to make the user carry signals around.

<a id="sn-handler-internet-websocket"></a>
#### `sn.Handler.Internet.webSocket`[ ↑](#toc)

```js
function signalViaWS(url)
```

Signals via a [`WebSocket`](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket). Have to pass it the URL before passing it as the `signaler` option.

<a id="sn-handler-random"></a>
### `sn.Handler.Random`[ ↑](#toc)

Random -1…1 feedback, for debugging.

<a id="sn-meta"></a>
## `sn.meta`[ ↑](#toc)

<a id="sn-meta-docs"></a>
### `sn.meta.docs`[ ↑](#toc)

```js
function docs()
```

Returns the Markdown string containing all the sensor network's documentation.

Objects need to define `.docs` to be either a string or a function to that.

<a id="sn-meta-tests"></a>
### `sn.meta.tests`[ ↑](#toc)

```js
async function tests()
```

Runs all sensor-network tests, and returns `null` if OK, else an array of `[failedTestName, value1, value2]`.

If not `null`, things are very wrong.

Internally, it calls `.tests()` which return `[…, [testName, value1, value2], …]`. String representations must match exactly to succeed.

<a id="sn-meta-metric"></a>
### `sn.meta.metric`[ ↑](#toc)

```js
function metric(key, value)
```

Call this with a string key & string/number value to display/measure something, if `E.meta.bench` controls execution.

<a id="sn-meta-bench"></a>
### `sn.meta.bench`[ ↑](#toc)

```js
async function bench(secPerBenchmark = 30, benchFilter=null, onBenchFinished=null)
```

Very slowly, runs all sensor-network benchmarks.

Can `JSON.stringify` the result: `{ …, .name:{ …, key:[...values], … }, … }`.

Arguments:
- `secPerBenchmark = 30`: how many seconds each step should last.
- `benchFilter(obj) = null`: return `true` to process a benchmark, else skip it.
- `onBenchFinished(obj, id, { …, key: String | [...values], …}, progress) = null`: the optional callback. If specified, there is no result.

Note that [Firefox and Safari don't support measuring memory](https://developer.mozilla.org/en-US/docs/Web/API/Performance/memory).

Benchmarks are contained in `.bench()` near to code that they benchmark. Report metrics via `E.meta.metric(key, value)`.
Those methods return objects (such as arrays) that contain start functions, which return stop functions.


<a id="sn-meta-ui"></a>
### `sn.meta.UI`[ ↑](#toc)

```js
function UI()
```

Creates UI for convenient configuration of sensors. Append it to `document.body` or something.

CSS not included. Markdown parsing not included.

(It doesn't affect code size that much, was quick to develop, and made apparent a few bugs, so why not?)