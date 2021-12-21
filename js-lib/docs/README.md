<a id=toc></a>
# Table of contents

Sensor network:

- [sn.Sensor](#sn-sensor)

- [sn.Accumulator](#sn-accumulator)

- [sn.Handler](#sn-handler)

- [sn.meta](#sn-meta)

    - [sn.meta.docs](#sn-meta-docs)

    - [sn.meta.tests](#sn-meta-tests)

    - [sn.meta.metric](#sn-meta-metric)

    - [sn.meta.bench](#sn-meta-bench)

    - [sn.meta.save](#sn-meta-save)

<a id="sn"></a>
# `sn`[ ↑](#toc)

<a id="sn-sensor"></a>
## `sn.Sensor`[ ↑](#toc)

Generalization of eyes and ears and hands, hotswappable and differentiable.

- `constructor({ name, values, onValues=null, channel='', noFeedback=false, rewardName=[], userName=[], emptyValues=0, hasher=undefined })`
    - `name`: a human-readable string, or an array of that or a -1…1 number or a function from `dataStart, dataEnd, dataLen` to a -1…1 number.
    - `values`: how many values each packet will have. To mitigate misalignment, try to stick to powers-of-2.
    - `onValues(sensor)`: the regularly-executed function that reports data, by calling `sensor.send` inside.
        - Can return a promise.
    - Extra flexibility:
        - `channel`: the human-readable name of the channel. Communication only happens within the same channel.
        - `noFeedback`: set to `true` if applicable to avoid some processing. Otherwise, feedback is the data that should have been.
        - `rewardName`: the name of the currently-optimized task, in case accumulators want to change it and inform handlers.
        - `userName`: the name of the machine that sources data. Makes it possible to reliably distinguish sources.
        - `emptyValues`: the guaranteed extra padding, for fractal folding. See `._dataNamer.fill`.
        - `hasher(…)(…)(…)`: see `._dataNamer.hasher`. The default mainly hashes strings in `rewardName`/`userName`/`name` with MD5 and rescales bytes into -1…1.
    - To change any of this, `pause()` and recreate.

- `send(values, error = null, reward = 0) → Promise<null|feedback>`
    - `values`: owned flat data, -1…1 `Float32Array` of length `values`. Do not perform ANY operations on it once called.
        - (Can use `._allocF32(length)` and fill that to reduce allocations via reuse.)
    - `error`: can be owned flat data, -1…1 `Float32Array` of length `values`: `max abs(truth - observation) - 1`. Do not perform ANY operations on it once called.
    - `reward`: every sensor can tell handlers what to maximize, -1…1. (What is closest in your mind? Localized pain and pleasure? Satisfying everyone's needs rather than the handler's? …Money? Close enough.)
        - Can be a number or a function from `valueStart, valueEnd, valuesTotal` to that.
    - (Result: `feedback` is NOT owned by you. Do NOT deallocate with `._deallocF32(feedback)`.)

- `pause()`, `resume()`

<a id="sn-accumulator"></a>
## `sn.Accumulator`[ ↑](#toc)

Modifies data/feedback, after sensors and before handlers.

- `constructor({ onValues=null, onFeedback=null, priority=0, channel='' })`
    - Needs one or both:
        - `onValues(data, error, cellShape) → extra`: can modify `data` and the optional `error` in-place.
            - `cellShape: [reward, user, name, data]`
            - Data is split into cells, each made up of `cellShape.reduce((a,b)=>a+b)` -1…1 numbers.
            - Can return a promise.
        - `onFeedback(feedback, cellShape, extra)`: can modify `feedback` in-place.
            - Can return a promise.
    - Extra flexibility:
        - `priority`: accumulators run in order, highest priority first.
        - `channel`: the human-readable name of the channel. Communication only happens within the same channel.
    - To change any of this, `pause()` and recreate.

- `pause()`, `resume()`

<a id="sn-handler"></a>
## `sn.Handler`[ ↑](#toc)

Given data, gives feedback: human or AI model.

- `constructor({ onValues, partSize=8, rewardParts=0, userParts=1, nameParts=3, dataSize=64, noFeedback=false, priority=0, channel='' })`
    - `onValues(data, error, cellShape, writeFeedback, feedback)`: process.
        - (`data` and `error` are not owned; do not write.)
        - `error` and `feedback` can be `null`s.
        - If `writeFeedback`, write something to `feedback`, else read `feedback`.
        - At any time, there is only one *main* handler, and only that can write feedback.
    - Cell sizes:
        - `partSize`: how many numbers each part in the cell ID takes up, where each string in a name takes up a whole part:
            - `rewardParts`
            - `userParts`
            - `nameParts`
        - `dataSize`: numbers in the data segment.
    - Extra flexibility:
        - `noFeedback`: can't provide feedback if `true`, only observe it.
        - `priority`: the highest-priority handler without `noFeedback` will be the *main* handler, and give feedback.
        - `channel`: the human-readable name of the channel. Communication only happens within the same channel.

- `pause()`, `resume()`

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


<a id="sn-meta-save"></a>
### `sn.meta.save`[ ↑](#toc)

```js
function save(...fs)
```

Preserves closed-over dependencies, so that code can be loaded, via `const [...funcs] = new Function('sn', result)(sensorNetwork)`.

Those dependencies do have to be explicitly preserved, such as via `a => Object.assign(b => a+b, { save:{a} })`.

Note that directly-referenced methods (`{ f(){} }`) have to be written out fully (`{ f: function(){} }`), and sensor-network dependencies should not be specified.

Safe to save+load if `.toString` is not overriden by any dependency, though not safe to use the loaded functions.