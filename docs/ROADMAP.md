Know what needs to be done, and do it.

No deleting, only adding (at the end) and keeping track of progress. Rephrasing/refinement that spans items is allowed.

- ✓ Track completion with Unicode characters: ⋯ to-do, ✓ complete, ❌ canceled.

- ✓ Have directories `/rs`, `/js-lib`, `/js-ext`. Init the Git repo.
    - ✓ In `/js-lib`, `npm init`.

---

## Platform

The Sensor Network aims to expose all real-time data accessible to a machine in a modular, dynamically-reconfigurable, AI-enabled, fashion.

It has to be efficient, and easily accessible.

- ⋯ To that end, have 2 interoperable (via WebRTC) platforms:
    - ⋯ The IPC-based [Rust](https://www.rust-lang.org/) implementation in `/rs`. The OS is the environment.
    - ⋯ [JS](https://developer.mozilla.org/en-US/docs/Web/JavaScript):
        - ⋯ In-page library in `/js-lib`, for programmatic use of the API. The web page is the environment.
        - ⋯ Extension in `/js-ext`, for ad-hoc cross-tab setup and use of sensors, and for Chrome's [`tabCapture` API](https://developer.chrome.com/docs/extensions/reference/tabCapture/). Makes the browser the environment instead, and the human the end-user.

### Rust API

The OS ecosystem, where communication happens through IPC. Each extra module should have its own crate, prefixed with `sensor-network-`.

The functioning of the Sensor Network proceeds as such:
- Each step: first, `-1`…`1` numbers are collected into named cells of a fixed size by *senders*, then all *handlers* of the environment receive those cells and create feedback (of the same size), then feedback is fed back to the *senders*. This composable architecture is made possible by *order invariance* of cells.
- Each cell has structure: first 1 number for the reward (because prediction of what others did is *not* enough to distinguish preferable solutions) (always 0 in no-action senders), then the name (split into equally-sized parts), then data. By default, `64` numbers in data, and `63` numbers in the name, `16` numbers per name part.
    - The name (positional embedding) can be specified as an array, where strings are hashed and turned into basically-unpredictable number sequences and put into parts in-order, and numbers are put wherever. At least 1 part is always for numbers. The first part is always for the user ID, the same per machine, `"self"` by default; allows to compose many machines into a network.
    - ✓ Nail down some simple string-hashing strategy. Such as putting MD5 byte-by-byte, rescaling to `-1`..`1`, fractally folding each part if needed.
    - ⋯ Allow specifying user ID (a string), or telling that it should be re-generated each time, or a place where it is stored.

This allows pretty much any interaction to happen, from simple observation of data, through actions in the environment, to corrections of observations [if they can change some](https://arxiv.org/abs/2006.12057)[how](https://powerlisting.fandom.com/wiki/Mind_Link).

- ⋯ Per-machine named [IPC](https://docs.rs/interprocess/latest/interprocess/) broadcasting, in this repo.
    - ⋯ The trait `Sensor`, which specifies a sensor:
        - ⋯ `new(name:&[&StringOrNumberOrClosure], data_size: u64, channel: Option<String>)`.
            - ⋯ Each closure in `name` is called with start & end indices, and returns a `-1`…`1` number to put. This allows individual blocks to have different metadata, so that models can adapt easily.
        - ⋯ `data_size(&self)->u32`. `0` by default.
        - ⋯ `on_data(&self, feedback: Vec<f32>, reward_feedback: f32)->Future<(data:Vec<f32>, reward:f32)>`: if implemented, this will be called automatically whenever handlers are ready.
        - ⋯ Pre-implemented `send(&self, data: Vec<f32>, reward: f32)->Future<(feedback: Vec<f32>, reward_feedback: f32)>`: send data, receive feedback, [eventually](https://crates.io/crates/futures).
        - ⋯ Implement `std::io::Write` (sending data) and `std::io::Read` (getting feedback) on top of `send`, which make the reward `0`. (Unless Rust complains about conflicting implementations.)
        - ⋯ `no_feedback(&self)->bool`, `true` by default, for no-action things. Handlers shouldn't bother giving feedback on these.
    - ⋯ The trait `Transform`, which changes a frame's data (post-sending pre-handling):
        - ⋯ For efficiency, transforms are not IPC (which would have needed at least one IPC copy per transform per message), but have to be created in the same process as the main `Handler`.
        - ⋯ `new(channel: Option<String>)`.
        - ⋯ `on_data(&self, data: Vec<f32>, cell_shape:&[u32])->Future<(Vec<f32>, Extra)>` returns a value [eventually](https://crates.io/crates/futures).
        - ⋯ `on_feedback(&self, feedback: Vec<f32>, cell_shape:&[u32], extra: Extra)->Future<Vec<f32>>`.
        - ⋯ `priority(&self) -> f64`: all transforms are called in a chain, highest-priority-first. `0` by default.
    - ⋯ The trait `Handler`, which gives feedback to sensors:
        - ⋯ `new(channel: Option<String>)`.
        - ⋯ `on_data(&self, data: Vec<f32>, error: Vec<f32>, cellShape: &[f32], feedback: Option<Vec<f32>>) -> Future<Option<Vec<f32>>>`: give feedback to data, or observe another handler's data+feedback.
            - ⋯ On each step, turn observations into corrections:
                - ⋯ A periodic loop of fulfilling corrections to smooth network latency, released in the order that they were requested, sending back data if not handled when needed. Try to match the latency of observation-correction things.
                    - ⋯ Benchmark the latency deviation, mean & stdev. Both when messages are sent evenly, and in bursts of 2/4/8/16/32/64/128.
                    - ⋯ Give per-number max error along with data.
            - ⋯ On each sent message, wait a bit before handling messages, to make inputs more coherent. (And, benchmark the coherence, as the % of senders transformed, avg per step.)
            - ⋯ Only send feedback to those senders that have `.no_feedback()->false`.
        - ⋯ `no_feedback(&self) -> bool`, `true` by default.
        - ⋯ `priority(&self) -> f64`, `0` by default: only the one max-priority handler with `no_feedback=false` will give feedback, the rest will simply observe.
        - ⋯ `data_size(&self) -> u32`, `64` by default.
        - ⋯ `name_size(&self) -> u32`, `64` by default.
        - ⋯ `name_part_size(&self) -> u32`, `16` by default.
    - ⋯ Sensors first request cell shapes from handlers, then for each unique shape, allocate the actual positions. On step, limit f32 numbers to `-1`…`1`, put them in places, fill in reward (`0`) & name, compress if specified, then send to handlers.
    - ⋯ Test that all data is indeed transformed correctly, via bogus senders/transforms.
    - ⋯ Benchmark throughput and latency with bogus data, in time-per-cell.

- ⋯ Sensors, each with read and write modes:
    - ⋯ Actual sensors, as in actual hardware interfaces, each exposed with both read (handlers observe the human acting) and write (handlers specify actions) modes:
        - ⋯ Keyboard.
        - ⋯ Mouse.
        - ⋯ Video+audio of a window or desktop, or camera/microphone.
            - ⋯ Research libraries: [X11Cap](https://github.com/bryal/X11Cap) is Linux-only; [FFmpeg](http://www.ffmpeg.org/) may be good, though is an external dependency.
            - ⋯ Video, in spatially-coalesced chunks, with x/y coords of each chunk's center in the name.
                - ⋯ Full stream.
                - ⋯ Around-mouse rect.
                - ⋯ Around-mouse fovea.
                - ⋯ Around-mouse progressively coarser grids, each zoomed out 2×, so if starting at 16×16 cells, only need 6 more levels to go to 1024×1024. (May act the same as a zooming data augmentation.)
                    - ⋯ Make the "mouse" point configurable.
            - ⋯ Audio:
                - ⋯ Mono (average all channels, with -1 & 0 in the name).
                - ⋯ Stereo (expose each channel, with 1 & i in the name).
            - ⋯ Write-mode: create a window and draw in it for video, and/or play the audio that we get. Debugging, essentially.
    - ⋯ System resources: CPU (% free mem and per-core % used) and GPU if available (roughly, % free mem and % used; align if possible).
    - ⋯ Time, as sines-of-time-multiples, with 100FPS as the most-frequent-wave-period.
    - ⋯ Read from Internet, through WebRTC. Many machines can thus gather into one sensor network.
        - ⋯ Each data packet (1+ cells) references its meta-data (cell shape) by ID; when/if meta-data changes, it's re-sent, and the other side can request it if it doesn't know it (such as when the packet got lost). Though, cell shape shouldn't ever change, or there's a big problem in ML models.
        - ⋯ Discourage disengagements: on user disconnect, hold its last block (`0` everywhere except the user in the name) with `-1` reward, for as many frames as specified (`8` by default). Dying is bad.
        - ⋯ Benchmark throughput, over localhost, with the default data (a file, preferably always the same one).
    - ⋯ Read from file/s.
    - ⋯ Launched-by-another-process STDIO. Model's outputs are sent, and model inputs are received as feedback (`0` or random noise in the first frame); inputs and outputs have separate cells, separated by type if possible. (With prediction & compute, one brain/model can download any knowledge and intuition for free, in the background. All models can gather in one.)
        - ⋯ Connect CPU-side GPT-2, which acts word-per-word, or even letter-per-letter and integrates like the keyboard sensor, by sharing parts of the name.
        - ⋯ Connect a GAN's generator, from a random or drifting vector to some simple data, such as MNIST digits, or even a very simple tabular dataset. See whether listening to this can somehow give an understanding.
    - ⋯ A [Puppeteer](https://pptr.dev/)ed browser, where the JS extension is installed, and we make it inject interfaces and collect data by calling a Puppeteer-injected function (from base64 data, to a promise of base64 feedback) for us.

- ⋯ Transforms:
    - ⋯ Shuffle cells, to make models/brains that are not fully order-independent become such.
    - ⋯ Reward sender, which replaces `0`s in all cells' first number with the reward. To specify per-user reward in one place. (The idealized job: you give it your situation, it makes your number go up.)
        - ⋯ Configurable reward, via a closure that's called each frame. By default, Ctrl+Down is `-1` reward, Ctrl+Up is `+1` reward, otherwise `0`.
    - ⋯ An alternative string-hashing strategy, namely, "ask the user" (display the string somewhere for at least a few seconds, send `0`s as the name, and record suggestions; the most distant one from all names in the database wins, and the mapping from string-hash to actual-data is preserved, so that even file recordings can be replayed comfortably).

- ⋯ Handlers (launch the main handler first, the rest will not give feedback):
    - ⋯ Sound output (speakers), no feedback: like machine-to-brain Neuralink, but everyone already has it. (Can even listen to what an AI model predicts and decides, for zero-effort human-AI merging.)
        - ⋯ Test, which sounds best and most recognizable, and what bandwidth we can achieve without making users tear out their ears: raw PCM output, +x -x PCM output, frequency-domain output.
    - ⋯ Sound input (microphone). May be trash though.
    - ⋯ Send to Internet.
        - ⋯ Username: our local MAC address or IP or a stored randomly-generated number, added to each cell's label before anything. (In a model, to batch per-user rather than integrate integrate cross-user data, group by username.)
        - ⋯ Research libraries that can carry messages: [Rabbi](https://github.com/CleverCloud/lapin)[tMQ](https://crates.io/crates/amiquip); [raw WebRTC](https://webrtc.rs/).
        - ⋯ Test stability: a remote sender that fails periodically, and a remote handler that fails periodically: the system has to re-establish connection automatically.
    - ⋯ Store to file/s.
        - ⋯ Each file consists of 16KB blocks, at the start of each is an i32: either a pointer to the next block or negated length of this one; it can then store streams of bytes; the list starting at block 0 specifies metadata (compression version) and all pointers to stream beginnings. So we need API for write-stream and enum-streams and sample-random-stream and read-stream.
        - ⋯ Max cell count in one stream, 0 for unlimited; auto-split when needed, without saving model state.
        - ⋯ Max byte-count of a file; when specified, store to a directory of named files. Also allow limiting the file count, to act as a circular buffer.
        - ⋯ Possibly, each stream should have a vector label, and we should allow nearest-neighbor lookups. Possibly separately from the actual data. (Indirectly allows things like priority queues and filter-by-creation-date.)
    - ⋯ Launched-by-another-process STDIO.
        - ⋯ Communicate in packets: cell-count and 1+value-size (so that names can be resized if needed) and cell-size and all-cell-data (`-1`…`1`). Uncompressed for simplicity of integration.
            - ⋯ Benchmark actual throughput.
        - ⋯ A [Perceiver IO](https://arxiv.org/abs/2107.14795) model to do next-frame prediction and first-cell-number maximization.
            - ⋯ To take advantage of exponential improvement of learning, ML must be a journey, not a series of separate steps like it is today. Sensor network being able to represent all datasets in one format is a prerequisite. So, experiment: implement weight decay weighted by gradient to [learn precise behaviors](https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf) without much inter-task interference, or weight-magnitude-based evolution of weight groups for cache-aware sparsification, then make the same model fully learn very small datasets. Are there then benefits to learning bigger text-based datasets? Might be a dead-end though.
            - ⋯ Prediction is technically a number to optimize. So make a transform that rewards a cell's next-frame prediction, and try to fully-learn that, so that models can then choose to predict without being forced to.
    - ⋯ A Neuralink device. Once it, you know, exists. (Maybe it would be a [HID](https://web.dev/hid/).)

- ⋯ Compression, for Internet and files:
    - ⋯ API for compressing and decompressing a stream of messages: a trait, with the constructor (taking cell-shape) and the compression version and message-compression (from message and context, to data and context) and message-decompression (taking data and context, returning message and max error and context).
        - ⋯ Handle historical context, via sync points: knowing past states can help compression, but if messages can be dropped (such as in RTC), decompression can't proceed if it needs history, so we need history-less points. Files won't need this, but WebRTC will.
        - ⋯ No std lib, for wasm.
    - ⋯ No-op compression, as the default.
    - ⋯ A benchmark that measures compression ratio of a file, preferably always the same one, stored in the repo. Fail if uncompressed data does not match actual data, within the returned tolerance. And compression speed, per cell.
    - ⋯ Try compression options, and find the best one.

- ⋯ PyTorch (or NumPy) integration, via easily-comprehensible blocking "gimme more data" and "give more data" functions.
    - ⋯ Research dataset/environment libraries, and how one-line we can make sensors of those. (Take a bath in data. Rub it into your eyes.)

- ⋯ A server for nearest-neighbor search. Accept requests: "register/update this ID, with a removal token" (the -1…1 vector label is decided randomly), "remove this ID, given a removal token" (the label still persists), "get up-to-N nearest neighbors, current URL/IP & ID & current label", "update this ID with this suggested vector label" (with an update method that effectively averages inter-user most-recent votes, and slowly nudges the label toward that, notifying the register-ee along the way, removing the ID if not connected). (Connect AI models & humans: train a worldwide AI model collaboratively, and vote with far more detail than democracy can ever hope for.)

- ⋯ If we can somehow find TBs of storage for hosting, create a server that stores all incoming connections to file, and samples from that file when requested, or reads by stream ID.

Isn't implementing all this such a joyful learning opportunity?

### JS API

Intelligence can do anything. But how to support the utter formlessness of generality of intelligence? By supporting every form at once. With reasonable defaults.

- ⋯ One library that puts everything into the global `sn`, populated as variables:
    - ✓ The basics:
        - ✓ Have a name-hasher, from name and available-parts and part-size to Float32Array, possibly written-to in-place.
            - ✓ To not waste space, numbers fill up their cells (and all no-string cells) with fractally-folded versions of themselves; each fold turns the line `{ 0: -1, 1: 1 }` into `{ 0: -1, .5: 1, 1: -1 }`, so, `x → 1-2*abs(x)`. (The listener can then make out details more easily.)
            - ✓ Data should also do that if unused, with feedback adding up the details too so that the reported feedback is nudged appropriately. No holes, only more detail.
        - ✓ Decide whether we handle `NaN | -1…1` or `-1…1`. Verdict: `-1…1`, because what even are holes in sensors.
        - ✓ `.Sensor`, used as `new Sensor({ name:['keyboard', 'a'], values:1, async onValues(s) { console.log((await s.send([1]))[0]) } })`:
            - ✓ `.constructor({ name, values=0, channel=null, onValues=null })`.
                - ❌ The options object can be modified after construction. (No, users should just destroy and recreate, emphasizing that this should be rare.)
                - ✓ `onValues(sensor) -> Promise<void>`: send data & receive feedback via `sensor.send(…)`, as often as possible, possibly async.
                    - ✓ Send at most 4 at once. Measure the average time between the main handler's feedbacks (even `noFeedback` empty feedback is feedback for this), and match it as exactly as we can.
            - ✓ `.pause()`, `.resume()`
            - ✓ `.send(values: Float32Array|null, error: Float32Array|null, reward=0, noFeedback=false) -> Promise<Float32Array|null>`: send data, receive feedback, once. (Reward is not fed back.)
                - ✓ "Allocate" the name into one array by creating a closure that writes, and re-use it, copying values into proper places.
                - ✓ Make transforms & handlers accept `noData` and `noFeedback` per-cell arrays as named args. After the main handler, replace `noData` cells with their feedback.
            - ❌ For convenience, if [`FinalizationRegistry`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/FinalizationRegistry) is present, `.pause()` when the sender is no longer needed. (The only reason for this is use in browser console, which is too iffy to justify such an unreliable functionality. Besides, forcing storage is just annoying when the user wants to fire-and-forget.)
        - ✓ `.Transform`:
            - ✓ Rename from `Accumulator` to `Transform` to stop being confusing.
            - ✓ `.constructor({ channel=null, priority=0, onValues=null, onFeedback=null })`.
                - ❌ The options object can be modified after construction.
                - ✓ Transforms run highest-priority-first.
                - ✓ `onValues({data: Float32Array, error: Float32Array|null, cellShape: [user, name, data], noData: Array<bool>, noFeedback: Array<bool>}) -> Promise<extra>`: prepares to modify data in-place, possibly async. The sum of numbers in `cellShape` always divides `data.length`.
                - ✓ `onFeedback(feedback: Float32Array, cellShape: [user, name, data], extra) -> Promise<void>`: modifies data's feedback. Maybe you want privacy, or maybe not all input sources are equally easy to activate.
            - ✓ `.pause()`, `.resume()`
            - ❌ For convenience, if [`FinalizationRegistry`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/FinalizationRegistry) is present, stop when the transform is no longer needed.
        - ✓ `.Handler`:
            - ✓ `.constructor({ onValues, channel=null, priority=0, noFeedback=false, dataSize=64, nameSize=64, namePartSize=16 })`.
                - ❌ The options object can be modified after construction.
                - ✓ `dataSize` is how many data numbers each cell can hold, `nameSize` is how many numbers the cell is identified with, split into `namePartSize`-sized blocks. First name then data; the first name part is used for the reward (always the first number) and the user ID, the rest are taken up be senders' string hashes and numbers.
                - ✓ `onValues({data: Float32Array, error: Float32Array|null, cellShape: [user, name, data], noData: Array<bool>, noFeedback: Array<bool>}, writeFeedback: bool, feedback: null|Float32Array)->Promise<void>`: receive data, and modify it in-place to send feedback (modify synchronously when the promise returns, to prevent data races).
            - ✓ `.pause()`, `.resume()`
            - ✓ On each sent message, wait a bit before handling messages, to make inputs more coherent.
                - ✓ Benchmark the coherence, as the cell-count transformed at each step. (Seems fully coherent.)
        - ✓ A function that runs all unit tests, `.tests()`, which traverses `sn` (not through prototypes) and calls every `tests` method.
            - ✓ `test.html`, which imports the `main.js` module and runs `sn.tests()`.
        - ✓ A function that runs all benchmarks, `.bench()`: traverses `sn` (not through prototypes) and calls every `bench` method.
            - ✓ Ability to log arbitrary metrics at arbitrary points, with the final report being the average metric per log.
            - ✓ Allow benchmarking in `test.html`.
                - ❌ Auto-save in `localStorage`, keyed by source-code-hash, only running no-results benchmarks or those explicitly requested. (Better to invalidate all dependents too, which is quite hard to do in JS.)
                - ✓ Display each object's benchmark results, with all per-metric plots (copy plotting code from Conceptual).
                - ✓ Allow switching between mean/median/all views for a fuller view of performance.
            - ✓ Benchmark sending+handling, from `1`-filled data to `-1`-filled feedback. We want to know the throughput (bytes/sec) and memory pressure (bytes/cell), for 4-number and 64-number cells, measured over █ minutes of running.
            - ✓ Transition from promises to callbacks, because promises are THE major allocation cause. (In per-packet `handle`, anyway; `handleLoop`, which calls `handle`, would get an annoyingly-infinite async-stack with callbacks.)
        - ✓ `.docs()`, which traverses `sn` and transformes all `docs` strings into a Markdown string.
            - ✓ Parents should become sections, into which their children belong.
            - ✓ Make a table of contents at the top, with refs to the top at every section heading.
            - ✓ Make `npm run doc` import the library and call this and write its result to `docs/DOCS.md`.
        - ✓ Ability to de/serialize sensors/transforms/handlers, so that users can pick up power-ups at the press of a button.
            - ✓ Have the `.needsExtensionAPI() → null|string` method on `Sensor`s, returning `null` by default, but can return `''` or `'tabs'`. Let users control which parts of the `chrome` API the extension can see.
                - ⋯ Actually use it in an extension.
        - ⋯ A better loop:
            - ⋯ A better time-per-step estimation scheme than moving-average-over-32-steps (limited to *1.1+11). It was intended to make small deviations insignificant, but it takes too long to actually adjust to a new rhythm. Maybe, keep track of moving-average-over-4, and when that average is too consistent for 4 steps (so, 4-length subsequences of a 7-steps array have a small standard deviation), update the actual ms-per-step… Or maybe just median instead of mean.
            - ⋯ Keep track of the "bottleneck": on `Handler`s, have `.bottleneck` (0…1), exponentially-updated avg of 0/1, where the last-returning non-main handler is 1 and the rest are 0s (also, compare non/main handlers, and the fastest-returning one is 0). And make `UI` display a message when `.bottleneck>.99`.
    - ⋯ Reasonable defaults, decided by the user and not the handler, in separate `import`ed files, or maybe their own NPM modules (though they *are* small):
        - ⋯ Make `main.js` import modules that import it and export classes that inherit sensors/transforms/handlers, by having getters that patch themselves on use.
        - ⋯ `.Sensor`:
            - ⋯ Actual sensors, with "observe the hardware" (no feedback) and "visualize effects in-page" (feedback, with data's error being `1`) modes, and UI visualization where possible:
                - ⋯ Keyboard.
                    - ❌ Put all keys in one strip, in lexicographically-first order. Or use a spatially-grouped QWERTY key layout. Or have a separate cell for every possible key, for max precision. (Too data-inefficient.)
                    - ⋯ One-hot-encode or MD5-hash the key, and have like 3 observation or action cells.
                - ⋯ Pointer (mouse/touch).
                    - ⋯ `targets: [..., {x,y,active}, ...] = Video.pointers()`: the objects to update. Share this with `Video` to be able to move virtual pointers.
                        - ⋯ `Video.pointers` → `Pointer.tab`.
                        - ⋯ Make `Pointer.tab()` return the exact same array if called many times, don't attach new event listeners.
                        - ⋯ Optionally have `.update()`, set by `Pointer.tab()`:
                            - ⋯ Movement fires pointer move/over/out events; rising/falling `active` edge fires up/down events.
                            - ⋯ The first item `.isPrimary` and `'mouse'`, the rest are `'touch'`.
                            - ⋯ Test whether mouse/touch event listeners are triggered by pointer events, and if not, dispatch them too.
                            - ⋯ To display hover-states, use ancient magic: go through all CSS rules in all stylesheets and in every new stylesheet, and duplicate those with `:hover` to use a class, which main-mouse-movement sets.
                    - ⋯ `noFeedback = true`:
                        - ⋯ If an action: on feedback, update the `{x,y}` objects, and create [non-trusted](https://developer.mozilla.org/en-US/docs/Web/API/Event/isTrusted)[ pointer events](https://developer.mozilla.org/en-US/docs/Web/API/PointerEvent/PointerEvent) (the first item is primary and `'mouse'`, the rest are `'touch'`) if possible to make the DOM aware.
                        - ⋯ If an observation: a cell per pointer (dynamically created/destroyed), report x (0…1), y (0…1), is-it-pressed, isPrimary, pointerType, width (screen-fraction), height (screen-fraction), pressure, tangentialPressure, tiltX, tiltY, twist.
                    - ⋯ `visualize({data, cellShape}, elem)`, showing round points.
                - ⋯ Scroll, exposing not just at top-level but in hierarchy levels: current X/Y and max X/Y scroll position; non-existent ones are 0s. Occupy only 1 cell.
                - ✓ Video: `Video`.
                    - ⋯ `name`, integrated into the actual `name`.
                    - ✓ `source`: `<canvas>` or `<video>` or `<img>` or `MediaStream` or a function to one of those.
                        - ✓ `static stitchTab()`, which draws the viewport's visible `<canvas>`es into a hidden `<canvas>`/`<video>`/`<img>`. This is the default in non-extensions, because it requires no user interaction.
                            - ⋯ Ask the extension for the stream if it allows us. (For security, the extension needs a per-tab checkbox "allow the page to read its own video/audio".)
                        - ✓ `static requestDisplay()`, which uses `getDisplayMedia`.
                        - ✓ `static requestCamera()`, which uses `getUserMedia`.
                    - ✓ Data on context2D: draw `source` into tiles.
                        - ⋯ To make blindly copying differently-shaped-cell data at least half-correct, collect pixels along diagonals, not row-by-row.
                    - ⋯ Feedback on context2D: draw tiles into `source`-shaped spots, by having `source.onFeedback(feedbackCanvas)`. Unzoom and position tiles properly, and make sure that max-detail information always wins.
                    - ✓ Coalesce tiles spatially, with x/y coords of the center in the name, with each tile dimension being `tileDimension`. 1 tile per cell: when `cellShape[-1]` is too small, cut off; when too big, zero-fill.
                        - ✓ Each cell's name: `['video', ''+tileDimension, x(), y(), zoom(), color()]`, where un/zoom level is -1 for 1× and 1 for 1024×, and color is -1 for monochrome and -⅓ for red and ⅓ for green and 1 for blue.
                    - ✓ The points `targets`: `[..., {x,y}, ...]`, 0…1 viewport coordinates.
                        - ✓ If empty, downsample the *full* stream, else follow the targets.
                        - ✓ By default, is `static pointers() → Array` for `Video`: every `.pointerId` that is in a pointer event is in here.
                    - ✓ Zooming-out, steps & magnitude-per-step, `zoomSteps` &  `zoomStep`; for example, with 6 & 2 with an 8×8 initial rect also generates 16×16 and 32×32 and 64×64 and 128×128 and 256×256 and 512×512, each downscaled to 8×8.
                    - ✓ Tiling, steps, `tilingSteps`; 1 is just the one rect, 2 is a 2×2 grid of rects with the center at the middle, and so on.
                    - ❌ Internally, for efficiency, render images to a WebGL texture if `gpuDecode:true`, and download data from there. (We already rescale via `drawImage` in Canvas 2D.)
                    - ⋯ `visualize` into a user-resizable canvas.
                    - ✓ A benchmark of reading from a `<canvas>`-sourced `MediaStream`, 2048×2048, as fast as possible.
                        - ❌ Use [`VideoFrame` and `MediaStreamTrackProcessor`](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrackProcessor). (Not a bottleneck.)
                        - ✓ To minimize GPU→CPU transfer, make reading 1 or more frames behind writing. (Helps a very tiny bit.)
                        - ❌ Tone down `MediaStream` settings dynamically (`frameRate`, `width:{ideal:512}`), according to how much we can/can't accept. (Doesn't work with canvas streams in Firefox; doesn't help in Chrome.)
                        - ⋯ To not draw invisible elems in `stitchTab`, if available, use [`IntersectionObserver`](https://developer.mozilla.org/en-US/docs/Web/API/IntersectionObserver) and [`MutationObserver`](https://developer.mozilla.org/en-US/docs/Web/API/MutationObserver).
                - ⋯ Audio.
                    - ⋯ Mono, by averaging all channels.
                    - ⋯ 2+ channels, each exposed directly.
                    - ⋯ Create an audio context that reads PCM data from the media stream.
                        - ⋯ Request from extension if possible.
                    - ⋯ Expose data and feedback as [audio](https://developer.mozilla.org/en-US/docs/Web/API/MediaStream) [streams](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamAudioDestinationNode): `.soundData()=>dataStream` and `.soundFeedback()=>feedbackStream`.
                        - ⋯ For more efficiency, allow initializing from an audio context by attaching all our nodes to it, in case all sound is played through that: `.onAudioContext(ctx, outputNode)`.
                    - ⋯ In its visualization, two `<audio>` elements, for data and feedback.
                        - ⋯ And a volume slider.
                        - ⋯ Report data/feedback volumes with color, possibly with `box-shadow`.
                - ⋯ `Text`. (The ability to *annotate* what you're doing. No need to guess intentions of humans if they can just tell you.)
                    - ✓ `name`, integrated into the actual `name`.
                    - ✓ `tokens=64`, `tokenSize=64`. One token per cell.
                    - ⋯ `text() → str` or `text:{ feedback(str) }`:
                        - ✓ `Text.readSelection(n=2048)`: `getSelection()`, `<input>`, `<textarea>`. If selection is empty, returns up-to-`n` characters before that, else only the selection.
                        - ✓ `Text.readHover(pos=Video.pointers(), n=2048)`: gets the text position [under ](https://developer.mozilla.org/en-US/docs/Web/API/Document/caretRangeFromPoint)[cursor](https://developer.mozilla.org/en-US/docs/Web/API/Document/caretPositionFromPoint) or under an `{x,y}` object (a virtual pointer), goes to end-of-word if trivial, and reads `n` characters before that.
                        - ❌ `Text.readChanges(n=2048)`, using a [`MutationObserver`](https://developer.mozilla.org/en-US/docs/Web/API/MutationObserver). (Why, does this happen often enough to be useful?)
                        - ✓ `Text.writeSelection()`: [modify the current selection](https://developer.mozilla.org/en-US/docs/Web/API/HTMLInputElement/setRangeText), if `contenteditable` or `<input>` or `<textarea>`. (Is autocomplete or autocorrect.)
                    - ✓ `textToTokens(str, max)→tokens = s => s.split('').slice(-max)`, with `.feedback(tokens)→str = a => a.join('')`.
                    - ✓ `tokenToDataMD5(token, data, start, end)=…`, with `.feedback(feedback, start, end)→token`. Use MD5 hashes because it's easy to, though feedback may suffer.
            - ⋯ Chrome/Edge/Opera (Firefox has no direct hardware access):
                - ⋯ Raw bytes of [HID](https://web.dev/hid/), remapped to -1…1.
            - ⋯ Mobile device [sensor readings](https://developer.mozilla.org/en-US/docs/Web/API/Sensor_APIs) (a Chrome-only API). Or [through ](https://developer.mozilla.org/en-US/docs/Web/API/DeviceMotionEvent)[events](https://developer.mozilla.org/en-US/docs/Web/API/DeviceOrientationEvent)? Why are there two APIs?
            - ✓ Time, as sines of exponentially decreasing frequency, with 100FPS as the most-frequent-wave-period.
            - ❌ System resources, if exposed: `m=performance.memory, m.usedJSHeapSize / m.totalJSHeapSize`. (Doesn't report a good number. Nor would have been useful even with a good estimate of RAM usage, because if JS over-allocates, it's usually already too late to do anything from JS.)
            - ⋯ Read from another channel: insert a hidden handler to that channel, and read-through.
            - ⋯ Read from file.
            - ⋯ In extension, read from tabs.
            - ⋯ Read from Internet, with WebRTC, RabbitMQ preferable.
                - ⋯ Each data packet (1+ cells) references its meta-data (cellShape & partSize & noData & noFeedback) by ID; when/if meta-data changes, it's re-sent, and the other side can request it if it doesn't know it (such as when the packet got lost).
                - ⋯ To communicate userName/name, fill them with closures, that read from received data.
                - ⋯ Discourage disengagements: on user disconnect, hold its last cell (now `0` everywhere except the user in the name) with `-1` reward, for as many frames as specified (`8` by default). Dying is bad.
                - ⋯ Benchmark throughput, over localhost, with the default data (a file, preferably always the same one).
            - ⋯ Search: in a distributed database (sync with search-server URL/s if given, updating a few fitting entries on demand), lookup the nearest-neighbor of values' feedback (the label) (must not be linked elsewhere), and connect via read-from-Internet.
            - ⋯ In-extension [tabs](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/tabs/Tab): `.active, .audible, .mutedInfo.muted, .pinned, .status==='complete'`, `.index` (out of 32); `+new Date() - .lastAccessed` as time; [visible area](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/tabs/captureTab) as an image; and with an option checked and the `"tabs"` permission: maybe favicon from `.favIconUrl`, maybe reading out `.title` characters, maybe reading out `.url`. (Though, if we have bandwidth for that, then we might as well expose some proper programming language and/or rewriting system for direct use.)
        - ⋯ `.Transform`:
            - ✓ `Reward`, filling `0`s of 0th numbers of cells with the numeric result of calling a function unless it's `0` too.
                - ✓ By default, make Ctrl+Up/Ctrl+Down give +1/-1 reward.
            - ⋯ `RewardFeedback`, which waits 8 steps to get feedback then calls its function to get the reward. The most natural function is a discriminator between data & feedback, or a simple difference if lazy. (Really empathize with other AI models.)
                - ⋯ Allow inserting more cells, for discrimination between mechanisms.
                - ⋯ Collect a dataset with the sensor network, and replay it to train. Should be able to learn to predict at least 1 cell reasonably well, and from there it's just about getting better and scaling up, right?
            - ⋯ `Visualize`, with a list of `Sensor`s on which to call `.visualize({data, cellShape}, DOMelem)`, so that humans can match data to a familiar format and thus learn a new representation of it. Infer sensors by name (by turning names into byte-strings, and pre-constructing regexes from sensors' names), so that even old and remote data is visualizable.
                - ⋯ `Visualize.all()({data, cellShape})`, which returns a `<canvas>` on which green=1 black=0 red=-1 4×4 pixels are drawn, with empty pixels between reward/user/name and data.
            - ⋯ Shuffle cells.
            - ⋯ Add `-error…error` random numbers to `data`, if there is error.
            - ⋯ To save feedback of even non-noData cells, a transform that splits such cells into `noData` and `noFeedback` cells.
            - ⋯ Limiter: by-FPS, by-bandwidth.
            - ⋯ Try an echo-state network, and see whether that makes `Sound` better or worse.
                - ⋯ Also find a music GAN, and train an RNN-from-observations that maximizes the discriminator's score. (Procedural music.)
                - ⋯ Also try mangling feedback, and having keyboard and camera input and camera-with-some-SSL-NN. See whether we can get something even remotely passable to work. (And have the visualization UI that first collects feedback, trains a model on it when asked, and when ready, actually uses the model.)
            - ❌ Ask the user to rename cells using feedback. (Feedback is value-only, and besides, it's probably inferior to AI translation from meager user data to full feedback, which is already trivially achievable by observing the user, demanding actions, and a handler that defers to an AI model. BMI? Pfft, AI translation is superior.)
        - ⋯ `.Handler`:
            - TODO: So what do we do next?
                - If easiest-first:
                    - Keyboard, pointers, text, `indexedDB` files, WebRTC.
                - If most-important-first:
                    - WebRTC, text, pointers, keyboard, audio, files.
            - TODO: Text (and is text really any good if each word doesn't have infinite depth, and links are everywhere):
                - TODO: Design constraints:
                    - Position-invariance, of cells into which data is divided. This enables hotswappable and user-defined observations/actions, which is how humans [expect ](https://en.wikipedia.org/wiki/Process_(computing))[computers ](https://en.wikipedia.org/wiki/USB)[to operate ](https://en.wikipedia.org/wiki/Internet_of_things)[anyway.](https://en.wikipedia.org/wiki/Internet) In ML, [Transformers are dominant anyway.](https://arxiv.org/abs/1706.03762)
                    - -1…1 values, including the reward. Humans do not tolerate [overly-strong](https://www.reddit.com/r/NoStupidQuestions/comments/65o0gi/how_loud_is_a_nuclear_explosion_all_noise_is/) signals anyway. ML models [typically perform worse with unnormalized data.](https://en.wikipedia.org/wiki/Feature_scaling)
                    - That's all. A human can use it. AGI can use it.
            - ✓ No-feedback sound output (speakers).
                - ✓ IFFT, implemented manually because it's not in `AudioContext`, with upsampling of inputs.
                - ✓ Make it no-skips and no-huge-backlog. Make it reasonably-good UX, essentially.
                    - ⋯ Inspect the first and last numbers of packets: these most likely cause that clicking. Try to pick the wave phase (dependent on previous-packets-size) that minimizes that difference.
                    - ⋯ What we made is clearly not quite frequency-based, since adding/removing sensors really changes the sound, so fix it.
                    - ⋯ Fix the noticeable slowdown-then-gradual-speedup phenomenon, likely occuring because we mispredict ms-per-step and don't have a backlog. (It really takes the listener out of the experience.)
                    - ⋯ [Normalize perceived loudness.](https://en.wikipedia.org/wiki/Equal-loudness_contour)
                - ⋯ Be able to specify how many sound samples each value should occupy, for more detail and less bandwidth.
            - ❌ Sound input (microphone). Probably terrible, especially without an ML model to summarize it.
            - ⋯ Write to `indexedDB`, with visualization (`option()`?) allowing saving it all to a file.
            - ⋯ If extension is present, write to background page. (`chrome.runtime.sendMessage` seems to be exposed to pages for some reason, but only in Chrome. Elsewhere, have to communicate via DOM events with a content script that does the actual message-sending.)
            - ⋯ Write to Internet.
                - ⋯ Take on the remote cellShape and partSize.
                - ⋯ Preserve no-data-from-this-cell and no-feedback-to-this-cell arrays.
            - ⋯ Advertise this computer for search (sync with search-server URL/s if given), and when someone connects, write-to-Internet.
            - ⋯ Research AI translation, from human observations to rewardable actions. [Just-reward is most general task description](https://deepmind.com/research/publications/2021/Reward-is-Enough), and an endless complexity of ways to hopefully align with future rewards, broadly falling into prediction or uncertainty-maximization (getting more data to predict). The age-old question: can specifics be learned from the general description? [Probably](http://www.incompleteideas.net/IncIdeas/BitterLesson.html), but how to best do it? In the sensor network, each cell has its own reward so that the model can learn many tasks at once, and distill many AI models (and humans, and anything else) into one. Basically, for [a good prior](https://deepmind.com/research/publications/2021/Creating-Interactive-Agents-with-Imitation-Learning): should perform max-reward filling of non-`noFeedback` cells, while also predicting non-`noData` cells with a slightly-different cell-name, so that actions are as-data-does first (top-reward among human actions: [a quantilizer](https://intelligence.org/files/QuantilizersSaferAlternative.pdf)) and well-adapted second. (Easy to implement, hard to implement well. But nothing is created perfect, and all we can do is improve: the better the model, the more humans use it and trust it and the more well-pointed their observations and rewards are, which means "the better the model, the faster it improves", which means "exponential progress", which means that any level of performance is practically reachable, even AGI. We can start any time you want.)
                - (A personal AI model is basically soft mind uploading, realistically-gradual rather than instantly-perfect. The only real obstacle is that a typical PC doesn't have the compute to run some massive AI model full-time. But AI-training increases in efficiency rapidly, [2…10 ](https://openai.com/blog/ai-and-efficiency/)[times in ](https://venturebeat.com/2020/06/04/ark-invest-ai-training-costs-dropped-100-fold-between-2017-and-2019/)[a year](https://ark-invest.com/articles/analyst-research/ai-training/): what is today a [multi-million-dollar project](https://syncedreview.com/2020/04/30/ai21-labs-asks-how-much-does-it-cost-to-train-nlp-models/) may be doable on a personal computer in a decade. Phones and other devices that need a battery could connect; [Internet bandwidth is growing too, at 1.5× a year](https://www.nngroup.com/articles/law-of-bandwidth/).)
                - ⋯ Need concrete and relatively-compute-light tasks to strive for. Such as "in a game, learn to augment human play with non-human actions" or "predict near-term reward from facial expression".
        - ✓ `.UI()`.
            - ⋯ Allow multi-channel configuration, after we have a channel-sensor.
        - ❌ `.default()`, collecting non-`false` `.default`s. (UI makes it too easy to include everything you need.)

- ⋯ Compression. Try to share code with Rust if possible, via Wasm. (Possibly split this into another library/package, and provide the no-compression default here and a way to negotiate compression, to not bloat code too much.)

An attempt to make the playing field even for humans and AI, based on the best of the best considerations. An attempt to bring about fairness, even though nothing would let it happen. Straying from beaten paths only to make a better one that no one wants to travel, most likely. Only when humanity is swimming in compute can it be desirable. Toys in the meantime.

### Extension UI

The extension should be a control center that can manage a human's direct connection to their personal computer and Internet.

- ⋯ Infrastructure:
    - ⋯ Isolate the `chrome` and `browser` namespaces, of course. Snippets shouldn't have such power.
    - ⋯ When a tab sends a message that it needs a video stream, [give it](https://developer.chrome.com/docs/extensions/reference/tabCapture/#method-getMediaStreamId).
        - ⋯ Prompt the user if the page was not authorized. (It's only authorized if the extension injected the video-collecting script, with a nonce.)
    - ⋯ Inject a content script that listens to DOM events and sends those messages to the extension, and sends replies as DOM events.
    - ⋯ Inject a handler that defers to the extension with priority `Infinity`.
        - ⋯ If active, suppress other handlers with `onlyIfNoExtension()` → `true`.
    - ⋯ Allow extensions to enforce user-selected interfaces on the currently-active tab, disconnecting when the tab switches.
        - ⋯ Benchmark tab-switching.
    - ⋯ …Come up with something for in-extension DOM visualization of what it tells pages to do: `visualize({data, cellShape}, DOMelem)`.
    - ⋯ Benchmark throughput of `.defaults()`.
    - ⋯ Test that an infinite loop in extension's active snippets can be recovered from, in some way.

- ⋯ UI:
    - ⋯ Sliced-off corners for a sharper look, via `clip-path: polygon(0 5%, 5% 0, 95% 0, 100% 5%, 100% 95%, 95% 100%, 5% 100%, 0 95%);  clip-path: polygon(0 .4em, .4em 0, calc(100% - .4em) 0, 100% .4em, 100% calc(100% - .4em), calc(100% - .4em) 100%, .4em 100%, 0 calc(100% - .4em))`. (No box shadows like this, though. Unless SVG magic will help.)
    - ⋯ A collapsible hierarchy of what's possible, with left-borders with gradients (indicating depth with brightness, and vertical progress with a rainbow). With triangles to indicate collapsed-ness.
        - ⋯ A store of all snippets, put into that tree.
        - ⋯ The summary of each snippet: a checkbox of whether it's currently active, unchecked by default, preserved in storage too; followed by name parts.
            - ⋯ When a parent's activity changes, all child activity becomes parent's.
            - ⋯ When a child's activity changes, all parents' activity becomes undetermined (and turned off, if the parent is more than just a namespace).
            - ⋯ Color the name with data's L2 norm (blue), and feedback's L2 norm (red).
        - ⋯ Each snippet can define its settings and their types, and the UI elements will show up.
        - ⋯ Each snippet can have a collapsed visualization; when expanded, the element exists, and gets updated by the snippet's visualization code.
        - ⋯ Each snippet has Markdown documentation.
            - ⋯ Only 1 line is displayed by default, and more can be seen be uncollapsing it.
        - ⋯ Each snippet has a list of children. Drag-and-drop changes it.
        - ⋯ Each snippet has the full collapsed code. With syntax-highlighting of each code field, in case someone is actually interested.
            - ⋯ (If we copy Conceptual's editor and modify the parser, use a filter for drop-shadows, so that irregularly-shaped boxes don't look super ugly.)
            - ⋯ Ctrl+A and copying copies the JSON serialization. (Or a button copies the serialization.)
        - ⋯ Each snippet can be deleted, by drag-and-dropping to a trash icon.
            - ⋯ Have a queue of recently-deleted items, viewable by clicking on the trash icon, and make Ctrl+Z able to bring back the very-recently-deleted item.
        - ⋯ New snippets can be created, by clicking a + icon near the trash icon.
            - ⋯ A textarea, or a real editor that shares code with the collapsed-code viewer, for the full pastable serialization.
        - ⋯ The one default snippet documents this UI.
            - ⋯ And directs to a tutorial where more snippets can be picked up.
        - Not that useful with modern technology (not enough bandwidth to control these):
            - Ability to attach to the extension's page.
            - Ability to watch not just one tab, but many tabs (`requestAnimationFrame` suffers in background tabs too).

## Journey

The Sensor Network is a human-friendly environment for integrating general intelligence with computers.

Or will be. We will make sure of it, manually.

Via a series of Web-based tutorials (let's be real, no one's gonna install some native application just to go through some tutorials), for *calibration*.

- ⋯ No feedback, read only, basics:
    - ⋯ Keyboard: listen to someone else type, and type what they typed. (Learn an understanding that connects observations to actions.)
        - ⋯ 4 difficulties, incremented on 3 successive successes, decremented on 3 successive failures: "any 0..9 number", "any key", "EN words", "EN sentences".
    - ⋯ Video: A bullet-hell game, which exposes the image around the point-of-interest as sound. Train with the image rendered, and prove mastery without it.
- ⋯ With feedback, basics:
    - ⋯ Keyboard: type the displayed sentences, with each completed letter highlighted, and 3 mistakes in a row un-completing a letter. WPM is measured.
    - ⋯ Mouse+scroll: have a target somewhere on the vast vast page, its position shown on scrollbars; hovering over it gives reward and a reset; hovering at the viewport's outermost pixels gives negative reward. With 1+ scrolling-element levels.
        - ⋯ A bunch of buttons, able to be clicked with left/right clicks and drag-and-dropped around, and a textbox telling you what to do.
    - ⋯ Video: a `<canvas>`, slowly paintable with video feedback. (3D rendering with multiple angles, or audio composition, is probably too hard to learn.)