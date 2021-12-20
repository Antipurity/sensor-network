Know what needs to be done, and do it.

No deleting, only adding (at the end) and keeping track of progress.

- ✓ Track completion with Unicode characters: ⋯ to-do, ✓ complete, ❌ canceled.

- ✓ Have directories `/rs`, `/js-lib`, `/js-ext`. Init the Git repo.
    - ✓ In `/js-lib`, `npm init`.

---

## Platform

The Sensor Network aims to expose all real-time data accessible to a machine in a modular, dynamically-reconfigurable, fashion.

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
    - ⋯ The trait `Accumulator`, which changes a frame's data (post-sending pre-handling):
        - ⋯ For efficiency, accumulators are not IPC (which would have needed at least one IPC copy per accumulator per message), but have to be created in the same process as the main `Handler`.
        - ⋯ `new(channel: Option<String>)`.
        - ⋯ `on_data(&self, data: Vec<f32>, cell_shape:&[u32])->Future<(Vec<f32>, Extra)>` returns a value [eventually](https://crates.io/crates/futures).
        - ⋯ `on_feedback(&self, feedback: Vec<f32>, cell_shape:&[u32], extra: Extra)->Future<Vec<f32>>`.
        - ⋯ `priority(&self) -> f64`: all accumulators are called in a chain, highest-priority-first. `0` by default.
    - ⋯ The trait `Handler`, which gives feedback to sensors:
        - ⋯ `new(channel: Option<String>)`.
        - ⋯ `on_data(&self, data: Vec<f32>, error: Vec<f32>, cellShape: &[f32], feedback: Option<Vec<f32>>) -> Future<Option<Vec<f32>>>`: give feedback to data, or observe another handler's data+feedback.
            - ⋯ On each step, turn observations into corrections:
                - ⋯ A periodic loop of fulfilling corrections to smooth network latency, released in the order that they were requested, sending back data if not handled when needed. Try to match the latency of observation-correction things.
                    - ⋯ Benchmark the latency deviation, mean & stdev. Both when messages are sent evenly, and in bursts of 2/4/8/16/32/64/128.
                    - ⋯ Give per-number max error along with data.
            - ⋯ On each sent message, wait a bit before handling messages, to make inputs more coherent. (And, benchmark the coherence, as the % of senders accumulated, avg per step.)
            - ⋯ Only send feedback to those senders that have `.no_feedback()->false`.
        - ⋯ `no_feedback(&self) -> bool`, `true` by default.
        - ⋯ `priority(&self) -> f64`, `0` by default: only the one max-priority handler with `no_feedback=false` will give feedback, the rest will simply observe.
        - ⋯ `data_size(&self) -> u32`, `64` by default.
        - ⋯ `name_size(&self) -> u32`, `64` by default.
        - ⋯ `name_part_size(&self) -> u32`, `16` by default.
    - ⋯ Sensors first request cell shapes from handlers, then for each unique shape, allocate the actual positions. On step, limit f32 numbers to `-1`…`1`, put them in places, fill in reward (`0`) & name, compress if specified, then send to handlers.
    - ⋯ Test that all data is indeed accumulated correctly, via bogus senders/accumulators.
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

- ⋯ Accumulators:
    - ⋯ Shuffle cells, to make models/brains that are not fully order-independent become such.
    - ⋯ Reward sender, which replaces `0`s in all cells' first number with the reward. To specify per-user reward in one place. (The idealized job: you give it your situation, it makes your number go up.)
        - ⋯ Configurable reward, via a closure that's called each frame. By default, F11 is `-1` reward, F12 is `+1` reward, otherwise `0`.
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

- ⋯ If we can somehow find TBs of storage for hosting, create a server that stores all incoming connections to file, and samples from that file when requested, or reads by stream ID.

Isn't implementing this such a joyful learning opportunity?

### JS API

Intelligence can do anything. But how to support the utter formlessness of generality of intelligence? By supporting every form at once. With reasonable defaults.

- ⋯ One library that puts everything into the global `sn`, populated as variables:
    - ⋯ The basics:
        - ✓ Have a name-hasher, from name and available-parts and part-size to Float32Array, possibly written-to in-place.
            - ✓ To not waste space, numbers fill up their cells (and all no-string cells) with fractally-folded versions of themselves; each fold turns the line `{ 0: -1, 1: 1 }` into `{ 0: -1, .5: 1, 1: -1 }`, so, `x → 1-2*abs(x)`. (The listener can then make out details more easily.)
            - ✓ Data should also do that if unused, with feedback adding up the details too so that the reported feedback is nudged appropriately. No holes, only more detail.
        - ✓ Decide whether we handle `NaN | -1…1` or `-1…1`. Verdict: `-1…1`, because what even are holes in sensors.
        - ⋯ `.Sensor`, used as `new Sensor({ name:['keyboard', 'a'], values:1, async onValues(s) { console.log((await s.send([1]))[0]) } })`:
            - ⋯ `.constructor({ name, values=0, channel=null, onValues=null })`.
                - ❌ The options object can be modified after construction. (No, users should just destroy and recreate, emphasizing that this should be rare.)
                - ⋯ `onValues(sensor) -> Promise<void>`: send data & receive feedback via `sensor.send(…)`, as often as possible, possibly async.
                    - ⋯ Send at most 16 at once. Measure the average time between the main handler's feedbacks (even `noFeedback` empty feedback is feedback for this), and match it as exactly as we can.
            - ⋯ `.pause()`, `.resume()`
            - ⋯ `.send(values: Float32Array|null, error: Float32Array|null, reward=0, noFeedback=false) -> Promise<Float32Array|null>`: send data, receive feedback, once. (Reward is not fed back.)
                - ⋯ "Allocate" the name into one array by creating a closure that writes, and re-use it, copying values into proper places.
            - ⋯ For convenience, if [`FinalizationRegistry`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/FinalizationRegistry) is present, stop when the sender is no longer needed (else, set `onValues` to `null`).
        - ⋯ `.Accumulator`:
            - ⋯ `.constructor({ channel=null, priority=0, onValues=null, onFeedback=null })`.
                - ❌ The options object can be modified after construction.
                - ⋯ Accumulators run highest-priority-first.
                - ⋯ `onValues(data: Float32Array, error: Float32Array|null, cellShape: [reward=1, user, name, data]) -> Promise<extra>`: prepares to modify data in-place, possibly async. The sum of numbers in `cellShape` always divides `data.length`.
                - ⋯ `onFeedback(feedback: Float32Array, cellShape: [reward=1, user, name, data], extra) -> Promise<void>`: modifies data's feedback. Maybe you want privacy, or maybe not all input sources are equally easy to activate.
            - ⋯ `.pause()`, `.resume()`
            - ⋯ For convenience, if [`FinalizationRegistry`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/FinalizationRegistry) is present, stop when the accumulator is no longer needed.
        - ⋯ `.Handler`:
            - ⋯ `.constructor({ channel=null, priority=0, noFeedback=false, onValues=null, dataSize=64, nameSize=64, namePartSize=16 })`.
                - ❌ The options object can be modified after construction.
                - ⋯ `dataSize` is how many data numbers each cell can hold, `nameSize` is how many numbers the cell is identified with, split into `namePartSize`-sized blocks. First name then data; the first name part is used for the reward (always the first number) and the user ID, the rest are taken up be senders' string hashes and numbers.
                - ⋯ `onValues(data: Float32Array, error: Float32Array|null, cellShape: [reward=1, user, name, data], writeFeedback: bool, feedback: null|Float32Array)->Promise<void>`: receive data, and modify it in-place to send feedback (modify synchronously when the promise returns, to prevent data races).
            - ⋯ `.pause()`, `.resume()`
            - ⋯ On each sent message, wait a bit before handling messages, to make inputs more coherent. (And, benchmark the coherence, as the % of senders accumulated, avg per step.)
        - ⋯ A function that runs all unit tests, `.tests()`, which traverses `sn` (not through prototypes) and calls every `runTests` method.
            - ⋯ `test.html`, which imports the `main.js` module and runs `sn.tests()`.
        - ⋯ A function that runs all benchmarks, `.bench()`, once, and accumulates results in localStorage or indexedDB or file, keyed by path (erasing previous results if the source-code hash doesn't match): traverses `sn` (not through prototypes) and calls every `runBench` method, which return promises of numbers or arrays of numbers.
        - ⋯ `.docs()`, which traverses `sn` and accumulates all `docs` strings into a Markdown string.
            - ⋯ Parents should become sections, into which their children belong.
            - ⋯ Make a table of contents at the top, with refs to the top at every section heading.
            - ⋯ Make `npm doc` import the library and call this and write its result to `docs/DOCS.md`.
        - ⋯ Ability to de/serialize senders/accumulators/handlers, so that users can pick up power-ups at the press of a button. (Top-level is an array, where the last item is the actual object, and the rest are the function dependencies. JSON is fine.)
            - ⋯ Some way to specify which parts are done in-extension and which are in-content-script and which are in-page-JS, here.
    - ⋯ Reasonable defaults, decided by the user and not the handler, in separate `import`ed files:
        - ⋯ `.Sensor`:
            - ⋯ Actual sensors, with "observe the hardware" (no feedback) and "visualize effects in-page" (feedback, with data's error being `1`) modes, and UI visualization where possible:
                - ⋯ Keyboard. Variants:
                    - ⋯ Put all keys in one strip, in lexicographically-first order.
                    - ⋯ Spatially-grouped QWERTY key layout.
                    - ⋯ Every key has its own cell.
                - ⋯ Mouse.
                    - ⋯ Expose `{x:…,y:…}` on the object itself, 0…1.
                    - ⋯ The first mouse can control the actual mouse, though only with [non-trusted events](https://developer.mozilla.org/en-US/docs/Web/API/Event/isTrusted). The rest only update `{x:…,y:…}`, and can be used in videos.
                        - ⋯ Main mouse movement fires mouse/pointer move/over/out events.
                        - ⋯ To display hover-states, use ancient magic: go through all CSS rules in all stylesheets and in every new stylesheet, and duplicate those with `:hover` to use a class, which main-mouse-movement sets.
                    - ⋯ (May need an actual visualization elem, because pseudo-moving the mouse in the extension is not visualization.)
                - ⋯ Scroll, exposing not just at top-level but in hierarchy levels: current X/Y and max X/Y scroll position; non-existent ones are 0s. Occupy only 1 cell.
                - ⋯ Video+audio of a tab.
                    - ⋯ Allow initializing from a media stream, in case there's some full-screen canvas that we can use and avoid asking the user. (Else, do our best to check that the selection is actually the tab's stream.)
                        - ⋯ Auto-detect such a scenario.
                        - ⋯ If the extension is present, request a stream ID from it.
                    - ⋯ Video.
                        - ⋯ Types:
                            - ⋯ Full stream, resized to a target resolution, to see the whole picture.
                            - ⋯ Around-mouse rect.
                            - ⋯ Around-mouse fovea.
                            - ⋯ Around-mouse progressively coarser grids, each zoomed out 2×. So if starting at 8×8 cells, only need 7 more levels to go to 1024×1024, which with default cell-sizes and 44.1kHz sound-output is at most 43FPS for grayscale or 14FPS for full-color. (May act the same as a zooming data augmentation.)
                        - ⋯ Coalesce chunks spatially wherever possible, with x/y coords of the center in the name.
                        - ⋯ Expose `.points()->Array` and `.sendPointsRGB(data: Uint8Array)=>Promise<feedback>`, in case someone really wants to render only what is strictly necessary, and not have to prod the user for access.
                        - ⋯ Allow passing the mouse object as an option, which is any object that has `{x:…,y:…}` with 0…1 client coordinates.
                        - ⋯ In-page per-pixel feedback to DOM elements, through something like `sn.video.on(elem, imageData=>void)` and/or a hidden canvas. (Differentiable rendering boys, even though JS doesn't have libraries for that.)
                        - ⋯ Internally, for efficiency, render images to a WebGL texture, and download data from there.
                            - ⋯ Make each result a separate buffer, downloaded after some delay (say, 4 results max).
                            - ⋯ For efficiency in certain cases, expose methods that receive data from a WebGL texture & context at 1+ coordinates.
                    - ⋯ Audio.
                        - ⋯ Mono.
                        - ⋯ Stereo.
                        - ⋯ Create an audio context that reads PCM data from the media stream.
                            - ⋯ In the constructor's options, a volume multiplier, `1` by default.
                        - ⋯ Expose data and feedback as [audio](https://developer.mozilla.org/en-US/docs/Web/API/MediaStream) [streams](https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamAudioDestinationNode): `.soundData()=>dataStream` and `.soundFeedback()=>feedbackStream`.
                            - ⋯ For more efficiency, allow initializing from an audio context by attaching all our nodes to it, in case all sound is played through that: `.onAudioContext(ctx, outputNode)`.
                        - ⋯ In its visualization, two `<audio>` elements, for data and feedback.
                            - ⋯ In visualization, report data/feedback volumes with color, possibly with `box-shadow`.
                - ⋯ No in-page feedback, in Chrome (Firefox doesn't seem to care as much about direct hardware access):
                    - ⋯ Raw bytes of [HID](https://web.dev/hid/), remapped to -1..1.
                    - ⋯ Mobile device [sensor readings](https://developer.mozilla.org/en-US/docs/Web/API/Sensor_APIs).
            - ⋯ System resources, if exposed: `m=performance.memory`, `m.usedJSHeapSize / m.totalJSHeapSize * 2 - 1`.
            - ⋯ Time, as sines exponentially increasing frequency, with 100FPS as the most-frequent-wave-period.
            - ⋯ Read from Internet, with WebRTC, RabbitMQ preferable.
                - ⋯ Each data packet (1+ cells) references its meta-data (cell shape) by ID; when/if meta-data changes, it's re-sent, and the other side can request it if it doesn't know it (such as when the packet got lost). Though, cell shape shouldn't ever change, or there's a big problem in ML models.
                - ⋯ Discourage disengagements: on user disconnect, hold its last block (`0` everywhere except the user in the name) with `-1` reward, for as many frames as specified (`8` by default). Dying is bad.
                - ⋯ Benchmark throughput, over localhost, with the default data (a file, preferably always the same one).
            - ⋯ Read from file.
            - ⋯ In extension, read from tabs.
        - ⋯ `.Accumulator`:
            - ⋯ Shuffle blocks.
            - ⋯ Reward, filling `0`s of 0th numbers of cells with the numeric result of calling a function unless it's `0` too.
                - ⋯ By default, make F11/F12 give +1/-1 reward.
            - ⋯ An alternative string-hashing strategy, namely, "ask the user" (display the string somewhere for at least a few seconds, send 0s as the name, and record suggestions; the most distant one from all names in the database wins, and the mapping from string-hash to actual-data is preserved, so that even file recordings can be replayed comfortably). May need UI integration, though.
                - ⋯ Nearest-neighbor lookup by names, to extract more from less.
            - ⋯ Once we have something that sounds recognizable, try an echo-state network, and see whether that makes it better or worse.
                - ⋯ Also find a music GAN, and train an RNN-from-observations that maximizes the discriminator's score. (Procedural music.)
                - ⋯ Also try mangling feedback, and having keyboard and camera input and camera-with-some-SSL-NN. See whether we can get something even remotely passable to work. (And have the visualization UI that first collects feedback, trains a model on it when asked, and when ready, actually uses the model.)
                - (This would have been so much easier if we just had a Neuralink device or an equivalent.)
            - ⋯ Extension-oriented, mainly-for-visualization things:
                - ⋯ Name calibration, where each sensor's name can participate, and on uncalibrated `0`-data feedback collect suggestions of what it is, as long as it's opened in this UI.
                - ⋯ Ability to hold a cell's button to only route its feedback, since it looks like we'll really need to scrounge for human-to-machine bandwidth, and labels won't be enough.
                    - ⋯ And the ability to use in-page keybindings. (Might even be *usable*.)
        - ⋯ `.Handler`:
            - ⋯ No-feedback sound output (speakers).
            - ⋯ Sound input (microphone). Probably terrible.
            - ⋯ Write to Internet.
                - ⋯ Make sure that at least `error=1` ("no data") is preserved.
            - ⋯ Write to file.
                - ⋯ Replace `error=1` ("no data") with feedback, leave the rest as-is.
            - ⋯ If extension is present, write to background page. (`chrome.runtime.sendMessage` seems to be exposed to pages for some reason, but only in Chrome. Elsewhere, have to communicate via DOM events with a content script that does the actual message-sending.)
        - ⋯ `.defaults()`.

- ⋯ Compression. Try to share code with Rust if possible, via Wasm. (Possibly split this into another library/package, and provide the no-compression default here and a way to negotiate compression, to not bloat code too much.)

Doesn't sound too difficult. I've done a lot of this before.

### Extension UI

The extension should be a control center that can manage a human's direct connection to their personal computer and Internet.

- ⋯ Infrastructure:
    - ⋯ Isolate the `chrome` and `browser` namespaces, of course. Snippets shouldn't have such power.
    - ⋯ When a tab sends a message that it needs a video stream, [give it](https://developer.chrome.com/docs/extensions/reference/tabCapture/#method-getMediaStreamId).
        - ⋯ Prompt the user if the page was not authorized. (It's only authorized if the extension injected the video-collecting script, with a nonce.)
    - ⋯ Inject a content script that listens to DOM events and sends those messages to the extension, and sends replies as DOM events.
    - ⋯ Inject a handler that defers to the extension with priority `Infinity`.
    - ⋯ Allow extensions to enforce user-selected interfaces on the currently-active tab, disconnecting when the tab switches.
        - ⋯ Benchmark tab-switching.
    - ⋯ …Come up with something for in-extension DOM visualization of what it tells pages to do.
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