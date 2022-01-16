The sensor network is not born perfect, but improves instead.

`test.html` here runs unit tests and benchmarks. Run it to run them.

(To run it, you may need to [serve `/js-lib` ](https://www.npmjs.com/package/http-server)[locally](https://developer.mozilla.org/en-US/docs/Learn/Common_questions/set_up_a_local_testing_server), or in Firefox, in `about:config`, set `security.fileuri.strict_origin_policy` to `false` to allow JS modules to actually access what they [`import`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/import).)

Code should always pass all tests, and should improve benchmark numbers over time.

---

## Benchmarks

- Handler: one sensor that always sends `960` `1`s, and one handler that always responds with `-1`s. The simplest possible scenario, dominated by array-copying and garbage-collection time.
- Internet: sends synthetic float32 values over a WebRTC connection, to the same page, without compression.
- Sound: synthetic data, sent to `sn.Handler.Sound`. Ideally, no gaps, and low latency.
- Video: sends `canvas.captureStream()` to `sn.Sensor.Video`. Note that without `.captureStream()`, the performance is much better, so we limit ourselves to the worst case in case it helps.

## Current benchmark results

Reporting the mean, worst-performers first.

- Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
    - Video, 1024×1024:
        - Firefox:
            - <details><summary><code>50.86</code> FPS (↑) </summary><textarea readonly>{"Video: simultaneous steps":[1,1,1],"Video: step processed data, values":[4608,4608,4608],"Video: throughput, bytes/s":[2376579.39,937378.6,235272.81],"Video: resolution":[512,1024,2048]}</textarea></details>
        - Chrome:
            - <details><summary><code>7880</code> FPS (↑), allocations <code>9.20</code> MiB/s (↓) </summary><textarea readonly>{"Video: simultaneous steps":[1,1,1],"Video: step processed data, values":[4608,4608,4608],"Video: throughput, bytes/s":[162464744.52,145251546.15,1467812.53],"Video: allocations, bytes/s":[6525176.67,9644433.23,94388.96],"Video: resolution":[512,1024,2048]}</textarea></details>
    - Internet, 960 values per step:
        - Firefox:
            - <details><summary>throughput <code>3.89</code> MiB/s (↑), latency <code>1.75</code> ms (↓) </summary><textarea readonly>{"Internet: sent, bytes/step":[3870,7711,11552,15393,19235,23076,26917,30758,34595.36,38435.33],"Internet: simultaneous steps":[1.04,1,1,1,1,1,1,1,1.5,1.5],"Internet: step processed data, values":[960,1919.94,2879.93,3839.9,4799.88,5759.87,6719.85,7679.84,8641.16,9604.26],"Internet: processing latency, ms":[1.75,3.64,4.07,5.49,6.98,7.54,8.49,9.16,15.75,17.15],"Internet: feedback-is-correct fraction":[1,1,1,1,1,1,1,1,1,1],"Internet: throughput, bytes/s":[4074268.07,3828582.44,5170143.58,5120428.2,5101268.43,5677539.93,5889478.41,6207616.43,8732303.03,8763771.09]}</textarea></details>
        - Chrome:
            - <details><summary>throughput <code>9.40</code> MiB/s (↑), latency <code>1.63</code> ms (↓), allocations <code>6.45</code> MiB/s (↓) </summary><textarea readonly>{"Internet: sent, bytes/step":[3870,7711,11552,15393,19235,23076,26917,30758.03,34600,38437.78],"Internet: simultaneous steps":[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],"Internet: step processed data, values":[960,1920,2880,3840,4800,5759.94,6720,7680,8640,9602.42],"Internet: throughput, bytes/s":[9851620.94,11635536.37,13536605.28,12090658.19,13314531.76,13474484.89,14618824.84,3955811.76,14986484.49,15419542.38],"Internet: allocations, bytes/s":[6764217.18,5154298.96,6291477.1,6559355.97,5583979.48,6091787.7,6161667.04,6780100.37,6334486.65,6171583.45],"Internet: processing latency, ms":[1.63,2.45,2.91,4.36,4.89,5.89,6.12,340.02,8.33,9.76],"Internet: feedback-is-correct fraction":[1,1,1,1,1,1,1,1,1,1]}</textarea></details>
    - Sound, 1056 values per step:
        - Firefox:
            - <details><summary>gaps <code>0</code>% (↓), latency <code>70</code>ms (↓) </summary><textarea readonly>{"Sound: gap in sound, bool":[0,0,0.18],"Sound: latency, s":[0.07,0.07,0.06],"Sound: simultaneous steps":[1,1,1],"Sound: step processed data, values":[96,1056,2016],"Sound: throughput, bytes/s":[52006.78,51983.2,51211.8]}</textarea></details>
        - Chrome:
            - <details><summary>gaps <code>0</code>% (↓), latency <code>40</code>ms (↓), allocations <code>.74</code> MiB/s (↓) </summary><textarea readonly>{"Sound: gap in sound, bool":[0,0,0.13],"Sound: latency, s":[0.04,0.04,0.03],"Sound: simultaneous steps":[1,1,1],"Sound: step processed data, values":[96,1056,2016],"Sound: throughput, bytes/s":[51895.81,52001.03,51640.49],"Sound: allocations, bytes/s":[771781.15,937707.86,1133323.09]}</textarea></details>
    - Handler, 960 values per step:
        - Firefox:
            - <details><summary>throughput <code>264.66</code> MiB/s (↑) (initially <code>7</code> MiB/s) </summary><textarea readonly>{"Handler: simultaneous steps":[1,1,1,1,1,1,1,1,1,1],"Handler: step processed data, values":[960,1920,2880,3840,4800,5760,6720,7680,8640,9600],"Handler: throughput, bytes/s":[277519158.23,285459319.71,320735137.8,331741368.82,330409559.1,328647529.99,335366055.1,386600551.6,395020788.47,388029963.79]}</textarea></details>
        - Chrome:
            - <details><summary>throughput <code>309.66</code> MiB/s (↑), allocations <code>3.45</code> MiB/s (↓) </summary><textarea readonly>{"Handler: simultaneous steps":[1,1,1,1,1,1,1,1,1,1],"Handler: step processed data, values":[960,1920,2880,3840,4800,5760,6720,7680,8640,9600],"Handler: throughput, bytes/s":[324699589.36,463661121.35,550296511.35,602405022.57,628256184.59,670384090.42,680194222.65,708813153.62,717057157.67,737763050.37],"Handler: allocations, bytes/s":[3614393.12,6185168.18,5107712.34,5383724.62,4686222.08,2979254.18,4729827.47,4222324.5,4272075.71,3873158.4]}</textarea></details>

## Lessons

- Chrome has [`performance.memory`](https://developer.mozilla.org/en-US/docs/Web/API/Performance/memory).

- `E:{_allocF32()}` bad, for performance. `function allocF32()` good.

- `Promise`s bad, for allocations. Callbacks good.

- `for-of` loops allocate a bit, but not nearly as much as promises.