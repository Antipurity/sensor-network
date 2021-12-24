`test.html` here runs unit tests and benchmarks. Run it to run them.

(To run it, you may need to [serve `/js-lib` ](https://www.npmjs.com/package/http-server)[locally](https://developer.mozilla.org/en-US/docs/Learn/Common_questions/set_up_a_local_testing_server), or in Firefox, in `about:config`, set `security.fileuri.strict_origin_policy` to `false` to allow JS modules to actually access what they [`import`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/import).)

Code should always pass all tests, and should improve benchmark numbers over time.

---

## Benchmarks

- Handler: one sensor that always sends `960` `1`s, and one handler that always responds with `-1`s. The simplest possible scenario, dominated by array-copying and garbage-collection time.

## Current benchmark results

Reporting the mean, usually for about `1000` transferred numbers per packet.

- Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
    - Handler:
        - Firefox:
            - <details><summary>throughput <code>149.47</code> MiB/s (↑)</summary><textarea readonly>{"Handler: simultaneous steps":[1,1,1,1,1,1,1,1,1,1],"Handler: step processed data, values":[960,1920,2880,3840,4800,5760,6720,7680,8640,9600],"Handler: throughput, bytes/s":[156732776.19,224178555.02,219678606.78,232630245.52,247700313.94,260806358.86,267804259.4,268968967.16,224425665.5,236281398.63]}</textarea></details>
        - Chrome:
            - <details><summary>throughput <code>269.53</code> MiB/s (↑), allocations <code> 2.23</code> MiB/s (↓)</summary><textarea readonly>{"Handler: simultaneous steps":[1,1,1,1,1,1,1,1,1,1],"Handler: step processed data, values":[960,1920,2880,3840,4800,5760,6720,7680,8640,9600],"Handler: throughput, bytes/s":[282620133.46,399673983.06,475995719.95,523101943.34,547517998.98,573855195.11,592481823.9,612028460.51,620438791.99,627211231.18],"Handler: allocations, bytes/s":[2334897.29,5254187.94,4995715.23,5121319.35,2406621.44,3609019.36,4607187.94,3952371.38,2925589.37,2616792.3]}</textarea></details>

## Lessons

- `E:{_allocF32()}` bad, for performance. `function allocF32()` good.

- `Promise`s bad, for allocations. Callbacks good.

- `for-of` loops allocate a bit, but not nearly as much as promises.