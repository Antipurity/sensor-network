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
            - <details><summary>throughput <code>148.53</code> MiB/s (↑)</summary><textarea readonly>{"Handler: simultaneous steps":[1,1,1,1,1,1,1,1,1,1],"Handler: step processed data, values":[960,1920,2880,3840,4800,5760,6720,7680,8640,9600],"Handler: throughput, bytes/s":[155746845.59,207823644.67,203790433.77,246371608.24,260854304.92,270054227.75,265606367.58,286011449.4,290056239.88,293305333.44]}</textarea></details>
        - Chrome:
            - <details><summary>throughput <code>252.56</code> MiB/s (↑), allocations <code> 3.25</code> MiB/s (↓)</summary><textarea readonly>{"Handler: simultaneous steps":[1,1,1,1,1,1,1,1,1,1],"Handler: step processed data, values":[960,1920,2880,3840,4800,5760,6720,7680,8640,9600],"Handler: throughput, bytes/s":[264831014.82,381572379.67,455253033.98,501104131.18,524175730.05,560954490.83,586383352.87,594249972.29,610052237.9,620215557.81],"Handler: allocations, bytes/s":[3407715.58,4648851.37,3786856.92,4641209.19,2491384.5,2933324.58,4734019.96,4618951.49,4116209.01,3050580.04]}</textarea></details>

## Lessons

- `E:{_allocF32()}` bad, for performance. `function allocF32()` good.

- `Promise`s bad, for allocations. Callbacks good.

- `for-of` loops allocate a bit, but not nearly as much as promises.