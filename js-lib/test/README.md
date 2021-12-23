`test.html` here runs unit tests and benchmarks. Run it to run them.

(To run it, you may need to [serve `/js-lib` ](https://www.npmjs.com/package/http-server)[locally](https://developer.mozilla.org/en-US/docs/Learn/Common_questions/set_up_a_local_testing_server), or in Firefox, in `about:config`, set `security.fileuri.strict_origin_policy` to `false` to allow JS modules to actually access what they [`import`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/import).)

Code should always pass all tests, and should improve benchmark numbers over time.

---

## Benchmarks

- Handler: one sensor that always sends `960` `1`s, and one handler that always responds with `-1`s. The simplest possible scenario, dominated by array-copying time.

## Current benchmark results

Reporting the mean, usually for about `1000` transferred numbers per second.

- Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
    - Handler:
        - Firefox:
            - <details><summary>throughput <code>41.68</code> MiB/s (↑)</summary><textarea readonly>{"Handler: simultaneous steps":[2.98,2.98,2.97,2.96,2.96,2.96,2.95,2.95,2.94,2.94],"Handler: step processed data, values":[960,1920,2880,3840,4800,5760,6720,7680,8640,9600],"Handler: throughput, bytes/s":[43702489.64,63391796.95,74223281.83,75702844.35,89013130.95,98188888.85,102252543.63,105714524.03,107841315.8,110008642.75]}</textarea></details>
        - Chrome:
            - <details><summary>throughput <code>83.90</code> MiB/s (↑), allocations <code>1.13</code> MiB/s (↓)</summary><textarea readonly>{"Handler: simultaneous steps":[2.98,2.98,2.97,2.96,2.96,2.95,2.94,2.94,2.93,2.93],"Handler: step processed data, values":[960,1920,2880,3840,4800,5760,6720,7680,8640,9600],"Handler: throughput, bytes/s":[87973845.69,127264548.26,148152427.41,164654286.74,174877317.78,184397151.64,189017669.54,199339354.82,202965898.95,211741332.77],"Handler: allocations, bytes/s":[1181544.2,2146027.95,3558120.15,2875348.56,3241276.41,3480596.46,2750931.85,1855689.04,966435.22,844000.91]}</textarea></details>

## Lessons

- `E:{_allocF32()}` bad. `function allocF32()` good.