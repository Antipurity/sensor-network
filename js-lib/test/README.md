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
            - <details><summary>throughput <code>38.46</code> MiB/s (↑)</summary><textarea readonly>{"Handler: simultaneous steps":[2.98,2.97,2.97,2.96,2.96,2.95,2.94,2.94,2.93,2.93],"Handler: step processed data, values":[960,1920,2880,3840,4800,5760,6720,7680,8640,9600],"Handler: throughput, bytes/s":[40328886.36,51852526.83,66242457.94,75453420.72,81744097.66,89515398.16,87762512.68,98246072.54,94845157.51,97280910.01]}</textarea></details>
        - Chrome:
            - <details><summary>throughput <code>73.23</code> MiB/s (↑), allocations <code>5.04</code> MiB/s (↓)</summary><textarea readonly>{"Handler: simultaneous steps":[2.98,2.97,2.96,2.96,2.94,2.94,2.93,2.92,2.91,2.9],"Handler: step processed data, values":[960,1920,2880,3840,4800,5760,6720,7680,8640,9600],"Handler: throughput, bytes/s":[76784953.99,102231131.04,120946993.07,136811731.02,132228426.69,142528712.04,148185381.25,154876841.38,150812759.28,151921406.17],"Handler: allocations, bytes/s":[5284541.73,10316711.66,10744620.67,7812427.02,6818487.17,9045936.45,6146906.3,4279721.65,7144546.52,7684282.42]}</textarea></details>

## Lessons

- `E:{_allocF32()}` bad. `function allocF32()` good.