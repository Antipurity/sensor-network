`test.html` here runs unit tests and benchmarks. Run it to run them.

(To run it, you may need to [serve `/js-lib` ](https://www.npmjs.com/package/http-server)[locally](https://developer.mozilla.org/en-US/docs/Learn/Common_questions/set_up_a_local_testing_server), or in Firefox, in `about:config`, set `security.fileuri.strict_origin_policy` to `false` to allow JS modules to actually access what they [`import`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/import).)

Code should always pass all tests, and should improve benchmark numbers over time.

---

## Current benchmark results

Reporting the mean, usually for about `1000` transferred numbers per second.

- Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
    - Handler:
        - Firefox:
            - <details><summary>throughput <code>36.05</code> MiB/s (↑)</summary><textarea readonly>{"Handler: simultaneous steps":[2.98,2.95,2.94,2.93,2.92,2.91,2.89,2.88,2.88,2.87],"Handler: step processed data, values":[960,1920,2880,3840,4800,5760,6720,7680,8640,9600],"Handler: throughput, bytes/s":[37804289.09,31230502.09,32799110.39,36769437.49,42946813.78,44033431.93,45794305.34,48985210.89,51437525.88,52752784.34]}</textarea></details>
        - Chrome:
            - <details><summary>throughput <code>17.22</code> MiB/s (↑), allocations <code>3.48</code> MiB/s (↓)</summary><textarea readonly>{"Handler: simultaneous steps":[2.92,2.89,2.88,2.85,2.82,2.79,2.77,2.77,2.74,2.73],"Handler: step processed data, values":[960,1920,2880,3840,4800,5760,6720,7680,8640,9600],"Handler: throughput, bytes/s":[18057431.79,28048615.64,37257007.09,38770360.81,41685338.16,41774095.81,44441804.94,51476089.45,50008423.26,53523131.22],"Handler: allocations, bytes/s":[3653182.48,5906774.12,5735500.59,4846444.83,6723823.29,6241338.24,7445240.61,5524328.97,6553852.8,6461670.44]}</textarea></details>

## Benchmarks

- Handler: one sensor that always sends `960` `1`s, and one handler that always responds with `-1`s. The simplest possible scenario, dominated by array-copying time.