"""
Gives a convenient command-line interface for running `sn` envs.

Exposes `envs = run(sn, argv = sys.argv, random_names = False)`: when run, imports all the envs that were specified as command-line args; when run without those args, displays a help-message (detailing all available envs) and aborts execution.

Details:
- `script.py` must have the `env/` folder with it. When running, `python script.py test` results in importing `env/test.py`.
- To make name-collisions impossible, `sn.fork` will be used to override the last name-part with path-to-env. Prefer shorter data-names.
- Envs can be passed args, such as `simple_board(N=8)`, or left as-is as in `simple_board`. The help-message will outline what args can be specified.
- Env-files are modules with `class Env: def __call__(self, sn): ...`, instances of which are put into `sn.sensors`.
    - To allow args, define `def __init__(self, a=1, b=False, c=-0.7)`. All args must have default values.
    - Can define `def metric(self): {**metrics}`, for optionally `log`ging some numbers related to performance. Not handled here, but callers of `run` can iterate over values in the `envs` dict, for example:

```py
from log import log
n = 0
for name, env in envs.items():
    if hasattr(env, 'metric'): # The list of envs for which this is true shouldn't ever change.
        log(n, False, torch, **{name+'.'+k: v for k,v in env.metric().items()})
        n += 1
```
"""



import ast
import sys
import importlib
def run(sn, argv = sys.argv, random_names = False):
    def name_modifier(ctx_name):
        def modify_name(name):
            """Remove inter-env collisions by adding the group ID to the end of their names. (`sn.Float` and `sn.Int` prepend a name-part, so `name` contains one less.)"""
            i = len(sn.cell_shape) - 3 # (8,8,8,8,64) has 3 usable name-parts, so `i=2`.
            #   (After this, only 2 name-parts would be usable by others.)
            res = list(name)
            while len(res) <= i: res.append(None)
            assert not isinstance(res[i], tuple), "The group-id shouldn't be a tuple (to make our lives easier)"
            res[i] = ctx_name if res[i] is None else (ctx_name + '.' + res[i])
            return tuple(res)
        return modify_name
    if len(argv) < 2:
        # Print the help message. Has the side-effect of importing all envs.
        #   (No point in not importing, since just using the env would import it anyway.)
        import os
        at = os.path.join(os.path.dirname(os.path.abspath(argv[0])), 'env')
        print("""Also pass in the environments to act in, such as:""")
        for dirpath, dirnames, filenames in os.walk(at):
            if dirpath[:len(at)] == at:
                path = dirpath[len(at):].split(os.sep)[1:]
                for filename in filenames:
                    if filename[-3:] == '.py':
                        import_path = '.'.join([*path, filename[:-3]])
                        print()
                        print('→ ', import_path)
                        mod = importlib.import_module('env.' + import_path)
                        if hasattr(mod.Env, '__init__'):
                            f = mod.Env.__init__
                            if hasattr(f, '__code__') and f.__defaults__:
                                assert len(f.__code__.co_varnames) == 1 + len(f.__defaults__), "All args must have defaults"
                                args = ','.join(k+'='+repr(v) for k,v in zip(f.__code__.co_varnames[1:], f.__defaults__))
                                print(' ·', import_path + '(' + args + ')')
                        print(' ·  ', mod.Env.__doc__.strip().partition('\n')[0].strip())
        sys.exit()
    else:
        encountered = {}
        def prepare_env(path: str):
            argstart = path.find('(')
            path, args = (path[:argstart], path[argstart:]) if argstart >= 0 else (path, '()')
            assert args[0] == '(' and args[-1] == ')'
            pairs = [pair.partition('=') for pair in args[1:-1].split(',')]
            args = { k.trim(): ast.literal_eval(v.trim()) for k,_,v in pairs if k }
            mod = importlib.import_module('env.' + path)
            assert hasattr(mod, 'Env')
            sensor = mod.Env(**args)
            if not random_names:
                encountered[path] = path if path not in encountered else encountered[path]+'0'
            else:
                import string
                import random
                vocabulary = string.ascii_uppercase + string.digits
                encountered[path] = ''.join(random.choice(vocabulary) for _ in range(16))
            fork = sn.fork(name_modifier(encountered[path]))
            def sensor_with_name(_):
                return sensor(fork)
            sn.sensors.add(sensor_with_name)
            return sensor
        return { e: prepare_env(e) for e in argv[1:] }