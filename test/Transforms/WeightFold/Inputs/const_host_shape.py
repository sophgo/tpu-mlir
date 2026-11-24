"""Regression coverage for constant host-shape folding and ownership boundaries."""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOL = sys.argv[1]

def tensor(shape, dtype):
    return 'tensor<' + 'x'.join([*(str(x) for x in shape), dtype]) + '>'

def run_case(name, weights, operations, outputs, arguments='', disabled=False):
    with tempfile.TemporaryDirectory(prefix='const_host_shape_') as directory:
        root = Path(directory)
        np.savez(root / 'weights.npz', **weights)
        types = ', '.join(t for _, t in outputs)
        values = ', '.join(v for v, _ in outputs)
        source = f'''module @{name} attributes {{module.chip = "cv84x6", module.cores = 1 : i64, module.mode = "F16", module.platform = "ONNX", module.state = "TPU_LOWERED", module.top_run_mode = "DYNAMIC", module.weight_file = "weights.npz"}} {{
  func.func @main({arguments}) -> ({types}) {{
    {operations}
    return {values} : {types}
  }}
}}
'''
        (root / 'input.mlir').write_text(source)
        option = '--weight-fold=fold-const-shape=false' if disabled else '--weight-fold'
        command = [TOOL, 'input.mlir', option, '-o', 'out.mlir']
        result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0, (name, result.stdout, result.stderr)
        result = (root / 'out.mlir').read_text()
        weight_file = re.search(r'module.weight_file = "([^"]+)"', result)[1]
        arrays = {}
        if (root / weight_file).exists():
            with np.load(root / weight_file) as data:
                arrays = {k: data[k].copy() for k in data.files}
        return result, arrays

def chain(name, dtype='f32', outtype='i32', shape=(2,)):
    tin, tout = tensor(shape, dtype), tensor(shape, outtype)
    return f'''%w_{name} = "top.Weight"() : () -> {tin} loc("{name}")
    %h_{name} = "tpu.Device2Host"(%w_{name}) : ({tin}) -> {tin} loc("host_{name}")
    %c_{name} = "tpu.ShapeCast"(%h_{name}) : ({tin}) -> {tout} loc("cast_{name}")'''

def count(result, op):
    return result.count('"tpu.' + op + '"')

# Equal constants under different names become one host producer. The negative
# reshape sentinel is preserved exactly, as is the newly serialized INT32 data.
weights = {'a': np.array([-1, 512], np.float32), 'b': np.array([-1, 512], np.float32)}
ops = chain('a') + '\n' + chain('b')
outputs = [('%c_a', tensor((2,), 'i32')), ('%c_b', tensor((2,), 'i32'))]
r, data = run_case('fold_and_share', weights, ops, outputs)
assert count(r, 'Device2Host') == 1 and count(r, 'ShapeCast') == 0, r
assert any(a.dtype == np.int32 and np.array_equal(a, [-1, 512]) for a in data.values()), data
r, _ = run_case('disabled', weights, ops, outputs, disabled=True)
assert count(r, 'Device2Host') == 2 and count(r, 'ShapeCast') == 2, r

# A shared FP32 host value remains available to its unconverted consumer.
r, data = run_case('shared_float_consumer', {'a': weights['a']}, chain('a'),
                   [('%c_a', tensor((2,), 'i32')), ('%h_a', tensor((2,), 'f32'))])
assert count(r, 'Device2Host') == 2 and count(r, 'ShapeCast') == 0, r
assert any(a.dtype == np.float32 and np.array_equal(a, [-1, 512]) for a in data.values())

# Dynamic values must continue to use the runtime cast.
ops = '''%h = "tpu.Device2Host"(%x) : (tensor<2xf32>) -> tensor<2xf32> loc("host_dynamic")
    %c = "tpu.ShapeCast"(%h) : (tensor<2xf32>) -> tensor<2xi32> loc("cast_dynamic")'''
r, _ = run_case('runtime_input', {}, ops, [('%c', 'tensor<2xi32>')],
                 '%x: tensor<2xf32> loc("x")')
assert count(r, 'Device2Host') == 1 and count(r, 'ShapeCast') == 1, r

# The optimization must not apply C++ casts to nonintegral/nonfinite values or
# to 2^31 (whose FP32 representation can hide an INT32_MAX range-check bug).
for name, values in [('fraction', [0.5, 1]), ('nan', [np.nan, 1]),
                     ('infinity', [np.inf, 1]), ('overflow', [2**31, 1])]:
    r, _ = run_case(name, {'a': np.array(values, np.float32)}, chain('a'),
                     [('%c_a', 'tensor<2xi32>')])
    assert count(r, 'ShapeCast') == 1, (name, r)

# Deduplication compares raw integer bytes, not lossy float conversions.
weights = {'a': np.array([2147483647], np.int32),
           'b': np.array([2147483646], np.int32),
           'c': np.array([2147483647], np.int32)}
ops = '\n'.join(f'''%w_{n} = "top.Weight"() : () -> tensor<1xi32> loc("{n}")
    %h_{n} = "tpu.Device2Host"(%w_{n}) : (tensor<1xi32>) -> tensor<1xi32> loc("h_{n}")''' for n in weights)
r, _ = run_case('exact_integer_keys', weights, ops,
                [(f'%h_{n}', 'tensor<1xi32>') for n in weights])
assert count(r, 'Device2Host') == 2, r

# Equal bytes with different tensor shapes or element types remain separate.
weights = {'a': np.array([1, 2], np.int32), 'b': np.array([[1, 2]], np.int32),
           'c': np.array([1, 2], np.float32)}
ops = []
outputs = []
for n, shape, dtype in [('a', (2,), 'i32'), ('b', (1, 2), 'i32'), ('c', (2,), 'f32')]:
    t = tensor(shape, dtype)
    ops.append(f'''%w_{n} = "top.Weight"() : () -> {t} loc("{n}")
    %h_{n} = "tpu.Device2Host"(%w_{n}) : ({t}) -> {t} loc("h_{n}")''')
    outputs.append((f'%h_{n}', t))
r, _ = run_case('distinct_types', weights, '\n'.join(ops), outputs)
assert count(r, 'Device2Host') == 3, r

# Nested control-flow regions are intentionally left untouched. In particular,
# no host buffer may be hoisted from one conditional path into another.
ops = f'''%result = "tpu.If"(%cond) ({{
      {chain('a')}
      "tpu.Yield"(%c_a) : (tensor<2xi32>) -> () loc("yield_a")
    }}, {{
      {chain('b')}
      "tpu.Yield"(%c_b) : (tensor<2xi32>) -> () loc("yield_b")
    }}) : (tensor<1xi1>) -> tensor<2xi32> loc("conditional")'''
r, _ = run_case('conditional_regions', {'a': np.array([4, 128], np.float32),
                                       'b': np.array([4, 128], np.float32)},
                 ops, [('%result', 'tensor<2xi32>')], '%cond: tensor<1xi1> loc("cond")')
assert count(r, 'Device2Host') == 2 and count(r, 'ShapeCast') == 2, r
print('Constant host-shape folding: 11 regression cases passed')
