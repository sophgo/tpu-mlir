"""Group-Conv partitioning: guards, uneven groups and assigned view addresses."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

TOOL = sys.argv[1]
COUNT = 0


def tensor(shape, dtype='f16', address=None):
    encoding = f', {address} : i64' if address is not None else ''
    return 'tensor<' + 'x'.join([*(str(x) for x in shape), dtype]) + encoding + '>'


def run(name, *, groups=16, gic=64, goc=64, cores=4, batch=1,
        spatial=(519,), kernel=(128, 1), pads=(4, 0, 4, 0), stride=(1, 1),
        dilation=(1, 1), dtype='f16', bias=True, bias_dtype='f32',
        chip='cv84x6', dynamic=False, dynamic_weight=False, disabled=False,
        attrs='', filter_shape=None, encoded=False, expected=True,
        check_address=False, subgroup=False):
    global COUNT
    ic, oc = groups * gic, groups * goc
    hw = (*spatial, 1) if len(spatial) == 1 else spatial
    ohw = tuple((hw[i] + pads[i] + pads[i+2] - dilation[i]*(kernel[i]-1) - 1)//stride[i] + 1 for i in range(2))
    xs, ys = [batch, ic, *spatial], [batch, oc, *(ohw[:1] if len(spatial) == 1 else ohw)]
    ws = filter_shape or [1, oc, 1, math.ceil(gic / 16)*16*math.prod(kernel)]
    bs = [1, oc, 1, 1]
    address = 2199023255552 if encoded else None
    xt, yt, wt, bt = tensor(xs, dtype, address), tensor(ys, dtype, address), tensor(ws, dtype, address), tensor(bs, bias_dtype, address)
    arguments = f'%arg0: {xt} loc("x")'
    weight = f'%w = "top.Weight"() : () -> {wt} loc("w")'
    if dynamic_weight:
        arguments += f', %arg1: {wt} loc("w")'
        weight = f'%w = "top.Input"(%arg1) : ({wt}) -> {wt} loc("w")'
    bias_op = f'%b = "top.Weight"() : () -> {bt} loc("b")' if bias else ''
    bias_value, bias_type = ('%b', bt) if bias else ('%none', 'none')
    attributes = {'kernel_shape': str(list(kernel)), 'strides': str(list(stride)),
                  'pads': str(list(pads)), 'dilations': str(list(dilation)),
                  'group': f'{groups} : i64', 'with_bias': str(bias).lower(),
                  'multicore': 'true', 'coeff_merged': 'false',
                  'weight_is_coeff': '1 : i64', 'use_3ic_optimize': '0 : i64'}
    # Extra attributes override defaults so rejection cases remain valid IR.
    for entry in attrs.split(';'):
        if entry:
            k, v = entry.split('=', 1)
            attributes[k.strip()] = v.strip()
    attr_text = ', '.join(k + ' = ' + v for k, v in attributes.items())
    conv = f'%y = "tpu.Conv2D"(%x, %w, {bias_value}) {{{attr_text}}} : ({xt}, {wt}, {bias_type}) -> {yt} loc("conv")'
    if subgroup:
        conv = f'''%y = "tpu.CoreParallel"(%x, %w, {bias_value}) ({{
      %inner = "tpu.Conv2D"(%x, %w, {bias_value}) {{{attr_text}, do_core_parallel = false}} : ({xt}, {wt}, {bias_type}) -> {yt} loc("conv")
      "tpu.Yield"(%inner) : ({yt}) -> () loc("yield")
    }}) {{offset = 0 : i64, size = {cores} : i64}} : ({xt}, {wt}, {bias_type}) -> {yt} loc("existing_parallel")'''
    source = f'''module @{name} attributes {{module.chip = "{chip}", module.cores = {cores} : i64, module.mode = "{dtype.upper()}", module.platform = "ONNX", module.state = "TPU_DIVIDED", module.weight_file = "weights.npz"}} {{
  func.func @main({arguments}) -> {yt} attributes {{mode = #tpu<run_mode {'TPU_DYNAMIC' if dynamic else 'TPU_STATIC'}>}} {{
    %none = "top.None"() : () -> none loc("none")
    %x = "top.Input"(%arg0) : ({xt}) -> {xt} loc("x")
    {weight}
    {bias_op}
    {conv}
    return %y : {yt}
  }}
}}
'''
    with tempfile.TemporaryDirectory(prefix='group_conv_') as directory:
        root = Path(directory)
        # Actual tensor files exercise weight ownership and address assignment.
        np.savez(root/'weights.npz', w=np.zeros(ws, np.uint16), b=np.zeros(bs, np.float32))
        (root/'input.mlir').write_text(source)
        config = {'rewriter_rules': [{'pattern_name': 'GroupConvCoreParallel', 'params': {'use_pattern': 0}}] if disabled else []}
        (root/'config.json').write_text(json.dumps(config))
        env = dict(os.environ, TPU_DIALECT_REWRITER_CONFIG=str(root/'config.json'))
        def command(inp, out, *options):
            result = subprocess.run([TOOL, inp, *options, '-o', out], cwd=root, env=env, text=True, capture_output=True)
            assert result.returncode == 0, (name, result.stdout, result.stderr)
            return (root/out).read_text()
        text = command('input.mlir', 'out.mlir', '--core-parallel')
        if not expected:
            assert '_group_core' not in text and text.count('"tpu.Conv2D"') == 1, (name, text)
        else:
            n = min(cores, groups)
            sizes = [groups//n + (i < groups % n) for i in range(n)]
            lines = [l for l in text.splitlines() if '"tpu.Conv2D"' in l]
            assert len(lines) == n and text.count('"tpu.CoreParallel"') == 1, (name, text)
            for line, g in zip(lines, sizes):
                assert f'group = {g} : i64' in line, (name, line)
                assert 'multicore = false' in line and 'do_core_parallel = false' in line, (name, line)
                assert tensor([1, gic*g, *spatial], dtype) in line, (name, line)
                assert tensor([1, goc*g, *ws[2:]], dtype) in line, (name, line)
            assert text.count('"tpu.CoreSplit"') == (3 if bias else 2), (name, text)
            assert text.count('"tpu.CoreJoin"') == 1, (name, text)
            again = command('out.mlir', 'again.mlir', '--core-parallel')
            assert text == again, (name, 'not idempotent')
            if check_address:
                addressed = command('out.mlir', 'addressed.mlir', '--address-assign')
                for line in addressed.splitlines():
                    if not any(f'"tpu.{op}"' in line for op in ['CoreSplit', 'CoreJoin']):
                        continue
                    types = re.findall(r'tensor<([0-9x]+)x(f16|bf16|f32), ([0-9]+) : i64>', line)
                    assert len(types) == n+1, (name, line)
                    def info(item):
                        shape, dtype, address = item
                        size = math.prod(map(int, shape.split('x'))) * (4 if dtype == 'f32' else 2)
                        return int(address), size
                    ranges = list(map(info, types))
                    parent, children = (ranges[0], ranges[1:]) if 'CoreSplit' in line else (ranges[-1], ranges[:-1])
                    cursor = parent[0]
                    assert cursor > 0
                    for address, size in children:
                        assert address == cursor, (name, line, cursor)
                        cursor += size
                    assert cursor == parent[0]+parent[1], (name, line)
    COUNT += 1


# The actual pre-padded Diarizen positional Conv, and independent layouts.
run('diarizen', check_address=True)
run('uneven_groups', groups=7, gic=3, goc=5, spatial=(13, 11), kernel=(3,3), pads=(1,1,1,1), check_address=True)
run('three_groups', groups=3, gic=16, goc=32, spatial=(65,), kernel=(5,1), pads=(2,0,2,0), check_address=True)
run('no_bias', groups=2, gic=3, goc=5, spatial=(31,), kernel=(3,1), pads=(1,0,1,0), bias=False, check_address=True)
run('bf16', dtype='bf16', groups=6, gic=16, goc=32, spatial=(31,29), kernel=(3,3), pads=(2,2,2,2), stride=(2,2), dilation=(2,2), check_address=True)
run('two_cores', cores=2)
run('shared_layout', groups=5, gic=17, goc=7, spatial=(19,17), kernel=(3,3), pads=(1,1,1,1), check_address=True)
for name, kwargs in [
    ('one_core', {'cores':1}), ('normal_conv', {'groups':1}),
    ('depthwise', {'gic':1,'goc':1}), ('batch_two', {'batch':2}),
    ('dynamic_subnet', {'dynamic':True}), ('dynamic_weight', {'dynamic_weight':True}),
    ('float32', {'dtype':'f32'}), ('fp16_bias', {'bias_dtype':'f16'}),
    ('wrong_chip', {'chip':'bm1688'}), ('disabled', {'disabled':True}),
    ('already_addressed', {'encoded':True}), ('already_parallel', {'subgroup':True}),
    ('merged_coeff', {'attrs':'coeff_merged=true'}),
    ('raw_weight', {'attrs':'weight_is_coeff=0 : i64'}),
    ('three_ic', {'attrs':'use_3ic_optimize=1 : i64'}),
    ('winograd', {'attrs':'use_winograd=2 : i64'}),
    ('rotated', {'attrs':'do_kernel_rotate=true'}),
    ('backend_multicore_disabled', {'attrs':'multicore=false'}),
    ('op_disabled', {'attrs':'do_core_parallel=false'}),
    ('weight_bits', {'attrs':'weight_bits=16 : i64'}),
    ('large_stride', {'stride':(16,1)}), ('large_dilation', {'dilation':(16,1),'kernel':(3,1)}),
    ('wrong_packed_row', {'filter_shape':[1,1024,1,8193]}),
    ('wrong_packed_oc', {'filter_shape':[1,512,1,16384]}),
]:
    run(name, expected=False, **kwargs)
print(f'Group convolution: {COUNT} regression cases passed')
