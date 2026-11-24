"""Check the two channel halves, their order, folded constants and target guards."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def tensor(*dims):
    return 'tensor<' + 'x'.join(map(str, dims)) + 'xf32>'


def run_case(name, *, expect=False, sizes=(2, 3, 5), bias=True, reverse=False,
             numeric=False, chip='CV84X6', cores=4, mode='F16', dynamic=False,
             high_precision=False, axis=1, beta=1., log=False, relu=False,
             shared=None, duplicate=False, strided=False, overflow=False,
             config=1, order=(0, 2, 1, 3)):
    n, ic, c = sizes
    x, w, b = tensor(n, ic, 1, 1), tensor(2*c, ic, 1, 1), tensor(2*c)
    full, pair, perm, half = (tensor(n, 2*c, 1, 1), tensor(n, 1, 2, c),
                              tensor(n, 2, 1, c), tensor(n, c, 1, 1))
    bias_arg, bias_type = ('%b', b) if bias else ('%none', 'none')
    ops = [
        '%none = "top.None"() : () -> none loc("none")',
        f'%input = "top.Input"(%arg) : ({x}) -> {x} loc("input")',
        f'%w = "top.Weight"() : () -> {w} loc("w")',
        f'%b = "top.Weight"() : () -> {b} loc("b")',
        f'%conv = "top.Conv"(%input, %w, {bias_arg}) {{kernel_shape=[1,1], strides=[1,1], pads=[0,0,0,0], group=1:i64, do_relu={str(relu).lower()}}} : ({x}, {w}, {bias_type}) -> {full} loc("conv")',
        f'%pair = "top.Reshape"(%conv) {{shape=[{n},1,2,{c}]}} : ({full}) -> {pair} loc("pair")',
        f'%perm = "top.Permute"(%pair) {{order={list(order)}}} : ({pair}) -> {perm} loc("perm")',
        f'%sm = "top.Softmax"(%perm) {{axis={axis}:si32, beta={beta}:f64, log={str(log).lower()}}} : ({perm}) -> {perm} loc("sm")',
        f'%flat = "top.Reshape"(%sm) {{shape=[{n},{2*c},1,1]}} : ({perm}) -> {full} loc("flat")',
    ]
    for i in ((1, 0) if reverse else (0, 1)):
        start = 0 if duplicate or (strided and i == 1) else c*i
        step = 2 if strided and i == 1 else 1
        end = start + step*c
        ops.append(f'%p{i} = "top.Slice"(%flat,%none,%none,%none) {{axes=[], hasparamConvert_axes=[1], offset=[0,{start},0,0], steps=[1,{step},1,1], ends=[{n},{end},1,1]}} : ({full}, none, none, none) -> {half} loc("p{i}")')
    extra_types = {'conv': full, 'pair': pair, 'perm': perm, 'sm': perm, 'flat': full, 'w': w}
    outputs, types = ['%p0', '%p1'], [half, half]
    if shared:
        outputs.append('%' + shared)
        types.append(extra_types[shared])
    ops.append(f"return {', '.join(outputs)} : {', '.join(types)}")
    ir = f"""module @test attributes {{module.chip="{chip.lower()}", module.cores={cores}:i64, module.mode="{mode}", module.platform="ONNX", module.state="TOP_F32", module.top_run_mode="{'DYNAMIC' if dynamic else 'STATIC'}", module.high_precision={str(high_precision).lower()}, module.weight_file="weights.npz"}} {{
      func.func @main(%arg: {x} loc("arg")) -> ({', '.join(types)}) {{
        {chr(10).join(ops)}
      }}
    }}"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(888)
        filt = rng.normal(size=(2*c, ic, 1, 1)).astype('f4')
        biases = rng.normal(size=2*c).astype('f4')
        if overflow:
            filt[0, 0] = 40000
            filt[c, 0] = -40000
        np.savez(root/'weights.npz', w=filt, b=biases)
        (root/'input.mlir').write_text(ir)
        env = os.environ.copy()
        env.pop('TPU_DIALECT_REWRITER_CONFIG', None)
        if config is not None:
            path = root/'rules.json'
            path.write_text(json.dumps({'rewriter_rules': [
                {'pattern_name': 'FoldRadix2Softmax', 'params': {'use_pattern': config}}]}))
            env['TPU_DIALECT_REWRITER_CONFIG'] = str(path)
        def opt(src, dest):
            cmd = [sys.argv[1], src, '--processor-top-optimize', '-o', dest]
            subprocess.run(cmd, cwd=root, env=env, check=True, capture_output=True)
            return (root/dest).read_text()
        text = opt('input.mlir', 'out.mlir')
        assert text.count('"top.Sigmoid"') == 2*expect, (name, text)
        assert text.count('"top.Softmax"') == int(not expect), (name, text)
        if expect:
            assert text.count('"top.Conv"') == 1 and '"top.Slice"' not in text
            assert opt('out.mlir', 'again.mlir').count('"top.Sigmoid"') == 2
        if numeric:
            from tools.model_runner import mlir_inference
            previous = Path.cwd()
            os.chdir(root)
            try:
                data = rng.normal(size=(n, ic, 1, 1)).astype('f4')
                logits = data.reshape(n, ic) @ filt.reshape(2*c, ic).T
                if bias:
                    logits += biases
                logits = logits.reshape(n, 2, c)
                ref = np.exp(logits - logits.max(axis=1, keepdims=True))
                ref /= ref.sum(axis=1, keepdims=True)
                for path in ('input.mlir', 'out.mlir'):
                    result = mlir_inference({'input': data}, path, dump_all=False)
                    for i in (0, 1):
                        np.testing.assert_allclose(result[f'p{i}'].reshape(n, c),
                                                   ref[:, i], atol=3e-6, rtol=3e-6)
            finally:
                os.chdir(previous)
    print('PASS', name, flush=True)


if __name__ == '__main__':
    run_case('two_halves', expect=True, numeric=True)
    run_case('reverse_use_order', expect=True, reverse=True, numeric=True)
    run_case('real_stage3', sizes=(8, 128, 256), expect=True, numeric=True)
    run_case('no_bias', sizes=(1, 7, 9), bias=False, expect=True, numeric=True)
    run_case('shared_filter', shared='w', expect=True)
    run_case('default_off', config=None)
    run_case('disabled', config=0)
    run_case('bm1684x', chip='BM1684X')
    run_case('two_cores', cores=2)
    run_case('int8', mode='INT8')
    run_case('bf16', mode='BF16')
    run_case('float32', mode='F32')
    run_case('dynamic', dynamic=True)
    run_case('high_precision', high_precision=True)
    run_case('wrong_axis', axis=3)
    run_case('wrong_permutation', sizes=(1, 3, 5), order=(1, 2, 0, 3))
    run_case('scaled_softmax', beta=0.5)
    run_case('log_softmax', log=True)
    run_case('relu_projection', relu=True)
    run_case('duplicate_halves', duplicate=True)
    run_case('strided_slice', strided=True)
    run_case('fp16_fold_overflow', overflow=True)
    for value in ('conv', 'pair', 'perm', 'sm', 'flat'):
        run_case('shared_' + value, shared=value)
