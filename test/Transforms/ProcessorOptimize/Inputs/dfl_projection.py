"""DFL layout/projection rewrite: numeric equivalence and semantic guards."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

TOOL = sys.argv[1]
COUNT = 0


def tensor(shape):
    return 'tensor<' + 'x'.join([*(str(d) for d in shape), 'f32']) + '>'


def run(name, *, shape=(1, 4, 16, 5040), chip='CV84X6', mode='F16',
        cores=4, dynamic=False, high_precision=False, beta=1., log=False,
        axis=1, bias=False, relu=False, shared=None, weights=None,
        disabled=False, numeric=False, match=False):
    global COUNT
    n, d, k, s = shape
    x, p, y = tensor(shape), tensor((n, k, d, s)), tensor((n, 1, d, s))
    w = tensor((1, k, 1, 1))
    outputs = ['%result'] + ([f'%{shared}'] if shared else [])
    outtypes = [y] + ([p] if shared else [])
    biasval = '%bias' if bias else '%none'
    biastype = tensor((1,)) if bias else 'none'
    ops = [
        '%none = "top.None"() : () -> none loc("none")',
        f'%input = "top.Input"(%arg) : ({x}) -> {x} loc("input")',
        f'%layout = "top.Permute"(%input) {{order = [0, 2, 1, 3]}} : ({x}) -> {p} loc("layout")',
        f'%prob = "top.Softmax"(%layout) {{axis = {axis} : si32, beta = {beta} : f64, log = {str(log).lower()}}} : ({p}) -> {p} loc("prob")',
        f'%weight = "top.Weight"() : () -> {w} loc("weight")',
    ]
    if bias:
        ops.append(f'%bias = "top.Weight"() : () -> {biastype} loc("bias")')
    ops.append(f'%result = "top.Conv"(%prob, %weight, {biasval}) {{kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1], group = 1 : i64, do_relu = {str(relu).lower()}}} : ({p}, {w}, {biastype}) -> {y} loc("result")')
    source = f'''module @{name} attributes {{module.chip = "{chip.lower()}", module.cores = {cores} : i64, module.mode = "{mode}", module.platform = "ONNX", module.state = "TOP_F32", module.top_run_mode = "{'DYNAMIC' if dynamic else 'STATIC'}", module.high_precision = {str(high_precision).lower()}, module.weight_file = "weights.npz"}} {{
  func.func @main(%arg: {x} loc("input_arg")) -> ({', '.join(outtypes)}) {{
    {chr(10).join(ops)}
    return {', '.join(outputs)} : {', '.join(outtypes)}
  }}
}}
'''
    with tempfile.TemporaryDirectory(prefix='dfl_projection_') as tmp:
        root = Path(tmp)
        (root/'input.mlir').write_text(source)
        values = np.arange(k, dtype=np.float32) if weights is None else weights
        np.savez(root/'weights.npz', weight=values.reshape(1,k,1,1), bias=np.array([0.5],np.float32))
        env = dict(os.environ)
        env.pop('TPU_DIALECT_REWRITER_CONFIG', None)
        if disabled:
            (root/'disabled.json').write_text(json.dumps({'rewriter_rules':[{'pattern_name':'DFLProjectionToMatMul','params':{'use_pattern':0}}]}))
            env['TPU_DIALECT_REWRITER_CONFIG'] = str(root/'disabled.json')
        cmd = [TOOL,'input.mlir','--processor-top-optimize','-o','out.mlir']
        proc = subprocess.run(cmd,cwd=root,env=env,text=True,capture_output=True)
        assert proc.returncode == 0, (name,proc.stdout,proc.stderr)
        text = (root/'out.mlir').read_text()
        assert text.count('"top.MatMul"') == int(match), (name,text)
        assert text.count('"top.Conv"') == int(not match), (name,text)
        if match:
            assert 'order = [0, 1, 3, 2]' in text and 'axis = 3 : si32' in text, (name,text)
            assert f'tensor<{n*d*s}x{k}xf32>' in text and f'tensor<{k}x1xf32>' in text, (name,text)
            assert text.count('"top.Permute"') == 1 and text.count('"top.Softmax"') == 1, (name,text)
            proc = subprocess.run([TOOL,'out.mlir','--processor-top-optimize','-o','again.mlir'],cwd=root,env=env,text=True,capture_output=True)
            assert proc.returncode == 0, (name,proc.stdout,proc.stderr)
            assert (root/'again.mlir').read_text().count('"top.MatMul"') == 1, name
        if numeric:
            from tools.model_runner import mlir_inference
            previous = Path.cwd()
            os.chdir(root)
            try:
                data = np.random.default_rng(100).normal(0,4,shape).astype(np.float32)
                before = mlir_inference({'input':data},'input.mlir',dump_all=False)['result'].copy()
                after = mlir_inference({'input':data},'out.mlir',dump_all=False)['result'].copy()
                np.testing.assert_allclose(after,before,rtol=2e-6,atol=1e-5)
                probs = np.exp(data-data.max(axis=2,keepdims=True))
                probs /= probs.sum(axis=2,keepdims=True)
                ref = (probs*np.arange(k,dtype=np.float32).reshape(1,1,k,1)).sum(axis=2).reshape(n,1,d,s)
                np.testing.assert_allclose(after,ref,rtol=2e-6,atol=1e-5)
            finally:
                os.chdir(previous)
    COUNT += 1
    print('PASS',name,flush=True)


def main():
    run('teacher_yl11_real_shape',match=True,numeric=True)
    run('different_bins_positions_batch',shape=(2,3,8,37),match=True,numeric=True)
    run('larger_bins',shape=(1,4,32,240),match=True,numeric=True)
    run('one_position',shape=(1,4,16,1),match=True,numeric=True)
    run('bm1684x_unchanged',chip='BM1684X')
    run('bm1688_unchanged',chip='BM1688')
    run('bf16_unchanged',mode='BF16')
    run('int8_unchanged',mode='INT8')
    run('f32_unchanged',mode='F32')
    run('two_cores_unchanged',cores=2)
    run('dynamic_unchanged',dynamic=True)
    run('high_precision_unchanged',high_precision=True)
    run('wrong_softmax_axis',axis=3)
    run('log_softmax',log=True)
    run('scaled_softmax',beta=.5)
    run('bias_unchanged',bias=True)
    run('relu_unchanged',relu=True)
    run('shared_softmax',shared='prob')
    run('shared_permute',shared='layout')
    run('learned_projection',weights=np.arange(16,dtype=np.float32)+.25)
    run('nan_projection',weights=np.full(16,np.nan,np.float32))
    run('pattern_disabled',disabled=True)
    print(f'{COUNT} DFL projection cases passed')


if __name__ == '__main__':
    main()
