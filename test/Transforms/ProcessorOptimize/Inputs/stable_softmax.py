"""Stable-softmax attention: real pruned shapes, ownership and semantic guards."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

TOOL = sys.argv[1]
CASES = 0


def tensor(shape, dtype='f16'):
    return 'tensor<' + 'x'.join([*(str(x) for x in shape), dtype]) + '>'


class Graph:
    def __init__(self, dtype='f16'):
        self.dtype = dtype
        self.types = {'%none': 'none'}
        self.args = []
        self.ops = ['%none = "top.None"() : () -> none loc("none")']

    def arg(self, name, shape):
        self.types['%' + name] = tensor(shape, self.dtype)
        self.args.append(f'%arg_{name}: {self.types["%" + name]} loc("arg_{name}")')
        self.ops.append(f'%{name} = "top.Input"(%arg_{name}) : ({self.types["%" + name]}) -> {self.types["%" + name]} loc("{name}")')
        return '%' + name

    def op(self, name, kind, operands, shape, attrs=''):
        self.types['%' + name] = tensor(shape, self.dtype)
        self.ops.append(f'%{name} = "tpu.{kind}"({", ".join(operands)}) '
                        f'{{{attrs}}} : ({", ".join(self.types[x] for x in operands)}) '
                        f'-> {self.types["%" + name]} loc("{name}")')
        return '%' + name

    def source(self, name, outputs, chip):
        types = ', '.join(self.types[x] for x in outputs)
        return f'''module @{name} attributes {{module.chip = "{chip}", module.cores = 4 : i64, module.mode = "F16", module.platform = "ONNX", module.state = "TPU_LOWERED", module.weight_file = "unused.npz"}} {{
  func.func @main({', '.join(self.args)}) -> ({types}) {{
    {chr(10).join(self.ops)}
    return {', '.join(outputs)} : {types}
  }}
}}
'''


def stable(g, scores, shape, *, mode='ReduceMax', axis=3, sm_axis=3,
           log=False, beta=1., subattrs='', keepdims=True):
    reduced = list(shape)
    if keepdims:
        reduced[axis] = 1
    else:
        del reduced[axis]
    m = g.op('row_max', 'Reduce', [scores, '%none', '%none'], reduced,
             f'axes = [{axis}], keepdims = {str(keepdims).lower()}, mode = "{mode}"')
    s = g.op('centered', 'Sub', [scores, m], shape, subattrs)
    p = g.op('probabilities', 'Softmax', [s] + ['%none'] * 5, shape,
             f'axis = {sm_axis} : si32, beta = {beta} : f64, log = {str(log).lower()}')
    return p


def attention(*, heads=5, dim=64, batch=1, mq=399, mk=399, dtype='f16',
              scale=True, shared=None, bias=False, relu=False, **kwargs):
    g = Graph(dtype)
    qshape, kshape = [batch, mq, heads, dim], [batch, mk, heads, dim]
    q, k, v = [g.arg(n, s) for n, s in [('q', qshape), ('k', kshape), ('v', kshape)]]
    q = g.op('q_transposed', 'Permute', [q, '%none'], [batch, heads, mq, dim], 'order = [0, 2, 1, 3]')
    k = g.op('k_transposed', 'Permute', [k, '%none'], [batch, heads, dim, mk], 'order = [0, 2, 3, 1]')
    v = g.op('v_transposed', 'Permute', [v, '%none'], [batch, heads, mk, dim], 'order = [0, 2, 1, 3]')
    if scale:
        q = g.op('q_scaled', 'MulConst', [q], [batch, heads, mq, dim], 'const_val = 0.125 : f64')
    b = g.arg('bias', [1, 1, 1, mk]) if bias else '%none'
    shape = [batch, heads, mq, mk]
    s = g.op('scores', 'MatMul', [q, k, b, '%none', '%none'], shape,
             f'do_relu = {str(relu).lower()}')
    p = stable(g, s, shape, **kwargs)
    c = g.op('context', 'MatMul', [p, v] + ['%none'] * 3, [batch, heads, mq, dim])
    t = g.op('output_transposed', 'Permute', [c, '%none'], qshape, 'order = [0, 2, 1, 3]')
    o = g.op('result', 'Reshape', [t], [batch, mq, heads * dim])
    outputs = [o]
    if shared:
        outputs.append('%' + shared)
    return g, outputs


def permutation(*, order=(0, 2, 1, 3), inverse=None, shared=None,
                dtype='f16', shape=(2, 17, 5, 19), **kwargs):
    g = Graph(dtype)
    x = g.arg('x', shape)
    ps = [shape[i] for i in order]
    p = g.op('layout', 'Permute', [x, '%none'], ps, f'order = {list(order)}')
    s = stable(g, p, ps, **kwargs)
    inverse = inverse or tuple(order.index(i) for i in range(4))
    y = g.op('result', 'Permute', [s, '%none'], [ps[i] for i in inverse], f'order = {list(inverse)}')
    outputs = [y]
    if shared:
        outputs.append('%' + shared)
    return g, outputs


def count(text, op):
    return text.count(f'"tpu.{op}"')


def run(name, graph, *, fused=None, permutes=None, disabled=(), chip='cv84x6'):
    global CASES
    g, outputs = graph
    with tempfile.TemporaryDirectory(prefix='stable_softmax_') as directory:
        root = Path(directory)
        source = g.source(name, outputs, chip)
        (root / 'input.mlir').write_text(source)
        config = {'rewriter_rules': [{'pattern_name': p, 'params': {'use_pattern': 0}} for p in disabled]}
        (root / 'config.json').write_text(json.dumps(config))
        env = dict(os.environ, TPU_DIALECT_REWRITER_CONFIG=str(root / 'config.json'))
        result = subprocess.run([TOOL, 'input.mlir', '--processor-tpu-optimize', '-o', 'out.mlir'],
                                cwd=root, env=env, text=True, capture_output=True)
        assert result.returncode == 0, (name, result.stdout, result.stderr)
        text = (root / 'out.mlir').read_text()
        if fused is not None:
            assert count(text, 'FAttention') == int(fused), (name, text)
            if fused:
                assert not any(count(text, x) for x in ['MatMul', 'Reduce', 'Sub', 'Softmax', 'Permute']), (name, text)
                assert 'scale = 1.000000e+00' in text, (name, text)
                if '%q_scaled' in source:
                    assert count(text, 'MulConst') == 1, (name, text)
                    line = next(l for l in text.splitlines() if '"tpu.MulConst"' in l)
                    assert g.types['%q'] in line and '1.250000e-01' in line, (name, line)
        if permutes is not None:
            assert count(text, 'Permute') == permutes, (name, text)
            assert count(text, 'Reduce') == count(text, 'Sub') == count(text, 'Softmax') == 1, (name, text)
            if permutes == 0:
                assert re.search(r'"tpu.Sub".*-> ' + re.escape(g.types['%x']), text), (name, text)
                assert re.search(r'"tpu.Softmax".*-> ' + re.escape(g.types['%x']), text), (name, text)
        # Verify and reapply the pass to catch non-converging layout rewrites.
        again = subprocess.run([TOOL, 'out.mlir', '--processor-tpu-optimize', '-o', 'again.mlir'],
                               cwd=root, env=env, text=True, capture_output=True)
        assert again.returncode == 0, (name, again.stdout, again.stderr)
        if fused or permutes == 0:
            assert text == (root / 'again.mlir').read_text(), (name, 'not idempotent')
    CASES += 1


def main():
    # All head counts occurring in the 22 WavLM attention blocks, with real T/D.
    for h in [1, 2, 4, 5, 6, 7, 8, 12, 15]:
        run(f'pruned_h{h}', attention(heads=h), fused=True)
    run('bm1684x', attention(), fused=True, chip='bm1684x')
    run('bf16', attention(dtype='bf16'), fused=True)
    run('cross_attention', attention(batch=2, mq=17, mk=19, dim=32), fused=True)
    run('dim128', attention(dim=128), fused=True)
    run('no_q_scale', attention(scale=False), fused=True)
    run('negative_axes', attention(axis=-1, sm_axis=-1), fused=True)

    # Preserve side outputs and unsupported semantics rather than deleting them.
    for shared in ['scores', 'row_max', 'centered', 'probabilities', 'q_scaled', 'context']:
        run('shared_' + shared, attention(shared=shared), fused=False)
    for name, kw in [
        ('float32', {'dtype': 'f32'}),
        ('large_dim', {'dim': 192}),
        ('short_keys', {'mk': 3}),
        ('logsoftmax', {'log': True}),
        ('beta', {'beta': 0.5}),
        ('reduce_sum', {'mode': 'ReduceSum'}),
        ('reduce_axis', {'axis': 2}),
        ('softmax_axis', {'sm_axis': 2}),
        ('reverse_sub', {'subattrs': 'is_reverse = true'}),
        ('relu_sub', {'subattrs': 'do_relu = true'}),
        ('scaled_sub', {'subattrs': 'coeff = [1.0, 2.0]'}),
        ('biased_qk', {'bias': True}),
        ('relu_qk', {'relu': True}),
    ]:
        run(name, attention(**kw), fused=False)
    run('disable_fusion', attention(), fused=False, disabled=['StableSoftmaxFAttentionPattern'])
    run('disable_existing_fusion', attention(), fused=False, disabled=['MatMul2FAttention'])

    # Non-self-inverse permutations, B>1, and a non-square Q/K length.
    for order in [(0, 2, 1, 3), (1, 2, 0, 3)]:
        run('inverse_' + ''.join(map(str, order)), permutation(order=order), permutes=0)
    run('layout_f32', permutation(dtype='f32'), permutes=0)
    run('layout_bf16', permutation(dtype='bf16'), permutes=0)
    run('layout_negative_axes', permutation(axis=-1, sm_axis=-1), permutes=0)
    for shared in ['layout', 'row_max', 'centered', 'probabilities']:
        run('layout_shared_' + shared, permutation(shared=shared), permutes=2)
    for name, kw in [
        ('not_inverse', {'inverse': (1, 0, 2, 3)}),
        ('normalization_axis_moved', {'order': (0, 1, 3, 2)}),
        ('reduce_sum', {'mode': 'ReduceSum'}),
        ('reduce_axis', {'axis': 2}),
        ('softmax_axis', {'sm_axis': 2}),
        ('reverse_sub', {'subattrs': 'is_reverse = true'}),
        ('scaled_sub', {'subattrs': 'coeff = [1.0, 2.0]'}),
    ]:
        run('layout_' + name, permutation(**kw), permutes=2)
    run('disable_layout', permutation(), permutes=2, disabled=['PermuteStableSoftmaxPattern'])
    print(f'Stable softmax: {CASES} regression cases passed')


if __name__ == "__main__":
    main()
