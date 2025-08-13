#!/usr/bin/env python3
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def build_and_save(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "struct_optimize_pattern_test.onnx")

    # Input/Output
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 77, 512])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 77, 512])

    # Initializers
    rng = np.random.default_rng(0)
    W = numpy_helper.from_array(rng.standard_normal((512, 1536), dtype=np.float32), "W")
    B = numpy_helper.from_array(rng.standard_normal((1, 1, 1536), dtype=np.float32), "B")
    B1D = numpy_helper.from_array(rng.standard_normal((1536, ), dtype=np.float32), "B1D")
    W2 = numpy_helper.from_array(np.ones((512, 512), dtype=np.float32), "W2")

    shape_77b3_512 = numpy_helper.from_array(np.array([77, 1, 3, 512], dtype=np.int64),
                                             "shape_77b3_512")
    shape_77b512 = numpy_helper.from_array(np.array([77, 1, 512], dtype=np.int64), "shape_77b512")
    shape_77b8_64 = numpy_helper.from_array(np.array([77, 1, 8, 64], dtype=np.int64),
                                            "shape_77b8_64")
    shape_77b64 = numpy_helper.from_array(np.array([77, 8, 64], dtype=np.int64), "shape_77b64")
    shape_b8_77_64 = numpy_helper.from_array(np.array([1, 8, 77, 64], dtype=np.int64),
                                             "shape_b8_77_64")
    shape_b8_77_77 = numpy_helper.from_array(np.array([1, 8, 77, 77], dtype=np.int64),
                                             "shape_b8_77_77")
    shape_8_77_64 = numpy_helper.from_array(np.array([8, 77, 64], dtype=np.int64), "shape_8_77_64")
    shape_8_64_77 = numpy_helper.from_array(np.array([8, 64, 77], dtype=np.int64), "shape_8_64_77")
    shape_8_77_77 = numpy_helper.from_array(np.array([8, 77, 77], dtype=np.int64), "shape_8_77_77")
    shape_77_512 = numpy_helper.from_array(np.array([77, 512], dtype=np.int64), "shape_77_512")
    shape_77_1_512 = numpy_helper.from_array(np.array([77, 1, 512], dtype=np.int64),
                                             "shape_77_1_512")
    shape_1_77_512 = numpy_helper.from_array(np.array([1, 77, 512], dtype=np.int64),
                                             "shape_1_77_512")
    shape_1536 = numpy_helper.from_array(np.array([1536], dtype=np.int64), "shape_1536")
    shape_77_1536 = numpy_helper.from_array(np.array([77, 1536], dtype=np.int64), "shape_77_1536")

    axes0 = numpy_helper.from_array(np.array([0], dtype=np.int64), "axes0")
    axes3 = numpy_helper.from_array(np.array([3], dtype=np.int64), "axes3")

    # LayerNorm params (1D over last dim 512)
    ln_gamma = numpy_helper.from_array(np.ones((512, ), dtype=np.float32), "ln_gamma")
    ln_beta = numpy_helper.from_array(np.zeros((512, ), dtype=np.float32), "ln_beta")
    # remove manual LN pow/epsilon initializers; use ONNX LayerNormalization ops

    # Use scalar indices so Gather drops the gathered axis and returns 3D directly
    gather_idx0 = numpy_helper.from_array(np.array(0, dtype=np.int64), "gather_idx0")
    gather_idx1 = numpy_helper.from_array(np.array(1, dtype=np.int64), "gather_idx1")
    gather_idx2 = numpy_helper.from_array(np.array(2, dtype=np.int64), "gather_idx2")

    # Nodes (simplified per request): no head Transpose/LayerNorm

    # As requested: split after Transpose: one branch to LN (below), the other direct to final Add
    # As requested: keep 3D along this branch: (77,1,512)->MatMul->Add->Reshape->Unsqueeze
    mm3d = helper.make_node("MatMul", ["input", "W"], ["mm3d"])  # (1,77,1536)
    add3d = helper.make_node("Add", ["mm3d", "B"], ["add3d"])  # (77,1,1536)

    rs1 = helper.make_node("Reshape", ["add3d", "shape_77b3_512"], ["rs1"])  # (77,1,3,512)
    unsq = helper.make_node("Unsqueeze", ["rs1", "axes0"], ["unsq"])  # (1,77,1,3,512)
    tr1 = helper.make_node("Transpose", ["unsq"], ["tr1"], perm=[3, 1, 2, 0, 4])  # (3,77,1,1,512)
    sq = helper.make_node("Squeeze", ["tr1", "axes3"], ["sq"])  # (3,77,1,512)

    g0 = helper.make_node("Gather", ["sq", "gather_idx0"], ["g0"], axis=0)  # (77,1,512)
    g1 = helper.make_node("Gather", ["sq", "gather_idx1"], ["g1"], axis=0)
    g2 = helper.make_node("Gather", ["sq", "gather_idx2"], ["g2"], axis=0)

    # Stream 1
    rs1_1 = helper.make_node("Reshape", ["g0", "shape_77b64"],
                             ["rs1_1"])  # (77,1*8,64) but 3D (77,-1,64)
    tr1_1 = helper.make_node("Transpose", ["rs1_1"], ["tr1_1"], perm=[1, 0, 2])  # (1*8,77,64)
    rs1_1f = helper.make_node("Reshape", ["tr1_1", "shape_b8_77_64"], ["final1"])  # (1,8,77,64)

    # Stream 2
    rs2_1 = helper.make_node("Reshape", ["g1", "shape_77b64"], ["rs2_1"])  # (77,1*8,64) 3D
    tr2_1 = helper.make_node("Transpose", ["rs2_1"], ["tr2_1"], perm=[1, 0, 2])  # (1*8,77,64)
    rs2_1f = helper.make_node("Reshape", ["tr2_1", "shape_b8_77_64"], ["final2"])  # (1,8,77,64)

    # Stream 3
    rs3_1 = helper.make_node("Reshape", ["g2", "shape_77b64"], ["rs3_1"])  # (77,1*8,64) 3D
    tr3_1 = helper.make_node("Transpose", ["rs3_1"], ["tr3_1"], perm=[1, 0, 2])  # (1*8,77,64)
    rs3_1f = helper.make_node("Reshape", ["tr3_1", "shape_b8_77_64"], ["final3"])  # (1,8,77,64)

    # final2 permute (0,1,3,2)
    tr_final2 = helper.make_node("Transpose", ["final2"], ["final2_t"], perm=[0, 1, 3,
                                                                              2])  # (1,8,64,77)

    # Stable: convert middle MatMuls to 3D batched
    final3_3d = helper.make_node("Reshape", ["final3", "shape_8_77_64"], ["final3_3d"])  # (8,77,64)
    final2t_3d = helper.make_node("Reshape", ["final2_t", "shape_8_64_77"],
                                  ["final2t_3d"])  # (8,64,77)
    mm23_3d = helper.make_node("MatMul", ["final3_3d", "final2t_3d"], ["mm23_3d"])  # (8,77,77)
    mm23_rs = helper.make_node("Reshape", ["mm23_3d", "shape_b8_77_77"], ["mm23_rs"])  # (1,8,77,77)
    mm23_3d_back = helper.make_node("Reshape", ["mm23_rs", "shape_8_77_77"],
                                    ["mm23_3d_back"])  # (8,77,77)
    final1_3d = helper.make_node("Reshape", ["final1", "shape_8_77_64"], ["final1_3d"])  # (8,77,64)
    mm_final_3d = helper.make_node("MatMul", ["mm23_3d_back", "final1_3d"],
                                   ["mm_final_3d"])  # (8,77,64)
    mm_final_rs = helper.make_node("Reshape", ["mm_final_3d", "shape_b8_77_64"],
                                   ["mm_final_rs"])  # (1,8,77,64)

    tr_final = helper.make_node("Transpose", ["mm_final_rs"], ["tr_final"], perm=[2, 0, 1,
                                                                                  3])  # (77,1,8,64)
    # Merge two reshapes: directly reshape (77,1,8,64) -> (77,512)
    rs_final_2d = helper.make_node("Reshape", ["tr_final", "shape_77_512"],
                                   ["rs_final_2d"])  # (77,512)
    mm2_2d = helper.make_node("MatMul", ["rs_final_2d", "W2"], ["mm2_2d"])  # (77,512)
    mm2 = helper.make_node("Reshape", ["mm2_2d", "shape_77_1_512"], ["mm2"])  # (77,1,512)

    # Residual: add input path (fixed LN output, 1,77,512) back, keep (1,77,512)
    # removed residual Add/Transpose/final LayerNormalization; output directly from the reshape before Add

    graph = helper.make_graph(
        nodes=[
            mm3d,
            add3d,
            rs1,
            unsq,
            tr1,
            sq,
            g0,
            g1,
            g2,
            rs1_1,
            tr1_1,
            rs1_1f,
            rs2_1,
            tr2_1,
            rs2_1f,
            rs3_1,
            tr3_1,
            rs3_1f,
            tr_final2,
            final3_3d,
            final2t_3d,
            mm23_3d,
            mm23_rs,
            mm23_3d_back,
            final1_3d,
            mm_final_3d,
            mm_final_rs,
            tr_final,
            rs_final_2d,
            mm2_2d,
            mm2,
        ],
        name="struct_optimize_pattern_graph",
        inputs=[inp],
        outputs=[helper.make_tensor_value_info("mm2", TensorProto.FLOAT, [77, 1, 512])],
        initializer=[
            W,
            B,
            W2,
            shape_77b3_512,
            shape_77b8_64,
            shape_77b64,
            shape_b8_77_64,
            shape_b8_77_77,
            shape_8_77_64,
            shape_8_64_77,
            shape_8_77_77,
            shape_77_512,
            shape_77_1_512,
            axes0,
            axes3,
            gather_idx0,
            gather_idx1,
            gather_idx2,
        ],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.save(model, onnx_path)
    return onnx_path


def main():
    out_dir = os.environ.get("OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
    model_path = build_and_save(out_dir)
    print(f"[OK] Exported ONNX: {model_path}")
    # save input for regression like struct_optimize_pattern_test.py
    input_npz = os.path.join(out_dir, "struct_optimize_pattern_test_input.npz")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 77, 512), dtype=np.float32)
    np.savez(input_npz, **{"input": x})
    print(f"[OK] Saved input npz: {input_npz}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import transform.TpuLang as tpul
from transform.TpuLangConverter import TpuLangConverter
from transform.TpuLang import TpuLang
import numpy as np
import math


def rand_data(shape, dtype, min=-10, max=10):
    if dtype in ['float32', 'float16']:
        return np.clip(np.random.randn(*shape).astype(dtype), min, max)
    if dtype == 'int32' or 'uint32' or 'int16' or 'uint16' or 'int8' or 'uint8':
        return np.random.randint(0, 127, size=shape).astype(dtype)
    raise Exception("Not supported data type: {}!".format(dtype))


def coeff_tensor(shape, dtype, data=None, scale=None, zero_point=None):
    if data is None:
        data = rand_data(shape, dtype)
        data = data * scale if (dtype in ['float32', 'float16'] and scale is not None) else data
    if dtype in ["int8", "uint8"]:
        return tpul.Tensor(dtype=dtype,
                           shape=shape,
                           data=data,
                           ttype="coeff",
                           scale=scale,
                           zero_point=zero_point)
    else:
        return tpul.Tensor(dtype=dtype, shape=shape, data=data, ttype="coeff")


def conv_int_op(x,
                kshape,
                stride,
                pad=None,
                group=1,
                dilation=[1, 1],
                bias=False,
                zp=[0, 0],
                out_dtype="int32"):
    oc = kshape[0]
    weight = coeff_tensor(kshape,
                          x.dtype,
                          scale=1 / (math.sqrt(kshape[1] * kshape[2] * kshape[3])),
                          zero_point=0)
    bias = coeff_tensor(oc, out_dtype) if bias else None
    conv = tpul.conv_int(x,
                         weight,
                         bias=bias,
                         stride=stride,
                         pad=pad,
                         dilation=dilation,
                         group=group,
                         input_zp=zp[0],
                         weight_zp=zp[1],
                         out_dtype=out_dtype)
    return conv


def resnet_quant(x):
    rq0 = tpul.requant_fp_to_int(x, 1.0, 0, 0, 'int8')
    # conv1 = conv_block(rq0, [64, 3, 7, 7], [2, 2], [3,3,3,3], [2030043136, -13, 0])
    conv1 = conv_int_op(rq0, [64, 3, 7, 7], [2, 2], [3, 3, 3, 3], zp=[0, 0], out_dtype='int32')
    rq1 = tpul.requant_int(conv1, 2030043136, -7, 0, 0, 'int8', round_mode='half_away_from_zero')
    # relu1 = tpul.relu(rq1)
    conv2 = conv_int_op(rq1, [96, 64, 3, 3], [2, 2], [1, 1, 1, 1], zp=[0, 0], out_dtype='int32')
    rq2 = tpul.requant_int(conv2, 1748893696, -10, 0, 0, 'int8', round_mode='half_away_from_zero')
    coeff3 = coeff_tensor([1, 96, 1, 1], 'int8', scale=10.0)
    mul3 = tpul.mul(rq2, coeff3, scale=[0.25, 10.0, 2.5], out_dtype='int8')
    coeff4 = coeff_tensor([1, 96, 1, 1], 'int8', scale=2.0)
    add4 = tpul.add(mul3, coeff4, scale=[2.5, 2.0, 4.0], out_dtype='int8')
    return add4


if __name__ == "__main__":
    tpul.init("bm1684x")
    in_shape = [1, 3, 224, 224]
    x_data = rand_data(in_shape, 'float32')
    x = tpul.Tensor(dtype='float32', shape=in_shape, data=x_data)
    out = resnet_quant(x)
    TpuLang.graph.inputs = [x]
    TpuLang.graph.outputs = [out]
    TpuLang.graph.quantized_type_inference()
    # convert to mlir
    model_name = "resnetquant"
    converter = TpuLangConverter(name=model_name, graph=TpuLang.graph, mode="quantized")
    mlir_origin = model_name + '_origin.mlir'
    converter.generate_mlir(mlir_origin)
    tpul.deinit()
