//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "cnpy.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"

fp8matmul_attr_t top::Fp8MatMulOp::parseParam() {
  fp8matmul_attr_t p = {0};
  auto a_s = SmallVector<int64_t>(module::getShape(getInput()));
  auto w_s = SmallVector<int64_t>(module::getShape(getWeight()));
  auto w_scale_s = SmallVector<int64_t>(module::getShape(getWeightScale()));
  auto o_s = SmallVector<int64_t>(module::getShape(getOutput()));
  auto block_size = getBlockSize();

  p.block_size = block_size;
  p.with_bias = !module::isNone(getBias());
  auto a_dims = a_s.size();
  auto w_dims = w_s.size();
  auto w_scale_dims = w_scale_s.size();
  auto o_dims = o_s.size();
  p.weight_transpose = getWeightTranspose();

  // TODO: only support w_dims == 2 temporarily
  ASSERT_THIS(w_dims == 2);
  ASSERT_THIS(w_scale_dims == 2);

  for (int i = 0; i < a_dims; i++) {
    p.L_shape.push_back(a_s[i]);
  }

  for (int i = 0; i < w_dims; i++) {
    p.R_shape.push_back(w_s[i]);
    p.scale_shape.push_back(w_scale_s[i]);
  }

  p.N = p.weight_transpose ? w_s[w_dims - 2] : w_s[w_dims - 1];
  p.K = p.weight_transpose ? w_s[w_dims - 1] : w_s[w_dims - 2];
  p.M = std::accumulate(o_s.begin(), o_s.begin() + o_dims - 1, 1,
                        std::multiplies<int64_t>());
  p.batch = 1;
  return p;
}

int64_t top::Fp8MatMulOp::getFLOPs() {
  auto p = parseParam();
  auto extra = p.with_bias ? 1 : 0;
  return p.batch * (2 * p.K + extra) * p.N * p.M;
}

LogicalResult top::Fp8MatMulOp::init(InferenceParameter &p) {
  return success();
}

void top::Fp8MatMulOp::deinit(InferenceParameter &p) { return; }

LogicalResult top::Fp8MatMulOp::inference(InferenceParameter &p) {
  auto matmul = new MatMul();
  auto a = parseParam();
  auto weight_len = a.N * a.K;

  // dequant weight as shape (N * K)
  auto new_weight = std::vector<float>(weight_len, 0);
  auto block_size = a.block_size;
  matmul->dequant_fp8_weight(new_weight.data(), p.inputs[1], p.inputs[2],
                             weight_len, block_size, a.N, a.K);

  matmul->setup(p.inputs[0], new_weight.data(), p.inputs[3], p.outputs[0],
                a.batch, 1, a.M, a.K, a.N, false, -1.0, 0, 0,
                a.weight_transpose, 0, 0, 0, a.L_shape, a.R_shape, false);
  matmul->run();
  delete matmul;
  return success();
}

void top::Fp8MatMulOp::shape_inference() {
  std::vector<int64_t> in0_shape = module::getShape(getInput());
  int in0_dims = in0_shape.size();
  auto k = in0_shape[in0_dims - 1];
  std::vector<int64_t> in1_shape = module::getShape(getWeight());
  int in1_dims = in1_shape.size();
  bool w_transpose = getWeightTranspose();
  int k_idx = in1_dims - (w_transpose ? 1 : 2);
  int n_idx = in1_dims - (w_transpose ? 2 : 1);
  auto n = in1_shape[n_idx];
  std::vector<int64_t> out_shape;
  if (in0_dims > in1_dims) {
    out_shape = in0_shape;
  } else if (in0_dims == in1_dims) {
    out_shape = in0_shape;
    for (int i = out_shape.size() - 3; i >= 0; i--) {
      out_shape[i] = std::max(in0_shape[i], in1_shape[i]);
    }
  } else {
    out_shape = in1_shape;
    for (int i = 1; i <= 2; i++) {
      out_shape[out_shape.size() - i] = in0_shape[in0_dims - i];
      if (i > in0_dims) {
        out_shape[out_shape.size() - i] = 1;
      }
    }
  }

  // if (in1_shape[k_idx] * 4 == k) {
  if (in1_shape[k_idx] == k) {
    out_shape[out_shape.size() - 1] = n;
  } else {
    dump();
    llvm_unreachable("shape is illegal");
  }

  module::setShapeOrVerify(getOutput(), out_shape);
}
