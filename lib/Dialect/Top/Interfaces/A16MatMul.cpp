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

// clang-format off
// case 1: [5, 6] * [6, 7] = [5, 7] => batch = 1, M = 5, K = 6, N = 7
// case 2: [1, 512, 7, 7] * [25088, 4096] = [1, 4096] => batch = 1, M = 1, K = 25088, N = 4096
// case 3: [3, 4, 5, 6] * [3, 4, 6, 7] = [3, 4, 5, 7] => batch = 12, M = 5, K = 6, N = 7
// case 4: [4, 5, 6] * [6,7] = [4, 5, 7] => batch =1, M = 20, K = 6, N = 7
// case 5: [4, 5, 6] * [6] = [4, 5] => batch =1, M = 20, K = 6, N = 1
// case 6: [4096] * [4096, 12884] = [1,12884] => batch =1, M = 1, K = 4096, N = 12884
// clang-format on

a16matmul_attr_t top::A16MatMulOp::parseParam() {
  a16matmul_attr_t p = {0};
  auto a_s = SmallVector<int64_t>(module::getShape(getInput()));
  auto w_s = SmallVector<int64_t>(module::getShape(getWeight()));
  auto s_s = SmallVector<int64_t>(module::getShape(getScale()));
  auto z_s = SmallVector<int64_t>(module::getShape(getZp()));
  auto o_s = SmallVector<int64_t>(module::getShape(getOutput()));
  auto q_group_size = getQGroupSize();
  auto weight_bits = getWeightBits();

  p.q_group_size = q_group_size;
  p.weight_bits = weight_bits;
  p.with_bias = !module::isNone(getBias());
  auto a_dims = a_s.size();
  auto w_dims = w_s.size();
  auto s_dims = s_s.size();
  auto z_dims = z_s.size();
  auto o_dims = o_s.size();
  p.right_transpose = getRightTranspose();

  // TODO: only support w_dims == 2 temporarily
  ASSERT_THIS(w_dims == 2);
  ASSERT_THIS(s_dims == 2);
  ASSERT_THIS(z_dims == 2);

  for (int i = 0; i < a_dims; i++) {
    p.L_shape.push_back(a_s[i]);
  }

  for (int i = 0; i < w_dims; i++) {
    p.R_shape.push_back(w_s[i]);
    p.scale_shape.push_back(s_s[i]);
    p.zp_shape.push_back(z_s[i]);
  }

  p.N = p.right_transpose ? w_s[w_dims - 2] : w_s[w_dims - 1];
  if (!p.output_transpose) {
    ASSERT_THIS(p.N == o_s[o_dims - 1]);
  }
  p.K = p.right_transpose ? w_s[w_dims - 1] : w_s[w_dims - 2];
  p.K *= 8 / weight_bits;
  p.M = std::accumulate(o_s.begin(), o_s.begin() + o_dims - 1, 1,
                        std::multiplies<int64_t>());
  p.batch = 1;
  p.dims_merge_2_M = o_dims - w_dims;
  return p;
}

int64_t top::A16MatMulOp::getFLOPs() {
  auto p = parseParam();
  auto extra = p.with_bias ? 1 : 0;
  return p.batch * (2 * p.K + extra) * p.N * p.M;
}

LogicalResult top::A16MatMulOp::init(InferenceParameter &p) {
  auto matmul = new MatMul();
  p.handle = (void *)matmul;
  return success();
}

void top::A16MatMulOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto matmul = (MatMul *)p.handle;
    delete matmul;
    p.handle = nullptr;
  }
  return;
}

LogicalResult top::A16MatMulOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto matmul = (MatMul *)p.handle;
  auto a = parseParam();
  auto weight_len = a.N * a.K;

  // dequant weight as shape (N * K)
  auto new_weight = std::vector<float>(weight_len, 0);
  matmul->dequant_weight(new_weight.data(), p.inputs[1], p.inputs[2],
                         p.inputs[3], weight_len, a.q_group_size,
                         a.weight_bits);

  // try {
  //   cnpy::npz_save("/workspace/tpu-mlir/qwen2_vl/tpulang_test_bm1684x/A16Matmul/dqweights.npz",
  //           "dq_weight", new_weight.data(), { static_cast<unsigned
  //           long>(new_weight.size()) }, "w");
  //   std::cout << "Success: dqweights.npz" << std::endl;
  // }
  // catch (const std::exception& e) {
  //   std::cerr << "Failed: " << e.what() << std::endl;
  // }

  matmul->setup(p.inputs[0], new_weight.data(), p.inputs[4], p.outputs[0],
                a.batch, 1, a.M, a.K, a.N, false, -1.0, 0, 0, a.right_transpose,
                0, 0, 0, a.L_shape, a.R_shape, a.dims_merge_2_M);
  matmul->run();
  return success();
}

void top::A16MatMulOp::shape_inference() {
  std::vector<int64_t> in0_shape = module::getShape(getInput());
  int in0_dims = in0_shape.size();
  auto k = in0_shape[in0_dims - 1];
  std::vector<int64_t> in1_shape = module::getShape(getWeight());
  int in1_dims = in1_shape.size();
  bool r_transpose = getRightTranspose();
  auto weight_bits = getWeightBits();
  int k_idx = in1_dims - (r_transpose ? 1 : 2);
  int n_idx = in1_dims - (r_transpose ? 2 : 1);
  auto n = in1_shape[n_idx];
  in1_shape[k_idx] = in1_shape[k_idx] * 8 / weight_bits;
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

  if (in1_shape[k_idx] == k) {
    out_shape[out_shape.size() - 1] = n;
  } else {
    dump();
    llvm_unreachable("shape is illegal");
  }

  module::setShapeOrVerify(getOutput(), out_shape);
}
