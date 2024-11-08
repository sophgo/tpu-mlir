//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

// clang-format off
// case 1: [5, 6] * [6, 7] = [5, 7] => batch = 1, M = 5, K = 6, N = 7
// case 2: [1, 512, 7, 7] * [25088, 4096] = [1, 4096] => batch = 1, M = 1, K = 25088, N = 4096
// case 3: [3, 4, 5, 6] * [3, 4, 6, 7] = [3, 4, 5, 7] => batch = 12, M = 5, K = 6, N = 7
// case 4: [4, 5, 6] * [6,7] = [4, 5, 7] => batch =1, M = 20, K = 6, N = 7
// case 5: [4, 5, 6] * [6] = [4, 5] => batch =1, M = 20, K = 6, N = 1
// case 6: [4096] * [4096, 12884] = [1,12884] => batch =1, M = 1, K = 4096, N = 12884
// clang-format on

matmul_attr_t top::MatMulOp::parseParam() {
  matmul_attr_t p = {0};
  auto a_s = SmallVector<int64_t>(module::getShape(getInput()));
  auto b_s = SmallVector<int64_t>(module::getShape(getRight()));
  auto o_s = SmallVector<int64_t>(module::getShape(getOutput()));
  p.with_bias = !module::isNone(getBias());
  p.do_relu = getDoRelu();
  p.relu_limit = this->getReluLimit().convertToDouble();
  auto a_dims = a_s.size();
  auto b_dims = b_s.size();
  auto o_dims = o_s.size();
  p.right_transpose = getRightTranspose();
  p.output_transpose = getOutputTranspose();
  if (b_dims == 1) {
    ASSERT_THIS(p.right_transpose == false);
    b_s.push_back(1);
    o_s.push_back(1);
    b_dims += 1;
    o_dims += 1;
  }
  for (int i = 0; i < a_dims; i++) {
    p.L_shape.push_back(a_s[i]);
  }
  for (int i = 0; i < b_dims; i++) {
    p.R_shape.push_back(b_s[i]);
  }
  if (a_dims == 1) {
    ASSERT_THIS(p.left_transpose == false);
    a_s.insert(a_s.begin(), 1);
    o_s.insert(o_s.begin(), 1);
    a_dims += 1;
    o_dims += 1;
  }
  p.N = p.right_transpose ? b_s[b_dims - 2] : b_s[b_dims - 1];
  if (!p.output_transpose) {
    ASSERT_THIS(p.N == o_s[o_dims - 1]);
  }
  p.K = p.right_transpose ? b_s[b_dims - 1] : b_s[b_dims - 2];
  p.batch = 1;
  for (int i = 0; i < b_dims - 2; i++) {
    if (a_s[i] == b_s[i])
      p.batch *= b_s[i];
    else if (a_s[i] == 1 || b_s[i] == 1)
      p.batch *= std::max(a_s[i], b_s[i]);
    else
      ASSERT_THIS(a_s[i] == b_s[i] || a_s[i] == 1 || b_s[i] == 1);
  }
  p.dims_merge_2_M = 0;
  for (int i = b_dims - 3; i > 0; i--) {
    if (b_s[i] == 1) {
      p.dims_merge_2_M++;
    } else {
      break;
    }
  }
  if (a_s[0] == b_s[0] && b_dims > 2 && p.dims_merge_2_M) {
    p.batch = b_s[0];
    int a_temp = 1;
    for (int i = a_dims - 3; i > a_dims - 3 - p.dims_merge_2_M; i--) {
      a_temp *= a_s[i];
    }
    p.M = a_s[o_dims - 2] * a_temp;
  } else {
    if (p.batch > 1 || o_dims <= 2) {
      p.M = o_s[o_dims - 2];
    } else {
      p.M = std::accumulate(o_s.begin(), o_s.begin() + o_dims - 1, 1,
                            std::multiplies<int64_t>());
      p.dims_merge_2_M = o_dims - b_dims;
    }
  }
  return p;
}

matmul_attr_t top::MatMulOp::dynparseParam() {
  std::vector<int64_t> in0_shape = module::getShape(getInput());
  int in0_dims = in0_shape.size();
  auto k = in0_shape[in0_dims - 1];
  std::vector<int64_t> in1_shape = module::getShape(getRight());
  int in1_dims = in1_shape.size();
  bool r_transpose = getRightTranspose();
  bool keep_dims_ = getKeepDims();
  int k_idx = in1_dims - (r_transpose ? 1 : 2);
  int n_idx = in1_dims - (r_transpose ? 2 : 1);
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
    }
  }
  if (in1_dims == 1) {
    ASSERT_THIS(in1_shape[0] == k);
    out_shape.pop_back();
  } else if (in1_shape[k_idx] == k) {
    if (module::getPlatform() == module::Platform::CAFFE) {
      // for caffe case
      // shape case:[1, 1, 1, 4832] * [4832, 126] = [1, 126]
      // shape case:[8, 1, 1, 4832] * [4832, 136] = [8, 136]
      for (int i = 1; i < out_shape.size(); i++) {
        if (out_shape[i] == 1) {
          out_shape.erase(out_shape.begin() + i);
          i--;
        }
      }
      out_shape[out_shape.size() - 1] = n;
    } else {
      out_shape[out_shape.size() - 1] = n;
    }
  } else if (in1_dims == 2) {
    auto sum = in1_shape[k_idx];
    while (out_shape.size() > 0 && sum % out_shape.back() == 0 && sum != 1) {
      sum = sum / out_shape.back();
      out_shape.pop_back();
    }
    if (sum != 1) {
      UNREACHABLE_THIS("shape is illegal");
    }
    out_shape.push_back(n);
  } else {
    out_shape[out_shape.size() - 1] = n;
  }
  if (!keep_dims_) {
    int64_t batch_size = std::accumulate(out_shape.begin(), out_shape.end() - 1,
                                         1, std::multiplies<int64_t>());
    out_shape.resize(2);
    out_shape[0] = batch_size;
    out_shape[1] = n;
  }
  module::setShape(getOutput(), out_shape);
  matmul_attr_t p = {0};
  auto a_s = SmallVector<int64_t>(module::getShape(getInput()));
  auto b_s = SmallVector<int64_t>(module::getShape(getRight()));
  auto o_s = SmallVector<int64_t>(module::getShape(getOutput()));
  p.with_bias = !module::isNone(getBias());
  p.do_relu = getDoRelu();
  p.relu_limit = this->getReluLimit().convertToDouble();
  auto a_dims = a_s.size();
  auto b_dims = b_s.size();
  auto o_dims = o_s.size();
  p.right_transpose = getRightTranspose();
  p.output_transpose = getOutputTranspose();
  if (b_dims == 1) {
    ASSERT_THIS(p.right_transpose == false);
    b_s.push_back(1);
    o_s.push_back(1);
    b_dims += 1;
    o_dims += 1;
  }
  for (int i = 0; i < a_dims; i++) {
    p.L_shape.push_back(a_s[i]);
  }
  for (int i = 0; i < b_dims; i++) {
    p.R_shape.push_back(b_s[i]);
  }
  if (a_dims == 1) {
    ASSERT_THIS(p.left_transpose == false);
    a_s.insert(a_s.begin(), 1);
    o_s.insert(o_s.begin(), 1);
    a_dims += 1;
    o_dims += 1;
  }
  p.N = p.right_transpose ? b_s[b_dims - 2] : b_s[b_dims - 1];
  if (!p.output_transpose) {
    ASSERT_THIS(p.N == o_s[o_dims - 1]);
  }
  p.K = p.right_transpose ? b_s[b_dims - 1] : b_s[b_dims - 2];
  p.batch = 1;
  for (int i = 0; i < b_dims - 2; i++) {
    if (a_s[i] == b_s[i])
      p.batch *= b_s[i];
    else if (a_s[i] == 1 || b_s[i] == 1)
      p.batch *= std::max(a_s[i], b_s[i]);
    else
      ASSERT_THIS(a_s[i] == b_s[i] || a_s[i] == 1 || b_s[i] == 1);
  }
  p.dims_merge_2_M = 0;
  for (int i = b_dims - 3; i > 0; i--) {
    if (b_s[i] == 1) {
      p.dims_merge_2_M++;
    } else {
      break;
    }
  }
  if (a_s[0] == b_s[0] && b_dims > 2 && p.dims_merge_2_M) {
    p.batch = b_s[0];
    int a_temp = 1;
    for (int i = a_dims - 3; i > a_dims - 3 - p.dims_merge_2_M; i--) {
      a_temp *= a_s[i];
    }
    p.M = a_s[o_dims - 2] * a_temp;
  } else {
    if (p.batch > 1 || o_dims <= 2) {
      p.M = o_s[o_dims - 2];
    } else {
      p.M = std::accumulate(o_s.begin(), o_s.begin() + o_dims - 1, 1,
                            std::multiplies<int64_t>());
      p.dims_merge_2_M = o_dims - b_dims;
    }
  }
  return p;
}

int64_t top::MatMulOp::getFLOPs() {
  auto p = parseParam();
  auto extra = (p.with_bias ? 1 : 0) + (p.do_relu ? 1 : 0);
  return p.batch * (2 * p.K + extra) * p.N * p.M;
}

LogicalResult top::MatMulOp::init(InferenceParameter &p) {
  auto matmul = new MatMul();
  p.handle = (void *)matmul;
  return success();
}

void top::MatMulOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto matmul = (MatMul *)p.handle;
    delete matmul;
    p.handle = nullptr;
  }
  return;
}

LogicalResult top::MatMulOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto matmul = (MatMul *)p.handle;
  auto a = dynparseParam();
  matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], a.batch, 1,
                a.M, a.K, a.N, a.do_relu, a.relu_limit, 0, 0, a.right_transpose,
                0, 0, 0, a.L_shape, a.R_shape, a.dims_merge_2_M);
  matmul->run();
  return success();
}

// shape case:
// case 1: [5, 6] * [6, 7] = [5, 7]
// case 2: [1, 512, 7, 7] * [25088, 4096] = [1, 4096]
// case 3: [3, 4, 5, 6] * [3, 4, 6, 7] = [3, 4, 5, 7]
// case 4: [4, 5, 6] * [6, 7] = [4, 5, 7]
// case 5: [4, 5, 6] * [6] = [4, 5]
// case 6: keep_dims == false, [4, 5, 6] * [6, 7] = [20, 7]
// case 7: [3] * [3, 256] = [1,256]
void top::MatMulOp::shape_inference() {
  std::vector<int64_t> in0_shape = module::getShape(getInput());
  int in0_dims = in0_shape.size();
  auto k = in0_shape[in0_dims - 1];
  std::vector<int64_t> in1_shape = module::getShape(getRight());
  int in1_dims = in1_shape.size();
  bool r_transpose = getRightTranspose();
  bool keep_dims_ = getKeepDims();
  int k_idx = in1_dims - (r_transpose ? 1 : 2);
  int n_idx = in1_dims - (r_transpose ? 2 : 1);
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
  if (in1_dims == 1) {
    ASSERT_THIS(in1_shape[0] == k);
    out_shape.pop_back();
  } else if (in1_shape[k_idx] == k) {
    if (module::getPlatform() == module::Platform::CAFFE) {
      // for caffe case
      // shape case:[1, 1, 1, 4832] * [4832, 126] = [1, 126]
      // shape case:[8, 1, 1, 4832] * [4832, 136] = [8, 136]
      for (int i = 1; i < out_shape.size(); i++) {
        if (out_shape[i] == 1) {
          out_shape.erase(out_shape.begin() + i);
          i--;
        }
      }
      out_shape[out_shape.size() - 1] = n;
    } else {
      out_shape[out_shape.size() - 1] = n;
    }
  } else if (in1_dims == 2) {
    auto sum = in1_shape[k_idx];
    while (out_shape.size() > 0 && sum % out_shape.back() == 0 && sum != 1) {
      sum = sum / out_shape.back();
      out_shape.pop_back();
    }
    if (sum != 1) {
      dump();
      llvm_unreachable("shape is illegal");
    }
    out_shape.push_back(n);
  } else {
    out_shape[out_shape.size() - 1] = n;
  }
  if (!keep_dims_) {
    int64_t batch_size = std::accumulate(out_shape.begin(), out_shape.end() - 1,
                                         1, std::multiplies<int64_t>());
    out_shape.resize(2);
    out_shape[0] = batch_size;
    out_shape[1] = n;
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}
