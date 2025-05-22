//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"

// clang-format off
// case 1: [5, 6] * [6, 7] = [5, 7] => batch = 1, M = 5, K = 6, N = 7
// case 2: [1, 512, 7, 7] * [25088, 4096] = [1, 4096] => batch = 1, M = 1, K = 25088, N = 4096
// case 3: [3, 4, 5, 6] * [3, 4, 6, 7] = [3, 4, 5, 7] => batch = 12, M = 5, K = 6, N = 7
// case 4: [4, 5, 6] * [6,7] = [4, 5, 7] => batch =1, M = 20, K = 6, N = 7
// case 5: [4, 5, 6] * [6] = [4, 5] => batch =1, M = 20, K = 6, N = 1
// case 6: [4096] * [4096, 12884] = [1,12884] => batch =1, M = 1, K = 4096, N = 12884
// clang-format on
matmul_attr_t tpu::MatMulLutOp::parseParam() {
  matmul_attr_t p = {0};
  auto a_s = SmallVector<int64_t>(module::getShape(getInput()));
  auto b_s = SmallVector<int64_t>(module::getShape(getRight()));
  auto o_s = SmallVector<int64_t>(module::getShape(getOutput()));
  p.input_zp = getInputZp();
  p.with_bias = !module::isNone(getBias());
  p.do_relu = getDoRelu();
  p.relu_limit = this->getReluLimit().convertToDouble();
  p.right_zp = getRightZp();
  p.right_transpose = getRightTranspose();
  p.left_transpose = getLeftTranspose();
  p.output_transpose = getOutputTranspose();
  p.hdim_is_batch = getHdimIsBatch();
  p.left_reuse = getLeftReuse();
  auto a_dims = a_s.size();
  auto b_dims = b_s.size();
  auto o_dims = o_s.size();
  p.batch = 1;
  p.batch_low = 1;
  if (b_dims == 1) {
    assert(p.right_transpose == false);
    b_s.push_back(1);
    o_s.push_back(1);
    b_dims += 1;
    o_dims += 1;
  }
  if (a_dims == 1) {
    assert(p.left_transpose == false);
    a_s.insert(a_s.begin(), 1);
    o_s.insert(o_s.begin(), 1);
    a_dims += 1;
    o_dims += 1;
  }
  // for hdim_is_batch = true,
  // BM1684x: (B0, M, B1, K) x (B0, K, B1, N) = (B0, M, B1, N)
  // CV18xx:  (B0, B1, M, K) x (B0, K, B1, N) = (B0, B1, M, N)
  // up to now bm168x right_trans, left_trans, output_trans always be true
  //           cv18xx support either one to be true
  if (p.right_transpose) {
    if (p.hdim_is_batch) {
      p.K = b_s[b_dims - 3];
      p.N = b_s[b_dims - 1];
      // fix bias_merge_izp size for bm1684x
      if (module::isBM1684XFamily() || module::isBM1690Family()) {
        p.N = b_s[b_dims - 3];
        p.K = b_s[b_dims - 1];
      }
    } else {
      // trans hw
      p.N = b_s[b_dims - 2];
      p.K = b_s[b_dims - 1];
    }
  } else {
    p.N = b_s[b_dims - 1];
    p.K = b_s[b_dims - 2];
  }

  if (p.left_transpose) {
    if (p.hdim_is_batch) {
      p.M = a_s[a_dims - 3];
    } else {
      // trans hw
      p.M = a_s[a_dims - 1];
      for (int i = 0; i < a_dims - 2; i++) {
        p.batch *= a_s[i];
      }
    }
  } else {
    p.M = a_s[a_dims - 2];
  }
  // parse batch info from output
  for (int i = 0; i < o_dims - 2; i++) {
    p.batch *= o_s[i];
  }
  if (p.hdim_is_batch) {
    p.batch = o_s[0];
    if (!p.output_transpose && module::isCV18xx()) {
      p.batch_low = o_s[1];
    } else {
      p.batch_low = o_s[2];
      p.output_transpose = true; // tmp code remove later
    }
  }
  if (!p.hdim_is_batch) {
    // if right batch dim is broadcast, merge left batch to M
    int right_batch = 1;
    for (int i = 0; i < b_dims - 2; i++) {
      right_batch *= b_s[i];
    }
    if (right_batch != p.batch && right_batch == 1) {
      p.batch = 1;
    }
    if (p.batch > 1 || o_dims <= 2) {
      p.M = o_s[o_dims - 2];
    } else {
      p.M = std::accumulate(o_s.begin(), o_s.begin() + o_dims - 1, 1,
                            std::multiplies<int64_t>());
    }
    int b_temp = 1;
    for (int i = 1; i < b_dims - 2; i++) {
      b_temp *= b_s[i];
    }
    if (a_s[0] == b_s[0] && b_temp == 1 && b_dims > 2) {
      p.batch = b_s[0];
      int a_temp = 1;
      for (int i = 1; i < a_dims - 2; i++) {
        a_temp *= a_s[i];
      }
      // consider left_transpose
      p.M = a_s[o_dims - 2 + p.left_transpose] * a_temp;
    }
  }
  return p;
}

LogicalResult tpu::MatMulLutOp::init(InferenceParameter &p) {
  return success();
}

void tpu::MatMulLutOp::deinit(InferenceParameter &p) { return; }

LogicalResult tpu::MatMulLutOp::inference(InferenceParameter &p) {
  return success();
}

bool tpu::MatMulLutOp::support_multi_core() { return false; }

mlir::Type tpu::MatMulLutOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return type_verify_case_same(getOperation(), opd_idx, mode);
}
