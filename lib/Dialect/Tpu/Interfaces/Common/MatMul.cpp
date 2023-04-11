//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Support/MathUtils.h"

// clang-format off
// case 1: [5, 6] * [6, 7] = [5, 7] => batch = 1, M = 5, K = 6, N = 7
// case 2: [1, 512, 7, 7] * [25088, 4096] = [1, 4096] => batch = 1, M = 1, K = 25088, N = 4096
// case 3: [3, 4, 5, 6] * [3, 4, 6, 7] = [3, 4, 5, 7] => batch = 12, M = 5, K = 6, N = 7
// case 4: [4, 5, 6] * [6,7] = [4, 5, 7] => batch =1, M = 20, K = 6, N = 7
// case 5: [4, 5, 6] * [6] = [4, 5] => batch =1, M = 20, K = 6, N = 1
// clang-format on
matmul_attr_t tpu::MatMulOp::parseParam() {
  matmul_attr_t p = {0};
  auto a_s = module::getShape(getInput());
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
  // for hdim_is_batch = true,
  // BM1684x: (B0, M, B1, K) x (B0, K, B1, N) = (B0, M, B1, N)
  // CV18xx:  (B0, B1, M, K) x (B0, K, B1, N) = (B0, B1, M, N)
  // up to now bm168x right_trans, left_trans, output_trans always be true
  //           cv18xx support either one to be true
  if (p.right_transpose) {
    if (p.hdim_is_batch) {
      p.K = b_s[b_dims - 3];
      p.N = b_s[b_dims - 1];
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
  }
  return p;
}

LogicalResult tpu::MatMulOp::init(InferenceParameter &p) {
  auto matmul = new MatMul();
  auto a = parseParam();
  matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], a.batch,
                a.batch_low, a.M, a.K, a.N, a.do_relu, a.relu_limit, a.right_zp,
                a.input_zp, a.right_transpose, a.left_transpose,
                a.output_transpose, a.hdim_is_batch);
  p.handle = (void *)matmul;
  return success();
}

void tpu::MatMulOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto matmul = (MatMul *)p.handle;
    delete matmul;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::MatMulOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto matmul = (MatMul *)p.handle;

  matmul->run();
  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());
  bool is_cv18xx = module::isCV18xx();
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (module::isUniformQuantized(getOutput())) {
    auto qmode = getQuantMode();
    if (is_cv18xx) {
      auto a = parseParam();
      auto full_batch = a.batch * a.batch_low;
      bool is_fc = isa<top::WeightOp>(getRight().getDefiningOp());
      i64_array_t rshift_v;
      i64_array_t multiplier_v;
      if (is_fc) {
        rshift_v = module::getI64Array(getRshifts(), full_batch, 0);
        multiplier_v = module::getI64Array(getMultipliers(), full_batch, 1);
      } else {
        rshift_v = module::getI64Array(getRshifts(), 1, 0);
        multiplier_v = module::getI64Array(getMultipliers(), 1, 1);
        rshift_v->resize(full_batch, rshift_v->at(0));
        multiplier_v->resize(full_batch, multiplier_v->at(0));
      }
      int64_t isz = a.M * a.N;
      for (int64_t i = 0; i < full_batch; ++i) {
#pragma omp parallel for schedule(static, omp_schedule(isz))
        for (int64_t j = 0; j < isz; ++j) {
          int64_t offset = i * isz + j;
          int64_t v = 0;
          v = applyMultiplierAndRShift(p.outputs[0][offset],
                                       multiplier_v->at(i), rshift_v->at(i),
                                       qmode, ROUNDING_HALF_AWAY_FROM_ZERO);
          p.outputs[0][offset] = saturate(v, out_type);
        }
      }
    } else {
      auto o_qtype = module::getUniformQuantizedType(getOutput());
      auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
      auto multiplier_v = module::getI64Array(getMultipliers(), 1, 1);
      assert(rshift_v->size() == 1);
      assert(multiplier_v->size() == 1);
      auto num_output = module::getNumElements(getOutput());
      if (qmode == tpu::RequantMode::TFLite_LShift ||
          qmode == tpu::RequantMode::TFLite) {
#pragma omp parallel for schedule(static, omp_schedule(num_output))
        for (int64_t i = 0; i < num_output; i++) {
          // auto v = (((int64_t)(p.outputs[0][i] * mlti) + (1 << (rft - 1))) >>
          // rft);
          auto v = MultiplyByQuantizedMultiplier((int32_t)(p.outputs[0][i]),
                                                 (int32_t)multiplier_v->at(0),
                                                 -(int32_t)rshift_v->at(0)) +
                   o_qtype.getZeroPoint();
          p.outputs[0][i] = saturate(v, out_type);
        }
      } else if (qmode == tpu::RequantMode::MultiplierShift) {
#pragma omp parallel for schedule(static, omp_schedule(num_output))
        for (int i = 0; i < num_output; ++i) {
          auto v = applyMultiplierAndRShift(
                       p.outputs[0][i], multiplier_v->at(0), rshift_v->at(0)) +
                   o_qtype.getZeroPoint();
          p.outputs[0][i] = saturate(v, out_type);
        }
      }
    }
  }

  return success();
}

LogicalResult tpu::MatMulOp::LocalGenSupport() {
  if (module::isCV18xx()) {
    return failure();
  }

  auto ins = getOperands();
  if (backend::BM168x::getDataType(ins[0]) == DTYPE_FP32) {
    return failure();
  }

  auto Lshape = module::getShape(ins[0]);
  auto Rshape = module::getShape(ins[1]);
  int left_num_dims = module::getShape(ins[0]).size();
  int right_num_dims = module::getShape(ins[1]).size();
  if (((left_num_dims == 4 && Lshape[1] < Lshape[2]) ||
       (left_num_dims == 5 && Lshape[1] < Lshape[3])) &&
      right_num_dims == 2) {
    // GROUP_SMALL_C
    return success();
  } else if (left_num_dims == 3 && right_num_dims == 3) {
    // (1, M, K) x (1, K, N)
    return success();
  } else if (left_num_dims == 4 && right_num_dims == 4 && getHdimIsBatch()) {
    // (B1, M, B2, K) x (B1, K, B2, N)
    return success();
  }
  return failure();
}

LogicalResult tpu::MatMulOp::AllowDataSplit(int64_t axis,
                                            group_type_t group_type) {
  if (axis == 0) {
    return success();
  }

  auto lshape = module::getShape(getInput());
  if (lshape.size() == 4 && axis == 2 && getHdimIsBatch()) {
    return success();
  }

  return failure();
}

mlir::Type tpu::MatMulOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  if (opd_idx == 0 || opd_idx == 1) {
    return type_verify_case_i32(getOperation(), opd_idx, mode);
  }
  return type_verify_case_same(getOperation(), opd_idx, mode);
}
