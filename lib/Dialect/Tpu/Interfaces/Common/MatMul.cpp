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
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Float16.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

// clang-format off
// case 1: [5, 6] * [6, 7] = [5, 7] => batch = 1, M = 5, K = 6, N = 7
// case 2: [1, 512, 7, 7] * [25088, 4096] = [1, 4096] => batch = 1, M = 1, K = 25088, N = 4096
// case 3: [3, 4, 5, 6] * [3, 4, 6, 7] = [3, 4, 5, 7] => batch = 12, M = 5, K = 6, N = 7
// case 4: [4, 5, 6] * [6,7] = [4, 5, 7] => batch =1, M = 20, K = 6, N = 7
// clang-format on
void tpu::MatMulOp::parseParam(int64_t &batch, int64_t &M, int64_t &K,
                               int64_t &N, bool &with_bias, bool &relu,
                               double &limit, int64_t &zp) {
  auto a_s = Module::getShape(input());
  auto b_s = Module::getShape(right());
  auto o_s = Module::getShape(output());
  with_bias = !bias().getType().isa<mlir::NoneType>();
  relu = do_relu();
  limit = this->relu_limit().convertToDouble();
  zp = right_zp();
  auto b_dims = b_s.size();
  auto o_dims = o_s.size();
  assert(b_dims >= 2);
  N = b_s[b_dims - 1];
  assert(N == o_s[o_dims - 1]);
  K = b_s[b_dims - 2];
  batch = 1;
  for (int i = 0; i < b_dims - 2; i++) {
    batch *= b_s[i];
  }
  if (batch > 1 || o_dims <= 2) {
    M = o_s[o_dims - 2];
  } else {
    M = std::accumulate(o_s.begin(), o_s.begin() + o_dims - 1, 1,
                        std::multiplies<int64_t>());
  }
}

LogicalResult tpu::MatMulOp::init(InferenceParameter &p) {
  auto matmul = new MatMul();
  int64_t batch, M, K, N, zp;
  bool relu, with_bias;
  double limit;
  parseParam(batch, M, K, N, with_bias, relu, limit, zp);

  matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], batch, M,
                K, N, relu, limit, zp);
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
  auto out_type = Module::getStorageType(output());
  auto num_elem = Module::getNumElements(output());
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (Quant::isUniformQuantized(output())) {
    auto o_qtype = Quant::getUniformQuantizedType(output());
    auto rft = rshift();
    auto mlti = multiplier();
    auto num_output = Module::getNumElements(output());
    if (quant_mode() == tpu::RequantMode::TFlite_Lshift ||
        quant_mode() == tpu::RequantMode::TFlite) {
#pragma omp parallel for schedule(static, omp_schedule(num_output))
      for (int64_t i = 0; i < num_output; i++) {
        // auto v = (((int64_t)(p.outputs[0][i] * mlti) + (1 << (rft - 1))) >> rft);
        auto v = MultiplyByQuantizedMultiplier(
                                (int32_t)(p.outputs[0][i]),
                                (int32_t)mlti, -(int32_t)rft);
        if (out_type.isUnsignedInteger(8)) {
          p.outputs[0][i] = Quant::to_uint8(v + o_qtype.getZeroPoint());
        } else {
          p.outputs[0][i] = Quant::to_int8(v + o_qtype.getZeroPoint());
        }
      }
    } else if (quant_mode() == tpu::RequantMode::Normal) {
#pragma omp parallel for schedule(static, omp_schedule(num_output))
      for (int i = 0; i < num_output; ++i) {
        auto v = applyMultiplierAndRShift(p.outputs[0][i], mlti, rft);
        if (out_type.isUnsignedInteger(8)) {
          p.outputs[0][i] = Quant::to_uint8(v + o_qtype.getZeroPoint());
        } else {
          p.outputs[0][i] = Quant::to_int8(v + o_qtype.getZeroPoint());
        }
      }
    }
  }

  return success();
}
