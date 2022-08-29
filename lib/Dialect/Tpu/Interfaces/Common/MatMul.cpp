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

void tpu::MatMulOp::parseParam(int64_t &batch, int64_t &M, int64_t &K,
                               int64_t &N, bool &with_bias, bool &relu,
                               double &limit) {
  auto i_s = input().getType().cast<RankedTensorType>().getShape();
  auto r_s = right().getType().cast<RankedTensorType>().getShape();
  auto o_s = output().getType().cast<RankedTensorType>().getShape();
  with_bias = !bias().getType().isa<mlir::NoneType>();
  relu = do_relu();
  limit = relu_limit().convertToDouble();
  auto r_dims = r_s.size();
  auto i_dims = i_s.size();
  N = r_s[r_dims - 1];
  K = r_s[r_dims - 2];
  if (r_dims > 2) {
    M = i_s[i_dims - 2];
    assert(i_s[i_dims - 1] == K);
    batch = std::accumulate(r_s.begin(), r_s.begin() + r_dims - 2, 1,
                            std::multiplies<int64_t>());
  } else {
    batch = 1;
    M = std::accumulate(i_s.begin(), i_s.begin() + i_dims - 1, 1,
                        std::multiplies<int64_t>());
  }
}

LogicalResult tpu::MatMulOp::init(InferenceParameter &p) {
  auto matmul = new MatMul();
  int64_t batch, M, K, N;
  bool relu, with_bias;
  double limit;
  parseParam(batch, M, K, N, with_bias, relu, limit);

  matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], batch, M,
                K, N, relu, limit);
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
    if (quant_mode() == 0 || quant_mode() == 1) {
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
    } else if (quant_mode() == 2) {
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
