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

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

LogicalResult tpu::MulConstOp::init(InferenceParameter &p) { return success(); }

void tpu::MulConstOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::MulConstOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  auto asym = module::isAsymmetric();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = p.inputs[0][i] * getConstVal().convertToDouble();
  }
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (module::isUniformQuantized(getOutput())) {
    if (asym == false) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // coeff has been merge in multiplier&&rshift
        double sum = applyMultiplierAndRShift(p.inputs[0][i], getMultiplier(),
                                              getRshift());
        if (getDoRelu() && sum < 0) {
          sum = 0;
        }
        p.outputs[0][i] = saturate(sum, out_type);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // inputs has been requant
        double sum = p.inputs[0][i];
        if (getDoRelu() && sum < 0) {
          sum = 0;
        }
        p.outputs[0][i] = saturate(sum, out_type);
      }
    }
  }
  return success();
}

LogicalResult tpu::MulConstOp::LocalGenSupport() { return success(); }

void tpu::MulConstOp::assign_fw_param(void *param) {
  fw_const_binary_layer_param_t fw_const_binary_layer_param = {0};
  fw_const_binary_layer_param.binary_op = BINARY_MUL;
  fw_const_binary_layer_param.b_value = getConstVal().convertToDouble();
  fw_const_binary_layer_param.inversed = 0;
  int out_sign = module::isSign(getOutput());
  auto data_size = get_dynamic_compiler_tensor_datasize(getInput());
  if (getDoRelu() || (DSIZE_8 == data_size && !out_sign)) {
    fw_const_binary_layer_param.if_relu = 1;
  } else {
    fw_const_binary_layer_param.if_relu = 0;
  }
  fw_const_binary_layer_param.relu_upper_limit =
      getReluLimit().convertToDouble();
  for (int idx = 0; idx < 2; ++idx) {
    fw_const_binary_layer_param.scale[idx] = getMultiplier();
    fw_const_binary_layer_param.rshift_num[idx] = getRshift();
  }
  fw_const_binary_layer_param.opd_sign[0] = module::isSign(getInput());
  memcpy(param, &fw_const_binary_layer_param,
         sizeof(fw_const_binary_layer_param_t));
}
