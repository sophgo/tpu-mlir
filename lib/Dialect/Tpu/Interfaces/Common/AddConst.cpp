//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::AddConstOp::init(InferenceParameter &p) { return success(); }

void tpu::AddConstOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::AddConstOp::inference(InferenceParameter &p) {
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
  auto num_elem = module::getNumElements(getOutput());
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  auto asym = module::isAsymmetric();
  if (module::isUniformQuantized(getOutput())) {
    auto i_qtype = module::getUniformQuantizedType(getInput());
    double izp = 0;
    if (module::isUniformQuantized(getInput()))
      izp = i_qtype.getZeroPoint();
    auto out_qtype = module::getUniformQuantizedType(getOutput());
    double ozp = 0;
    auto out_type = module::getStorageType(getOutput());
    if (asym == true)
      ozp = out_qtype.getZeroPoint();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      double sum = p.inputs[0][i] - izp;
      sum = applyMultiplierAndRShift(sum, getMultiplier(), 0);
      sum += getConstVal().convertToDouble();
      sum = applyMultiplierAndRShift(sum, 1, getRshift());

      if (getDoRelu() && sum < 0) {
        sum = 0;
      }
      p.outputs[0][i] = saturate(sum, out_type);
    }
  } else {
    if (in_type.isFloat8E4M3FN()) {
      double scale = getF8Scale().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] =
            p.inputs[0][i] * scale + getConstVal().convertToDouble();
      }
      if (getDoRelu()) {
        auto limit = getReluLimit().convertToDouble();
        function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] = p.inputs[0][i] + getConstVal().convertToDouble();
      }
      if (getDoRelu()) {
        auto limit = getReluLimit().convertToDouble();
        function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
      }
    }
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isFloat8E4M3FN()) {
      F8E4M3(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
    } else if (out_type.isFloat8E5M2()) {
      F8E5M2(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
    }
  }
  return success();
}

mlir::Type tpu::AddConstOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  auto op = getOperation();
  auto in_stype = module::getStorageType(getInput());
  auto out_stype = module::getStorageType(getOutput());
  if (in_stype.isIntOrIndex() && out_stype.isIntOrIndex()) {
    return do_nothing(mode);
  }
  return type_verify_case_same(op, opd_idx, mode);
}

void tpu::AddConstOp::assign_fw_param(void *param) {
  fw_const_binary_layer_param_t fw_const_binary_layer_param = {0};
  fw_const_binary_layer_param.binary_op = BINARY_ADD;
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

ArrayAttr tpu::AddConstOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::AddConstOp::support_multi_core() { return false; }
