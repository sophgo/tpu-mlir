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

LogicalResult tpu::BinaryConstShiftOp::init(InferenceParameter &p) { return success(); }

void tpu::BinaryConstShiftOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::BinaryConstShiftOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  int64_t const_val = getScale();
  int32_t shift_val = getShift();
  auto rmode = round_mode_convert(getRoundMode());
  // bool is_satu = getSaturation();
  if (getMode().str() == "Add") {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      int64_t sum = p.inputs[0][i] + const_val;
      sum = RightShiftRound(sum, -shift_val, rmode);
      p.outputs[0][i] = saturate(sum, out_type, rmode);
    }
  } else if (getMode().str() == "Sub") {
    if (getIsReverse() == false) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        int64_t sum = p.inputs[0][i] - const_val;
        sum = RightShiftRound(sum, -shift_val, rmode);
        p.outputs[0][i] = saturate(sum, out_type, rmode);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        int64_t sum = const_val - p.inputs[0][i];
        sum = RightShiftRound(sum, -shift_val, rmode);
        p.outputs[0][i] = saturate(sum, out_type, rmode);
      }
    }
  } else if (getMode().str() == "Mul") {
    #pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      int64_t sum = p.inputs[0][i] * const_val;
      sum = RightShiftRound(sum, -shift_val, rmode);
      p.outputs[0][i] = saturate(sum, out_type, rmode);
    }
  } else {
    llvm_unreachable("Not Implemented");
  }

  return success();
}

// LogicalResult tpu::BinaryConstShiftOp::canonicalize(BinaryConstShiftOp op,
//                                             PatternRewriter &rewriter) {
//   bool is_type_match = module::getStorageType(op.getInput()) ==
//                        module::getStorageType(op.getResult());
//   bool is_identity = ((std::abs(op.getScale()) == 0 && op.getMode().str() == "Add") ||
//                       (std::abs(op.getScale()) == 0 && op.getMode().str() == "Sub" && op.getIsReverse() == false) ||
//                       (std::abs(op.getScale()) == 1 && op.getMode().str() == "Mul")) &&
//                      op.getRshift() == 0;

//   if (is_type_match && is_identity) {
//     rewriter.replaceOp(op, op.getInput());
//     return success();
//   }
//   return failure();
// };

// mlir::Type tpu::BinaryConstShiftOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
//   return do_nothing(mode);
// }

// void tpu::BinaryConstShiftOp::assign_fw_param(void *param) {
//   fw_const_binary_layer_param_t fw_const_binary_layer_param = {0};
//   fw_const_binary_layer_param.binary_op = BINARY_ADD;
//   fw_const_binary_layer_param.b_value = getConstVal().convertToDouble();
//   fw_const_binary_layer_param.inversed = 0;
//   int out_sign = module::isSign(getOutput());
//   auto data_size = get_dynamic_compiler_tensor_datasize(getInput());
//   if (getDoRelu() || (DSIZE_8 == data_size && !out_sign)) {
//     fw_const_binary_layer_param.if_relu = 1;
//   } else {
//     fw_const_binary_layer_param.if_relu = 0;
//   }
//   fw_const_binary_layer_param.relu_upper_limit =
//       getReluLimit().convertToDouble();
//   for (int idx = 0; idx < 2; ++idx) {
//     fw_const_binary_layer_param.scale[idx] = getMultiplier();
//     fw_const_binary_layer_param.rshift_num[idx] = getRshift();
//   }
//   fw_const_binary_layer_param.opd_sign[0] = module::isSign(getInput());
//   memcpy(param, &fw_const_binary_layer_param,
//          sizeof(fw_const_binary_layer_param_t));
// }

ArrayAttr tpu::BinaryConstShiftOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map = AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};
