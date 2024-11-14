//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/CastUtils.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"

static void cvi_int8_to_bf16(float *p_src, float *p_dst, float scale, int num,
                             bool is_tpu) {
  // int8 / uint8 ==> bf16 / fp32
  if (is_tpu) {
    scale = BF16(scale);
#pragma omp parallel for schedule(static, omp_schedule(num))
    for (int i = 0; i < num; i++) {
      p_dst[i] = bf16_mul(BF16(p_src[i], false), scale);
    }
  } else {
#pragma omp parallel for schedule(static, omp_schedule(num))
    for (int i = 0; i < num; i++) {
      p_dst[i] = p_src[i] * scale;
    }
  }
}

LogicalResult tpu::DtypeCastOp::init(InferenceParameter &p) {
  return success();
}
void tpu::DtypeCastOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::DtypeCastOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  auto num_elem = module::getNumElements(getOutput());
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());

  if (in_type.isF32() && out_type.isF16()) {
    F16(p.inputs[0], p.outputs[0], num_elem);
  };

  return success();
}

mlir::Type tpu::DtypeCastOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return do_nothing(mode);
}

// LogicalResult tpu::DtypeCastOp::LocalGenSupport() {
//   if (module::isCV18xx()) {
//     auto in_type = module::getStorageType(getInput());
//     auto out_type = module::getStorageType(getOutput());
//     int64_t n, c, h, w;
//     module::getNCHW(getOutput(), n, c, h, w);
//     if (c > MAX_TIU_CHL || w > MAX_TIU_CHL) {
//       return failure();
//     }
//     // type.isSignedInteger()
//     if ((in_type.getIntOrFloatBitWidth() == 8 && out_type.isBF16()) ||
//         (in_type.isBF16() && out_type.isSignedInteger())) {
//       return success();
//     }
//     return failure();
//   }
//   if (module::isBM1684Family()) {
//     auto in_dtype = BM168x::getDataType(getInput());
//     if (in_dtype == DTYPE_INT32) {
//       return failure();
//     }
//   }
//   return success();
// }

// void tpu::DtypeCastOp::assign_fw_param(void *param) {
//   fw_dtype_convert_layer_param_t fw_param = {0};
//   fw_param.src_type = BM168x::getDataType(getInput());
//   fw_param.dst_type = BM168x::getDataType(getOutput());
//   fw_param.src_stmode = BM1684::getStoreMode(getInput());
//   fw_param.dst_stmode = BM1684::getStoreMode(getOutput());
//   fw_param.round_mode = ROUND_INF; // if support other round_mode, change
//   here memcpy(param, &fw_param, sizeof(fw_dtype_convert_layer_param_t));
// }

ArrayAttr tpu::DtypeCastOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::DtypeCastOp::support_multi_core() { return false; }
