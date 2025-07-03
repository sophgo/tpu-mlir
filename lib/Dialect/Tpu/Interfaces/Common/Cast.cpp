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
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

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

LogicalResult tpu::CastOp::init(InferenceParameter &p) { return success(); }
void tpu::CastOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CastOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  auto num_elem = module::getNumElements(getOutput());
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  bool isInQuant = module::isUniformQuantized(getInput());
  bool isOutQuant = module::isUniformQuantized(getOutput());
  bool fInput = in_type.isIntOrIndex() == false;
  bool fOutput = out_type.isIntOrIndex() == false;
  auto op = getOperation();
  bool is_cv18xx = module::isCV18xx();
  auto round_mode =
      is_cv18xx ? ROUNDING_HALF_TO_EVEN : round_mode_convert(getRoundMode());
  bool is_tpu = module::isTpuOp(op);

  if (in_type.isF32() && out_type.isF16()) {
    F16(p.inputs[0], p.outputs[0], num_elem);
  } else if (in_type.isF32() && out_type.isBF16()) {
    BF16(p.inputs[0], p.outputs[0], num_elem, false);
  } else if ((in_type.isF32() || in_type.isF16()) &&
             out_type.isFloat8E4M3FN()) {
    F8E4M3(p.inputs[0], p.outputs[0], num_elem, 1.0, true);
  } else if ((in_type.isF32() || in_type.isF16()) && out_type.isFloat8E5M2()) {
    F8E5M2(p.inputs[0], p.outputs[0], num_elem, 1.0, true);
  } else if (in_type.isFloat8E4M3FN() &&
             (out_type.isF32() || out_type.isF16())) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      p.outputs[0][i] = p.inputs[0][i];
    }
  } else if (in_type.isFloat8E5M2() && (out_type.isF32() || out_type.isF16())) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      p.outputs[0][i] = p.inputs[0][i];
    }
  } else if (isOutQuant && fInput) {
    // FP32|BF16|F16|... => INT8|UINT8|...
    auto qtype = module::getUniformQuantizedType(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      float v;
      if (is_cv18xx) {
        v = bf16_mul(BF16(p.inputs[0][i], false), BF16(1. / qtype.getScale()));
      } else {
        if (in_type.isBF16()) {
          v = requant(BF16(p.inputs[0][i], false), qtype);
        } else {
          v = requant(p.inputs[0][i], qtype);
        }
      }
      p.outputs[0][i] = saturate(v, out_type, round_mode);
    }
  } else if (isInQuant && fOutput) {
    // INT8|UINT8|... ==> FP32|BF16|F16|...
    auto qtype = module::getUniformQuantizedType(getInput());
    if (is_cv18xx) {
      cvi_int8_to_bf16(p.inputs[0], p.outputs[0], qtype.getScale(), num_elem,
                       is_tpu);
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] = dequant(p.inputs[0][i], qtype);
      }
    }
    //   } else if (isInQuant && isOutQuant)  {
    //     auto in_qtype = module::getUniformQuantizedType(getInput());
    //     auto out_qtype = module::getUniformQuantizedType(getOutput());
    //     if (in_qtype.getScale() == out_qtype.getScale() &&
    //         in_type.isInteger(8) && out_type.isInteger(8)) {
    //       int zero_diff = in_qtype.getZeroPoint() - out_qtype.getZeroPoint();
    //       if (zero_diff == 0) {
    //         std::copy(p.inputs[0], p.inputs[0] + num_elem, p.outputs[0]);
    //       } else {
    // #pragma omp parallel for schedule(static, omp_schedule(num_elem))
    //         for (int64_t i = 0; i < num_elem; i++) {
    //           p.outputs[0][i] = (p.inputs[0][i] - zero_diff);
    //         }
    //       }
    //     } else {
    //       int64_t multi, shift_val;
    //       QuantizeMultiplier(in_qtype.getScale() / out_qtype.getScale(),
    //       &multi, &shift_val); for (int64_t i = 0; i < num_elem; ++i) {
    //         auto v = out_qtype.getZeroPoint() +
    //         MultiplyByQuantizedMultiplier(
    //                                     (int32_t)(p.inputs[0][i]) -
    //                                     in_qtype.getZeroPoint(),
    //                                     (int32_t)multi, (int32_t)shift_val);
    //         p.outputs[0][i] = saturate(v, out_type);
    //       }
    //     }
  } else if (in_type.isF32() && out_type.isInteger(32)) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = round(p.inputs[0][i]);
    }
  } else {
    std::copy(p.inputs[0], p.inputs[0] + num_elem, p.outputs[0]);
  }

  return success();
}

mlir::Type tpu::CastOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return do_nothing(mode);
}

LogicalResult tpu::CastOp::LocalGenSupport() {
  if (module::isCV18xx()) {
    auto in_type = module::getStorageType(getInput());
    auto out_type = module::getStorageType(getOutput());
    int64_t n, c, h, w;
    module::getNCHW(getOutput(), n, c, h, w);
    if (c > MAX_TIU_CHL || w > MAX_TIU_CHL) {
      return failure();
    }
    // type.isSignedInteger()
    if ((in_type.getIntOrFloatBitWidth() == 8 && out_type.isBF16()) ||
        (in_type.isBF16() && out_type.isSignedInteger())) {
      return success();
    }
    return failure();
  }
  if (module::isBM1684Family()) {
    auto in_dtype = BM168x::getDataType(getInput());
    if (in_dtype == DTYPE_INT32) {
      return failure();
    }
  }
  return success();
}

void tpu::CastOp::assign_fw_param(void *param) {
  fw_dtype_convert_layer_param_t fw_param = {0};
  fw_param.src_type = BM168x::getDataType(getInput());
  fw_param.dst_type = BM168x::getDataType(getOutput());
  fw_param.src_stmode = BM1684::getStoreMode(getInput());
  fw_param.dst_stmode = BM1684::getStoreMode(getOutput());
  fw_param.round_mode = ROUND_INF; // if support other round_mode, change here
  memcpy(param, &fw_param, sizeof(fw_dtype_convert_layer_param_t));
}

void tpu::CastOp::assign_sec_info(int64_t n_step, int64_t c_step,
                                  int64_t h_step, int64_t d_step,
                                  int64_t w_step, group_type_t group_type,
                                  local_sec_info_t &sec_info) {
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;
  int64_t n, c, d, h, w, on, oc, od, oh, ow;
  auto input = getOperand();
  auto output = getResult();
  module::getNCDHW(input, n, c, d, h, w, group_type);
  module::getNCDHW(output, on, oc, od, oh, ow, group_type);
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(input, n_step, h_step, d_step,
                                               w_step, c_step);
  sec_info.n_slice = in_gi.n_slice;
  sec_info.d_slice = in_gi.d_slice;
  sec_info.h_slice = gi.h_slice;
  sec_info.w_slice = gi.w_slice;
  sec_info.c_slice = gi.c_slice;
  sec_info.n_idx = in_gi.n_idx;
  sec_info.d_idx = in_gi.d_idx;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == h);
  sec_info.w_idx = in_gi.w_idx;
  sec_info.is_w_split = !(in_gi.w_idx == 0 && in_gi.w_slice == w);
  sec_info.c_idx = gi.c_idx;
  sec_info.is_c_split = !(in_gi.c_idx == 0 && in_gi.c_slice == c);
  // set margins
  setHWMargins(sec_info.hw_margins_opdA, in_gi, gi);
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_idx = gi.w_idx;
  sec_info.out_w_slice = gi.w_slice;
}

ArrayAttr tpu::CastOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::CastOp::support_multi_core() { return false; }
