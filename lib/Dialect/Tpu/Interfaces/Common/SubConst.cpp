//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::SubConstOp::init(InferenceParameter &p) { return success(); }

void tpu::SubConstOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SubConstOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  auto asym = module::isAsymmetric();

  auto do_relu = getDoRelu();
  auto relu_limit = getReluLimitAttr().getValueAsDouble();

  if (getIsReverse()) {
    if (in_type.isFloat8E4M3FN()) {
      double scale = getF8Scale().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] =
            getConstVal().convertToDouble() - p.inputs[0][i] * scale;
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        auto tmp_input =
            module::isUniformQuantized(getOutput())
                ? applyMultiplierAndRShift(p.inputs[0][i], getMultiplier(), 0)
                : p.inputs[0][i];
        p.outputs[0][i] = getConstVal().convertToDouble() - tmp_input;
      }
    }
  } else {
    if (in_type.isFloat8E4M3FN()) {
      double scale = getF8Scale().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] =
            p.inputs[0][i] * scale - getConstVal().convertToDouble();
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        auto tmp_input =
            module::isUniformQuantized(getOutput())
                ? applyMultiplierAndRShift(p.inputs[0][i], getMultiplier(), 0)
                : p.inputs[0][i];
        p.outputs[0][i] = tmp_input - getConstVal().convertToDouble();
      }
    }
  }
  if (do_relu) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = std::max(p.outputs[0][i], 0.0f);
    }
    if (relu_limit > 0) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] = std::min(p.outputs[0][i], (float)relu_limit);
      }
    }
  }
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isFloat8E4M3FN()) {
      F8E4M3(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
    } else if (out_type.isFloat8E5M2()) {
      F8E5M2(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
    }
  } else if (module::isUniformQuantized(getOutput())) {
    auto o_qtype = module::getUniformQuantizedType(getOutput());
    if (asym == false) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // coeff has been merge in multiplier&&rshift
        double sum = applyMultiplierAndRShift(p.outputs[0][i], 1, getRshift());
        if (getDoRelu() && sum < 0) {
          sum = 0;
        }
        p.outputs[0][i] = saturate(sum, out_type);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // inputs has been requant
        double sum = p.outputs[0][i] + o_qtype.getZeroPoint();
        if (getDoRelu() && sum < o_qtype.getZeroPoint()) {
          sum = o_qtype.getZeroPoint();
        }
        p.outputs[0][i] = saturate(sum, out_type);
      }
    }
  }
  return success();
}

void tpu::SubConstOp::assign_sec_info_kernel(
    group_type_t group_type, local_sec_info_t &sec_info,
    std::vector<group_info_t> &in_group_infos,
    std::vector<group_info_t> &out_group_infos) {
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;
  int64_t n, c, d, h, w, on, oc, od, oh, ow;
  auto input = getOperand();
  auto output = getResult();
  module::getNCDHW(input, n, c, d, h, w, group_type);
  module::getNCDHW(output, on, oc, od, oh, ow, group_type);
  auto gi = out_group_infos[0];
  auto in_gi = in_group_infos[0];
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

ArrayAttr tpu::SubConstOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::SubConstOp::support_multi_core() { return false; }
