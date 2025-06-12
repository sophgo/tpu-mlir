//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::RequantIntOp::init(InferenceParameter &p) {
  return success();
}
void tpu::RequantIntOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RequantIntOp::inference(InferenceParameter &p) {
  auto o_sType = module::getStorageType(getOutput());
  auto o_qtype = module::getUniformQuantizedType(getOutput());
  auto mode = getQuantMode();
  auto shape = module::getShape(getOutput());
  int64_t inner = 1;
  for (int i = 2; i < shape.size(); ++i) {
    inner *= shape[i];
  }
  int64_t zp_x = 0;
  if (module::isUniformQuantized(getInput())) {
    auto i_qtype = module::getUniformQuantizedType(getInput());
    zp_x = i_qtype.getZeroPoint();
    assert(mode == tpu::RequantMode::MultiplierShift);
  }
  int64_t shift_val = -getRshift();
  int64_t multi = getMultiplier();
  int64_t zero_point = o_qtype.getZeroPoint();
  auto round_mode = round_mode_convert(getRoundMode());

  if (shape.size() == 1) {
    return success();
  }

  if (mode == tpu::RequantMode::TFLite_LShift ||
      mode == tpu::RequantMode::TFLite) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          auto v =
              zero_point + MultiplyByQuantizedMultiplier(
                               (int32_t)(p.inputs[0][offset]), (int32_t)multi,
                               (int32_t)shift_val, round_mode);
          p.outputs[0][offset] = saturate(v, o_sType);
        }
      }
    }
  } else if (mode == tpu::RequantMode::MultiplierShift) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          auto v =
              zero_point + applyMultiplierAndRShift(
                               (p.inputs[0][offset] - zp_x), multi, -shift_val,
                               tpu::RequantMode::MultiplierShift, round_mode);
          p.outputs[0][offset] = saturate(v, o_sType);
        }
      }
    }
  }
  return success();
}

mlir::Type tpu::RequantIntOp::type_verify(uint64_t opd_idx,
                                          TypeCastMode &mode) {
  if (opd_idx == 0) {
    auto op = getOperation();
    auto stype = module::getStorageType(getInput());
    if (stype.isIntOrIndex()) {
      return do_nothing(mode);
    }
    if ((stype.isF32() || stype.isF16() || stype.isBF16()) &&
        module::isUniformQuantized(getOutput())) {
      mode = TypeCastMode::DO_QUANTIZE;
      return module::getStorageType(getOutput());
    }
    mode = TypeCastMode::DO_CAST;
    auto bitwith = stype.getIntOrFloatBitWidth();
    return Builder(op).getIntegerType(bitwith);
  }
  return do_nothing(mode);
}

void tpu::RequantIntOp::DumpQuantAgnosticAttrs(llvm::raw_string_ostream &os) {
  for (auto attr : getOperation()->getAttrs()) {
    auto attr_name = attr.getName().str();
    if (attr_name == "ginfo" || attr_name == "multiplier" || attr_name == "rshift") {
      continue;
    }
    os << attr_name << "=";
    attr.getValue().print(os);
    os << "; ";
  }
  auto rshift_v = getRshift();
  auto multiplier_v = getMultiplier();
  if (rshift_v == 0) {
    // do-nothing.
  } else {
    os << "rshift_len=1; ";
  }
  if (multiplier_v == 1) {
    // do-nothing.
  } else {
    os << "multiplier=1; ";
  }
}

ArrayAttr tpu::RequantIntOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::RequantIntOp::support_multi_core() { return false; }
