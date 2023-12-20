//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/CastUtils.h"
#include "tpu_mlir/Support/Float16.h"

namespace tpu_mlir {

float requant(const float &data, const quant::UniformQuantizedType &qtype) {
  auto stype = qtype.getExpressedType();
  if (stype.isF32()) {
    return std::round(data * (float)(1.0 / qtype.getScale())) +
           qtype.getZeroPoint();
  }
  if (stype.isF16()) {
    return std::round(F16(data * F16(1.0 / qtype.getScale()))) +
           qtype.getZeroPoint();
  }
  if (stype.isBF16()) {
    return std::round(BF16(data * BF16(1.0 / qtype.getScale()))) +
           qtype.getZeroPoint();
  }
  qtype.dump();
  llvm_unreachable("Unsupport type");
}

float dequant(const float &data, const quant::UniformQuantizedType &qtype) {
  auto stype = qtype.getExpressedType();
  if (stype.isF32()) {
    return (float)qtype.getScale() * (data - (float)qtype.getZeroPoint());
  }
  if (stype.isF16()) {
    return F16(F16(qtype.getScale()) * F16(data - (float)qtype.getZeroPoint()));
  }
  if (stype.isBF16()) {
    return BF16(BF16(qtype.getScale()) *
                BF16(data - (float)qtype.getZeroPoint()));
  }
  qtype.dump();
  llvm_unreachable("Unsupport type");
}

} // namespace tpu_mlir
