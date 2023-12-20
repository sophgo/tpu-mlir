//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

namespace tpu_mlir {

float requant(const float &data, const quant::UniformQuantizedType &qtype);

float dequant(const float &data, const quant::UniformQuantizedType &qtype);

} // namespace tpu_mlir
