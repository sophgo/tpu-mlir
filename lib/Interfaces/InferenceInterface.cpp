//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "sophgo/Interfaces/InferenceInterface.h"
#include "sophgo/Support/Helper/Quant.h"

using namespace mlir;
using namespace sophgo::helper;

namespace sophgo {

int omp_schedule(int count) {
  return (count + omp_get_num_threads() - 1) / omp_get_num_threads();
}

void relu(float *src, float *dst, int64_t size, mlir::Type elem_type) {
#pragma omp parallel for schedule(static, omp_schedule(size))
  for (int64_t i = 0; i < size; ++i) {
    dst[i] = src[i] > 0 ? src[i] : 0;
    if (elem_type && elem_type.isInteger(8)) {
      //float max = elem_type.isUnsignedInteger(8) ? 255.0: 127.0;
      //dst[i] = std::min(dst[i], max);
      dst[i] = Quant::to_int8(dst[i]);
    }
  }
}

} // namespace sophgo

#include "sophgo/Interfaces/InferenceInterface.cpp.inc"
