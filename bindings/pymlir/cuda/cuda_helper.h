//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cudnn.h>
#include <cuda_runtime.h>

// mode 0: 1.5 -> 2   -1.5 -> -2
// mode refer to RoundingMode defined in MathUtils.h
void cudaQuantizeToInt8_0(void *input, void *output, float scale, int size);

void cudaScaleToF32(void *input, void *output, float scale,
                    int size); // for bm168x

void cudaCVScaleToF32(void *input, void *output, float scale,
                      int size); // for cv18xx

// for cv18xx add: (int8 * int32 + int8 * int32) >> shift = int8 (half up)
void cudaAddInt8(void *input0, void *input1, void *output, int mul0,
                 int mul1, int shift, int size);
