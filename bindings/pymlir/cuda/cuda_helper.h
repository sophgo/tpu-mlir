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
void cudaAddInt8(void *input0, void *input1, void *output, int mul0, int mul1,
                 int shift, int size);

void cudaConvInt8(void *input, void *filter, void *bias, void *output,
                  void *multipliers, void *shifts, int n, int ic, int ih,
                  int iw, int oc, int kh, int kw, int stride_h, int stride_w,
                  int pad_h, int pad_w);
void cudaMatMulF32(void *input, void *right, void *output, int m, int k, int n);
void cudaRequantInt8Perchannel(void *input, void *output, void *multipliers,
                               void *shifts, int n, int c, int h, int w,
                               bool out_sign, bool qdm = false,
                               bool relu = false);

void cudaRequantInt8(void *input, void *output, int32_t multiplier,
                     int32_t shift, int num, bool out_sign, bool qdm = false,
                     bool relu = false);

cudaError_t cudaTransform(void *src, void *dst, int size,
                          cudnnDataType_t src_type, cudnnDataType_t dst_type);

void cudaPrint(void *data, int size, cudnnDataType_t type);
