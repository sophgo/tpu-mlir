//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>

namespace tpu_mlir {
/*
Convert an Activation tensor into INT8 / UINT8
For cv18xx
*/
int8_t cvi_f32_to_int8(float v, int round_mode);
uint8_t cvi_f32_to_uint8(float v, int round_mode);

void cvi_f32_to_int8(float *p_src, float *p_dst,
                    float scale, int zero_point,
                    int num, bool is_unsigned,
                    bool is_tpu=true);
} // namespace tpu_mlir
