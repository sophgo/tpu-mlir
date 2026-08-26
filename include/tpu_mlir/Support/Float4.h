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

float get_f4e2m1_max();

/*
convert f32 to f4e2m1 by uint8
*/
uint8_t f32_to_f4e2m1(float src);
float f4e2m1_to_f32(uint8_t src);

/*
  convert f32 to f8e4m3 and back to f32
*/
float F4E2M1(float src, float step);
void F4E2M1(const float *p_src, float *p_dst, int num, float step);

/*
  NVF4 two-level scaling:
  - global_scale = FP8_E4M3_MAX * FP4_E2M1_MAX / tensor_amax
  - per-block scale quantized to FP8 E4M3
*/
float compute_nvf4_global_scale(float tensor_amax);
float compute_nvf4_output_scale(float block_amax, float global_scale);
float NVF4_E4M3(float src, float block_amax, float global_scale);
void NVF4_E4M3(const float *p_src, float *p_dst, int num, float block_amax,
               float global_scale);

/*
  NVF4 UE5M3 two-level scaling:
  - global_scale = FP8_UE5M3_MAX * FP4_E2M1_MAX / tensor_amax
  - per-block scale quantized to unsigned FP8 UE5M3
*/
float compute_nvf4_ue5m3_global_scale(float tensor_amax);
float compute_nvf4_ue5m3_output_scale(float block_amax, float global_scale);
float NVF4_UE5M3(float src, float block_amax, float global_scale);
void NVF4_UE5M3(const float *p_src, float *p_dst, int num, float block_amax,
                float global_scale);

} // namespace tpu_mlir
