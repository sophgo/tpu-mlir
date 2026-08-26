#include "limits.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"

namespace tpu_mlir {

const std::vector<float> FP4_MAPPING = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,
                                        4.0,  6.0,  -0.0, -0.5, -1.0, -1.5,
                                        -2.0, -3.0, -4.0, -6.0};

uint8_t f32_to_f4e2m1(float src) {
  uint8_t nearest = 0;
  float min_diff = std::abs(src - FP4_MAPPING[0]);
  for (uint8_t i = 1; i < FP4_MAPPING.size(); i++) {
    float diff = std::abs(src - FP4_MAPPING[i]);
    if (diff < min_diff) {
      min_diff = diff;
      nearest = i;
    }
  }
  return nearest;
}

float f4e2m1_to_f32(uint8_t src) { return FP4_MAPPING[src]; }

float get_f4e2m1_max() { return float(6.0); }

float F4E2M1(float src, float step) {
  return f4e2m1_to_f32(f32_to_f4e2m1(src / step));
}

void F4E2M1(const float *p_src, float *p_dst, int num, float step) {
#pragma omp parallel for schedule(static, omp_schedule(num))
  for (int i = 0; i < num; i++) {
    p_dst[i] = F4E2M1(p_src[i], step);
  }
}

float compute_nvf4_global_scale(float tensor_amax) {
  if (tensor_amax == 0.0f)
    return 1.0f;
  return get_f8e4m3_max() * get_f4e2m1_max() / tensor_amax;
}

float compute_nvf4_output_scale(float block_amax, float global_scale) {
  float f4_max = get_f4e2m1_max();
  float f8_max = get_f8e4m3_max();
  float scale_raw =
      std::clamp(global_scale * block_amax / f4_max, -f8_max, f8_max);
  float fp8_scale = F8E4M3(scale_raw, 1.0, true);
  if (fp8_scale == 0.0f)
    return 1.0f;
  return 1.0f / (fp8_scale / global_scale);
}

float NVF4_E4M3(float src, float block_amax, float global_scale) {
  float f4_max = get_f4e2m1_max();
  float output_scale = compute_nvf4_output_scale(block_amax, global_scale);
  if (output_scale == 0.0f)
    return 0.0f;
  float scaled = std::clamp(src * output_scale, -f4_max, f4_max);
  return f4e2m1_to_f32(f32_to_f4e2m1(scaled));
}

void NVF4_E4M3(const float *p_src, float *p_dst, int num, float block_amax,
               float global_scale) {
  float f4_max = get_f4e2m1_max();
  float output_scale = compute_nvf4_output_scale(block_amax, global_scale);
  if (output_scale == 0.0f) {
    for (int i = 0; i < num; i++)
      p_dst[i] = 0.0f;
    return;
  }
#pragma omp parallel for schedule(static, omp_schedule(num))
  for (int i = 0; i < num; i++) {
    float scaled = std::clamp(p_src[i] * output_scale, -f4_max, f4_max);
    p_dst[i] = f4e2m1_to_f32(f32_to_f4e2m1(scaled));
  }
}

// =============================================================================
// NVF4 UE5M3 two-level scaling (unsigned FP8 UE5M3 scale, max=114688)
// =============================================================================

float compute_nvf4_ue5m3_global_scale(float tensor_amax) {
  if (tensor_amax == 0.0f)
    return 1.0f;
  return get_f8ue5m3_max() * get_f4e2m1_max() / tensor_amax;
}

float compute_nvf4_ue5m3_output_scale(float block_amax, float global_scale) {
  float f4_max = get_f4e2m1_max();
  float f8_max = get_f8ue5m3_max();
  float scale_raw =
      std::clamp(global_scale * block_amax / f4_max, 0.0f, f8_max);
  float fp8_scale = F8UE5M3(scale_raw, 1.0, true);
  if (fp8_scale == 0.0f)
    return 1.0f;
  return 1.0f / (fp8_scale / global_scale);
}

float NVF4_UE5M3(float src, float block_amax, float global_scale) {
  float f4_max = get_f4e2m1_max();
  float output_scale =
      compute_nvf4_ue5m3_output_scale(block_amax, global_scale);
  if (output_scale == 0.0f)
    return 0.0f;
  float scaled = std::clamp(src * output_scale, -f4_max, f4_max);
  return f4e2m1_to_f32(f32_to_f4e2m1(scaled));
}

void NVF4_UE5M3(const float *p_src, float *p_dst, int num, float block_amax,
                float global_scale) {
  float f4_max = get_f4e2m1_max();
  float output_scale =
      compute_nvf4_ue5m3_output_scale(block_amax, global_scale);
  if (output_scale == 0.0f) {
    for (int i = 0; i < num; i++)
      p_dst[i] = 0.0f;
    return;
  }
#pragma omp parallel for schedule(static, omp_schedule(num))
  for (int i = 0; i < num; i++) {
    float scaled = std::clamp(p_src[i] * output_scale, -f4_max, f4_max);
    p_dst[i] = f4e2m1_to_f32(f32_to_f4e2m1(scaled));
  }
}

} // namespace tpu_mlir
