#include "limits.h"
#include "tpu_mlir/Support/Float16.h"
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

} // namespace tpu_mlir
