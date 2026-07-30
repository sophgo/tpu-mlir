//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace tpu_mlir {

// Portable equivalent of llama.cpp nearest_int (ggml-quants.c:563).
static inline int mse_nearest_int(float fval) { return (int)std::round(fval); }

float make_qx_quants(int n, int nmax, const float *x, int8_t *L, int rmse_type,
                     const float *qw) {
  float max = 0;
  float amax = 0;
  for (int i = 0; i < n; ++i) {
    float ax = fabsf(x[i]);
    if (ax > amax) {
      amax = ax;
      max = x[i];
    }
  }
  if (amax < 1e-15f) { // all zero
    for (int i = 0; i < n; ++i) {
      L[i] = 0;
    }
    return 0.f;
  }
  float iscale = -nmax / max;
  if (rmse_type == 0) {
    for (int i = 0; i < n; ++i) {
      int l = mse_nearest_int(iscale * x[i]);
      L[i] = nmax + std::max(-nmax, std::min(nmax - 1, l));
    }
    return 1 / iscale;
  }
  bool return_early = false;
  if (rmse_type < 0) {
    rmse_type = -rmse_type;
    return_early = true;
  }
  float sumlx = 0;
  float suml2 = 0;
  for (int i = 0; i < n; ++i) {
    int l = mse_nearest_int(iscale * x[i]);
    l = std::max(-nmax, std::min(nmax - 1, l));
    L[i] = l + nmax;
    float w = qw               ? qw[i]
              : rmse_type == 1 ? x[i] * x[i]
              : rmse_type == 2 ? 1.f
              : rmse_type == 3 ? fabsf(x[i])
                               : sqrtf(fabsf(x[i]));
    sumlx += w * x[i] * l;
    suml2 += w * l * l;
  }
  float scale = suml2 ? sumlx / suml2 : 0.0f;
  if (return_early)
    return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
  float best = scale * sumlx;
  for (int is = -9; is <= 9; ++is) {
    if (is == 0) {
      continue;
    }
    iscale = -(nmax + 0.1f * is) / max;
    sumlx = suml2 = 0;
    for (int i = 0; i < n; ++i) {
      int l = mse_nearest_int(iscale * x[i]);
      l = std::max(-nmax, std::min(nmax - 1, l));
      float w = qw               ? qw[i]
                : rmse_type == 1 ? x[i] * x[i]
                : rmse_type == 2 ? 1.f
                : rmse_type == 3 ? fabsf(x[i])
                                 : sqrtf(fabsf(x[i]));
      sumlx += w * x[i] * l;
      suml2 += w * l * l;
    }
    if (suml2 > 0 && sumlx * sumlx > best * suml2) {
      for (int i = 0; i < n; ++i) {
        int l = mse_nearest_int(iscale * x[i]);
        L[i] = nmax + std::max(-nmax, std::min(nmax - 1, l));
      }
      scale = sumlx / suml2;
      best = scale * sumlx;
    }
  }
  return scale;
}

bool mse_quant_enabled() {
  static bool enabled = (getenv("TPU_MLIR_USE_MSE") != nullptr);
  return enabled;
}

} // namespace tpu_mlir
