//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::Yuv2rgbFormulaOp::init(InferenceParameter &p) {
  return success();
}
void tpu::Yuv2rgbFormulaOp::deinit(InferenceParameter &p) {}

typedef unsigned char u8;
#define CLIP(val, max, min)                                                    \
  val = (val <= min) ? min : val;                                              \
  val = (val >= max) ? max : val

typedef enum {
  FLOAT32,
  UINT8,
} image_data_format_ext;

typedef enum {
  FORMAT_MAPPING_YUV420P,
  FORMAT_MAPPING_YUV422P,
  FORMAT_MAPPING_YUV444P,
  FORMAT_MAPPING_NV12,
  FORMAT_MAPPING_NV21,
  FORMAT_MAPPING_NV16,
  FORMAT_MAPPING_NV61,
  FORMAT_MAPPING_NV24,
  FORMAT_MAPPING_RGB,
  FORMAT_MAPPING_BGR,
} kernel_image_format_t;

typedef enum {
  _601_limited,
  _601_full,
} formula_mode;

inline void YCrCb2RGB_601_limited(u8 y, u8 u, u8 v, float *r, float *g,
                                  float *b, bool isUINT8,
                                  RoundingMode round_mode) {
  float r_fp, g_fp, b_fp;
  int Y = (int)y - 16;
  int U = (int)u - 128;
  int V = (int)v - 128;
  r_fp = 1.16438 * Y + 1.59603 * V;
  g_fp = 1.16438 * Y - 0.39176 * U - 0.81297 * V;
  b_fp = 1.16438 * Y + 2.01723 * U;
  CLIP(r_fp, 255.0, 0.0);
  CLIP(g_fp, 255.0, 0.0);
  CLIP(b_fp, 255.0, 0.0);
  if (isUINT8) {
    r_fp = (u8)(to_uint8(r_fp, round_mode));
    g_fp = (u8)(to_uint8(g_fp, round_mode));
    b_fp = (u8)(to_uint8(b_fp, round_mode));
  }
  *r = r_fp;
  *g = g_fp;
  *b = b_fp;
}

inline void YCrCb2RGB_601_full(u8 y, u8 u, u8 v, float *r, float *g, float *b,
                               bool isUINT8, RoundingMode round_mode) {
  float r_fp, g_fp, b_fp;
  int Y = (int)y;
  int U = (int)u - 128;
  int V = (int)v - 128;
  r_fp = Y + 1.40189 * V;
  g_fp = Y - 0.34581 * U - 0.71490 * V;
  b_fp = Y + 1.77098 * U;
  CLIP(r_fp, 255.0, 0.0);
  CLIP(g_fp, 255.0, 0.0);
  CLIP(b_fp, 255.0, 0.0);
  if (isUINT8) {
    r_fp = (u8)(to_uint8(r_fp, round_mode));
    g_fp = (u8)(to_uint8(g_fp, round_mode));
    b_fp = (u8)(to_uint8(b_fp, round_mode));
  }
  *r = r_fp;
  *g = g_fp;
  *b = b_fp;
}

inline void YCrCb2RGB(u8 y, u8 u, u8 v, float *r, float *g, float *b,
                      formula_mode formula_mode, bool isUINT8,
                      RoundingMode round_mode) {
  if (formula_mode == _601_limited) {
    YCrCb2RGB_601_limited(y, u, v, r, g, b, isUINT8, round_mode);
  } else if (formula_mode == _601_full) {
    YCrCb2RGB_601_full(y, u, v, r, g, b, isUINT8, round_mode);
  }
}

LogicalResult tpu::Yuv2rgbFormulaOp::inference(InferenceParameter &p) {
  auto width = static_cast<unsigned int>(getWidth());
  auto height = static_cast<unsigned int>(getHeight());
  auto src_format = static_cast<unsigned int>(getSrcFormat());
  auto dst_format = static_cast<unsigned int>(getDstFormat());
  auto output_data_format =
      static_cast<image_data_format_ext>(getImageFormat());
  auto formula_mode = static_cast<::formula_mode>(getFormulaMode());
  auto round_mode = static_cast<RoundingMode>(getRoundMode());

  auto Y_shape = module::getShape(getY());
  assert(width == Y_shape[Y_shape.size() - 1]);
  assert(height == Y_shape[Y_shape.size() - 2]);
  size_t product = 1;
  for (auto it = Y_shape.begin(); it != Y_shape.end() - 2; it++) {
    product *= *it;
  }

  auto fail_flag = false;
  for (size_t n = 0; n < product; n++) {
#pragma omp parallel for schedule(static, omp_schedule(height *width))
    for (size_t i = 0; i < height * width; i++) {
      u8 y, u, v;
      size_t h_index = i / width;
      size_t w_index = i % width;
      size_t uv_h_index, u_w_index, v_w_index;
      if (src_format == FORMAT_MAPPING_NV12 ||
          src_format == FORMAT_MAPPING_NV21) {
        uv_h_index = h_index / 2;
        u_w_index = w_index / 2;
        v_w_index = w_index / 2;
      } else if (src_format == FORMAT_MAPPING_YUV420P) {
        uv_h_index = 0;
        u_w_index = w_index / 2 + h_index / 2 * width / 2;
        v_w_index = w_index / 2 + h_index / 2 * width / 2;
      } else {
        fail_flag = true;
        continue;
      }

      y = p.inputs[0][i + n * height * width];
      u = p.inputs[1]
                  [u_w_index + uv_h_index * u_w_index + n / 4 * height * width];
      v = p.inputs[2]
                  [v_w_index + uv_h_index * v_w_index + n / 4 * height * width];

      float *c0 = ((float *)p.outputs[0]) + n * height * width +
                  h_index * width + w_index;
      float *c1 = ((float *)p.outputs[0]) + (n + 1) * height * width +
                  h_index * width + w_index;
      float *c2 = ((float *)p.outputs[0]) + (n + 2) * height * width +
                  h_index * width + w_index;

      if (dst_format == FORMAT_MAPPING_RGB) {
        YCrCb2RGB(y, u, v, c0, c1, c2, formula_mode,
                  output_data_format == UINT8, round_mode);
      } else {
        YCrCb2RGB(y, u, v, c2, c1, c0, formula_mode,
                  output_data_format == UINT8, round_mode);
      }
      // std::cout << h_index << ", " << w_index << std::endl;
      // std::cout << *c2 << ", " << *c1 << ", " << *c0 << std::endl;
      // std::cout << "\n" << std::endl;
    }
    if (fail_flag)
      return failure();
  }
  return success();
}
