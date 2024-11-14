//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include <malloc.h>
#include <string.h>

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
  FORMAT_MAPPING_YUV420P_YU12,
  FORMAT_MAPPING_YUV420P_YV12,
  FORMAT_MAPPING_NV12,
  FORMAT_MAPPING_NV21,
  FORMAT_MAPPING_RGB,
  FORMAT_MAPPING_BGR,
} kernel_image_format_t;

typedef enum {
  _601_limited,
  _601_full,
} formula_mode;

inline void YCrCb2RGB_601_limited(u8 y, u8 u, u8 v, float *r, float *g,
                                  float *b, bool isUINT8,
                                  tpu_mlir::RoundingMode round_mode) {
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
                               bool isUINT8,
                               tpu_mlir::RoundingMode round_mode) {
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
                      tpu_mlir::RoundingMode round_mode) {
  if (formula_mode == _601_limited) {
    YCrCb2RGB_601_limited(y, u, v, r, g, b, isUINT8, round_mode);
  } else if (formula_mode == _601_full) {
    YCrCb2RGB_601_full(y, u, v, r, g, b, isUINT8, round_mode);
  }
}

std::vector<float *> split_YUV(float *yuv, unsigned int src_format,
                               unsigned int width, unsigned int height) {
  std::vector<float *> yuv_split;
  float *y = yuv;
  float *u;
  float *v;

  if (src_format == FORMAT_MAPPING_YUV420P_YU12) {
    u = yuv + width * height;
    v = yuv + width * height + width * height / 4;
  } else if (src_format == FORMAT_MAPPING_YUV420P_YV12) {
    v = yuv + width * height;
    u = yuv + width * height + width * height / 4;
  } else if (src_format == FORMAT_MAPPING_NV12) {
    u = yuv + width * height;
    float *u_buf = (float *)malloc(sizeof(float) * width * height / 4);
    float *v_buf = (float *)malloc(sizeof(float) * width * height / 4);
    for (int i = 0; i < width * height / 4; i++) {
      memcpy(u_buf + i, u + 2 * i, sizeof(float));
      memcpy(v_buf + i, u + 2 * i + 1, sizeof(float));
    }
    u = u_buf;
    v = v_buf;
  } else if (src_format == FORMAT_MAPPING_NV21) {
    u = yuv + width * height;
    float *u_buf = (float *)malloc(sizeof(float) * width * height / 4);
    float *v_buf = (float *)malloc(sizeof(float) * width * height / 4);
    for (int i = 0; i < width * height / 4; i++) {
      memcpy(v_buf + i, u + 2 * i, sizeof(float));
      memcpy(u_buf + i, u + 2 * i + 1, sizeof(float));
    }
    u = u_buf;
    v = v_buf;
  }
  yuv_split.emplace_back(y);
  yuv_split.emplace_back(u);
  yuv_split.emplace_back(v);
  return yuv_split;
}

LogicalResult tpu::Yuv2rgbFormulaOp::inference(InferenceParameter &p) {
  // width and height must be even!
  // auto width = static_cast<unsigned int>(getWidth());
  // auto height = static_cast<unsigned int>(getHeight());
  auto YUV_shape = module::getShape(getYUV());
  auto width = static_cast<unsigned int>(YUV_shape[YUV_shape.size() - 1]);
  auto height =
      static_cast<unsigned int>(YUV_shape[YUV_shape.size() - 2] * 2 / 3);
  auto src_format = static_cast<unsigned int>(getSrcFormat());
  auto dst_format = static_cast<unsigned int>(getDstFormat());
  auto output_data_format =
      static_cast<image_data_format_ext>(getImageFormat());
  auto formula_mode = static_cast<::formula_mode>(getFormulaMode());
  auto round_mode = static_cast<tpu_mlir::RoundingMode>(getRoundMode());

  size_t product = 1;
  for (auto it = YUV_shape.begin(); it != YUV_shape.end() - 2; it++) {
    product *= *it;
  }

  auto fail_flag = false;
  std::vector<float *> yuv_split;
  float *current_batch_y, *current_batch_u, *current_batch_v;
  float *y_begin = p.inputs[0];
  float *rgb_begin = p.outputs[0];
  for (size_t n = 0; n < product; n++) {
    yuv_split = split_YUV(y_begin, src_format, width, height);
    current_batch_y = yuv_split[0];
    current_batch_u = yuv_split[1];
    current_batch_v = yuv_split[2];
    // #pragma omp parallel for schedule(static, omp_schedule(height *width))
    for (size_t i = 0; i < height * width; i++) {
      u8 y, u, v;
      size_t h_index = i / width;
      size_t w_index = i % width;

      size_t y_offset = i;
      size_t u_offset, v_offset;
      if (src_format == FORMAT_MAPPING_YUV420P_YU12 ||
          src_format == FORMAT_MAPPING_YUV420P_YV12 ||
          src_format == FORMAT_MAPPING_NV12 ||
          src_format == FORMAT_MAPPING_NV21) {
        u_offset = w_index / 2 + h_index / 2 * width / 2;
        v_offset = w_index / 2 + h_index / 2 * width / 2;
      } else {
        fail_flag = true;
        continue;
      }

      y = current_batch_y[y_offset];
      u = current_batch_u[u_offset];
      v = current_batch_v[v_offset];

      float *c0 = ((float *)rgb_begin) + h_index * width + w_index;
      float *c1 =
          ((float *)rgb_begin) + height * width + h_index * width + w_index;
      float *c2 =
          ((float *)rgb_begin) + 2 * height * width + h_index * width + w_index;

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
    y_begin += width * height * 3 / 2;
    rgb_begin += 3 * width * height;
    if (fail_flag)
      return failure();
  }
  return success();
}

bool tpu::Yuv2rgbFormulaOp::support_multi_core() { return false; }
