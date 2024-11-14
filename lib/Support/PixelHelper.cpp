//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/PixelHelper.h"
#include "tpu_mlir/Support/Module.h"
namespace tpu_mlir {
// namespace pixel_helper {
void setPixelAlign(std::string &pixel_format, int64_t &y_align,
                   int64_t &w_align, int64_t &channel_align) {
  if (module::isChip(module::Chip::CV183x)) {
    y_align = 32;
    w_align = 32;
    channel_align = 0x1000;
  } else {
    y_align = 64;
    w_align = 64;
    channel_align = 64;
  }
  if ("YUV420_PLANAR" == pixel_format) {
    y_align = w_align * 2;
  }
}

void setPixelAlign(std::string &pixel_format, std::string chip_name,
                   int64_t &y_align, int64_t &w_align, int64_t &channel_align) {
  if ("cv183x" == chip_name || "CV183X" == chip_name) {
    y_align = 32;
    w_align = 32;
    channel_align = 0x1000;
  } else {
    y_align = 64;
    w_align = 64;
    channel_align = 64;
  }
  if ("YUV420_PLANAR" == pixel_format) {
    y_align = w_align * 2;
  }
}

int aligned_image_size(int n, int c, int h, int w, std::string &pixel_format,
                       int y_align, int w_align, int channel_align) {
  if ("YUV420_PLANAR" == pixel_format) {
    assert(c == 3);
    int y_w_aligned = align_up(w, y_align);
    int uv_w_aligned = align_up(w / 2, w_align);
    int u = align_up(h * y_w_aligned, channel_align);
    int v = align_up(u + h / 2 * uv_w_aligned, channel_align);
    int n_stride = align_up(v + h / 2 * uv_w_aligned, channel_align);
    return n * n_stride;
  } else if ("YUV_NV21" == pixel_format || "YUV_NV12" == pixel_format) {
    int y_w_aligned = align_up(w, y_align);
    int uv_w_aligned = align_up(w, w_align);
    int uv = align_up(h * y_w_aligned, channel_align);
    int n_stride = align_up(uv + h / 2 * uv_w_aligned, channel_align);
    return n * n_stride;
  } else if ("RGB_PLANAR" == pixel_format || "BGR_PLANAR" == pixel_format ||
             "RGBA_PLANAR" == pixel_format) {
    int aligned_w = align_up(w, w_align);
    int n_stride = align_up(aligned_w * h, channel_align) * c;
    return n * n_stride;
  } else if ("RGB_PACKED" == pixel_format || "BGR_PACKED" == pixel_format ||
             "GRAYSCALE" == pixel_format) {
    int aligned_w = align_up(w * c, w_align);
    int n_stride = aligned_w * h;
    return n * n_stride;
  } else {
    assert(0 && "unsupported pixel format");
    return 0;
  }
}
//} // namespace pixel_helper
} // namespace tpu_mlir
