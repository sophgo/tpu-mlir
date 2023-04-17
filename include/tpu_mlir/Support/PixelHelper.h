//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <assert.h>
#include <string>
#include <atomic>
#include <fstream>

namespace tpu_mlir {

//namespace pixel_helper {

inline int align_up(int x, int n) {
  if (n <= 0) {
    return x;
  }
  return (x + n - 1) & ~(n - 1);
}

void setPixelAlign(std::string &pixel_format,
                   int64_t &y_align, int64_t &w_align,
                   int64_t &channel_align);

void setPixelAlign(std::string &pixel_format, std::string chip,
                   int64_t &y_align, int64_t &w_align,
                   int64_t &channel_align);

int aligned_image_size(int n, int c, int h, int w, std::string &pixel_format,
               int y_align, int w_align, int channel_align);

//} // namespace pixel_helper
} // namespace tpu_mlir
