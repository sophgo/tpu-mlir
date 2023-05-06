//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "tl_pooling"

namespace tpu_mlir {
namespace backend {
void cvi_backend_tl_pooling(uint32_t layer_id, laddr_t ifmap_laddr,
                            laddr_t ofmap_laddr, int input_n, int input_c,
                            int input_h, int input_w, int output_n,
                            int output_c, int output_h, int output_w,
                            uint32_t kh, uint32_t kw, uint32_t stride_h,
                            uint32_t stride_w, uint32_t pad_h_top,
                            uint32_t pad_h_bottom, uint32_t pad_w_left,
                            uint32_t pad_w_right, bool is_avg_pooling,
                            int8_t rshift, int8_t m_i8) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "cvi_backend_tl_pooling:\n"
                 "    ifmap_laddr 0x%lx, ofmap_laddr 0x%lx\n"
                 "    in(%d, %d, %d, %d), out(%d, %d, %d, %d)\n"
                 "    kernel(%d, %d), stride(%d, %d), pad(%d, %d, %d, %d)\n"
                 "    is_avg_pooling %d, rshift %d, m_i8 %d\n",
                 ifmap_laddr, ofmap_laddr, input_n, input_c, input_h, input_w,
                 output_n, output_c, output_h, output_w, kw, kh, stride_h,
                 stride_w, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
                 is_avg_pooling, rshift, m_i8));

  // Input feature map with assigned local memory address
  cvk_tl_shape_t ifmap_shape =
      CV18xx::tl_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_t ifmap;
  ifmap.start_address = ifmap_laddr;
  ifmap.fmt = CVK_FMT_I8;
  ifmap.shape = ifmap_shape;
  ifmap.stride =
      CV18xx::tl_default_stride(ifmap_shape, CVK_FMT_I8, 1); // EU-aligned

  // Output feature map with assigned local memory address
  cvk_tl_shape_t ofmap_shape =
      CV18xx::tl_shape_t4(output_n, output_c, output_h, output_w);
  cvk_tl_t ofmap;
  ofmap.start_address = ofmap_laddr;
  ofmap.fmt = CVK_FMT_I8;
  ofmap.shape = ofmap_shape;
  ofmap.stride =
      CV18xx::tl_default_stride(ofmap_shape, CVK_FMT_I8, 1); // EU-aligned

  if (!is_avg_pooling) {
    // Max pooling
    cvk_tiu_max_pooling_param_t param = {0};
    param.ofmap = &ofmap;
    param.ifmap = &ifmap;
    param.kh = kh;
    param.kw = kw;
    param.pad_top = pad_h_top;
    param.pad_bottom = pad_h_bottom;
    param.pad_left = pad_w_left;
    param.pad_right = pad_w_right;
    param.stride_h = stride_h;
    param.stride_w = stride_w;
    param.layer_id = layer_id;
    param.ins_val = -128;
    param.ins_fp = 0xff7f;
    CV18xx::tiu_max_pooling(&param);
  } else {
    cvk_tiu_average_pooling_param_t param = {0};
    param.ofmap = &ofmap;
    param.ifmap = &ifmap;
    param.kh = kh;
    param.kw = kw;
    param.ins_h = 0;
    param.ins_last_h = 0;
    param.ins_w = 0;
    param.ins_last_w = 0;
    param.pad_top = pad_h_top;
    param.pad_bottom = pad_h_bottom;
    param.pad_left = pad_w_left;
    param.pad_right = pad_w_right;
    param.stride_h = stride_h;
    param.stride_w = stride_w;
    param.avg_pooling_const = m_i8;
    param.rshift_bits = rshift;
    param.layer_id = layer_id;
    param.ins_val = 0;
    param.ins_fp = CV18xx::convert_fp32_to_bf16(0.0);
    CV18xx::tiu_average_pooling(&param);
  }
}

void cvi_backend_tl_bf16_pooling(uint32_t layer_id, laddr_t ifmap_laddr,
                                 laddr_t ofmap_laddr, int input_n, int input_c,
                                 int input_h, int input_w, int output_n,
                                 int output_c, int output_h, int output_w,
                                 uint32_t kh, uint32_t kw, uint32_t stride_h,
                                 uint32_t stride_w, uint32_t pad_h_top,
                                 uint32_t pad_h_bottom, uint32_t pad_w_left,
                                 uint32_t pad_w_right, bool is_avg_pooling) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "cvi_backend_tl_bf16_pooling:\n"
                 "    ifmap_laddr 0x%lx, ofmap_laddr 0x%lx\n"
                 "    in(%d, %d, %d, %d), out(%d, %d, %d, %d)\n"
                 "    kernel(%d, %d), stride(%d, %d), pad(%d, %d, %d, %d)\n"
                 "    is_avg_pooling %d\n",
                 ifmap_laddr, ofmap_laddr, input_n, input_c, input_h, input_w,
                 output_n, output_c, output_h, output_w, kw, kh, stride_h,
                 stride_w, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
                 is_avg_pooling));

  // Input feature map with assigned local memory address
  cvk_tl_shape_t ifmap_shape =
      CV18xx::tl_shape_t4(input_n, input_c, input_h, input_w);
  cvk_tl_t ifmap;
  ifmap.start_address = ifmap_laddr;
  ifmap.fmt = CVK_FMT_BF16;
  ifmap.shape = ifmap_shape;
  ifmap.stride =
      CV18xx::tl_default_stride(ifmap_shape, CVK_FMT_BF16, 1); // EU-aligned

  // Output feature map with assigned local memory address
  cvk_tl_shape_t ofmap_shape =
      CV18xx::tl_shape_t4(output_n, output_c, output_h, output_w);
  cvk_tl_t ofmap;
  ofmap.start_address = ofmap_laddr;
  ofmap.fmt = CVK_FMT_BF16;
  ofmap.shape = ofmap_shape;
  ofmap.stride =
      CV18xx::tl_default_stride(ofmap_shape, CVK_FMT_BF16, 1); // EU-aligned

  if (!is_avg_pooling) {
    // Max pooling
    cvk_tiu_max_pooling_param_t param = {0};
    param.ofmap = &ofmap;
    param.ifmap = &ifmap;
    param.kh = kh;
    param.kw = kw;
    param.pad_top = pad_h_top;
    param.pad_bottom = pad_h_bottom;
    param.pad_left = pad_w_left;
    param.pad_right = pad_w_right;
    param.stride_h = stride_h;
    param.stride_w = stride_w;
    param.layer_id = layer_id;
    param.ins_val = -128;
    param.ins_fp = 0xff7f; // CV18xx::convert_fp32_to_bf16(0.0);
    CV18xx::tiu_max_pooling(&param);

  } else {
    cvk_tiu_average_pooling_param_t param = {0};
    param.ofmap = &ofmap;
    param.ifmap = &ifmap;
    param.kh = kh;
    param.kw = kw;
    param.ins_h = 0;
    param.ins_last_h = 0;
    param.ins_w = 0;
    param.ins_last_w = 0;
    param.pad_top = pad_h_top;
    param.pad_bottom = pad_h_bottom;
    param.pad_left = pad_w_left;
    param.pad_right = pad_w_right;
    param.stride_h = stride_h;
    param.stride_w = stride_w;
    param.layer_id = layer_id;
    param.avg_pooling_const = CV18xx::convert_fp32_to_bf16(1.0);
    param.ins_val = 0;
    param.ins_fp = CV18xx::convert_fp32_to_bf16(0.0);
    CV18xx::tiu_average_pooling(&param);
  }
}

} // namespace backend
} // namespace tpu_mlir
