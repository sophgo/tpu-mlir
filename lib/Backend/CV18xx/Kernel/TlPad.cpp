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

#define DEBUG_TYPE "tl_pad"

namespace tpu_mlir {
namespace backend {
void cvi_backend_tl_pad(uint32_t layer_id, int64_t *input_dim,
                        int64_t *output_dim, laddr_t la_input,
                        laddr_t la_output, float const_val, int32_t *pads) {

  LLVM_DEBUG(llvm::errs() << llvm::format("cvi_backend_tl_pad:\n"
                                          "  layer_id %d\n",
                                          layer_id));

  LLVM_DEBUG(llvm::errs() << llvm::format("la_input:%d\n", la_input));

  LLVM_DEBUG(llvm::errs() << llvm::format("la_output:%d\n", la_output));

  uint32_t in = input_dim[0];
  uint32_t ic = input_dim[1];
  uint32_t ih = input_dim[2];
  uint32_t iw = input_dim[3];

  uint32_t on = output_dim[0];
  uint32_t oc = output_dim[1];
  uint32_t oh = output_dim[2];
  uint32_t ow = output_dim[3];

  assert(on == pads[0] + pads[4] + in);
  assert(oc == pads[1] + pads[5] + ic);
  assert(oh == pads[2] + pads[6] + ih);
  assert(ow == pads[3] + pads[7] + iw);

  cvk_tl_shape_t input_shape = {in, ic, ih, iw};

  cvk_tl_shape_t output_shape = {on, oc, oh, ow};

  auto output_offset = output_shape.w * pads[2] + pads[3];

  uint32_t out_addr = la_output + output_offset;
  cvk_tl_t tl_input = {};
  tl_input.start_address = la_input;
  tl_input.fmt = CVK_FMT_I8;
  tl_input.shape = input_shape;
  tl_input.stride = CV18xx::tl_default_stride(input_shape, CVK_FMT_I8, 1);

  cvk_tl_t tl_output;
  tl_output.start_address = la_output;
  tl_output.fmt = CVK_FMT_I8;
  tl_output.shape = output_shape;
  tl_output.stride = CV18xx::tl_default_stride(output_shape, CVK_FMT_I8, 1);

  cvk_tdma_g2l_tensor_fill_constant_param_t p1 = {0};
  p1.constant = const_val;
  p1.dst = &tl_output;
  p1.layer_id = layer_id;
  CV18xx::parallel_disable();
  CV18xx::tdma_g2l_tensor_fill_constant(&p1);

  tl_output.start_address = out_addr;
  tl_output.shape = input_shape;
  cvk_tdma_l2l_tensor_copy_param_t p2 = {0};
  p2.dst = &tl_output;
  p2.src = &tl_input;
  p2.layer_id = layer_id;
  CV18xx::tdma_l2l_tensor_copy(&p2);
  CV18xx::parallel_enable();
}

void cvi_backend_tl_bf16_pad(uint32_t layer_id, int64_t *input_dim,
                             int64_t *output_dim, laddr_t la_input,
                             laddr_t la_output, float const_val,
                             int32_t *pads) {

  LLVM_DEBUG(llvm::errs() << llvm::format("cvi_backend_tl_pad:\n"
                                          "  layer_id %d\n",
                                          layer_id));

  LLVM_DEBUG(llvm::errs() << llvm::format("la_input:%d\n", la_input));

  LLVM_DEBUG(llvm::errs() << llvm::format("la_output:%d\n", la_output));

  uint32_t in = input_dim[0];
  uint32_t ic = input_dim[1];
  uint32_t ih = input_dim[2];
  uint32_t iw = input_dim[3];

  uint32_t on = output_dim[0];
  uint32_t oc = output_dim[1];
  uint32_t oh = output_dim[2];
  uint32_t ow = output_dim[3];

  assert(on == pads[0] + pads[4] + in);
  assert(oc == pads[1] + pads[5] + ic);
  assert(oh == pads[2] + pads[6] + ih);
  assert(ow == pads[3] + pads[7] + iw);

  cvk_tl_shape_t input_shape = {in, ic, ih, iw};

  cvk_tl_shape_t output_shape = {on, oc, oh, ow};

  auto output_offset = (output_shape.w * pads[2] + pads[3]) * sizeof(uint16_t);

  uint32_t out_addr = la_output + output_offset;
  cvk_tl_t tl_input = {};
  tl_input.start_address = la_input;
  tl_input.fmt = CVK_FMT_BF16;
  tl_input.shape = input_shape;
  tl_input.stride = CV18xx::tl_default_stride(input_shape, CVK_FMT_BF16, 1);

  cvk_tl_t tl_output = {};
  tl_output.start_address = la_output;
  tl_output.fmt = CVK_FMT_BF16;
  tl_output.shape = output_shape;
  tl_output.stride = CV18xx::tl_default_stride(output_shape, CVK_FMT_BF16, 1);

  cvk_tdma_g2l_tensor_fill_constant_param_t p1 = {0};
  p1.constant = CV18xx::convert_fp32_to_bf16(const_val);
  p1.dst = &tl_output;
  p1.layer_id = layer_id;
  CV18xx::parallel_disable();
  CV18xx::tdma_g2l_tensor_fill_constant(&p1);

  tl_output.start_address = out_addr;
  tl_output.shape = input_shape;
  cvk_tdma_l2l_tensor_copy_param_t p2 = {0};
  p2.dst = &tl_output;
  p2.src = &tl_input;
  p2.layer_id = layer_id;
  CV18xx::tdma_l2l_tensor_copy(&p2);
  CV18xx::parallel_enable();
}
} // namespace backend
} // namespace tpu_mlir
