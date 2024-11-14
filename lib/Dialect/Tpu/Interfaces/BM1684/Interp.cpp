//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;

void tpu::InterpOp::codegen_global_bm1684() {
  uint64_t bottom_global_offset = module::getAddress(getInput());
  uint64_t top_global_offset = module::getAddress(getResult());

  int64_t in, ic, ih, iw;
  module::getNCHW(getInput(), in, ic, ih, iw, true);
  int64_t on, oc, oh, ow;
  module::getNCHW(getOutput(), on, oc, oh, ow, true);

  // >>>>the followed param are Aligned with nntc bmneto's interp_layer.py
  int pad_bag = 0, pad_end = 0;
  bool align_corners, half_pixel_centers = 0;
  PLATFORM_SUPPORT platform_sp;
  if (getMode() == tpu::ResizeMode::linear) {
    if (getCoordMode() == tpu::ResizeCoordMode::half_pixel) {
      platform_sp = TENSORFLOW_SUPPORT;
      half_pixel_centers = 1;
      align_corners = 0;
    } else if (getCoordMode() == tpu::ResizeCoordMode::pytorch_half_pixel) {
      platform_sp = PYTORCH_SUPPORT;
      half_pixel_centers = 0;
      align_corners = 0;
    } else if (getCoordMode() == tpu::ResizeCoordMode::align_corners) {
      platform_sp = TENSORFLOW_SUPPORT;
      half_pixel_centers = 0;
      align_corners = 1;
    } else {
      llvm_unreachable("BM1684 DO NOT Support Such Attribute.!!!!");
    }
  } else if (getMode() == tpu::ResizeMode::nearest) {
    if (getCoordMode() == tpu::ResizeCoordMode::half_pixel) {
      platform_sp = ONNX_NEAREST;
      half_pixel_centers = 1;
      align_corners = 0;
    } else if (getCoordMode() == tpu::ResizeCoordMode::pytorch_half_pixel) {
      platform_sp = PYTORCH_NEAREST;
      half_pixel_centers = 1;
      align_corners = 0;
    } else if (getCoordMode() == tpu::ResizeCoordMode::asymmetric) {
      platform_sp = ONNX_NEAREST;
      half_pixel_centers = 0;
      align_corners = 0;
    } else {
      llvm_unreachable("BM1684 DO NOT Support Such Attribute.!!!!");
    }
  }
  // <<<<

  BM1684::instance().dl_nodechip_interp_forward_parallel(
      bottom_global_offset, top_global_offset, in, ic, ih, iw, pad_bag, pad_end,
      oh, ow, align_corners, half_pixel_centers, platform_sp,
      (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
}

uint32_t tpu::InterpOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::InterpOp::get_fw_type_bm1684() { return -1; }
