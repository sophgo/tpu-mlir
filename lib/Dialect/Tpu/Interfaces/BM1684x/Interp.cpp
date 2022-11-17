//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
  CAFFE_SUPPORT = 0,
  TENSORFLOW_SUPPORT = 1,
  CAFFE_NEAREST = 2,
  TENSORFLOW_NEAREST = 3,
  PYTORCH_SUPPORT = 4,
  PYTORCH_NEAREST = 5,
  OPENCV_BILINEAR = 6,
  ONNX_NEAREST = 7,
} PLATFORM_SUPPORT;

typedef struct {
    unsigned int       input_addr;
    unsigned int       output_addr;
    int                input_n;
    int                input_c;
    int                input_h;
    int                input_w;
    int                output_h;
    int                output_w;
    int                pad_bag;
    int                pad_end;
    bool               align_corners;
    bool               half_pixel_centers;
    PLATFORM_SUPPORT   platform_sp;
    DATA_TYPE_T        dtype;
} interp_local_param_t;

typedef struct {
    unsigned long long input_addr;
    unsigned long long output_addr;
    int                input_n;
    int                input_c;
    int                input_h;
    int                input_w;
    int                output_h;
    int                output_w;
    int                pad_bag;
    int                pad_end;
    bool               align_corners;
    bool               half_pixel_centers;
    PLATFORM_SUPPORT   platform_sp;
    DATA_TYPE_T        dtype;
} interp_global_param_t;
#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================
void tpu::InterpOp::codegen_global_bm1684x() {
  int64_t n, c, ih, iw, oh, ow;
  Module::getNCHW(input(), n, c, ih, iw);
  interp_global_param_t param = {0};
  param.input_addr = Module::getAddress(input());
  param.output_addr = Module::getAddress(output());
  param.input_n = n;
  param.input_c = c;
  param.input_h = ih;
  param.input_w = iw;
  Module::getNCHW(output(), n, c, oh, ow);
  param.output_h = oh;
  param.output_w = ow;
  param.pad_bag = 0;
  param.pad_end = 0;
  param.dtype = BM168x::getDataType(input());
  int coord = 0;
  bool align_corners = (coord_mode() == tpu::ResizeCoordMode::align_corners);
  bool half_pixel = (coord_mode() == tpu::ResizeCoordMode::half_pixel);
  if (coord_mode() == tpu::ResizeCoordMode::half_pixel)
    coord = 0;
  else if (coord_mode() == tpu::ResizeCoordMode::pytorch_half_pixel)
    coord = 1;
  else if (coord_mode() == tpu::ResizeCoordMode::align_corners)
    coord = 2;
  if (mode() == tpu::ResizeMode::nearest) {
    param.platform_sp = ONNX_NEAREST;
    param.align_corners = true;
    param.half_pixel_centers = false;
  } else if (mode() == tpu::ResizeMode::linear) {
    param.platform_sp = PYTORCH_SUPPORT;
    param.align_corners = (coord == 2) ? 1: 0;
    param.half_pixel_centers = (coord == 0 || coord == 1) ? 1 : 0;
  }
  auto op = getOperation();
  BM168x::instance(Module::getChip(op))->call_global_func("backend_api_interp_global", &param,
                                         sizeof(interp_global_param_t));
}
#if 0
// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::InterpOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::InterpOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  int64_t n, c, ih, iw, oh, ow;
  Module::getNCHW(input(), n, c, ih, iw);
  Module::getNCHW(output(), n, c, oh, ow);
  interp_local_param_t param = {0};
  param.input_addr = in_gi.out_addr;
  param.output_addr = gi.out_addr;
  param.input_n = static_cast<int32_t>(in_gi.n_slice);
  param.input_c = static_cast<int32_t>(c);
  param.input_h = static_cast<int32_t>(in_gi.h_slice);
  param.input_w = static_cast<int32_t>(iw);
  param.output_h = gi.h_slice;
  param.output_w = ow;
  param.pad_bag = 0;
  param.pad_end = 0;
  param.dtype = BM168x::getDataType(input());

  int coord = 0;
  bool align_corners = (coord_mode() == tpu::ResizeCoordMode::align_corners);
  bool half_pixel = (coord_mode() == tpu::ResizeCoordMode::half_pixel);
  if (coord_mode() == tpu::ResizeCoordMode::half_pixel)
    coord = 0;
  else if (coord_mode() == tpu::ResizeCoordMode::pytorch_half_pixel)
    coord = 1;
  else if (coord_mode() == tpu::ResizeCoordMode::align_corners)
    coord = 2;
  if (mode() == tpu::ResizeMode::nearest) {
    param.platform_sp = ONNX_NEAREST;
    param.align_corners = true;
    param.half_pixel_centers = false;
  } else if (mode() == tpu::ResizeMode::linear) {
    param.platform_sp = PYTORCH_SUPPORT;
    param.align_corners = (coord == 2) ? 1: 0;
    param.half_pixel_centers = (coord == 0 || coord == 1) ? 1 : 0;
  }
  auto op = getOperation();
  BM168x::instance(Module::getChip(op))->call_local_func("backend_api_interp_local", &param,
                                      sizeof(param));
}
#endif
