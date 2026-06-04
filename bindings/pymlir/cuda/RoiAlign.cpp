//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaRoiAlignOp(tpu::RoiAlignOp op) {
  auto in_shape = module::getShape(op.getInput());
  int N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
  auto roi_shape = module::getShape(op.getRois());
  int num_rois = roi_shape[0];
  int output_h = op.getOutputHeight();
  int output_w = op.getOutputWidth();
  int sampling_ratio = op.getSamplingRatio();
  float spatial_scale = op.getSpatialScale().convertToDouble();
  bool align_corners = op.getAlignCorners();
  bool avg_mode = op.getMode().str() == "Avg";

  cuda::bmRoiAlign(getCudaData(op.getInput()), getCudaData(op.getRois()),
                   getCudaData(op.getOutput()),
                   N, C, H, W,
                   num_rois, output_h, output_w,
                   sampling_ratio, spatial_scale, align_corners, avg_mode);
}

void py_cuda::cudaRoiAlignOp(top::RoiAlignOp op) {
  auto in_shape = module::getShape(op.getInput());
  int N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
  auto roi_shape = module::getShape(op.getRois());
  int num_rois = roi_shape[0];
  int output_h = op.getOutputHeight();
  int output_w = op.getOutputWidth();
  int sampling_ratio = op.getSamplingRatio();
  float spatial_scale = op.getSpatialScale().convertToDouble();
  bool align_corners = op.getAlignCorners();
  bool avg_mode = op.getMode().str() == "Avg";

  cuda::bmRoiAlign(getCudaData(op.getInput()), getCudaData(op.getRois()),
                   getCudaData(op.getOutput()),
                   N, C, H, W,
                   num_rois, output_h, output_w,
                   sampling_ratio, spatial_scale, align_corners, avg_mode);
}
