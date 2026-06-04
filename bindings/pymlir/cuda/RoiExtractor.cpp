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

void py_cuda::cudaRoiExtractorOp(tpu::RoiExtractorOp op) {
  auto roi_shape = module::getShape(op.getRois());
  int num_rois = roi_shape[0];
  int output_h = op.getOutputHeight();
  int output_w = op.getOutputWidth();
  int sampling_ratio = op.getSamplingRatio();
  int num_levels = op.getNumLevels();
  bool align_corners = op.getAlignCorners();
  bool avg_mode = op.getMode().str() == "Avg";
  auto spatial_scales = module::getF64Array(op.getSpatialScales(), num_levels, 1.0);

  // Read rois and target_lvls to host
  std::vector<float> h_rois(num_rois * 5);
  CHECK_CUDA(cudaMemcpy(h_rois.data(), getCudaData(op.getRois()),
                         num_rois * 5 * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<int> h_lvls(num_rois);
  CHECK_CUDA(cudaMemcpy(h_lvls.data(), getCudaData(op.getTargetLvls()),
                         num_rois * sizeof(int), cudaMemcpyDeviceToHost));

  auto inputs = op.getInputs();
  // First feature map determines C (all levels share same C)
  auto feat_shape = module::getShape(inputs[0]);
  int C = feat_shape[1];

  auto dst = (float *)getCudaData(op.getOutput());

  for (int l = 0; l < num_levels; l++) {
    // Filter rois belonging to this level
    std::vector<float> level_rois;
    std::vector<int> roi_indices;
    for (int i = 0; i < num_rois; i++) {
      if (h_lvls[i] == l) {
        level_rois.insert(level_rois.end(),
                          h_rois.begin() + i * 5,
                          h_rois.begin() + (i + 1) * 5);
        roi_indices.push_back(i);
      }
    }
    if (level_rois.empty()) continue;

    int level_count = roi_indices.size();
    auto level_feat = inputs[l];
    auto fshape = module::getShape(level_feat);
    int H = fshape[2], W = fshape[3];

    // Upload filtered rois to GPU
    auto level_rois_gpu = cuda_malloc(level_count * 5 * sizeof(float));
    CHECK_CUDA(cudaMemcpy(level_rois_gpu.get(), level_rois.data(),
                           level_count * 5 * sizeof(float),
                           cudaMemcpyHostToDevice));

    // Run RoiAlign for this level
    auto temp_out = cuda_malloc(level_count * C * output_h * output_w * sizeof(float));
    cuda::bmRoiAlign(getCudaData(level_feat), level_rois_gpu.get(),
                     temp_out.get(),
                     1, C, H, W,
                     level_count, output_h, output_w,
                     sampling_ratio, (float)(*spatial_scales)[l],
                     align_corners, avg_mode);

    // Scatter temp_out to output at correct roi indices
    int per_roi_elems = C * output_h * output_w;
    for (int i = 0; i < level_count; i++) {
      int dst_offset = roi_indices[i] * per_roi_elems;
      int src_offset = i * per_roi_elems;
      CHECK_CUDA(cudaMemcpy(dst + dst_offset,
                             (float *)temp_out.get() + src_offset,
                             per_roi_elems * sizeof(float),
                             cudaMemcpyDeviceToDevice));
    }
  }
}

void py_cuda::cudaRoiExtractorOp(top::RoiExtractorOp op) {
  auto roi_shape = module::getShape(op.getRois());
  int num_rois = roi_shape[0];
  int output_h = op.getOutputHeight();
  int output_w = op.getOutputWidth();
  int sampling_ratio = op.getSamplingRatio();
  int num_levels = op.getNumLevels();
  bool align_corners = op.getAlignCorners();
  bool avg_mode = op.getMode().str() == "Avg";
  auto spatial_scales = module::getF64Array(op.getSpatialScales(), num_levels, 1.0);

  std::vector<float> h_rois(num_rois * 5);
  CHECK_CUDA(cudaMemcpy(h_rois.data(), getCudaData(op.getRois()),
                         num_rois * 5 * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<int> h_lvls(num_rois);
  CHECK_CUDA(cudaMemcpy(h_lvls.data(), getCudaData(op.getTargetLvls()),
                         num_rois * sizeof(int), cudaMemcpyDeviceToHost));

  auto inputs = op.getInputs();
  auto feat_shape = module::getShape(inputs[0]);
  int C = feat_shape[1];

  auto dst = (float *)getCudaData(op.getOutput());

  for (int l = 0; l < num_levels; l++) {
    std::vector<float> level_rois;
    std::vector<int> roi_indices;
    for (int i = 0; i < num_rois; i++) {
      if (h_lvls[i] == l) {
        level_rois.insert(level_rois.end(),
                          h_rois.begin() + i * 5,
                          h_rois.begin() + (i + 1) * 5);
        roi_indices.push_back(i);
      }
    }
    if (level_rois.empty()) continue;

    int level_count = roi_indices.size();
    auto level_feat = inputs[l];
    auto fshape = module::getShape(level_feat);
    int H = fshape[2], W = fshape[3];

    auto level_rois_gpu = cuda_malloc(level_count * 5 * sizeof(float));
    CHECK_CUDA(cudaMemcpy(level_rois_gpu.get(), level_rois.data(),
                           level_count * 5 * sizeof(float),
                           cudaMemcpyHostToDevice));

    auto temp_out = cuda_malloc(level_count * C * output_h * output_w * sizeof(float));
    cuda::bmRoiAlign(getCudaData(level_feat), level_rois_gpu.get(),
                     temp_out.get(),
                     1, C, H, W,
                     level_count, output_h, output_w,
                     sampling_ratio, (float)(*spatial_scales)[l],
                     align_corners, avg_mode);

    int per_roi_elems = C * output_h * output_w;
    for (int i = 0; i < level_count; i++) {
      int dst_offset = roi_indices[i] * per_roi_elems;
      int src_offset = i * per_roi_elems;
      CHECK_CUDA(cudaMemcpy(dst + dst_offset,
                             (float *)temp_out.get() + src_offset,
                             per_roi_elems * sizeof(float),
                             cudaMemcpyDeviceToDevice));
    }
  }
}
