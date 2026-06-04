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

void py_cuda::cudaMaskRCNNGetBboxBOp(top::MaskRCNNGetBboxBOp op) {
  auto rois = getCudaData(op.getPtrRois());
  auto bbox = getCudaData(op.getPtrBbox());
  auto scores = getCudaData(op.getPtrScore());
  auto max_val = getCudaData(op.getMaxVal());
  auto det_bboxes = getCudaData(op.getResultDetBboxes());
  auto det_labels = getCudaData(op.getResultDetLabels());

  float threshold_score = op.getThresholdScoreEq().convertToDouble();
  float wh_ratio_log = op.getWhRatioLog().convertToDouble();
  float nms_iou_thr = op.getNmsIouThr().convertToDouble();
  float delta2bbox_means = op.getDelta2bboxMeans().convertToDouble();
  float delta2bbox_stds_0 = op.getDelta2bboxStds_0().convertToDouble();
  float delta2bbox_stds_1 = op.getDelta2bboxStds_1().convertToDouble();
  int num_indexes = op.getNUM_INDEXES();
  int num_classes = op.getNUM_CLASSESGetBboxB(); // use GetBboxB-specific class count
  int max_per_img = op.getMAX_PER_IMG_GetBboxB();

  auto roi_shape = module::getShape(op.getPtrRois());
  int64_t roi_elems = 1;
  for (auto d : roi_shape)
    roi_elems *= d;
  int roi_len = num_classes + num_indexes; // should be 5
  int total_rois = roi_elems / roi_len;

  int max_candidates = total_rois * num_classes;

  // Stage 1: decode bbox deltas + score filter
  auto cand_boxes_buf = cuda_malloc(max_candidates * 4 * sizeof(float));
  auto cand_scores_buf = cuda_malloc(max_candidates * sizeof(float));
  auto cand_indices_buf = cuda_malloc(max_candidates * sizeof(int));
  auto cand_count_buf = cuda_malloc(sizeof(int));
  CHECK_CUDA(cudaMemset(cand_count_buf.get(), 0, sizeof(int)));

  cuda::getBboxBDecode(rois, bbox, scores, max_val, cand_boxes_buf.get(),
                       cand_scores_buf.get(), cand_indices_buf.get(),
                       cand_count_buf.get(), total_rois, num_classes,
                       num_indexes, delta2bbox_means, delta2bbox_stds_0,
                       delta2bbox_stds_1, threshold_score,
                       wh_ratio_log /* max_scalar_c */, max_candidates);

  // Read candidate count back
  int num_candidates = 0;
  CHECK_CUDA(cudaMemcpy(&num_candidates, cand_count_buf.get(), sizeof(int),
                        cudaMemcpyDeviceToHost));
  if (num_candidates > max_candidates)
    num_candidates = max_candidates;

  // Stage 2: NMS + collect results
  auto processed_buf = cuda_malloc(max_candidates * sizeof(int));
  CHECK_CUDA(cudaMemset(processed_buf.get(), 0, max_candidates * sizeof(int)));

  cuda::getBboxBCollect(cand_boxes_buf.get(), cand_scores_buf.get(),
                        cand_indices_buf.get(), num_candidates, det_bboxes,
                        det_labels, max_per_img, nms_iou_thr,
                        processed_buf.get());
}

void py_cuda::cudaMaskRCNNGetBboxBOp(tpu::MaskRCNNGetBboxBOp op) {
  // Same logic as top dialect; tpu has extra buffer inputs but core computation is identical
  auto rois = getCudaData(op.getPtrRois());
  auto bbox = getCudaData(op.getPtrBbox());
  auto scores = getCudaData(op.getPtrScore());
  auto max_val = getCudaData(op.getMaxVal());
  auto det_bboxes = getCudaData(op.getResultDetBboxes());
  auto det_labels = getCudaData(op.getResultDetLabels());

  float threshold_score = op.getThresholdScoreEq().convertToDouble();
  float wh_ratio_log = op.getWhRatioLog().convertToDouble();
  float nms_iou_thr = op.getNmsIouThr().convertToDouble();
  float delta2bbox_means = op.getDelta2bboxMeans().convertToDouble();
  float delta2bbox_stds_0 = op.getDelta2bboxStds_0().convertToDouble();
  float delta2bbox_stds_1 = op.getDelta2bboxStds_1().convertToDouble();
  int num_indexes = op.getNUM_INDEXES();
  int num_classes = op.getNUM_CLASSESGetBboxB();
  int max_per_img = op.getMAX_PER_IMG_GetBboxB();

  auto roi_shape = module::getShape(op.getPtrRois());
  int64_t roi_elems = 1;
  for (auto d : roi_shape)
    roi_elems *= d;
  int roi_len = num_classes + num_indexes;
  int total_rois = roi_elems / roi_len;

  int max_candidates = total_rois * num_classes;

  auto cand_boxes_buf = cuda_malloc(max_candidates * 4 * sizeof(float));
  auto cand_scores_buf = cuda_malloc(max_candidates * sizeof(float));
  auto cand_indices_buf = cuda_malloc(max_candidates * sizeof(int));
  auto cand_count_buf = cuda_malloc(sizeof(int));
  CHECK_CUDA(cudaMemset(cand_count_buf.get(), 0, sizeof(int)));

  cuda::getBboxBDecode(rois, bbox, scores, max_val, cand_boxes_buf.get(),
                       cand_scores_buf.get(), cand_indices_buf.get(),
                       cand_count_buf.get(), total_rois, num_classes,
                       num_indexes, delta2bbox_means, delta2bbox_stds_0,
                       delta2bbox_stds_1, threshold_score, wh_ratio_log,
                       max_candidates);

  int num_candidates = 0;
  CHECK_CUDA(cudaMemcpy(&num_candidates, cand_count_buf.get(), sizeof(int),
                        cudaMemcpyDeviceToHost));
  if (num_candidates > max_candidates)
    num_candidates = max_candidates;

  auto processed_buf = cuda_malloc(max_candidates * sizeof(int));
  CHECK_CUDA(cudaMemset(processed_buf.get(), 0, max_candidates * sizeof(int)));

  cuda::getBboxBCollect(cand_boxes_buf.get(), cand_scores_buf.get(),
                        cand_indices_buf.get(), num_candidates, det_bboxes,
                        det_labels, max_per_img, nms_iou_thr,
                        processed_buf.get());
}
