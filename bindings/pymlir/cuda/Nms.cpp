//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include <algorithm>

static float calcIoU(float ax1, float ay1, float ax2, float ay2,
                      float bx1, float by1, float bx2, float by2) {
  float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
  float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
  float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
  float areaA = (ax2 - ax1) * (ay2 - ay1), areaB = (bx2 - bx1) * (by2 - by1);
  return (iw * ih) / (areaA + areaB - iw * ih + 1e-8f);
}

static void runNms(const float *boxes, const float *scores,
                    int batch, int num_classes, int num_boxes,
                    float iou_threshold, float score_threshold,
                    int max_output_per_class,
                    std::vector<float> &selected) {
  struct Candidate { float score; int idx; };
  for (int b = 0; b < batch; ++b) {
    const float *batch_boxes = boxes + b * num_boxes * 4;
    for (int c = 0; c < num_classes; ++c) {
      std::vector<Candidate> cands;
      for (int i = 0; i < num_boxes; ++i) {
        float s = scores[b * num_classes * num_boxes + c * num_boxes + i];
        if (s > score_threshold) cands.push_back({s, i});
      }
      std::sort(cands.begin(), cands.end(),
                [](const Candidate &a, const Candidate &b) {
                  return a.score > b.score;
                });

      std::vector<bool> suppressed(cands.size(), false);
      int out_count = 0;
      for (size_t i = 0; i < cands.size(); ++i) {
        if (suppressed[i]) continue;
        int keep = cands[i].idx;
        selected.push_back((float)b);
        selected.push_back((float)c);
        selected.push_back((float)keep);

        ++out_count;
        if (max_output_per_class > 0 && out_count >= max_output_per_class)
          break;

        float bx1 = batch_boxes[keep * 4], by1 = batch_boxes[keep * 4 + 1];
        float bx2 = batch_boxes[keep * 4 + 2], by2 = batch_boxes[keep * 4 + 3];
        for (size_t j = i + 1; j < cands.size(); ++j) {
          if (suppressed[j]) continue;
          int other = cands[j].idx;
          float iou = calcIoU(bx1, by1, bx2, by2,
                              batch_boxes[other * 4], batch_boxes[other * 4 + 1],
                              batch_boxes[other * 4 + 2], batch_boxes[other * 4 + 3]);
          if (iou > iou_threshold) suppressed[j] = true;
        }
      }
    }
  }
}

void py_cuda::cudaNmsOp(tpu::NmsOp op) {
  auto inputs = op.getInputs();
  int num_inputs = inputs.size();
  auto box_shape = module::getShape(inputs[0]);
  auto score_shape = module::getShape(inputs[1]);
  int batch = score_shape[0];
  int num_classes = score_shape[1];
  int num_boxes = score_shape[2];
  if (box_shape.size() == 3) {
    batch = box_shape[0];
    num_boxes = box_shape[1];
  }

  std::vector<float> h_boxes(batch * num_boxes * 4);
  CHECK_CUDA(cudaMemcpy(h_boxes.data(), getCudaData(inputs[0]),
                        h_boxes.size() * sizeof(float), cudaMemcpyDeviceToHost));

  int num_scores = batch * num_classes * num_boxes;
  std::vector<float> h_scores(num_scores);
  CHECK_CUDA(cudaMemcpy(h_scores.data(), getCudaData(inputs[1]),
                        num_scores * sizeof(float), cudaMemcpyDeviceToHost));

  float iou_threshold = 0.5f, score_threshold = 0.5f;
  int max_output = op.getMaxOutputSize();
  if (num_inputs >= 4 && !module::isNone(inputs[3]))
    CHECK_CUDA(cudaMemcpy(&iou_threshold, getCudaData(inputs[3]),
                          sizeof(float), cudaMemcpyDeviceToHost));
  if (num_inputs >= 5 && !module::isNone(inputs[4]))
    CHECK_CUDA(cudaMemcpy(&score_threshold, getCudaData(inputs[4]),
                          sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> selected;
  runNms(h_boxes.data(), h_scores.data(), batch, num_classes, num_boxes,
         iou_threshold, score_threshold, max_output, selected);

  auto out_bytes = module::getBytes(op.getOutput());
  CHECK_CUDA(cudaMemset(getCudaData(op.getOutput()), 0, out_bytes));
  if (!selected.empty()) {
    auto copy_bytes = std::min(selected.size() * sizeof(float),
                               static_cast<size_t>(out_bytes));
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), selected.data(),
                          copy_bytes,
                          cudaMemcpyHostToDevice));
  }
}

void py_cuda::cudaNmsOp(top::NmsOp op) {
  auto inputs = op.getInputs();
  int num_inputs = inputs.size();
  auto box_shape = module::getShape(inputs[0]);
  auto score_shape = module::getShape(inputs[1]);
  int batch = score_shape[0];
  int num_classes = score_shape[1];
  int num_boxes = score_shape[2];
  if (box_shape.size() == 3) {
    batch = box_shape[0];
    num_boxes = box_shape[1];
  }

  std::vector<float> h_boxes(batch * num_boxes * 4);
  CHECK_CUDA(cudaMemcpy(h_boxes.data(), getCudaData(inputs[0]),
                        h_boxes.size() * sizeof(float), cudaMemcpyDeviceToHost));

  int num_scores = batch * num_classes * num_boxes;
  std::vector<float> h_scores(num_scores);
  CHECK_CUDA(cudaMemcpy(h_scores.data(), getCudaData(inputs[1]),
                        num_scores * sizeof(float), cudaMemcpyDeviceToHost));

  float iou_threshold = 0.5f, score_threshold = 0.5f;
  int max_output = op.getMaxOutputSize();
  if (num_inputs >= 4 && !module::isNone(inputs[3]))
    CHECK_CUDA(cudaMemcpy(&iou_threshold, getCudaData(inputs[3]),
                          sizeof(float), cudaMemcpyDeviceToHost));
  if (num_inputs >= 5 && !module::isNone(inputs[4]))
    CHECK_CUDA(cudaMemcpy(&score_threshold, getCudaData(inputs[4]),
                          sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> selected;
  runNms(h_boxes.data(), h_scores.data(), batch, num_classes, num_boxes,
         iou_threshold, score_threshold, max_output, selected);

  auto out_bytes = module::getBytes(op.getOutput());
  CHECK_CUDA(cudaMemset(getCudaData(op.getOutput()), 0, out_bytes));
  if (!selected.empty()) {
    auto copy_bytes = std::min(selected.size() * sizeof(float),
                               static_cast<size_t>(out_bytes));
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), selected.data(),
                          copy_bytes,
                          cudaMemcpyHostToDevice));
  }
}
