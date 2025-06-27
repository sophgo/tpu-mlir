//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"

#include "tpu_mlir/Support/DeformConv2D.h"
#include <queue>

using namespace tpu_mlir::backend;
namespace tpu_mlir {

static void sigmoid_batch(float *x, const int n) {
  for (int i = 0; i < n; ++i) {
    x[i] = 1.0f / (1.0f + exp(-x[i]));
  }
}

static float softplus_activate(float x) { return std::log(std::exp(x) + 1); }

static inline float tanh_activate(float x) {
  return (2 / (1 + std::exp(-2 * x)) - 1);
}

float my_mish_activate(float x_val) {
  return x_val * tanh_activate(softplus_activate(x_val));
}

static bool SortScoreCmp0(const std::pair<float, int> &pair1,
                          const std::pair<float, int> &pair2) {
  return pair1.first > pair2.first;
}

static bool SortScoreCmp1(const std::pair<float, std::pair<int, int>> &pair1,
                          const std::pair<float, std::pair<int, int>> &pair2) {
  return pair1.first > pair2.first;
}

static void GetConfidenceScores_opt(
    const float *conf_data, const int num, const int num_preds_per_class,
    const int num_classes, const float score_threshold,
    std::vector<std::map<int, std::vector<std::pair<float, int>>>>
        *conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    std::map<int, std::vector<std::pair<float, int>>> &label_scores =
        (*conf_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      for (int c = 0; c < num_classes; ++c) {
        if (conf_data[start_idx + c] > score_threshold) {
          label_scores[c].push_back(
              std::make_pair(conf_data[start_idx + c], p));
        }
      }
    }
    conf_data += num_preds_per_class * num_classes;
  }
}

static void GetLocPredictions_opt(const float *loc_data, const int num,
                                  const int num_preds_per_class,
                                  const int num_loc_classes,
                                  const bool share_location,
                                  float *decode_index,
                                  std::vector<LabelBBox_l> *loc_preds) {
  loc_preds->clear();
  if (share_location) {
    assert(num_loc_classes == 1);
  }
  loc_preds->resize(num);
  float *decode_pos = decode_index;
  for (int i = 0; i < num; ++i) {
    if (share_location) {
      decode_pos = decode_index + i * num_preds_per_class;
    }
    LabelBBox_l &label_bbox = (*loc_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_loc_classes * 4;
      for (int c = 0; c < num_loc_classes; ++c) {
        if (!share_location) {
          decode_pos = decode_index +
                       num_preds_per_class * num_loc_classes * i +
                       num_preds_per_class * c;
        }
        int label = share_location ? -1 : c;
        if (label_bbox.find(label) == label_bbox.end()) {
          label_bbox[label].resize(num_preds_per_class);
        }
        if (decode_pos[p] != 1) {
          continue;
        }
        label_bbox[label][p].xmin = loc_data[start_idx + c * 4];
        label_bbox[label][p].ymin = loc_data[start_idx + c * 4 + 1];
        label_bbox[label][p].xmax = loc_data[start_idx + c * 4 + 2];
        label_bbox[label][p].ymax = loc_data[start_idx + c * 4 + 3];
      }
    }
    loc_data += num_preds_per_class * num_loc_classes * 4;
  }
}

static void DecodeBBoxesAll_opt(const std::vector<LabelBBox_l> &all_loc_preds,
                                int num_priors, const float *prior_data,
                                const int num, const bool share_location,
                                const int num_loc_classes,
                                const int background_label_id,
                                const CodeType code_type,
                                const bool variance_encoded_in_target,
                                const bool clip, float *decode_index,
                                std::vector<LabelBBox_l> *all_decode_bboxes) {
  assert(all_loc_preds.size() == (size_t)num);
  all_decode_bboxes->clear();
  all_decode_bboxes->resize(num);
  float *decode_pos = decode_index;
  for (int i = 0; i < num; ++i) {
    if (share_location) {
      decode_pos = decode_index + i * num_priors;
    }
    // Decode predictions into bboxes.
    for (int c = 0; c < num_loc_classes; ++c) {
      int label = share_location ? -1 : c;
      if (label == background_label_id) {
        // Ignore background class.
        continue;
      }
      if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
        llvm::errs() << "Could not find location predictions for label "
                     << label;
      }
      const std::vector<BBox_l> &bboxes = all_loc_preds[i].find(label)->second;
      LabelBBox_l &decode_bboxes = (*all_decode_bboxes)[i];
      std::vector<BBox_l> *p = &(decode_bboxes[label]);
      p->clear();

      if (!share_location) {
        decode_pos =
            decode_index + num_priors * num_loc_classes * i + num_priors * c;
      }
      for (int k = 0; k < num_priors; ++k) {
        // NormalizedBBox decode_bbox;
        BBox_l decode_bbox;
        if (decode_pos[k] != 1) {
          p->push_back(decode_bbox);
          continue;
        }
        // opt CENTER_SIZE
        assert(code_type == PriorBoxParameter_CodeType_CENTER_SIZE);
        // prior_bboxes
        int start_idx = k * 4;
        const float *p0 = prior_data + start_idx;
        const float *p1 = prior_data + start_idx + 4 * num_priors;
        float prior_width = p0[2] - p0[0];
        assert(prior_width > 0);
        float prior_height = p0[3] - p0[1];
        assert(prior_height > 0);
        float prior_center_x = (p0[0] + p0[2]) * 0.5;
        float prior_center_y = (p0[1] + p0[3]) * 0.5;

        float decode_bbox_center_x, decode_bbox_center_y;
        float decode_bbox_width, decode_bbox_height;
        if (variance_encoded_in_target) {
          // variance is encoded in target, we simply need to retore the offset
          // predictions.
          decode_bbox_center_x = bboxes[k].xmin * prior_width + prior_center_x;
          decode_bbox_center_y = bboxes[k].ymin * prior_height + prior_center_y;
          decode_bbox_width = exp(bboxes[k].xmax) * prior_width;
          decode_bbox_height = exp(bboxes[k].ymax) * prior_height;
        } else {
          // variance is encoded in bbox, we need to scale the offset
          // accordingly.
          decode_bbox_center_x =
              p1[0] * bboxes[k].xmin * prior_width + prior_center_x;
          decode_bbox_center_y =
              p1[1] * bboxes[k].ymin * prior_height + prior_center_y;
          decode_bbox_width = exp(p1[2] * bboxes[k].xmax) * prior_width;
          decode_bbox_height = exp(p1[3] * bboxes[k].ymax) * prior_height;
        }
        decode_bbox.xmin = decode_bbox_center_x - decode_bbox_width * 0.5;
        decode_bbox.ymin = decode_bbox_center_y - decode_bbox_height * 0.5;
        decode_bbox.xmax = decode_bbox_center_x + decode_bbox_width * 0.5;
        decode_bbox.ymax = decode_bbox_center_y + decode_bbox_height * 0.5;
        decode_bbox.CalcSize();
        p->push_back(decode_bbox);
      }
    }
  }
}

static void GetConfidenceScores_v2_opt(
    const float *conf_data, const int num, const int num_preds_per_class,
    const int num_classes, const float score_threshold,
    std::vector<std::map<int, std::vector<std::pair<float, int>>>>
        *conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    std::map<int, std::vector<std::pair<float, int>>> &label_scores =
        (*conf_preds)[i];
    for (int p = 0; p < num_classes; ++p) {
      int start_idx = p * num_preds_per_class;
      for (int c = 0; c < num_preds_per_class; ++c) {
        if (conf_data[start_idx + c] > score_threshold) {
          label_scores[p].push_back(
              std::make_pair(conf_data[start_idx + c], c));
        }
      }
    }
    conf_data += num_preds_per_class * num_classes;
  }
}

static void
DecodeBBoxesAll_v2_opt(const std::vector<LabelBBox_l> &all_loc_preds,
                       int num_priors, const float *prior_data, const int num,
                       const bool share_location, const int num_loc_classes,
                       const int background_label_id, const CodeType code_type,
                       const bool variance_encoded_in_target, const bool clip,
                       float *decode_index,
                       std::vector<LabelBBox_l> *all_decode_bboxes) {
  assert(all_loc_preds.size() == (size_t)num);
  all_decode_bboxes->clear();
  all_decode_bboxes->resize(num);
  float *decode_pos = decode_index;
  for (int i = 0; i < num; ++i) {
    if (share_location) {
      decode_pos = decode_index + i * num_priors;
    }
    // Decode predictions into bboxes.
    for (int c = 0; c < num_loc_classes; ++c) {
      int label = share_location ? -1 : c;
      if (label == background_label_id) {
        // Ignore background class.
        continue;
      }
      if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
        llvm::errs() << "Could not find location predictions for label "
                     << label;
      }
      const std::vector<BBox_l> &bboxes = all_loc_preds[i].find(label)->second;
      LabelBBox_l &decode_bboxes = (*all_decode_bboxes)[i];
      std::vector<BBox_l> *p = &(decode_bboxes[label]);
      p->clear();

      if (!share_location) {
        decode_pos =
            decode_index + num_priors * num_loc_classes * i + num_priors * c;
      }
      for (int k = 0; k < num_priors; ++k) {
        // NormalizedBBox decode_bbox;
        BBox_l decode_bbox;
        if (decode_pos[k] != 1) {
          p->push_back(decode_bbox);
          continue;
        }

        decode_bbox.xmin = bboxes[k].xmin;
        decode_bbox.ymin = bboxes[k].ymin;
        decode_bbox.xmax = bboxes[k].xmax;
        decode_bbox.ymax = bboxes[k].ymax;
        decode_bbox.CalcSize();
        p->push_back(decode_bbox);
      }
    }
  }
}

static void
ApplyNMSFast_opt(const std::vector<BBox_l> &bboxes,
                 const std::vector<std::pair<float, int>> &conf_score,
                 const float score_threshold, const float nms_threshold,
                 const float eta, int top_k,
                 std::vector<std::pair<float, int>> *indices) {
  // Do nms.
  float adaptive_threshold = nms_threshold;
  int i = 0;
  int length = (top_k < (int)conf_score.size()) ? top_k : conf_score.size();
  while (length != i) {
    bool keep = true;
    for (int k = 0; k < (int)indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k].second;
        const BBox_l &b1 = bboxes[conf_score[i].second];
        const BBox_l &b2 = bboxes[kept_idx];
        if (b2.xmin > b1.xmax || b2.xmax < b1.xmin || b2.ymin > b1.ymax ||
            b2.ymax < b1.ymin) {
          keep = true;
        } else {
          const float inter_xmin = std::max(b1.xmin, b2.xmin);
          const float inter_ymin = std::max(b1.ymin, b2.ymin);
          const float inter_xmax = std::min(b1.xmax, b2.xmax);
          const float inter_ymax = std::min(b1.ymax, b2.ymax);
          const float inter_width = inter_xmax - inter_xmin;
          const float inter_height = inter_ymax - inter_ymin;
          const float inter_size = inter_width * inter_height;
          const float total_size = b1.size + b2.size;
          keep = (inter_size * (adaptive_threshold + 1) <=
                  total_size * adaptive_threshold)
                     ? true
                     : false;
        }
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(conf_score[i]);
    }
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
    i++;
  }
}

static inline float exp_fast(float x) {
  union {
    unsigned int i;
    float f;
  } v;
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);

  return v.f;
}

static inline float _sigmoid(float x, bool fast) {
  if (fast)
    return 1.0f / (1.0f + exp_fast(-x));
  else
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float _softmax(float *probs, float *data, int input_stride,
                             int num_of_class, int *max_cls, bool fast) {
  // assert(num_of_class == 80);
  float x[num_of_class];
  float max_x = -INFINITY;
  float min_x = INFINITY;
  for (int i = 0; i < num_of_class; i++) {
    x[i] = data[i * input_stride];
    if (x[i] > max_x) {
      max_x = x[i];
    }
    if (x[i] < min_x) {
      min_x = x[i];
    }
  }
#define t (-100.0f)
  float exp_x[num_of_class];
  float sum = 0;
  for (int i = 0; i < num_of_class; i++) {
    x[i] = x[i] - max_x;
    if (min_x < t)
      x[i] = x[i] / min_x * t;
    if (fast)
      exp_x[i] = exp_fast(x[i]);
    else
      exp_x[i] = std::exp(x[i]);
    sum += exp_x[i];
  }
  float max_prob = 0;
  for (int i = 0; i < num_of_class; i++) {
    probs[i] = exp_x[i] / sum;
    if (probs[i] > max_prob) {
      max_prob = probs[i];
      *max_cls = i;
    }
  }
  return max_prob;
}

// feature in shape [3][5+80][grid_size][grid_size]
#define GET_INDEX(cell_idx, box_idx_in_cell, data_idx, num_cell, class_num)    \
  (box_idx_in_cell * (class_num + 5) * num_cell + data_idx * num_cell +        \
   cell_idx)

static void process_feature(detection *det, int *det_idx, float *feature,
                            std::vector<int64_t> grid_size, double *anchor,
                            std::vector<int64_t> yolo_size,
                            int64_t num_of_class, float obj_threshold,
                            std::string version = "yolov3") {
  int yolo_w = yolo_size[1];
  int yolo_h = yolo_size[0];
  // TPU_LOG_DEBUG("grid_size_h: %d\n", grid_size[0]);
  // TPU_LOG_DEBUG("grid_size_w: %d\n", grid_size[1]);
  // TPU_LOG_DEBUG("obj_threshold: %f\n", obj_threshold);
  int num_boxes_per_cell = 3;
  // assert(num_of_class == 80);

// 255 = 3 * (5 + 80)
// feature in shape [3][5+80][grid_size][grid_size]
#define COORD_X_INDEX (0)
#define COORD_Y_INDEX (1)
#define COORD_W_INDEX (2)
#define COORD_H_INDEX (3)
#define CONF_INDEX (4)
#define CLS_INDEX (5)
  int num_cell = grid_size[0] * grid_size[1];
  // int box_dim = 5 + num_of_class;

  int idx = *det_idx;
  int hit = 0, hit2 = 0;
  ;
  for (int i = 0; i < num_cell; i++) {
    for (int j = 0; j < num_boxes_per_cell; j++) {
      float box_confidence = _sigmoid(
          feature[GET_INDEX(i, j, CONF_INDEX, num_cell, num_of_class)], false);
      if (box_confidence < obj_threshold) {
        continue;
      }
      hit++;
      float box_class_probs[num_of_class];
      int box_max_cls = -1;
      float box_max_prob =
          _softmax(box_class_probs,
                   &feature[GET_INDEX(i, j, CLS_INDEX, num_cell, num_of_class)],
                   num_cell, num_of_class, &box_max_cls, false);
      float box_max_score = box_confidence * box_max_prob;
      if (box_max_score < obj_threshold) {
        continue;
      }
      // get coord now
      int grid_x = i % grid_size[1];
      int grid_y = i / grid_size[1];
      if (version == "yolov5") {
        float io_r_h = yolo_size[0] / grid_size[0];
        float io_r_w = yolo_size[1] / grid_size[1];
        float box_x = _sigmoid(
            feature[GET_INDEX(i, j, COORD_X_INDEX, num_cell, num_of_class)],
            false);
        box_x = (box_x * 2 + grid_x - 0.5) * io_r_w;
        float box_y = _sigmoid(
            feature[GET_INDEX(i, j, COORD_Y_INDEX, num_cell, num_of_class)],
            false);
        box_y = (box_y * 2 + grid_y - 0.5) * io_r_h;
        // anchor is in shape [3][2]
        float box_w = _sigmoid(
            feature[GET_INDEX(i, j, COORD_W_INDEX, num_cell, num_of_class)],
            false);
        box_w = (box_w * 2) * (box_w * 2) * anchor[j * 2];
        float box_h = _sigmoid(
            feature[GET_INDEX(i, j, COORD_H_INDEX, num_cell, num_of_class)],
            false);
        box_h = (box_h * 2) * (box_h * 2) * anchor[j * 2 + 1];
        hit2++;
        // DBG("  hit2 %d, conf = %f, cls = %d, coord = [%f, %f, %f, %f]\n",
        //    hit2, box_max_score, box_max_cls, box_x, box_y, box_w, box_h);
        det[idx].bbox = box{box_x, box_y, box_w, box_h};
        det[idx].score = box_max_score;
        det[idx].cls = box_max_cls;
        idx++;
        assert(idx <= MAX_DET);
      } else {
        float box_x = _sigmoid(
            feature[GET_INDEX(i, j, COORD_X_INDEX, num_cell, num_of_class)],
            false);
        box_x += grid_x;
        box_x /= grid_size[1];
        float box_y = _sigmoid(
            feature[GET_INDEX(i, j, COORD_Y_INDEX, num_cell, num_of_class)],
            false);
        box_y += grid_y;
        box_y /= grid_size[0];
        // anchor is in shape [3][2]
        float box_w = std::exp(
            feature[GET_INDEX(i, j, COORD_W_INDEX, num_cell, num_of_class)]);
        box_w *= anchor[j * 2];
        box_w /= yolo_w;
        float box_h = std::exp(
            feature[GET_INDEX(i, j, COORD_H_INDEX, num_cell, num_of_class)]);
        box_h *= anchor[j * 2 + 1];
        box_h /= yolo_h;
        hit2++;
        // DBG("  hit2 %d, conf = %f, cls = %d, coord = [%f, %f, %f, %f]\n",
        //    hit2, box_max_score, box_max_cls, box_x, box_y, box_w, box_h);
        det[idx].bbox = box{box_x, box_y, box_w, box_h};
        det[idx].score = box_max_score;
        det[idx].cls = box_max_cls;
        idx++;
        assert(idx <= MAX_DET);
      }
    }
  }
  *det_idx = idx;
}

// https://github.com/ChenYingpeng/caffe-yolov3/blob/master/box.cpp
static float overlap(float x1, float w1, float x2, float w2) {
  float l1 = x1 - w1 / 2;
  float l2 = x2 - w2 / 2;
  float left = l1 > l2 ? l1 : l2;
  float r1 = x1 + w1 / 2;
  float r2 = x2 + w2 / 2;
  float right = r1 < r2 ? r1 : r2;
  return right - left;
}

static float box_intersection(box a, box b) {
  float w = overlap(a.x, a.w, b.x, b.w);
  float h = overlap(a.y, a.h, b.y, b.h);
  if (w < 0 || h < 0)
    return 0;
  float area = w * h;
  return area;
}

static float box_union(box a, box b) {
  float i = box_intersection(a, b);
  float u = a.w * a.h + b.w * b.h - i;
  return u;
}

//
// more aboud iou
//   https://github.com/ultralytics/yolov3/blob/master/utils/utils.py
// IoU = inter / (a + b - inter), can't handle enclosure issue
// GIoU, DIoU, CIoU?
//
static float box_iou(box a, box b) {
  return box_intersection(a, b) / box_union(a, b);
}

static void nms(detection *det, int num, float nms_threshold) {
  for (int i = 0; i < num; i++) {
    if (det[i].score == 0) {
      // erased already
      continue;
    }
    for (int j = i + 1; j < num; j++) {
      if (det[j].score == 0) {
        // erased already
        continue;
      }
      if (det[i].cls != det[j].cls) {
        // not the same class
        continue;
      }
      float iou = box_iou(det[i].bbox, det[j].bbox);
      assert(iou <= 1.0f);
      if (iou > nms_threshold) {
        // overlapped, select one to erase
        if (det[i].score < det[j].score) {
          det[i].score = 0;
        } else {
          det[j].score = 0;
        }
      }
    }
  }
}
DetectionOutputFunc::DetectionOutputFunc(DetParam &param) : param_(param) {}

void DetectionOutputFunc::invoke() {
  int num = param_.loc_shape[0];
  int num_priors =
      param_.onnx_nms ? param_.loc_shape[1] : param_.prior_shape[2] / 4;
  int num_loc_classes = param_.share_location ? 1 : param_.num_classes;
  float eta = 1.0;
  bool variance_encoded_in_target = false;
  std::vector<std::map<int, std::vector<std::pair<float, int>>>>
      all_conf_scores;
  if (!param_.onnx_nms) {
    GetConfidenceScores_opt(param_.conf_data, num, num_priors,
                            param_.num_classes, param_.confidence_threshold,
                            &all_conf_scores);
  } else {
    GetConfidenceScores_v2_opt(param_.conf_data, num, num_priors,
                               param_.num_classes, param_.confidence_threshold,
                               &all_conf_scores);
  }
  for (int i = 0; i < num; ++i) {
    for (int c = 0; c < param_.num_classes; ++c) {
      if (all_conf_scores[i].find(c) == all_conf_scores[i].end()) {
        continue;
      }
      std::vector<std::pair<float, int>> &scores =
          all_conf_scores[i].find(c)->second;

      if (param_.top_k < (int)scores.size()) {
        std::partial_sort(scores.begin(), scores.begin() + param_.top_k,
                          scores.end(), SortScoreCmp0);
      } else {
        std::sort(scores.begin(), scores.end(), SortScoreCmp0);
      }
    }
  }

  // build keep for decode ,recode vilad index
  float *decode_keep_index;
  int buf_length = 0;
  if (param_.share_location) {
    buf_length = num * num_priors;
  } else {
    buf_length = num * num_priors * param_.num_classes;
  }
  decode_keep_index = new float[buf_length];
  memset(decode_keep_index, 0, buf_length * 4);
  float *ptr = decode_keep_index;
  for (int i = 0; i < num; ++i) {
    if (param_.share_location) {
      ptr = decode_keep_index + num_priors * i;
    }
    for (int c = 0; c < param_.num_classes; ++c) {
      if (!param_.share_location) {
        ptr = decode_keep_index + num_priors * param_.num_classes * i +
              num_priors * c;
      }
      if (c == param_.background_label_id) {
        // Ignore background class.
        continue;
      }

      if (all_conf_scores[i].find(c) == all_conf_scores[i].end())
        continue;
      std::vector<std::pair<float, int>> &scores =
          all_conf_scores[i].find(c)->second;
      int length =
          param_.top_k < (int)scores.size() ? param_.top_k : scores.size();
      for (int k = 0; k < length; ++k) {
        ptr[scores[k].second] = 1;
      }
    }
  }

  // Retrieve all location predictions.
  std::vector<LabelBBox_l> all_loc_preds;
  GetLocPredictions_opt(param_.loc_data, num, num_priors, num_loc_classes,
                        param_.share_location, decode_keep_index,
                        &all_loc_preds);

  // Decode all loc predictions to bboxes.
  std::vector<LabelBBox_l> all_decode_bboxes;
  const bool clip_bbox = false;
  if (!param_.onnx_nms) {
    DecodeBBoxesAll_opt(all_loc_preds, num_priors, param_.prior_data, num,
                        param_.share_location, num_loc_classes,
                        param_.background_label_id, param_.code_type,
                        variance_encoded_in_target, clip_bbox,
                        decode_keep_index, &all_decode_bboxes);
  } else {
    DecodeBBoxesAll_v2_opt(all_loc_preds, num_priors, param_.prior_data, num,
                           param_.share_location, num_loc_classes,
                           param_.background_label_id, param_.code_type,
                           variance_encoded_in_target, clip_bbox,
                           decode_keep_index, &all_decode_bboxes);
  }
  delete[] decode_keep_index;

  int num_kept = 0;
  std::vector<std::map<int, std::vector<std::pair<float, int>>>> all_indices;
  for (int i = 0; i < num; ++i) {
    const LabelBBox_l &decode_bboxes = all_decode_bboxes[i];
    const std::map<int, std::vector<std::pair<float, int>>> &conf_scores =
        all_conf_scores[i];
    std::map<int, std::vector<std::pair<float, int>>> indices;
    int num_det = 0;
    for (int c = 0; c < param_.num_classes; ++c) {
      if (c == param_.background_label_id) {
        // Ignore background class.
        continue;
      }
      if (conf_scores.find(c) == conf_scores.end())
        continue;
      int label = param_.share_location ? -1 : c;
      if (decode_bboxes.find(label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current
        // label.
        llvm::errs() << "Could not find location predictions for label "
                     << label;
        continue;
      }
      const std::vector<BBox_l> &bboxes = decode_bboxes.find(label)->second;
      const std::vector<std::pair<float, int>> &aa =
          conf_scores.find(c)->second;
      ApplyNMSFast_opt(bboxes, aa, param_.confidence_threshold,
                       param_.nms_threshold, eta, param_.top_k, &(indices[c]));

      num_det += indices[c].size();
    }

    if (param_.keep_top_k > -1 && num_det > param_.keep_top_k) {
      std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
      for (auto it = indices.begin(); it != indices.end(); ++it) {
        int label = it->first;

        const std::vector<std::pair<float, int>> &label_indices = it->second;
        for (int j = 0; j < (int)label_indices.size(); ++j) {
          score_index_pairs.push_back(
              std::make_pair(label_indices[j].first,
                             std::make_pair(label, label_indices[j].second)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScoreCmp1);
      score_index_pairs.resize(param_.keep_top_k);
      // Store the new indices.
      std::map<int, std::vector<std::pair<float, int>>> new_indices;
      for (int j = 0; j < (int)score_index_pairs.size(); ++j) {

        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        float s = score_index_pairs[j].first;

        new_indices[label].push_back(std::make_pair(s, idx));
      }
      all_indices.push_back(new_indices);
      num_kept += param_.keep_top_k;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  }
  float *top_data = (float *)param_.output_data;
  int output_size = num * param_.keep_top_k * 1 * 1 * 7;
  // init output buf
  for (int i = 0; i < output_size; ++i) {
    top_data[i] = -1;
  }

  if (num_kept == 0) {
    // Generate fake results per image.
    for (int i = 0; i < num; ++i) {
      top_data[0] = i;
      top_data += 7;
    }
  } else {
    int count = 0;
    for (int i = 0; i < num; ++i) {
      const LabelBBox_l &decode_bboxes = all_decode_bboxes[i];
      for (auto it = all_indices[i].begin(); it != all_indices[i].end(); ++it) {
        int label = it->first;
        int loc_label = param_.share_location ? -1 : label;
        if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
          // Something bad happened if there are no predictions for current
          // label.
          llvm::errs() << "Could not find location predictions for "
                       << loc_label;
          continue;
        }
        const std::vector<BBox_l> &bboxes =
            decode_bboxes.find(loc_label)->second;
        std::vector<std::pair<float, int>> &indices = it->second;
        for (int j = 0; j < (int)indices.size(); ++j) {

          int idx = indices[j].second;
          top_data[count * 7] = i;
          top_data[count * 7 + 1] = label;
          top_data[count * 7 + 2] = indices[j].first;
          const BBox_l &bbox = bboxes[idx];
          top_data[count * 7 + 3] = bbox.xmin;
          top_data[count * 7 + 4] = bbox.ymin;
          top_data[count * 7 + 5] = bbox.xmax;
          top_data[count * 7 + 6] = bbox.ymax;
          ++count;
        }
      }
    }
  }
}

YoloDetectionFunc::YoloDetectionFunc(YoloDetParam &param) : param_(param) {
  std::sort(param_.inputs.begin(), param_.inputs.end(),
            [](const tensor_list_t &a, const tensor_list_t &b) {
              return a.shape[3] > b.shape[3];
            });
}

void YoloDetectionFunc::invoke() {
  auto top_data = param_.output.ptr;
  memset(top_data, 0, param_.output.size);
  int batch = param_.output.shape[0];
  size_t bottom_count = param_.inputs.size();
  assert(param_.anchors.size() == bottom_count * 6);
  double(*anchors)[6] = (double(*)[6])param_.anchors.data();

  for (int b = 0; b < batch; ++b) {
    std::vector<std::vector<int64_t>> grid_size;
    std::vector<float *> features;

    for (size_t i = 0; i < bottom_count; ++i) {
      int offset = b * param_.inputs[i].shape[1] * param_.inputs[i].shape[2] *
                   param_.inputs[i].shape[3];
      grid_size.push_back(
          {param_.inputs[i].shape[2], param_.inputs[i].shape[3]});
      auto data = param_.inputs[i].ptr + offset;
      // auto size = param_.inputs[i].count() / batch;
      // std::vector<float> bottom_data(data, data + size);
      features.push_back(data);
    }

    detection det_raw[MAX_DET_RAW];
    detection dets[MAX_DET];
    int det_raw_idx = 0;
    for (size_t i = 0; i < features.size(); i++) {
      process_feature(det_raw, &det_raw_idx, features[i], grid_size[i],
                      &anchors[i][0], {param_.net_input_h, param_.net_input_w},
                      param_.class_num, param_.obj_threshold, param_.version);
    }
    nms(det_raw, det_raw_idx, param_.nms_threshold);
    int det_idx = 0;
    for (int i = 0; i < det_raw_idx; i++) {
      if (det_raw[i].score > 0) {
        dets[det_idx] = det_raw[i];
        det_idx++;
      }
    }

    auto keep_topk = param_.keep_topk;
    if (keep_topk > det_idx)
      keep_topk = det_idx;

    long long count = 0;
    auto batch_output_data = top_data + b * param_.output.shape[1] *
                                            param_.output.shape[2] *
                                            param_.output.shape[3];

    for (int i = 0; i < keep_topk; ++i) {
      batch_output_data[count++] = dets[i].bbox.x;
      batch_output_data[count++] = dets[i].bbox.y;
      batch_output_data[count++] = dets[i].bbox.w;
      batch_output_data[count++] = dets[i].bbox.h;
      batch_output_data[count++] = dets[i].cls;
      batch_output_data[count++] = dets[i].score;
      // TPU_LOG_DEBUG("x = %f, y = %f, w = %f, h = %f, class = %d, score =
      // %f\n",
      //               dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w,
      //               dets[i].bbox.h, dets[i].cls, dets[i].score);
    }
  }
}

static void _mkanchors(std::vector<float> ctrs, std::vector<float> &anchors) {
  anchors.push_back(ctrs[2] - 0.5 * (ctrs[0] - 1));
  anchors.push_back(ctrs[3] - 0.5 * (ctrs[1] - 1));
  anchors.push_back(ctrs[2] + 0.5 * (ctrs[0] - 1));
  anchors.push_back(ctrs[3] + 0.5 * (ctrs[1] - 1));
}

static void _whctrs(std::vector<float> anchor, std::vector<float> &ctrs) {
  float w = anchor[2] - anchor[0] + 1;
  float h = anchor[3] - anchor[1] + 1;
  float x_ctr = anchor[0] + 0.5 * (w - 1);
  float y_ctr = anchor[1] + 0.5 * (h - 1);
  ctrs.push_back(w);
  ctrs.push_back(h);
  ctrs.push_back(x_ctr);
  ctrs.push_back(y_ctr);
}

static void _ratio_enum(std::vector<float> anchor,
                        std::vector<float> anchor_ratio,
                        std::vector<float> &ratio_anchors) {
  std::vector<float> ctrs;
  _whctrs(anchor, ctrs);
  float size = ctrs[0] * ctrs[1];
  int ratio_num = anchor_ratio.size();
  for (int i = 0; i < ratio_num; i++) {
    float ratio = size / anchor_ratio[i];
    int ws = int(std::round(std::sqrt(ratio)));
    int hs = int(std::round(ws * anchor_ratio[i]));
    std::vector<float> ctrs_in;
    ctrs_in.push_back(ws);
    ctrs_in.push_back(hs);
    ctrs_in.push_back(ctrs[2]);
    ctrs_in.push_back(ctrs[3]);
    _mkanchors(ctrs_in, ratio_anchors);
  }
}

static void _scale_enum(std::vector<float> ratio_anchors,
                        std::vector<float> anchor_scale,
                        std::vector<float> &anchor_boxes) {
  int anchors_ratio_num = ratio_anchors.size() / 4;
  for (int i = 0; i < anchors_ratio_num; i++) {
    std::vector<float> anchor;
    anchor.push_back(ratio_anchors[i * 4]);
    anchor.push_back(ratio_anchors[i * 4 + 1]);
    anchor.push_back(ratio_anchors[i * 4 + 2]);
    anchor.push_back(ratio_anchors[i * 4 + 3]);
    std::vector<float> ctrs;
    _whctrs(anchor, ctrs);
    int scale_num = anchor_scale.size();
    for (int j = 0; j < scale_num; j++) {
      float ws = ctrs[0] * anchor_scale[j];
      float hs = ctrs[1] * anchor_scale[j];
      std::vector<float> ctrs_in;
      ctrs_in.push_back(ws);
      ctrs_in.push_back(hs);
      ctrs_in.push_back(ctrs[2]);
      ctrs_in.push_back(ctrs[3]);
      _mkanchors(ctrs_in, anchor_boxes);
    }
  }
}

static void generate_anchors(int anchor_base_size,
                             std::vector<float> anchor_scale,
                             std::vector<float> anchor_ratio,
                             std::vector<float> &anchor_boxes) {
  std::vector<float> base_anchor = {0, 0, (float)(anchor_base_size - 1),
                                    (float)(anchor_base_size - 1)};
  std::vector<float> ratio_anchors;
  _ratio_enum(base_anchor, anchor_ratio, ratio_anchors);
  _scale_enum(ratio_anchors, anchor_scale, anchor_boxes);
}

static void
anchor_box_transform_inv(float img_width, float img_height,
                         std::vector<std::vector<float>> bbox,
                         std::vector<std::vector<float>> select_anchor,
                         std::vector<std::vector<float>> &pred) {
  int num = bbox.size();
  for (int i = 0; i < num; i++) {
    float dx = bbox[i][0];
    float dy = bbox[i][1];
    float dw = bbox[i][2];
    float dh = bbox[i][3];
    float pred_ctr_x = select_anchor[i][0] + select_anchor[i][2] * dx;
    float pred_ctr_y = select_anchor[i][1] + select_anchor[i][3] * dy;
    float pred_w = select_anchor[i][2] * std::exp(dw);
    float pred_h = select_anchor[i][3] * std::exp(dh);
    std::vector<float> tmp_pred;
    tmp_pred.push_back(
        std::max(std::min((float)(pred_ctr_x - 0.5 * pred_w), img_width - 1),
                 (float)0.0));
    tmp_pred.push_back(
        std::max(std::min((float)(pred_ctr_y - 0.5 * pred_h), img_height - 1),
                 (float)0.0));
    tmp_pred.push_back(
        std::max(std::min((float)(pred_ctr_x + 0.5 * pred_w), img_width - 1),
                 (float)0.0));
    tmp_pred.push_back(
        std::max(std::min((float)(pred_ctr_y + 0.5 * pred_h), img_height - 1),
                 (float)0.0));
    pred.push_back(tmp_pred);
  }
}

static void anchor_box_nms(std::vector<std::vector<float>> &pred_boxes,
                           std::vector<float> &confidence,
                           float nms_threshold) {
  for (size_t i = 0; i < pred_boxes.size() - 1; i++) {
    float s1 = (pred_boxes[i][2] - pred_boxes[i][0] + 1) *
               (pred_boxes[i][3] - pred_boxes[i][1] + 1);
    for (size_t j = i + 1; j < pred_boxes.size(); j++) {
      float s2 = (pred_boxes[j][2] - pred_boxes[j][0] + 1) *
                 (pred_boxes[j][3] - pred_boxes[j][1] + 1);

      float x1 = std::max(pred_boxes[i][0], pred_boxes[j][0]);
      float y1 = std::max(pred_boxes[i][1], pred_boxes[j][1]);
      float x2 = std::min(pred_boxes[i][2], pred_boxes[j][2]);
      float y2 = std::min(pred_boxes[i][3], pred_boxes[j][3]);

      float width = x2 - x1;
      float height = y2 - y1;
      if (width > 0 && height > 0) {
        float IOU = width * height / (s1 + s2 - width * height);
        if (IOU > nms_threshold) {
          if (confidence[i] >= confidence[j]) {
            pred_boxes.erase(pred_boxes.begin() + j);
            confidence.erase(confidence.begin() + j);
            j--;
          } else {
            pred_boxes.erase(pred_boxes.begin() + i);
            confidence.erase(confidence.begin() + i);
            i--;
            break;
          }
        }
      }
    }
  }
}

ProposalFunc::ProposalFunc(ProposalParam &param) : param_(param) {
  generate_anchors(param_.anchor_base_size, anchor_scale, anchor_ratio,
                   anchor_boxes);
}

void ProposalFunc::invoke() {
  assert(param_.inputs.size() == 2);
  auto score_shape = param_.inputs[0].shape;
  auto bbox_shape = param_.inputs[1].shape;
  int batch = score_shape[0];
  int channel = score_shape[1];
  int height = score_shape[2];
  int width = score_shape[3];

  int feat_stride = param_.feat_stride;
  int net_input_h = param_.net_input_h;
  int net_input_w = param_.net_input_w;
  float rpn_nms_threshold = param_.rpn_nms_threshold;
  int rpn_nms_post_top_n = param_.rpn_nms_post_top_n;
  float thresh = param_.rpn_obj_threshold;

  float *score = param_.inputs[0].ptr;
  float *bbox_deltas = param_.inputs[1].ptr;
  float *output_data = param_.output.ptr;
  auto output_shape = param_.output.shape;
  auto anchor_boxes = this->anchor_boxes;

  for (int b = 0; b < batch; ++b) {
    auto batched_score = score + b * channel * height * width;
    auto batched_bbox_deltas =
        bbox_deltas + b * bbox_shape[1] * bbox_shape[2] * bbox_shape[3];
    std::vector<std::vector<float>> select_anchor;
    std::vector<float> confidence;
    std::vector<std::vector<float>> bbox;
    int anchor_num = anchor_scale.size() * anchor_ratio.size();

    for (int k = 0; k < anchor_num; k++) {
      float w = anchor_boxes[4 * k + 2] - anchor_boxes[4 * k] + 1;
      float h = anchor_boxes[4 * k + 3] - anchor_boxes[4 * k + 1] + 1;
      float x_ctr = anchor_boxes[4 * k] + 0.5 * (w - 1);
      float y_ctr = anchor_boxes[4 * k + 1] + 0.5 * (h - 1);

      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          if (batched_score[anchor_num * height * width +
                            (k * height + i) * width + j] >= thresh) {
            std::vector<float> tmp_anchor;
            std::vector<float> tmp_bbox;

            tmp_anchor.push_back(j * feat_stride + x_ctr);
            tmp_anchor.push_back(i * feat_stride + y_ctr);
            tmp_anchor.push_back(w);
            tmp_anchor.push_back(h);
            select_anchor.push_back(tmp_anchor);
            confidence.push_back(batched_score[anchor_num * height * width +
                                               (k * height + i) * width + j]);
            tmp_bbox.push_back(
                batched_bbox_deltas[(4 * k * height + i) * width + j]);
            tmp_bbox.push_back(
                batched_bbox_deltas[((4 * k + 1) * height + i) * width + j]);
            tmp_bbox.push_back(
                batched_bbox_deltas[((4 * k + 2) * height + i) * width + j]);
            tmp_bbox.push_back(
                batched_bbox_deltas[((4 * k + 3) * height + i) * width + j]);
            bbox.push_back(tmp_bbox);
          }
        }
      }
    }
    std::vector<std::vector<float>> pred_boxes;
    anchor_box_transform_inv(net_input_w, net_input_h, bbox, select_anchor,
                             pred_boxes);
    anchor_box_nms(pred_boxes, confidence, rpn_nms_threshold);
    int num = pred_boxes.size() > rpn_nms_post_top_n ? rpn_nms_post_top_n
                                                     : pred_boxes.size();

    auto batched_output =
        output_data + b * output_shape[1] * output_shape[2] * output_shape[3];
    for (int i = 0; i < num; i++) {
      batched_output[5 * i] = b;
      batched_output[5 * i + 1] = pred_boxes[i][0];
      batched_output[5 * i + 2] = pred_boxes[i][1];
      batched_output[5 * i + 3] = pred_boxes[i][2];
      batched_output[5 * i + 4] = pred_boxes[i][3];
    }
  }
}

ROIPoolingFunc::ROIPoolingFunc(ROIPoolingParam &param) : param_(param) {}

void ROIPoolingFunc::invoke() {
  assert(param_.inputs.size() == 2);
  float *input_data = param_.inputs[0].ptr;
  float *rois = param_.inputs[1].ptr;
  float *output_data = param_.output.ptr;
  auto input_shape = param_.inputs[0].shape;
  auto roi_shape = param_.inputs[1].shape;
  int64_t pooled_h = param_.pooled_h;
  int64_t pooled_w = param_.pooled_w;
  double spatial_scale = param_.spatial_scale;

  int batch = input_shape[0];
  int channel = input_shape[1];
  int height = input_shape[2];
  int width = input_shape[3];
  int num_rois = roi_shape[2];

  for (int b = 0; b < batch; ++b) {
    auto batched_rois = rois + b * num_rois * 5;
    auto batched_output =
        output_data + b * num_rois * channel * pooled_h * pooled_w;
    for (int n = 0; n < num_rois; ++n) {
      int roi_batch_ind = batched_rois[0];
      int roi_start_w = std::round(batched_rois[1] * spatial_scale);
      int roi_start_h = std::round(batched_rois[2] * spatial_scale);
      int roi_end_w = std::round(batched_rois[3] * spatial_scale);
      int roi_end_h = std::round(batched_rois[4] * spatial_scale);
      assert(roi_batch_ind < batch);

      int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
      int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
      const float bin_size_h =
          static_cast<float>(roi_height) / static_cast<float>(pooled_h);
      const float bin_size_w =
          static_cast<float>(roi_width) / static_cast<float>(pooled_w);

      float *batch_data = input_data + roi_batch_ind * channel * height * width;

      for (int c = 0; c < channel; ++c) {
        for (int ph = 0; ph < pooled_h; ++ph) {
          for (int pw = 0; pw < pooled_w; ++pw) {
            // Compute pooling region for this output unit:
            //  start (included) = floor(ph * roi_height / pooled_height_)
            //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
            int hstart = static_cast<int>(
                std::floor(static_cast<float>(ph) * bin_size_h));
            int wstart = static_cast<int>(
                std::floor(static_cast<float>(pw) * bin_size_w));
            int hend = static_cast<int>(
                std::ceil(static_cast<float>(ph + 1) * bin_size_h));
            int wend = static_cast<int>(
                std::ceil(static_cast<float>(pw + 1) * bin_size_w));

            hstart = std::min(std::max(hstart + roi_start_h, 0), height);
            hend = std::min(std::max(hend + roi_start_h, 0), height);
            wstart = std::min(std::max(wstart + roi_start_w, 0), width);
            wend = std::min(std::max(wend + roi_start_w, 0), width);

            bool is_empty = (hend <= hstart) || (wend <= wstart);

            const int pool_index = ph * pooled_w + pw;
            if (is_empty) {
              batched_output[pool_index] = 0;
            }

            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width + w;
                if (batch_data[index] > batched_output[pool_index]) {
                  batched_output[pool_index] = batch_data[index];
                }
              }
            }
          }
        }
        batch_data += height * width;
        batched_output += pooled_h * pooled_w;
      }
      batched_rois += 5;
    }
  }
}

class Reducer {
  roi_align_mode_t mode;
  int count;
  float result;

public:
  Reducer(roi_align_mode_t _mode) : mode(_mode) {}
  void init() {
    count = 0;
    if (mode == RoiAlignAvgMode) {
      result = 0;
    } else {
      result = __FLT_MIN__;
    }
  }
  void insert(float x) {
    count += 1;
    if (mode == RoiAlignAvgMode) {
      result += x;
    } else {
      result = std::max(result, x);
    }
  }
  float reduce() {
    if (mode == RoiAlignAvgMode) {
      const float avg_scaling = 1.0f / count;
      result *= avg_scaling;
    }
    return result;
  }
};

RoiAlignFunc::RoiAlignFunc(RoiAlignParam &param) : param_(param) {}

struct RoiAlignPreCalc {
  int pos[4];
  float w[4];
};

static inline void preCalcForBilinearInterp(
    const int input_h, const int input_w, const int pooled_h,
    const int pooled_w, const int sample_h, const int sample_w, float start_h,
    float start_w, float bin_h, float bin_w, int sampling_ratio,
    std::vector<RoiAlignPreCalc> &pre_calc) {
  RoiAlignPreCalc *p = pre_calc.data();
  float h_scale = bin_h / sample_h;
  float w_scale = bin_w / sample_w;
  for (int ph = 0; ph < pooled_h; ++ph) {
    for (int pw = 0; pw < pooled_w; ++pw) {
      float yiter = start_h + ph * bin_h + 0.5f * h_scale;
      for (int iy = 0; iy < sample_h; ++iy) {
        float y = yiter;
        if (y < -1.f || y > input_h) {
          memset(p, 0x0, sizeof(RoiAlignPreCalc) * sample_w);
          p += sample_w;
        } else {
          y = std::min(std::max(y, 0.f), (float)(input_h - 1));
          int yl = (int)std::floor(y);
          int yh = std::min(yl + 1, input_h - 1);
          float ly = y - yl;
          float hy = 1.f - ly;
          int yl_x_iw = yl * input_w;
          int yh_x_iw = yh * input_w;
          float xiter = start_w + pw * bin_w + 0.5f * w_scale;
          for (int ix = 0; ix < sample_w; ++ix) {
            float x = xiter;
            if (x < -1.f || x > input_w)
              *p = {0, 0, 0, 0, 0.f, 0.f, 0.f, 0.f};
            else {
              x = std::min(std::max(x, 0.f), (float)(input_w - 1));
              int xl = (int)std::floor(x);
              int xh = std::min(xl + 1, input_w - 1);
              float lx = x - xl;
              float hx = 1.0f - lx;
              *p = {yl_x_iw + xl, yl_x_iw + xh, yh_x_iw + xl, yh_x_iw + xh,
                    hy * hx,      hy * lx,      ly * hx,      ly * lx};
            }
            ++p;
            xiter += w_scale;
          }
        }
        yiter += h_scale;
      }
    }
  }
}

static inline void _RoiAlign(float *output, const float *input,
                             const float *rois, const float spatial_scale,
                             const int sampling_ratio, const int OH,
                             const int OW, const bool aligned,
                             const roi_align_mode_t mode,
                             const std::vector<int64_t> input_shape,
                             const std::vector<int64_t> rois_shape,
                             const int dim_input, const int dim_rois) {
  const int pooled_size = OH * OW;
  assert(dim_input == 4);
  assert(dim_rois == 2);
  assert(rois_shape[1] == 5);

  const int IC = input_shape[1];
  const int IH = input_shape[2];
  const int IW = input_shape[3];
  const int input_size = IH * IW;
  const int roi_num = rois_shape[0];

  const float roi_offset = aligned ? 0.5f : 0.f;
  const float pooled_h_inv = 1.0f / OH;
  const float pooled_w_inv = 1.0f / OW;

#pragma omp parallel for schedule(static, omp_schedule(roi_num))
  for (int i = 0; i < roi_num; ++i) {
    const int batch_idx = (int)rois[i * 5 + 0];
    float start_w = rois[i * 5 + 1] * spatial_scale - roi_offset;
    float start_h = rois[i * 5 + 2] * spatial_scale - roi_offset;
    float end_w = rois[i * 5 + 3] * spatial_scale - roi_offset;
    float end_h = rois[i * 5 + 4] * spatial_scale - roi_offset;
    float roi_h = end_h - start_h;
    float roi_w = end_w - start_w;
    if (!aligned) {
      roi_h = std::max(roi_h, 1.0f);
      roi_w = std::max(roi_w, 1.0f);
    }
    float bin_w = roi_w * pooled_w_inv;
    float bin_h = roi_h * pooled_h_inv;
    int sample_w = sampling_ratio > 0.0f ? sampling_ratio : ceil(bin_w);
    int sample_h = sampling_ratio > 0.0f ? sampling_ratio : ceil(bin_h);
    const int sample_size = sample_h * sample_w;
    std::vector<RoiAlignPreCalc> pre_calc(sample_size * pooled_size);
    preCalcForBilinearInterp(IH, IW, OH, OW, sample_h, sample_w, start_h,
                             start_w, roi_h * pooled_h_inv,
                             roi_w * pooled_w_inv, sampling_ratio, pre_calc);
    Reducer reducer(mode);
    int index_n = i * IC * pooled_size;
    for (int c = 0; c < IC; ++c) {
      int index_n_c = index_n + c * pooled_size;
      for (int p = 0; p < pooled_size; ++p) {
        const float *feat = input + (batch_idx * IC + c) * input_size;
        reducer.init();
        for (int s = 0; s < sample_size; ++s) {
          const auto &pc = pre_calc[p * sample_size + s];
          float x = 0.0f;
          for (int k = 0; k < 4; ++k) {
            x += pc.w[k] * feat[pc.pos[k]];
          }
          reducer.insert(x);
        }
        output[index_n_c + p] = reducer.reduce();
      }
    }
  }
}

void RoiAlignFunc::invoke() {
  assert(param_.inputs.size() == 2);
  const float spatial_scale = param_.spatial_scale;
  const int sampling_ratio = param_.sampling_ratio;
  const int OH = param_.pooled_h;
  const int OW = param_.pooled_w;
  const bool aligned = param_.aligned; // corresponse to new attr of opset16
  const roi_align_mode_t mode = param_.mode;

  const float *input = param_.inputs[0].ptr;
  const float *rois = param_.inputs[1].ptr;
  float *output = param_.output.ptr;

  const auto input_shape = param_.inputs[0].shape;
  const auto rois_shape = param_.inputs[1].shape;
  _RoiAlign(output, input, rois, spatial_scale, sampling_ratio, OH, OW, aligned,
            mode, input_shape, rois_shape, input_shape.size(),
            rois_shape.size());
}

RoiExtractorFunc::RoiExtractorFunc(RoiExtractorParam &param) : param_(param) {}

void RoiExtractorFunc::invoke() {
  const int offset_feature = 2;
  const int num_levels = param_.num_levels;
  assert(num_levels <= MAX_ROI_ALIGN_NUM_LEVELS);
  assert(param_.inputs.size() == param_.num_levels + 2);
  std::vector<float> spatial_scales(num_levels);
  for (int i = 0; i < num_levels; i++) {
    spatial_scales[i] = param_.spatial_scales[i];
  }
  const int sampling_ratio = param_.sampling_ratio;
  //[CHECK-0] ensure sampling_ratio >0, to close dynamic bins
  assert(sampling_ratio > 0);
  const int OH = param_.pooled_h;
  const int OW = param_.pooled_w;
  const bool aligned = param_.aligned; // corresponse to new attr of opset16
  const roi_align_mode_t mode = param_.mode;

  auto _rois = param_.inputs[0];
  auto _target_lvls = param_.inputs[1];

  const float *rois = _rois.ptr;
  const float *target_lvls = _target_lvls.ptr;

  const auto rois_shape = _rois.shape;
  const auto target_lvls_shape = _target_lvls.shape;
  const int total_roi_num = target_lvls_shape[0];
  //[CHECK-1] num_rois is same for 5len-rois and target_lvls
  //[Note] roi_len  is mmmdetection style 5-len
  assert(rois_shape[0] == total_roi_num);
  assert(rois_shape[1] == 5);
  assert(rois_shape.size() == 2);
  assert(_target_lvls.shape.size() == 1);
  const int input_n = param_.inputs[offset_feature].shape[0];
  const int input_c = param_.inputs[offset_feature].shape[1];
  //[CHECK-2] ensure each feature batch and channel is same
  for (int id_layer = 1; id_layer < num_levels; id_layer++) {
    assert(input_n == param_.inputs[offset_feature + id_layer].shape[0]);
    assert(input_c == param_.inputs[offset_feature + id_layer].shape[1]);
  }
  //[CHECK-3] ensure elements in target_lvls smaller than num_levels: #features
  for (int i = 0; i < total_roi_num; i++) {
    assert(target_lvls[i] < num_levels);
    assert(target_lvls[i] >= 0);
  }
  //[CHECK-4] ensure batch_id in 5len-rois smaller than input_n: batch_size
  for (int i = 0; i < total_roi_num; i++) {
    assert(rois[5 * i] < input_n);
    assert(rois[5 * i] >= 0);
  }
  //[CHECK-5] ensure valid coordinates for 5len-rois, x0 <= x1,  y0 <=y1
  for (int i = 0; i < total_roi_num; i++) {
    assert(rois[5 * i + 1] <= rois[5 * i + 3]);
    assert(rois[5 * i + 2] <= rois[5 * i + 4]);
  }

  float *output = param_.output.ptr;
  const int perRoiOutputSize = input_c * OH * OW;
  std::fill(output, output + input_n * perRoiOutputSize, 0.0f);

  for (int id_layer = 0; id_layer < num_levels; id_layer++) {
    const auto feature_shape = param_.inputs[offset_feature + id_layer].shape;

    // inds = target_lvls == i
    std::vector<bool> inds(total_roi_num);
    std::fill(inds.begin(), inds.end(), 0);
    std::transform(target_lvls, target_lvls + total_roi_num, inds.begin(),
                   [id_layer](int lvl) { return int(lvl) == id_layer; });

    // rois_ = ros[inds, :]
    float *rois_gather = new float[total_roi_num * rois_shape[1]];
    int idx_gather = 0;
    for (int i = 0; i < total_roi_num; i++) {
      if (inds[i]) {
        memcpy(&rois_gather[5 * idx_gather], &rois[5 * i], 5 * sizeof(float));
        idx_gather++;
      }
    }
    if (idx_gather == 0) {
      continue;
    }
    std::vector<int64_t> shape_rois_gather(_rois.shape.size());
    shape_rois_gather[0] = idx_gather;
    shape_rois_gather[1] = rois_shape[1];
    float *roi_feats = new float[idx_gather * input_c * OH * OW];
    _RoiAlign(roi_feats, param_.inputs[offset_feature + id_layer].ptr,
              rois_gather, spatial_scales[id_layer], sampling_ratio, OH, OW,
              aligned, mode, feature_shape, shape_rois_gather,
              feature_shape.size(), 2);
    // output[inds] = roi_feats
    int idx_scatter = 0;

    for (int i = 0; i < total_roi_num; i++) {
      if (inds[i]) {
        memcpy(&output[perRoiOutputSize * i],
               &roi_feats[perRoiOutputSize * idx_scatter],
               perRoiOutputSize * sizeof(float));
        idx_scatter++;
      }
    }
  }
}

static void bbox_transform_inv(const float *boxes, const float *deltas,
                               float *pred, int num, int class_num) {
  for (int i = 0; i < num; ++i) {
    float height = boxes[i * 4 + 3] - boxes[i * 4 + 1] + 1;
    float width = boxes[i * 4 + 2] - boxes[i * 4 + 0] + 1;
    float ctr_x = boxes[i * 4 + 0] + width * 0.5;
    float ctr_y = boxes[i * 4 + 1] + height * 0.5;

    for (int j = 0; j < class_num; ++j) {
      float dx = deltas[i * class_num * 4 + j * 4 + 0];
      float dy = deltas[i * class_num * 4 + j * 4 + 1];
      float dw = deltas[i * class_num * 4 + j * 4 + 2];
      float dh = deltas[i * class_num * 4 + j * 4 + 3];

      float pred_ctr_x = dx * width + ctr_x;
      float pred_ctr_y = dy * height + ctr_y;
      float pred_w = std::exp(dw) * width;
      float pred_h = std::exp(dh) * height;

      pred[i * class_num * 4 + j * 4 + 0] = pred_ctr_x - pred_w / 2;
      pred[i * class_num * 4 + j * 4 + 1] = pred_ctr_y - pred_h / 2;
      pred[i * class_num * 4 + j * 4 + 2] = pred_ctr_x + pred_w / 2;
      pred[i * class_num * 4 + j * 4 + 3] = pred_ctr_y + pred_h / 2;
    }
  }
}

static void nms(detections *dets, int num, float nms_threshold) {
  for (int i = 0; i < num; i++) {
    if (dets[i].score == 0) {
      // erased already
      continue;
    }

    float s1 = (dets[i].bbox.x2 - dets[i].bbox.x1 + 1) *
               (dets[i].bbox.y2 - dets[i].bbox.y1 + 1);
    for (int j = i + 1; j < num; j++) {
      if (dets[j].score == 0) {
        // erased already
        continue;
      }
      if (dets[i].cls != dets[j].cls) {
        // not the same class
        continue;
      }

      float s2 = (dets[j].bbox.x2 - dets[j].bbox.x1 + 1) *
                 (dets[j].bbox.y2 - dets[j].bbox.y1 + 1);

      float x1 = std::max(dets[i].bbox.x1, dets[j].bbox.x1);
      float y1 = std::max(dets[i].bbox.y1, dets[j].bbox.y1);
      float x2 = std::min(dets[i].bbox.x2, dets[j].bbox.x2);
      float y2 = std::min(dets[i].bbox.y2, dets[j].bbox.y2);

      float width = x2 - x1;
      float height = y2 - y1;
      if (width > 0 && height > 0) {
        float iou = width * height / (s1 + s2 - width * height);
        assert(iou <= 1.0f);
        if (iou > nms_threshold) {
          // overlapped, select one to erase
          if (dets[i].score < dets[j].score) {
            dets[i].score = 0;
          } else {
            dets[j].score = 0;
          }
        }
      }
    }
  }
}

FrcnDetctionFunc::FrcnDetctionFunc(FrcnDetParam &param) : param_(param) {}

void FrcnDetctionFunc::invoke() {
  assert(param_.inputs.size() == 3);
  float *bbox_deltas = param_.inputs[0].ptr;
  float *scores = param_.inputs[1].ptr;
  float *rois = param_.inputs[2].ptr;
  float *output_data = param_.output.ptr;

  auto rois_shape = param_.inputs[2].shape;
  auto output_shape = param_.output.shape;
  int64_t class_num = param_.class_num;
  int64_t keep_topk = param_.keep_topk;
  double nms_threshold = param_.nms_threshold;
  double obj_threshold = param_.obj_threshold;

  int batch = rois_shape[0];
  int num = rois_shape[2];

  for (int b = 0; b < batch; ++b) {
    auto batched_bbox_deltas = bbox_deltas + b * num * class_num * 4;
    auto batched_scores = scores + b * num * class_num;
    auto batched_rois = rois + b * num * 5;

    std::vector<float> boxes(num * 4, 0);
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < 4; ++j) {
        boxes[i * 4 + j] = batched_rois[i * 5 + j + 1];
      }
    }

    std::vector<float> pred(num * class_num * 4, 0);
    float *pred_data = pred.data();
    std::vector<float> deltas(batched_bbox_deltas,
                              batched_bbox_deltas + num * class_num * 4);
    bbox_transform_inv(boxes.data(), deltas.data(), pred_data, num, class_num);

    int det_num = 0;
    auto dets = new detections[num];

    for (int i = 0; i < num; ++i) {
      for (int j = 1; j < (int)class_num; ++j) {
        if (batched_scores[i * class_num + j] > obj_threshold) {
          dets[det_num].bbox.x1 = pred[i * class_num * 4 + j * 4 + 0];
          dets[det_num].bbox.y1 = pred[i * class_num * 4 + j * 4 + 1];
          dets[det_num].bbox.x2 = pred[i * class_num * 4 + j * 4 + 2];
          dets[det_num].bbox.y2 = pred[i * class_num * 4 + j * 4 + 3];
          dets[det_num].cls = j;
          dets[det_num].score = batched_scores[i * class_num + j];
          det_num++;
        }
      }
    }

    nms(dets, det_num, nms_threshold);
    auto dets_nms = new detections[det_num];
    int det_idx = 0;
    for (int i = 0; i < det_num; i++) {
      if (dets[i].score > 0) {
        dets_nms[det_idx] = dets[i];
        det_idx++;
      }
    }

    if (keep_topk > det_idx)
      keep_topk = det_idx;

    long long count = 0;
    auto batched_output =
        output_data + b * output_shape[1] * output_shape[2] * output_shape[3];
    for (int i = 0; i < keep_topk; ++i) {
      batched_output[count++] = dets_nms[i].bbox.x1;
      batched_output[count++] = dets_nms[i].bbox.y1;
      batched_output[count++] = dets_nms[i].bbox.x2;
      batched_output[count++] = dets_nms[i].bbox.y2;
      batched_output[count++] = dets_nms[i].cls;
      batched_output[count++] = dets_nms[i].score;
      // printf("x1: %f, y1: %f, x2: %f, y2: %f, cls: %d, score: %f\n",
      //     dets_nms[i].bbox.x1, dets_nms[i].bbox.y1, dets_nms[i].bbox.x2,
      //     dets_nms[i].bbox.y2, dets_nms[i].cls, dets_nms[i].score);
    }
    delete[] dets_nms;
    delete[] dets;
  }
}

void RetinaFaceDetectionFunc::setup(std::vector<tensor_list_t> &inputs,
                                    tensor_list_t &output,
                                    RetinaFaceDetectionParam &param) {
  // sort inputs by neuron shape size
  std::sort(inputs.begin(), inputs.end(),
            [](const tensor_list_t &a, const tensor_list_t &b) {
              if (a.shape[3] < b.shape[3]) {
                return true;
              } else if (a.shape[3] == b.shape[3]) {
                return a.shape[1] < b.shape[1];
              } else {
                return false;
              }
            });
  _bottoms = inputs;
  _tops = output;
  _nms_threshold = param.nms_threshold;
  _confidence_threshold = param.confidence_threshold;
  _keep_topk = param.keep_topk;
}

void RetinaFaceDetectionFunc::invoke() {
  auto top_data = _tops.ptr;
  memset(top_data, 0, _tops.size);

  size_t bottom_count = _bottoms.size();
  assert(bottom_count == 9);

  auto batch = _tops.shape[0];

  for (int b = 0; b < batch; ++b) {
    std::vector<FaceInfo> infos;
    for (size_t i = 0; i < _feature_stride_fpn.size(); ++i) {
      int stride = _feature_stride_fpn[i];

      size_t offset0 = b * _bottoms[3 * i].shape[1] * _bottoms[3 * i].shape[2] *
                       _bottoms[3 * i].shape[3];
      size_t count0 = _bottoms[3 * i].shape[0] * _bottoms[3 * i].shape[1] *
                      _bottoms[3 * i].shape[2] * _bottoms[3 * i].shape[3];
      auto score_data = _bottoms[3 * i].ptr + offset0;
      size_t score_count = count0 / batch;

      size_t offset1 = b * _bottoms[3 * i + 1].shape[1] *
                       _bottoms[3 * i + 1].shape[2] *
                       _bottoms[3 * i + 1].shape[3];
      size_t count1 =
          _bottoms[3 * i + 1].shape[0] * _bottoms[3 * i + 1].shape[1] *
          _bottoms[3 * i + 1].shape[2] * _bottoms[3 * i + 1].shape[3];
      auto bbox_data = _bottoms[3 * i + 1].ptr + offset1;
      size_t bbox_count = count1 / batch;

      size_t offset2 = b * _bottoms[3 * i + 2].shape[1] *
                       _bottoms[3 * i + 2].shape[2] *
                       _bottoms[3 * i + 2].shape[3];
      size_t count2 =
          _bottoms[3 * i + 2].shape[0] * _bottoms[3 * i + 2].shape[1] *
          _bottoms[3 * i + 2].shape[2] * _bottoms[3 * i + 2].shape[3];
      auto landmark_data = _bottoms[3 * i + 2].ptr + offset2;
      size_t landmark_count = count2 / batch;

      auto shape = _bottoms[3 * i].shape;
      size_t height = shape[2];
      size_t width = shape[3];

      std::vector<float> score(score_data + score_count / 2,
                               score_data + score_count);
      std::vector<float> bbox(bbox_data, bbox_data + bbox_count);
      std::vector<float> landmark(landmark_data,
                                  landmark_data + landmark_count);

      int count = height * width;
      std::string key = "stride" + std::to_string(stride);
      auto anchors_fpn = _anchors_fpn[key];
      auto num_anchors = _num_anchors[key];

      std::vector<AnchorBox> anchors =
          anchors_plane(height, width, stride, anchors_fpn);

      for (int num = 0; num < num_anchors; ++num) {
        for (int j = 0; j < count; ++j) {
          float confidence = score[j + count * num];
          if (confidence <= _confidence_threshold)
            continue;

          float dx = bbox[j + count * (0 + num * 4)];
          float dy = bbox[j + count * (1 + num * 4)];
          float dw = bbox[j + count * (2 + num * 4)];
          float dh = bbox[j + count * (3 + num * 4)];
          std::vector<float> bbox_deltas{dx, dy, dw, dh};
          auto bbox = bbox_pred(anchors[j + count * num], bbox_deltas);

          std::vector<float> landmark_deltas(10, 0);
          for (size_t k = 0; k < 5; ++k) {
            landmark_deltas[k] = landmark[j + count * (num * 10 + k * 2)];
            landmark_deltas[k + 5] =
                landmark[j + count * (num * 10 + k * 2 + 1)];
          }

          auto pts = landmark_pred(anchors[j + count * num], landmark_deltas);

          FaceInfo info;
          info.x1 = bbox[0];
          info.y1 = bbox[1];
          info.x2 = bbox[2];
          info.y2 = bbox[3];
          info.score = confidence;
          for (int idx = 0; idx < 5; ++idx) {
            info.x[idx] = pts[idx];
            info.y[idx] = pts[idx + 5];
          }

          infos.push_back(info);
        }
      }
    }

    auto preds = nms(infos, _nms_threshold);
    auto keep_topk = _keep_topk;
    if (keep_topk > (int)preds.size())
      keep_topk = (int)preds.size();

    long long count = 0;
    size_t top_offset = b * _tops.shape[1] * _tops.shape[2] * _tops.shape[3];
    auto batch_top_data = top_data + top_offset;
    for (int i = 0; i < keep_topk; ++i) {
      batch_top_data[count++] = preds[i].x1;
      batch_top_data[count++] = preds[i].y1;
      batch_top_data[count++] = preds[i].x2;
      batch_top_data[count++] = preds[i].y2;
      batch_top_data[count++] = preds[i].score;
      for (int j = 0; j < 5; ++j) {
        batch_top_data[count++] = preds[i].x[j];
        batch_top_data[count++] = preds[i].y[j];
      }
    }
  }
}

template <typename Dtype>
void ApplyNms_opt(std::vector<PredictionResult> &boxes, std::vector<int> &idxes,
                  Dtype threshold, int agnostic_nms = 0) {
  int bbox_cnt = (int)boxes.size();
  // init the map
  uint32_t map[bbox_cnt / 32 + 1];
  memset(map, 0xFF, sizeof(map));

  for (int i = 0; i < bbox_cnt - 1; ++i) {
    // skip the dropped bbox
    if (!(map[i / 32] & (1 << (i % 32))))
      continue;

    for (int j = i + 1; j < bbox_cnt; ++j) {
      // skip the dropped bbox
      if (!(map[j / 32] & (1 << (j % 32))))
        continue;

      box Bbox1, Bbox2;
      Bbox1.x = boxes[i].x;
      Bbox1.y = boxes[i].y;
      Bbox1.w = boxes[i].w;
      Bbox1.h = boxes[i].h;
      Bbox2.x = boxes[j].x;
      Bbox2.y = boxes[j].y;
      Bbox2.w = boxes[j].w;
      Bbox2.h = boxes[j].h;

      Dtype iou;
      if (agnostic_nms == 0 && boxes[i].classType == boxes[j].classType) {
        iou = box_iou(Bbox1, Bbox2);
      } else {
        iou = (Dtype)0;
      }
      if (iou >= threshold) {
        map[j / 32] &= ~(1 << (j % 32));
      }
    }
  }

  for (int i = 0; i < bbox_cnt; ++i) {
    if (map[i / 32] & (1 << (i % 32))) {
      idxes.push_back(i);
    }
  }
}

YoloDetectionFunc_v2::YoloDetectionFunc_v2(YoloDetParam &param)
    : param_(param) {
  std::sort(param_.inputs.begin(), param_.inputs.end(),
            [](const tensor_list_t &a, const tensor_list_t &b) {
              return a.shape[3] > b.shape[3];
            });
}

void YoloDetectionFunc_v2::invoke() {
  auto top_data = param_.output.ptr;
  memset(top_data, 0, param_.output.size);
  int batch_num = param_.inputs[0].shape[0];
  int bottom_num = param_.inputs.size();
  assert(param_.anchors.size() == bottom_num * 6);
  int len = 4 + param_.class_num + 1;
  int mask_offset = 0;
  const int num = batch_num;
  int total_num = 0;
  std::vector<PredictionResult> total_preds;
  total_preds.clear();

  // calc the threshold for Po
  float po_thres = -std::log(1 / param_.obj_threshold - 1);
  std::vector<float> class_score;
  for (int b = 0; b < batch_num; b++) {
    std::vector<PredictionResult> predicts;
    predicts.clear();
    mask_offset = 0;
    for (int index = 0; index < bottom_num; index++) {
      int h = (int)param_.inputs[index].shape[2];
      int w = (int)param_.inputs[index].shape[3];
      int stride = h * w;
      const float *input_data = param_.inputs[index].ptr;
      for (int cy = 0; cy < h; cy++) {
        for (int cx = 0; cx < w; cx++) {
          for (int n = 0; n < param_.num_boxes; n++) {
            int index = b * param_.num_boxes * len * stride + n * len * stride +
                        cy * w + cx;
            std::vector<float> pred;
            class_score.clear();

            // Po/Pmax/tx/ty/tw/th
            float swap_data[6] = {0};

            // filter bbxo by Po
            int index_po = 4 * stride + index;
            if (input_data[index_po] <= po_thres)
              continue;

            for (int c = 0; c < len; ++c) {
              int index2 = c * stride + index;
              if (c > 4) {
                class_score.push_back(input_data[index2]);
              } else {
                if (c == 4) {
                  swap_data[0] = input_data[index2];
                } else {
                  swap_data[c + 2] = input_data[index2];
                }
              }
            }

            PredictionResult predict;
            swap_data[1] =
                *std::max_element(class_score.begin(), class_score.end());
            int arg_max = std::distance(
                class_score.begin(),
                std::max_element(class_score.begin(), class_score.end()));
            if (param_.version == "yolov5") {
              sigmoid_batch(swap_data, 6);
              swap_data[1] = swap_data[0] * swap_data[1];
              if (swap_data[1] > param_.obj_threshold) {
                [&](std::vector<float> &b, float *x, double *biases, int n,
                    int i, int j, int lw, int lh, int w, int h) {
                  b.clear();
                  b.push_back((i + x[0] * 2 - 0.5) * w / lw);
                  b.push_back((j + x[1] * 2 - 0.5) * h / lh);
                  b.push_back((x[2] * 2) * (x[2] * 2) * biases[2 * n]);
                  b.push_back((x[3] * 2) * (x[3] * 2) * biases[2 * n + 1]);
                }(pred, &swap_data[2], param_.anchors.data(),
                  param_.mask[n + mask_offset], cx, cy, w, h,
                  param_.net_input_w, param_.net_input_h);

                predict.idx = b;
                predict.x = pred[0];
                predict.y = pred[1];
                predict.w = pred[2];
                predict.h = pred[3];
                predict.classType = arg_max;
                predict.confidence = swap_data[1];
                predicts.push_back(predict);
              }
            } else {
              sigmoid_batch(swap_data, 4);
              // Pmax = Pmax * Po
              swap_data[1] = swap_data[0] * swap_data[1];

              if (swap_data[1] > param_.obj_threshold) {
                [&](std::vector<float> &b, float *x, double *biases, int n,
                    int i, int j, int lw, int lh, int w, int h) {
                  b.clear();
                  b.push_back((i + (x[0])) / lw);
                  b.push_back((j + (x[1])) / lh);
                  b.push_back(exp(x[2]) * biases[2 * n] / (w));
                  b.push_back(exp(x[3]) * biases[2 * n + 1] / (h));
                }(pred, &swap_data[2], param_.anchors.data(),
                  param_.mask[n + mask_offset], cx, cy, w, h,
                  param_.net_input_w, param_.net_input_h);

                predict.idx = b;
                predict.x = pred[0];
                predict.y = pred[1];
                predict.w = pred[2];
                predict.h = pred[3];
                predict.classType = arg_max;
                predict.confidence = swap_data[1];
                predicts.push_back(predict);
              }
            }
          }
        }
      }
      mask_offset += param_.num_boxes;
    }

    // NMS for each image
    std::vector<int> idxes;
    idxes.clear();

    int num_kept = 0;
    if (predicts.size() > 0) {
      std::stable_sort(
          predicts.begin(), predicts.end(),
          [](const PredictionResult &box1, const PredictionResult &box2) {
            return box1.confidence > box2.confidence;
          });
      // sprintf(str, "Sort Box (batch %d)", b);

      ApplyNms_opt(predicts, idxes, param_.nms_threshold);
      num_kept = idxes.size();
      // sprintf(str, "NMS %d Boxes (batch %d)", num_kept, b);

      if (param_.keep_topk > 0) {
        if (num_kept > param_.keep_topk)
          num_kept = param_.keep_topk;
      } else {
        if (num_kept > KEEP_TOP_K)
          num_kept = KEEP_TOP_K;
      }

      for (int i = 0; i < num_kept; i++) {
        total_preds.push_back(predicts[idxes[i]]);
      }
      total_num += num_kept;
    }
  }

  if (total_num == 0) {
    total_num = num;
    // Generate fake results per image.
    for (int i = 0; i < num; ++i) {
      top_data[0] = i;
      for (int j = 1; j < 7; ++j)
        top_data[j] = -1;
      top_data += 7;
    }
  } else {
    for (int i = 0; i < total_num; i++) {
      top_data[i * 7 + 0] = total_preds[i].idx;        // Image_Id
      top_data[i * 7 + 1] = total_preds[i].classType;  // label
      top_data[i * 7 + 2] = total_preds[i].confidence; // confidence
      top_data[i * 7 + 3] = total_preds[i].x;
      top_data[i * 7 + 4] = total_preds[i].y;
      top_data[i * 7 + 5] = total_preds[i].w;
      top_data[i * 7 + 6] = total_preds[i].h;
    }
  }
}

Yolov5DetectionFunc::Yolov5DetectionFunc(YoloDetParam &param) : param_(param) {}

void Yolov5DetectionFunc::invoke() {
  auto top_data = param_.output.ptr;
  memset(top_data, 0, param_.output.size);
  int batch_num = param_.inputs[0].shape[0];
  int box_num = param_.inputs[0].shape[1];
  int box_len = param_.inputs[0].shape[2];
  int agnostic_nms = param_.agnostic_nms;

  std::vector<PredictionResult> total_preds;
  // #pragma omp parallel for schedule(static, omp_schedule(batch_num * 4))
  for (int b = 0; b < batch_num; b++) {
    std::vector<PredictionResult> preds;
    //================================
    // box decode
    //================================
    for (int j = 0; j < box_num; j++) {
      const float *input_data =
          param_.inputs[0].ptr + b * box_num * box_len + j * box_len;
      if (input_data[4] > param_.obj_threshold) {
        float max_value = -10000;
        int max_idx = 0;
        for (int l = 5; l < box_len; l++) {
          if (input_data[4] * input_data[l] > max_value) {
            max_value = input_data[4] * input_data[l];
            max_idx = l - 5;
          }
        }
        if (max_value >= param_.obj_threshold) {
          PredictionResult pred;
          pred.classType = max_idx;
          pred.confidence = max_value;
          pred.idx = b;
          pred.x = input_data[0];
          pred.y = input_data[1];
          pred.w = input_data[2];
          pred.h = input_data[3];
          preds.push_back(pred);
        }
      }
    }
    //================================
    // NMS for each image
    //================================
    std::vector<int> idxes;
    idxes.clear();

    int num_kept = 0;
    if (preds.size() > 0) {
      std::stable_sort(
          preds.begin(), preds.end(),
          [](const PredictionResult &box1, const PredictionResult &box2) {
            return box1.confidence > box2.confidence;
          });

      ApplyNms_opt(preds, idxes, param_.nms_threshold, agnostic_nms);
      num_kept = idxes.size();

      if (param_.keep_topk > 0) {
        if (num_kept > param_.keep_topk)
          num_kept = param_.keep_topk;
      } else {
        if (num_kept > KEEP_TOP_K)
          num_kept = KEEP_TOP_K;
      }

      for (int i = 0; i < num_kept; i++) {
        total_preds.push_back(preds[idxes[i]]);
      }
    }
  }

  for (int i = 0; i < total_preds.size(); i++) {
    top_data[i * 7 + 0] = total_preds[i].idx;        // Image_Id
    top_data[i * 7 + 1] = total_preds[i].classType;  // label
    top_data[i * 7 + 2] = total_preds[i].confidence; // confidence
    top_data[i * 7 + 3] = total_preds[i].x;
    top_data[i * 7 + 4] = total_preds[i].y;
    top_data[i * 7 + 5] = total_preds[i].w;
    top_data[i * 7 + 6] = total_preds[i].h;
  }
}
Yolov8DetectionFunc::Yolov8DetectionFunc(YoloDetParam &param) : param_(param) {}

void Yolov8DetectionFunc::invoke() {

  auto top_data = param_.output.ptr;
  memset(top_data, 0, param_.output.size);
  int batch_num = param_.inputs[0].shape[0];
  int box_len = param_.inputs[0].shape[1]; // 84   (x y w h + 80 class_score)
  int box_num = param_.inputs[0].shape[2]; // 8400 box
  int agnostic_nms = param_.agnostic_nms;

  std::vector<PredictionResult> total_preds;

  for (int b = 0; b < batch_num; b++) {
    std::vector<PredictionResult> preds;
    //================================
    // box decode
    //================================
    for (int j = 0; j < box_num; j++) {
      const float *input_data =
          param_.inputs[0].ptr + b * box_num * box_len +
          j; // TODO:transpose input_data [1,84,box_num]->[1,box_num,84] when
             // canonicalizing may faster
      float max_value = -10000;
      int max_idx = 0;
      for (int l = 4; l < box_len; l++) {
        const float *input_element = input_data + l * box_num;
        if (*input_element > max_value) {
          max_value = *input_element;
          max_idx = l - 4;
        }
        if (max_value >= param_.obj_threshold) {
          PredictionResult pred;
          pred.classType = max_idx;
          pred.confidence = max_value;
          pred.idx = b;
          pred.x = *(input_data);
          pred.y = *(input_data + box_num);
          pred.w = *(input_data + 2 * box_num);
          pred.h = *(input_data + 3 * box_num);
          preds.push_back(pred);
        }
      }
    }

    //================================
    // NMS for each image
    //================================
    std::vector<int> idxes;
    idxes.clear();

    int num_kept = 0;
    if (preds.size() > 0) {
      std::stable_sort(
          preds.begin(), preds.end(),
          [](const PredictionResult &box1, const PredictionResult &box2) {
            return box1.confidence > box2.confidence;
          });

      ApplyNms_opt(preds, idxes, param_.nms_threshold, agnostic_nms);
      num_kept = idxes.size();

      if (param_.keep_topk > 0) {
        if (num_kept > param_.keep_topk)
          num_kept = param_.keep_topk;
      } else {
        if (num_kept > KEEP_TOP_K)
          num_kept = KEEP_TOP_K;
      }

      for (int i = 0; i < num_kept; i++) {
        total_preds.push_back(preds[idxes[i]]);
      }
    }
  }

  for (int i = 0; i < total_preds.size(); i++) {
    top_data[i * 7 + 0] = total_preds[i].idx;        // Image_Id
    top_data[i * 7 + 1] = total_preds[i].classType;  // label
    top_data[i * 7 + 2] = total_preds[i].confidence; // confidence
    top_data[i * 7 + 3] = total_preds[i].x;
    top_data[i * 7 + 4] = total_preds[i].y;
    top_data[i * 7 + 5] = total_preds[i].w;
    top_data[i * 7 + 6] = total_preds[i].h;
  }
}

NmsFunc::NmsFunc(NmsParam &param) : param_(param) {}

float NmsFunc::iou(const float *box, const int i, const int j) {
  // box:[y1, x1, y2, x2]
  const float *box_i = box + i * 4;
  const float *box_j = box + j * 4;
  const float ymax_i = (box_i[0] > box_i[2]) ? box_i[0] : box_i[2];
  const float ymin_i = (box_i[0] < box_i[2]) ? box_i[0] : box_i[2];
  const float xmax_i = (box_i[1] > box_i[3]) ? box_i[1] : box_i[3];
  const float xmin_i = (box_i[1] < box_i[3]) ? box_i[1] : box_i[3];
  const float ymax_j = (box_j[0] > box_j[2]) ? box_j[0] : box_j[2];
  const float ymin_j = (box_j[0] < box_j[2]) ? box_j[0] : box_j[2];
  const float xmax_j = (box_j[1] > box_j[3]) ? box_j[1] : box_j[3];
  const float xmin_j = (box_j[1] < box_j[3]) ? box_j[1] : box_j[3];
  const float area_i = (ymax_i - ymin_i) * (xmax_i - xmin_i);
  if (area_i <= 0.f)
    return 0.f;
  const float area_j = (ymax_j - ymin_j) * (xmax_j - xmin_j);
  if (area_j <= 0.f)
    return 0.f;
  const float ymax_inter = (ymax_i < ymax_j) ? ymax_i : ymax_j;
  const float ymin_inter = (ymin_i > ymin_j) ? ymin_i : ymin_j;
  const float y_inter =
      (ymax_inter > ymin_inter) ? (ymax_inter - ymin_inter) : 0;
  if (y_inter == 0.f)
    return 0.f;
  const float xmax_inter = (xmax_i < xmax_j) ? xmax_i : xmax_j;
  const float xmin_inter = (xmin_i > xmin_j) ? xmin_i : xmin_j;
  const float x_inter =
      (xmax_inter > xmin_inter) ? (xmax_inter - xmin_inter) : 0;
  if (x_inter == 0.f)
    return 0.f;
  const float area_inter = y_inter * x_inter;
  const float iou = area_inter / (area_i + area_j - area_inter);
  return iou;
}

int NmsFunc::invoke() {
  // boxes: [num_batches, spatial_dimension, 4]
  // scores: [num_batches, num_classes, spatial_dimension]
  assert(2 == param_.inputs.size());
  float *box = param_.box;
  float *score = param_.score;
  const int num_boxes = param_.inputs[0].shape[1];
  assert(4 == param_.inputs[0].shape[2]);
  assert(num_boxes == param_.inputs[1].shape[2]);
  const int batch_num = param_.inputs[1].shape[0];
  const int num_class = param_.inputs[1].shape[1];
  float iou_threshold = param_.iou_threshold;
  float score_threshold = param_.score_threshold;
  int max_output_size = param_.max_output_boxes_per_class;
  max_output_size = (max_output_size > num_boxes) ? num_boxes : max_output_size;
  struct Candidate {
    int box_index;
    float score;
    int begin_index;
  };
  // align with tpu algorithm
  auto cmp = [](const Candidate i, const Candidate j) {
    if (i.score != j.score)
      return i.score < j.score;
    else {
      return i.box_index > j.box_index;
    }
  };

  int num_selected_indices = 0;
  for (int n = 0; n < batch_num; ++n) {
    for (int c = 0; c < num_class; ++c) {
      const int score_offset = (n * num_class + c) * num_boxes;
      const int box_offset = (n * num_boxes * 4);
      std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)>
          candicate_prior_queue(cmp);
      for (int i = 0; i < num_boxes; ++i) {
        if (score[score_offset + i] > score_threshold) {
          candicate_prior_queue.emplace(
              Candidate({i, score[score_offset + i], 0}));
        }
      }

      std::vector<int> selected_index;
      Candidate next_cand;
      float iou;
      while (selected_index.size() < max_output_size &&
             (!candicate_prior_queue.empty())) {
        next_cand = candicate_prior_queue.top();
        candicate_prior_queue.pop();

        bool selected = true;
        for (int i = static_cast<int>(selected_index.size()) - 1; i >= 0; --i) {
          iou = NmsFunc::iou((box + box_offset), next_cand.box_index,
                             selected_index[i]);
          if ((iou > iou_threshold) && iou != 0.f) {
            selected = false;
            break;
          }
        }

        if (selected == true) {
          selected_index.push_back(next_cand.box_index);
        }
      }
      int *output =
          reinterpret_cast<int *>(param_.output) + (num_selected_indices * 3);
      for (int i = 0; i < selected_index.size(); i++) {
        output[i * 3] = n;
        output[i * 3 + 1] = c;
        output[i * 3 + 2] = selected_index[i];
      }
      num_selected_indices += static_cast<int>(selected_index.size());
    }
  }
  return num_selected_indices * 3;
}

BMCpuOp::BMCpuOp(tpu::GenericCpuOp &op) : op_(op) {
  this->op_name = op.getCpuOpName().str();
  this->op_type = this->getCpuOpType();
  this->getCpuParam();
}

int BMCpuOp::getCpuOpType() {
  return llvm::StringSwitch<int>(op_.getCpuOpName())
      .Case("topk", CPU_TOPK)
      .Case("onnx_nms", CPU_ONNX_NMS)
      .Case("gathernd_tf", CPU_GATHERND_TF)
      .Case("gatherelements_pt", CPU_GATHER_PT)
      .Case("tensor_scatter", CPU_TENSOR_SCATTER_OP)
      .Case("grid_sampler", CPU_GRID_SAMPLER)
      .Case("deform_gather", CPU_DEFORM_GATHER)
      .Case("roi_align", CPU_PYTORCH_ROI_ALIGN)
      .Default(CPU_LAYER_UNKNOW);
}

void BMCpuOp::get_topk_param() {
  cpu_topk_param_t cpu_param{};
  mlir::DictionaryAttr paramDic = op_.getParam().value();
  cpu_param.k = paramDic.get("K").cast<IntegerAttr>().getInt();
  cpu_param.axis = paramDic.get("axis").cast<IntegerAttr>().getInt();
  cpu_param.sorted = paramDic.get("sorted").cast<BoolAttr>().getValue();
  cpu_param.descending = paramDic.get("largest").cast<BoolAttr>().getValue();
  cpu_param.values_used_only =
      paramDic.get("values_used_only").cast<BoolAttr>().getValue();
  this->param_size = sizeof(cpu_topk_param_t);
  this->param = (void *)malloc(this->param_size);
  memcpy(this->param, &cpu_param, this->param_size);
}

void BMCpuOp::get_onnx_nms_param() {
  cpu_onnx_nms_param_t cpu_param{};
  mlir::DictionaryAttr paramDic = op_.getParam().value();
  cpu_param.max_output_size =
      paramDic.get("max_output_size").cast<IntegerAttr>().getInt();
  cpu_param.center_point_box =
      paramDic.get("center_point_box").cast<IntegerAttr>().getInt();
  this->param_size = sizeof(cpu_onnx_nms_param_t);
  this->param = (void *)malloc(this->param_size);
  memcpy(this->param, &cpu_param, this->param_size);
}

void BMCpuOp::get_gather_nd_tf_param() {
  cpu_gathernd_t cpu_param{};
  mlir::DictionaryAttr paramDic = op_.getParam().value();
  cpu_param.indice_is_int = true;
  cpu_param.batch_dims =
      paramDic.get("batch_dims").cast<IntegerAttr>().getInt();
  this->param_size = sizeof(cpu_gathernd_t);
  this->param = (void *)malloc(this->param_size);
  memcpy(this->param, &cpu_param, this->param_size);
}

void BMCpuOp::get_gatherelements_pt_param() {
  cpu_gather_t cpu_param{};
  mlir::DictionaryAttr paramDic = op_.getParam().value();
  cpu_param.axis = paramDic.get("axis").cast<IntegerAttr>().getInt();
  this->param_size = sizeof(cpu_gather_t);
  this->param = (void *)malloc(this->param_size);
  memcpy(this->param, &cpu_param, this->param_size);
}

void BMCpuOp::get_tensor_scatter_param() {
  cpu_tensor_scatter_op_param_t cpu_param{};
  mlir::DictionaryAttr paramDic = op_.getParam().value();
  cpu_param.input_dtype =
      (CPU_DATA_TYPE_T)BM168x::getDataType(op_.getInputs()[0]);
  cpu_param.scatter_op =
      (CPU_SCATTER_OP_T)paramDic.get("reduction").cast<IntegerAttr>().getInt();
  this->param_size = sizeof(cpu_tensor_scatter_op_param_t);
  this->param = (void *)malloc(this->param_size);
  memcpy(this->param, &cpu_param, this->param_size);
}

void BMCpuOp::get_grid_sampler_param() {
  cpu_grid_sampler_param_t cpu_param{};
  mlir::DictionaryAttr paramDic = op_.getParam().value();
  cpu_param.mode = paramDic.get("mode").cast<IntegerAttr>().getInt();
  cpu_param.padding_mode =
      paramDic.get("padding_mode").cast<IntegerAttr>().getInt();
  cpu_param.align_corners =
      paramDic.get("align_corners").cast<BoolAttr>().getValue();
  this->param_size = sizeof(cpu_grid_sampler_param_t);
  this->param = (void *)malloc(this->param_size);
  memcpy(this->param, &cpu_param, this->param_size);
}

void BMCpuOp::get_deform_gather_param() {
  cpu_deform_gather_param_t cpu_param{};
  mlir::DictionaryAttr paramDic = op_.getParam().value();
  cpu_param.mode = DEFORM_TORCHVISION_MODE;
  cpu_param.modulated = paramDic.get("use_mask").cast<BoolAttr>().getValue();
  cpu_param.deform_groups =
      paramDic.get("deform_group").cast<IntegerAttr>().getInt();
  cpu_param.kh = paramDic.get("kh").cast<IntegerAttr>().getInt();
  cpu_param.kw = paramDic.get("kw").cast<IntegerAttr>().getInt();
  cpu_param.pad_t = paramDic.get("pad_t").cast<IntegerAttr>().getInt();
  cpu_param.pad_b = paramDic.get("pad_b").cast<IntegerAttr>().getInt();
  cpu_param.pad_l = paramDic.get("pad_l").cast<IntegerAttr>().getInt();
  cpu_param.pad_r = paramDic.get("pad_r").cast<IntegerAttr>().getInt();
  cpu_param.stride_h = paramDic.get("stride_h").cast<IntegerAttr>().getInt();
  cpu_param.stride_w = paramDic.get("stride_w").cast<IntegerAttr>().getInt();
  cpu_param.dilation_h =
      paramDic.get("dilation_h").cast<IntegerAttr>().getInt();
  cpu_param.dilation_w =
      paramDic.get("dilation_w").cast<IntegerAttr>().getInt();
  this->param_size = sizeof(cpu_deform_gather_param_t);
  this->param = (void *)malloc(this->param_size);
  memcpy(this->param, &cpu_param, this->param_size);
}

void BMCpuOp::get_roi_align_param() {
  cpu_pytorch_roi_align_param_t cpu_param{};
  mlir::DictionaryAttr paramDic = op_.getParam().value();
  cpu_param.pooled_height =
      paramDic.get("output_height").cast<IntegerAttr>().getInt();
  cpu_param.pooled_width =
      paramDic.get("output_width").cast<IntegerAttr>().getInt();
  cpu_param.spatial_scale =
      paramDic.get("spatial_scale").cast<FloatAttr>().getValueAsDouble();
  cpu_param.sampling_ratio =
      paramDic.get("sampling_ratio").cast<IntegerAttr>().getInt();
  cpu_param.align = paramDic.get("align_corners").cast<BoolAttr>().getValue();
  this->param_size = sizeof(cpu_pytorch_roi_align_param_t);
  this->param = (void *)malloc(this->param_size);
  memcpy(this->param, &cpu_param, this->param_size);
}

void BMCpuOp::getCpuParam() {
  switch (this->op_type) {
  case CPU_TOPK:
    get_topk_param();
    break;
  case CPU_ONNX_NMS:
    get_onnx_nms_param();
    break;
  case CPU_GATHERND_TF:
    get_gather_nd_tf_param();
    break;
  case CPU_GATHER_PT:
    get_gatherelements_pt_param();
    break;
  case CPU_TENSOR_SCATTER_OP:
    get_tensor_scatter_param();
    break;
  case CPU_GRID_SAMPLER:
    get_grid_sampler_param();
    break;
  case CPU_DEFORM_GATHER:
    get_deform_gather_param();
    break;
  case CPU_PYTORCH_ROI_ALIGN:
    get_roi_align_param();
    break;
  case CPU_LAYER_UNKNOW:
    llvm_unreachable("Unknow CPU Op");
  }
}

InstanceNormFunc::InstanceNormFunc(InstanceNormParam &param) : param_(param) {}

void InstanceNormFunc::invoke() {
  assert(param_.inputs.size() >= 2);
  int n = param_.output.shape[0];
  int c = param_.output.shape[1];
  int h = param_.output.shape.size() > 2 ? param_.output.shape[2] : 1;
  int w = param_.output.shape.size() > 3 ? param_.output.shape[3] : 1;

  // gamma_value * (x - mean_value) / np.sqrt(var_value + epsilon) + beta_value
  // epsilon default is 1e-5
  // please reference onnx
  // [implementation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#InstanceNormalization)
  // caffe2 cpu
  // [implementation](https://caffe2.ai/doxygen-c/html/instance__norm__op_8cc_source.html)

  std::vector<float> _mean(c);
  std::vector<float> _variance(c);
  int hw = h * w;

  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      int channel_shift = ni * c * h * w + ci * h * w;
      auto start = param_.inputs[0].ptr + channel_shift;
      _mean[ci] = std::accumulate(start, start + hw, 0.0) / hw;

      float var = 0;

      for (int i = 0; i < hw; ++i) {
        var += pow(param_.inputs[0].ptr[ni * c * h * w + ci * h * w + i] -
                       _mean[ci],
                   2);
      }
      var = (var) / hw;
      _variance[ci] = var;
    }
  }

  auto mean = &_mean;
  auto variance = &_variance;

  // duplicate code from bn
  float scale_factor = 1 / param_.inputs[1].ptr[0];
  for (int i = 0; i < c; ++i) {
    mean->at(i) = mean->at(i) * scale_factor;
    variance->at(i) = variance->at(i) * scale_factor;
  }
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int i = 0; i < h * w; ++i) {
        auto x = param_.inputs[0].ptr[ni * c * h * w + ci * h * w + i] -
                 mean->at(ci);
        auto d = sqrt(variance->at(ci) + param_.eps);
        param_.output.ptr[ni * c * h * w + ci * h * w + i] = x / d;
        if (fabs(variance->at(ci)) <= param_.eps &&
            fabs(mean->at(ci)) <= 1e-8 &&
            fabs(param_.inputs[0].ptr[ni * c * h * w + ci * h * w + i]) >=
                1.0e-4 &&
            fabs(param_.output.ptr[ni * c * h * w + ci * h * w + i]) >=
                1.0e-2) {
          llvm::errs()
              << "WARNING: BN: var too small, i=" << i
              << ", v=" << std::to_string(variance->at(ci))
              << ", m=" << std::to_string(mean->at(ci)) << "\n               "
              << ", i="
              << std::to_string(
                     param_.inputs[0].ptr[ni * c * h * w + ci * h * w + i])
              << ", x=" << std::to_string(x) << ", d=" << std::to_string(d)
              << ", o="
              << std::to_string(
                     param_.output.ptr[ni * c * h * w + ci * h * w + i])
              << "\n";
        }
      }
    }
  }
  for (int i = 0; i < c; ++i) {
    mean->at(i) = mean->at(i) * param_.inputs[1].ptr[0];
    variance->at(i) = variance->at(i) * param_.inputs[1].ptr[0];
  }
}

static void my_interp(const int channels, const float *data1, const int x1,
                      const int y1, const int height1, const int width1,
                      const int Height1, const int Width1, float *data2,
                      const int x2, const int y2, const int height2,
                      const int width2, const int Height2, const int Width2) {
  bool packed = false;

  assert(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 &&
         y2 >= 0 && height2 > 0 && width2 > 0);
  assert(Width1 >= width1 + x1 && Height1 >= height1 + y1 &&
         Width2 >= width2 + x2 && Height2 >= height2 + y2);

  // special case: just copy
  if (height1 == height2 && width1 == width2) {
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < width2; ++w2) {
        const int w1 = w2;
        if (packed) {
          const float *pos1 =
              &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
          float *pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
          for (int c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1++;
            pos2++;
          }
        } else {
          const float *pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
          float *pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
          for (int c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1 += Width1 * Height1;
            pos2 += Width2 * Height2;
          }
        }
      }
    }
    return;
  }
  const float rheight =
      (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth =
      (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = float(1.) - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const float w1lambda = w1r - w1;
      const float w0lambda = float(1.) - w1lambda;
      if (packed) {
        const float *pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
        float *pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
        for (int c = 0; c < channels; ++c) {
          pos2[0] =
              h0lambda *
                  (w0lambda * pos1[0] + w1lambda * pos1[channels * w1p]) +
              h1lambda * (w0lambda * pos1[channels * h1p * Width1] +
                          w1lambda * pos1[channels * (h1p * Width1 + w1p)]);
          pos1++;
          pos2++;
        }
      } else {
        const float *pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
        float *pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
        for (int c = 0; c < channels; ++c) {
          pos2[0] = h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
                    h1lambda * (w0lambda * pos1[h1p * Width1] +
                                w1lambda * pos1[h1p * Width1 + w1p]);
          pos1 += Width1 * Height1;
          pos2 += Width2 * Height2;
        }
      }
    }
  }
}

static float coordinate_transform(float x_resized, float x_scale,
                                  float length_resized, bool pytorch) {
  // please refer NativeCpuImplementation.cpp for more details
  if (pytorch) {
    return length_resized > 1 ? ((x_resized + 0.5f) / x_scale - 0.5f) : 0.0f;
  } else {
    return (x_resized + 0.5f) / x_scale - 0.5f;
  }
}

// copy from caffe_cpu_interp2
static void interp(const int channels, const float *data1, const int x1,
                   const int y1, const int height1, const int width1,
                   const int Height1, const int Width1, float *data2,
                   const int x2, const int y2, const int height2,
                   const int width2, const int Height2, const int Width2) {
  bool packed = false;

  assert(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 &&
         y2 >= 0 && height2 > 0 && width2 > 0);
  assert(Width1 >= width1 + x1 && Height1 >= height1 + y1 &&
         Width2 >= width2 + x2 && Height2 >= height2 + y2);

  // special case: just copy
  if (height1 == height2 && width1 == width2) {
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < width2; ++w2) {
        const int w1 = w2;
        if (packed) {
          const float *pos1 =
              &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
          float *pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
          for (int c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1++;
            pos2++;
          }
        } else {
          const float *pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
          float *pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
          for (int c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1 += Width1 * Height1;
            pos2 += Width2 * Height2;
          }
        }
      }
    }
    return;
  }
  const float rheight =
      (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth =
      (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = float(1.) - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const float w1lambda = w1r - w1;
      const float w0lambda = float(1.) - w1lambda;
      if (packed) {
        const float *pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
        float *pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
        for (int c = 0; c < channels; ++c) {
          pos2[0] =
              h0lambda *
                  (w0lambda * pos1[0] + w1lambda * pos1[channels * w1p]) +
              h1lambda * (w0lambda * pos1[channels * h1p * Width1] +
                          w1lambda * pos1[channels * (h1p * Width1 + w1p)]);
          pos1++;
          pos2++;
        }
      } else {
        const float *pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
        float *pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
        for (int c = 0; c < channels; ++c) {
          pos2[0] = h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
                    h1lambda * (w0lambda * pos1[h1p * Width1] +
                                w1lambda * pos1[h1p * Width1 + w1p]);
          pos1 += Width1 * Height1;
          pos2 += Width2 * Height2;
        }
      }
    }
  }
}

template <typename T>
static void upsampleBilinear(int64_t batch_size, int64_t num_channels,
                             int64_t input_height, int64_t input_width,
                             float height_scale, float width_scale,
                             const T *Xdata, T *Ydata, bool pytorch = false) {
  int64_t output_width = static_cast<int64_t>(input_width * width_scale);
  int64_t output_height = static_cast<int64_t>(input_height * height_scale);

  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t c = 0; c < num_channels; ++c) {
      for (int64_t y = 0; y < output_height; ++y) {
        float in_y =
            std::min(y / height_scale, static_cast<float>(input_height - 1));
        in_y = height_scale == 1
                   ? static_cast<float>(y)
                   : coordinate_transform(static_cast<float>(y), height_scale,
                                          static_cast<float>(output_height),
                                          pytorch);
        in_y = std::max(0.0f,
                        std::min(in_y, static_cast<float>(input_height - 1)));

        const int64_t in_y1 =
            std::min(static_cast<int64_t>(in_y), input_height - 1);
        const int64_t in_y2 = std::min(in_y1 + 1, input_height - 1);
        float dy1 = fabs(in_y - in_y1);
        float dy2 = fabs(in_y - in_y2);
        if (in_y1 == in_y2) {
          dy1 = 0.5f;
          dy2 = 0.5f;
        }

        const int64_t input_width_mul_y1 = input_width * in_y1;
        const int64_t input_width_mul_y2 = input_width * in_y2;

        for (int64_t x = 0; x < output_width; ++x) {
          float in_x =
              std::min(x / width_scale, static_cast<float>(input_width - 1));
          in_x = width_scale == 1
                     ? static_cast<float>(x)
                     : coordinate_transform(static_cast<float>(x), width_scale,
                                            static_cast<float>(output_width),
                                            pytorch);
          in_x = std::max(0.0f,
                          std::min(in_x, static_cast<float>(input_width - 1)));

          const int64_t in_x1 =
              std::min(static_cast<int64_t>(in_x), input_width - 1);
          const int64_t in_x2 = std::min(in_x1 + 1, input_width - 1);

          float dx1 = std::abs(in_x - in_x1);
          float dx2 = std::abs(in_x - in_x2);
          if (in_x1 == in_x2) {
            dx1 = 0.5f;
            dx2 = 0.5f;
          }

          T X11 = Xdata[input_width_mul_y1 + in_x1];
          T X21 = Xdata[input_width_mul_y1 + in_x2];
          T X12 = Xdata[input_width_mul_y2 + in_x1];
          T X22 = Xdata[input_width_mul_y2 + in_x2];

          Ydata[output_width * y + x] =
              static_cast<T>(dx2 * dy2 * X11 + dx1 * dy2 * X21 +
                             dx2 * dy1 * X12 + dx1 * dy1 * X22);
        }
      }
      Xdata += input_height * input_width;
      Ydata += output_width * output_height;
    }
  }
}

static void interp_linear(float *input, float *output, int n, int c, int ih,
                          int iw, int oh, int ow, bool pytorch = false) {
  float height_scale = (float)oh / (float)ih;
  float width_scale = (float)ow / (float)iw;
  upsampleBilinear<float>(n, c, ih, iw, height_scale, width_scale, input,
                          output, pytorch);
}

static void interp_neast(float *input, float *output, int n, int c, int ih,
                         int iw, int oh, int ow, bool half_pixel) {
  int nc = n * c;
  float scale_h = ((float)ih) / oh;
  float scale_w = ((float)iw) / ow;
#pragma omp parallel for schedule(static, omp_schedule(nc))
  for (int i = 0; i < nc; i++) {
    for (int h = 0; h < oh; h++) {
      for (int w = 0; w < ow; w++) {
        int o_index = i * oh * ow + h * ow + w;
        int h_resized = (int)(half_pixel ? std::ceil((h + 0.5) * scale_h - 1.0)
                                         : h * scale_h);
        int w_resized = (int)(half_pixel ? std::ceil((w + 0.5) * scale_w - 1.0)
                                         : w * scale_w);
        int i_index = i * ih * iw + h_resized * iw + w_resized;
        output[o_index] = input[i_index];
      }
    }
  }
}

static inline float value(float *input, int w, int ih, int iw) {
  return input[ih * w + iw];
}

static float value(float *input, int w, float fh, float fw) {
  int h0 = std::floor(fh);
  int h1 = std::ceil(fh);
  int w0 = std::floor(fw);
  int w1 = std::ceil(fw);
  if (h0 == fh && w0 == fw) {
    return value(input, w, h0, w0);
  }
  if (h0 == fh) {
    return value(input, w, h0, w0) * (w1 - fw) +
           value(input, w, h0, w1) * (fw - w0);
  }
  if (w0 == fw) {
    return value(input, w, h0, w0) * (h1 - fh) +
           value(input, w, h1, w0) * (fh - h0);
  }
  float scale0 = (w1 - fw) * (h1 - fh);
  float scale1 = (fw - w0) * (h1 - fh);
  float scale2 = (w1 - fw) * (fh - h0);
  float scale3 = (fw - w0) * (fh - h0);
  return value(input, w, h0, w0) * scale0 + value(input, w, h0, w1) * scale1 +
         value(input, w, h1, w0) * scale2 + value(input, w, h1, w1) * scale3;
}

static void interp_asymmetric(float *input, float *output, int n, int c, int ih,
                              int iw, int oh, int ow) {
  int nc = n * c;
  float scale_h = (float)ih / oh;
  float scale_w = (float)iw / ow;
#pragma omp parallel for schedule(static, omp_schedule(nc))
  for (int i = 0; i < nc; i++) {
    for (int h = 0; h < oh; h++) {
      for (int w = 0; w < ow; w++) {
        int o_index = i * oh * ow + h * ow + w;
        float fh = std::min(h * scale_h, (float)(ih - 1));
        float fw = std::min(w * scale_w, (float)(iw - 1));
        output[o_index] = value(input + i * ih * iw, iw, fh, fw);
      }
    }
  }
}

InterpFunc::InterpFunc(InterpParam &param) : param_(param) {}

void InterpFunc::invoke() {
  auto &input_shape = param_.inputs[0].shape;
  auto &output_shape = param_.output.shape;
  int in = input_shape[0];
  int ic = input_shape[1];
  int ih = input_shape[2];
  int iw = input_shape[3];
  int oh = output_shape[2];
  int ow = output_shape[3];
  int height_in_ = ih;
  int width_in_ = iw;
  int height_in_eff_ = height_in_ + param_.pad_beg + param_.pad_end;
  int width_in_eff_ = width_in_ + param_.pad_beg + param_.pad_end;
  int height_out_ = -1;
  int width_out_ = -1;
  if (param_.shrink_factor && !param_.zoom_factor) {
    assert(param_.shrink_factor >= 1 && "Shrink factor must be positive");
    height_out_ = (height_in_eff_ - 1) / param_.shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / param_.shrink_factor + 1;
  } else if (param_.zoom_factor && !param_.shrink_factor) {
    assert(param_.zoom_factor >= 1 && "Zoom factor must be positive");
    height_out_ =
        height_in_eff_ + (height_in_eff_ - 1) * (param_.zoom_factor - 1);
    width_out_ = width_in_eff_ + (width_in_eff_ - 1) * (param_.zoom_factor - 1);
  } else if (param_.height && param_.width) {
    height_out_ = param_.height;
    width_out_ = param_.width;
  } else if (param_.zoom_factor && param_.shrink_factor) {
    assert(param_.shrink_factor >= 1 && "Shrink factor must be positive");
    assert(param_.zoom_factor >= 1 && "Zoom factor must be positive");

    height_out_ = (height_in_eff_ - 1) / param_.shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / param_.shrink_factor + 1;
    height_out_ = height_out_ + (height_out_ - 1) * (param_.zoom_factor - 1);
    width_out_ = width_out_ + (width_out_ - 1) * (param_.zoom_factor - 1);
  }
  if (param_.coordinate_transformation_mode == "align_corners") {
    // TODO: verify pad_end_ > 0
    my_interp(in * ic, param_.inputs[0].ptr, -param_.pad_beg, -param_.pad_beg,
              height_in_eff_, width_in_eff_, height_in_, width_in_,
              param_.output.ptr, 0, 0, height_out_, width_out_, height_out_,
              width_out_);
  } else if (param_.coordinate_transformation_mode == "half_pixel") {
    interp_linear(param_.inputs[0].ptr, param_.output.ptr, in, ic, ih, iw, oh,
                  ow);
  } else if (param_.coordinate_transformation_mode == "pytorch_half_pixel") {
    interp_linear(param_.inputs[0].ptr, param_.output.ptr, in, ic, ih, iw, oh,
                  ow, true);
  } else if (param_.coordinate_transformation_mode == "nearest_half_pixel") {
    interp_neast(param_.inputs[0].ptr, param_.output.ptr, in, ic, ih, iw, oh,
                 ow, true);
  } else if (param_.coordinate_transformation_mode == "nearest") {
    interp_neast(param_.inputs[0].ptr, param_.output.ptr, in, ic, ih, iw, oh,
                 ow, false);
  } else if (param_.coordinate_transformation_mode == "asymmetric") {
    interp_asymmetric(param_.inputs[0].ptr, param_.output.ptr, in, ic, ih, iw,
                      oh, ow);
  } else {
    llvm_unreachable("coordinate_transformation_model not support");
  }
}

EmbeddingFunc::EmbeddingFunc(EmbeddingParam &param) : param_(param) {}

void EmbeddingFunc::invoke() {
  auto &input_shape = param_.inputs[0].shape;
  auto &table_shape = param_.inputs[1].shape;
  auto &output_shape = param_.output.shape;
  auto input_data = param_.inputs[0].ptr;
  auto table_data = param_.inputs[1].ptr;
  auto output_data = param_.output.ptr;

  auto feature_dim = table_shape.back();
  assert(output_shape.back() == feature_dim && "must be the same feature dim");
  int64_t count = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                  std::multiplies<int64_t>());
  for (int64_t i = 0; i < count; i++) {
    auto index = (size_t)input_data[i];
    size_t table_offset = (size_t)index * feature_dim;
    auto out_offset = i * feature_dim;
    memcpy(output_data + out_offset, table_data + table_offset,
           feature_dim * sizeof(float));
    // if (mix_bf16 == false) {
    //   memcpy(output_data + out_offset, table_data + table_offset,
    //         feature_dim * sizeof(float));
    // } else {
    //   for (int64_t j = 0; j < feature_dim; j++) {
    //     output[out_offset + j] =
    //         BF16(BF16(table[table_offset + j] * scale_data->at(j)) +
    //         zeropoint_data->at(j));
    //   }
    // }
  }
}

ArgMaxFunc::ArgMaxFunc(ArgMaxParam &param) : param_(param) {}

void ArgMaxFunc::invoke() {
  auto data = param_.inputs[0].ptr;
  auto map = param_.inputs[1].ptr;
  auto indices = param_.outputs[0].ptr;
  float *values = nullptr;
  if (param_.outputs.size() > 1) {
    values = param_.outputs[1].ptr;
  }

  int outer_dim = std::accumulate(param_.inputs[0].shape.begin(),
                                  param_.inputs[0].shape.begin() + param_.axis,
                                  1, std::multiplies<int64_t>());
  int inner_dim = param_.inputs[0].shape[param_.axis];
  int tile_size = 256;
  int tile_num = (inner_dim + tile_size - 1) / tile_size;
  float max_val_fp32 = 0;
  for (int i = 0; i < outer_dim; ++i) {
    auto map_ptr = map + i * tile_num;
    float max_val = map_ptr[0];
    int idx = 0;
    // find max_val
    for (int j = 1; j < tile_num; j++) {
      if (map_ptr[j] > max_val) {
        max_val = map_ptr[j];
        idx = j;
      }
    }
    int offset = idx * tile_size;
    int len = std::min(inner_dim - offset, tile_size);
    auto ptr = data + i * inner_dim + offset;
    idx = 0;
    for (int j = 0; j < len; ++j) {
      if (ptr[j] == max_val) {
        idx = j;
        break;
      }
    }
    indices[i] = (float)(idx + offset);
    if (values) {
      if (!param_.fmt_i8) {
        max_val_fp32 = max_val;
      } else {
        max_val_fp32 = max_val * param_.scale;
      }
      values[i] = max_val_fp32;
    }
  }
}

GatherndFunc::GatherndFunc(GatherNDParam &param) : param_(param) {}

uint64_t GatherndFunc::gather_offset(std::vector<int64_t> input_shape,
                                     std::vector<int> gather_index) {
  uint64_t offset = 0;
  int dim_size = gather_index.size();
  uint64_t gap = 1;
  for (int i = dim_size - 1; i >= 0; i--) {
    offset += gather_index[i] * gap;
    gap *= input_shape[i];
  }
  return offset;
}

void GatherndFunc::invoke() {
  int batch_dims_size = 1;
  auto batch_dims = param_.batch_dims;
  auto input_info = param_.inputs[0];
  auto indices_info = param_.inputs[1];
  auto indices_shape = indices_info.shape;
  auto input_shape = input_info.shape;
  const float *input = input_info.ptr;
  const float *indices = indices_info.ptr;
  std::vector<int> indices_v(indices_info.size);
  for (int i = 0; i < indices_info.size; ++i) {
    indices_v[i] = (int)indices[i];
  }
  float *out = param_.output.ptr;

  for (int i = 0; i < batch_dims; ++i) {
    batch_dims_size *= indices_shape[i];
  }

  int channel = (indices_info.size / batch_dims_size) /
                indices_shape[indices_shape.size() - 1];
  assert(channel * indices_shape[indices_shape.size() - 1] * batch_dims_size ==
         indices_info.size);
  std::vector<int64_t> indices_new_shape = {
      batch_dims_size, channel, indices_shape[indices_shape.size() - 1]};
  std::vector<int64_t> input_new_shape = {batch_dims_size};
  for (int i = batch_dims; i < input_shape.size(); ++i) {
    input_new_shape.push_back(input_shape[i]);
  }

  uint64_t gather_eltment =
      param_.output.size / (indices_new_shape[0] * indices_new_shape[1]);
  assert(gather_eltment * indices_new_shape[0] * indices_new_shape[1] ==
         param_.output.size);
  for (int b = 0; b < indices_new_shape[0]; ++b) {
    for (int c = 0; c < indices_new_shape[1]; ++c) {
      std::vector<int> gather_index(indices_new_shape[2]);
      memcpy(gather_index.data(),
             (int *)indices_v.data() +
                 b * indices_new_shape[1] * indices_new_shape[2] +
                 c * indices_new_shape[2],
             indices_new_shape[2] * sizeof(int));
      gather_index.insert(gather_index.begin(), b);
      uint64_t offset = gather_offset(input_new_shape, gather_index);
      memcpy(out + (b * indices_new_shape[1] + c) * gather_eltment,
             input + offset * gather_eltment, gather_eltment * sizeof(float));
    }
  }
}

// tianjia
GatherElementsFunc::GatherElementsFunc(GatherElementsParam &param)
    : param_(param) {}

static inline void gather_dim1_0(float *dst, const float *src, const int *idx,
                                 int64_t *shape, int64_t *org_shape) {
  for (int i = 0; i < shape[0]; ++i) {
    *dst = src[*idx];
    ++dst;
    ++idx;
  }
}
static inline void gather_dim2_0(float *dst, const float *src, const int *idx,
                                 int64_t *shape, int64_t *org_shape) {
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      *dst = src[*idx * org_shape[1] + j];
      ++dst;
      ++idx;
    }
  }
}
static inline void gather_dim2_1(float *dst, const float *src, const int *idx,
                                 int64_t *shape, int64_t *org_shape) {
  for (int i = 0; i < shape[0]; ++i) {
    int idx_i = i * org_shape[1];
    for (int j = 0; j < shape[1]; ++j) {
      *dst = src[idx_i + *idx];
      ++dst;
      ++idx;
    }
  }
}
static inline void gather_dim3_0(float *dst, const float *src, const int *idx,
                                 int64_t *shape, int64_t *org_shape) {
  int shape_1_2 = org_shape[1] * org_shape[2];
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      int idx_j = j * org_shape[2];
      for (int k = 0; k < shape[2]; ++k) {
        *dst = src[*idx * shape_1_2 + idx_j + k];
        ++dst;
        ++idx;
      }
    }
  }
}
static inline void gather_dim3_1(float *dst, const float *src, const int *idx,
                                 int64_t *shape, int64_t *org_shape) {
  int shape_1_2 = org_shape[1] * org_shape[2];
  for (int i = 0; i < shape[0]; ++i) {
    int idx_i = i * shape_1_2;
    for (int j = 0; j < shape[1]; ++j) {
      for (int k = 0; k < shape[2]; ++k) {
        *dst = src[idx_i + *idx * org_shape[2] + k];
        ++dst;
        ++idx;
      }
    }
  }
}
static inline void gather_dim3_2(float *dst, const float *src, const int *idx,
                                 int64_t *shape, int64_t *org_shape) {
  int shape_1_2 = org_shape[1] * org_shape[2];
  for (int i = 0; i < shape[0]; ++i) {
    int idx_i = i * shape_1_2;
    for (int j = 0; j < shape[1]; ++j) {
      int idx_j = idx_i + j * org_shape[2];
      for (int k = 0; k < shape[2]; ++k) {
        *dst = src[idx_j + *idx];
        ++dst;
        ++idx;
      }
    }
  }
}
static inline void gather_dim4_0(float *dst, const float *src, const int *idx,
                                 int64_t *shape, int64_t *org_shape) {
  int shape_1_2_3 = org_shape[1] * org_shape[2] * org_shape[3];
  int shape_2_3 = org_shape[2] * org_shape[3];
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j) {
      int idx_j = j * shape_2_3;
      for (int k = 0; k < shape[2]; ++k) {
        int idx_k = idx_j + k * org_shape[3];
        for (int g = 0; g < shape[3]; ++g) {
          *dst = src[*idx * shape_1_2_3 + idx_k + g];
          ++dst;
          ++idx;
        }
      }
    }
  }
}
static inline void gather_dim4_1(float *dst, const float *src, const int *idx,
                                 int64_t *shape, int64_t *org_shape) {
  int shape_1_2_3 = org_shape[1] * org_shape[2] * org_shape[3];
  int shape_2_3 = org_shape[2] * org_shape[3];
  for (int i = 0; i < shape[0]; ++i) {
    int idx_i = i * shape_1_2_3;
    for (int j = 0; j < shape[1]; ++j) {
      for (int k = 0; k < shape[2]; ++k) {
        int idx_k = k * org_shape[3];
        for (int g = 0; g < shape[3]; ++g) {
          *dst = src[idx_i + *idx * shape_2_3 + idx_k + g];
          ++dst;
          ++idx;
        }
      }
    }
  }
}
static inline void gather_dim4_2(float *dst, const float *src, const int *idx,
                                 int64_t *shape, int64_t *org_shape) {
  int shape_1_2_3 = org_shape[1] * org_shape[2] * org_shape[3];
  int shape_2_3 = org_shape[2] * org_shape[3];
  for (int i = 0; i < shape[0]; ++i) {
    int idx_i = i * shape_1_2_3;
    for (int j = 0; j < shape[1]; ++j) {
      int idx_j = idx_i + j * shape_2_3;
      for (int k = 0; k < shape[2]; ++k) {
        for (int g = 0; g < shape[3]; ++g) {
          *dst = src[idx_j + *idx * org_shape[3] + g];
          ++dst;
          ++idx;
        }
      }
    }
  }
}
static inline void gather_dim4_3(float *dst, const float *src, const int *idx,
                                 int64_t *shape, int64_t *org_shape) {
  int shape_1_2_3 = org_shape[1] * org_shape[2] * org_shape[3];
  int shape_2_3 = org_shape[2] * org_shape[3];
  for (int i = 0; i < shape[0]; ++i) {
    int idx_i = i * shape_1_2_3;
    for (int j = 0; j < shape[1]; ++j) {
      int idx_j = idx_i + j * shape_2_3;
      for (int k = 0; k < shape[2]; ++k) {
        int idx_k = idx_j + k * org_shape[3];
        for (int g = 0; g < shape[3]; ++g) {
          *dst = src[idx_k + *idx];
          ++dst;
          ++idx;
        }
      }
    }
  }
}
void GatherElementsFunc::invoke() {
  auto axis = param_.axis;
  auto input_info = param_.inputs[0];
  auto indices_info = param_.inputs[1];
  auto indices_shape = indices_info.shape;
  auto input_shape = input_info.shape;
  const float *input = input_info.ptr;
  float *indices = indices_info.ptr;
  std::vector<int> indices_v(indices_info.size);
  for (int i = 0; i < indices_info.size; ++i) {
    indices_v[i] = (int)indices[i];
  }
  auto output_info = param_.output;
  auto output_shape = output_info.shape;
  float *output = output_info.ptr;
  // float *output = param_.output.ptr;
  switch (input_shape.size()) {
  case 1:
    gather_dim1_0(output, input, indices_v.data(), indices_shape.data(),
                  input_shape.data());
    break;
  case 2:
    if (axis == 0)
      gather_dim2_0(output, input, indices_v.data(), indices_shape.data(),
                    input_shape.data());
    else if (axis == 1)
      gather_dim2_1(output, input, indices_v.data(), indices_shape.data(),
                    input_shape.data());
    break;
  case 3:
    if (axis == 0)
      gather_dim3_0(output, input, indices_v.data(), indices_shape.data(),
                    input_shape.data());
    else if (axis == 1)
      gather_dim3_1(output, input, indices_v.data(), indices_shape.data(),
                    input_shape.data());
    else if (axis == 2)
      gather_dim3_2(output, input, indices_v.data(), indices_shape.data(),
                    input_shape.data());
    break;
  case 4:
    if (axis == 0)
      gather_dim4_0(output, input, indices_v.data(), indices_shape.data(),
                    input_shape.data());
    else if (axis == 1)
      gather_dim4_1(output, input, indices_v.data(), indices_shape.data(),
                    input_shape.data());
    else if (axis == 2)
      gather_dim4_2(output, input, indices_v.data(), indices_shape.data(),
                    input_shape.data());
    else if (axis == 3)
      gather_dim4_3(output, input, indices_v.data(), indices_shape.data(),
                    input_shape.data());
    break;
  default:
    printf("error: %s: %d: invalid input dimension: %d. \n", __FILE__, __LINE__,
           static_cast<int>(input_shape.size()));
    exit(-1);
  }
  output_shape = indices_shape;
  return;
}

ScatterNDFunc::ScatterNDFunc(ScatterNDParam &param) : param_(param) {}

void ScatterNDFunc::scatternd_update_core(float *data, const float *updates,
                                          int len, CPU_SCATTER_OP_T op) {
  if (op == CPU_SCATTER_ASSIGN) {
    memcpy(data, updates, len * sizeof(float));
  } else if (op == CPU_SCATTER_ADD) {
    for (int i = 0; i < len; i++) {
      data[i] += updates[i];
    }
  } else if (op == CPU_SCATTER_SUB) {
    for (int i = 0; i < len; i++) {
      data[i] -= updates[i];
    }
  } else if (op == CPU_SCATTER_SUB_REVERSE) {
    for (int i = 0; i < len; i++) {
      data[i] = updates[i] - data[i];
    }
  } else if (op == CPU_SCATTER_MUL) {
    for (int i = 0; i < len; i++) {
      data[i] *= updates[i];
    }
  } else if (op == CPU_SCATTER_MAX) {
    for (int i = 0; i < len; i++) {
      data[i] += std::max(data[i], updates[i]);
    }
  } else if (op == CPU_SCATTER_MIN) {
    for (int i = 0; i < len; i++) {
      data[i] += std::min(data[i], updates[i]);
    }
  } else {
    llvm_unreachable("error scatter_nd op_type");
  }
}

void ScatterNDFunc::invoke() {
  auto input_info = param_.inputs[0];
  auto indices_info = param_.inputs[1];
  auto updates_info = param_.inputs[2];

  const float *input = input_info.ptr;
  const float *indices = indices_info.ptr;
  const float *updates = updates_info.ptr;
  float *out = param_.output.ptr;
  auto input_shape = input_info.shape;
  auto index_shape = indices_info.shape;
  auto updates_shape = updates_info.shape;
  auto index_depth = index_shape.back();

  auto input_dim = input_shape.size();
  auto index_dim = index_shape.size();
  auto updates_dim = updates_shape.size();

  auto updates_elems = 1;
  auto slice_elems = 1;
  for (int i = 0; i < index_dim - 1; ++i) {
    assert(index_shape[i] == updates_shape[i]);
    updates_elems *= index_shape[i];
  }
  for (int i = index_depth; i < input_dim; ++i) {
    assert(input_shape[i] == updates_shape[i]);
    slice_elems *= input_shape[i];
  }

  assert(updates_dim == input_dim + index_dim - index_depth - 1);
  // outer_shape = input_shape[:index_depth]
  std::vector<int> outer_shape(input_shape.begin(),
                               input_shape.begin() + index_depth);
  auto type_len = sizeof(float);
  std::vector<int> outer_stride(outer_shape.size(), slice_elems);
  for (int i = outer_stride.size() - 2; i >= 0; i--) {
    outer_stride[i] = outer_stride[i + 1] * outer_shape[i + 1];
  }
  // init output with input
  memcpy(out, input, input_info.size * type_len);

  for (int idx = 0; idx < updates_elems; ++idx) {
    int index_depth_ = outer_stride.size();
    auto index_data = indices + idx * index_depth_;
    int out_offset = 0;
    for (int i = 0; i < outer_stride.size(); ++i) {
      int real_index_data = (int)index_data[i];
      if ((int)index_data[i] < 0) {
        real_index_data += input_shape[i];
      }
      out_offset += real_index_data * outer_stride[i];
    }
    auto out_ = out + out_offset;
    auto updates_data = updates + idx * outer_stride.back();
    scatternd_update_core(out_, updates_data, slice_elems, param_.op_code);
  }
}

GridSamplerFunc::GridSamplerFunc(GridSamplerParam &param) : param_(param) {}

#define FLOAT_PTR(p) (reinterpret_cast<float *>(p))
#define INT_PTR(p) (reinterpret_cast<int *>(p))
#define FLOAT(val) (static_cast<float>(val))
#define INT(val) (static_cast<int>(val))

template <typename scalar_t>
scalar_t GridSamplerFunc::clip_coordinates(scalar_t in, int64_t clip_limit) {
  return std::min(static_cast<scalar_t>(clip_limit - 1),
                  std::max(in, static_cast<scalar_t>(0)));
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template <typename scalar_t>
scalar_t GridSamplerFunc::reflect_coordinates(scalar_t in, int64_t twice_low,
                                              int64_t twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = std::fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = std::fmod(in, span);
  int flips = static_cast<int>(std::floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

float GridSamplerFunc::computeIndex(float coord, int size, int paddingMode,
                                    bool alignCorners) {
  float res = 0.f;

  // Unnormalize coordinate
  // From [-1, 1] to pixel index
  if (alignCorners)
    res = ((coord + 1.f) * .5f) * (size - 1);
  else
    res = ((coord + 1.f) * size - 1.f) * .5f;

  switch (paddingMode) {
  case GridSamplerZeros:
    break;
  case GridSamplerBorder:
    res = clip_coordinates(res, size);
    break;
  case GridSamplerReflection:
    if (alignCorners) {
      res = reflect_coordinates(res, 0, 2 * (size - 1));
    } else {
      res = reflect_coordinates(res, -1, 2 * size - 1);
    }
    res = clip_coordinates(res, size);
    break;
  default:
    assert(0);
  }
  return res;
}
void GridSamplerFunc::invoke() {
  std::vector<int64_t> input_shapes = param_.inputs[0].shape;
  std::vector<int64_t> grid_shape = param_.inputs[1].shape;
  tensor_list_t input_tensor = param_.inputs[0];
  tensor_list_t grid_tensor = param_.inputs[1];
  tensor_list_t output_tensor = param_.output;
  int mode = param_.mode;
  int padding_mode = param_.padding_mode;
  bool align_corners = param_.align_corners;
  assert((grid_shape.size() == 4 && grid_shape[3] == 2) ||
         (grid_shape.size() == 5 && grid_shape[4] == 3));
  const int N = input_shapes[0];
  const int C = input_shapes[1];
  if (grid_shape.size() == 4) {
    const int IH = input_shapes[2];
    const int IW = input_shapes[3];
    const int OH = grid_shape[1];
    const int OW = grid_shape[2];

#pragma omp parallel for schedule(static, omp_schedule(N *C))
    for (int n = 0; n < N; ++n) {
      const float *input = input_tensor.ptr + n * C * IH * IW;
      const float *grid = grid_tensor.ptr + n * OH * OW * 2;
      float *output = output_tensor.ptr + n * C * OH * OW;
      for (int h = 0; h < OH; ++h) {
        for (int w = 0; w < OW; ++w) {
          auto fx = computeIndex(*grid, IW, padding_mode, align_corners);
          ++grid;
          auto fy = computeIndex(*grid, IH, padding_mode, align_corners);
          ++grid;
          switch (mode) {
          case GridSamplerBilinear: {
            int x = INT(std::floor(fx));
            int y = INT(std::floor(fy));
            float dx = fx - x;
            float dy = fy - y;
            float tx = 1.f - dx;
            float ty = 1.f - dy;
            float txty = tx * ty, dxty = dx * ty, txdy = tx * dy,
                  dxdy = dx * dy;
            bool yBound_0 = y >= 0 && y < IH;
            bool yBound_1 = y + 1 >= 0 && y + 1 < IH;
            bool xBound_0 = x >= 0 && x < IW;
            bool xBound_1 = x + 1 >= 0 && x + 1 < IW;
            const float *iiter = input + y * IW + x;
            float *oiter = output;
            for (int c = 0; c < C; ++c) {
              *oiter = 0.f;
              if (yBound_0) {
                if (xBound_0)
                  *oiter += iiter[0] * txty;
                if (xBound_1)
                  *oiter += iiter[1] * dxty;
              }
              if (yBound_1) {
                if (xBound_0)
                  *oiter += iiter[IW] * txdy;
                if (xBound_1)
                  *oiter += iiter[IW + 1] * dxdy;
              }
              iiter += IH * IW;
              oiter += OH * OW;
            }
          } break;
          case GridSamplerNearest: {
            int x = INT(std::nearbyint(fx));
            int y = INT(std::nearbyint(fy));
            const float *iiter = input + y * IW + x;
            float *oiter = output;
            for (int c = 0; c < C; ++c) {
              *oiter = y >= 0 && y < IH && x >= 0 && x < IW ? *iiter : 0.f;
              iiter += IH * IW;
              oiter += OH * OW;
            }
          } break;
          default:
            assert(0);
          }
          ++output;
        }
      }
    }
    output_tensor.shape = {N, C, OH, OW};

  } else {
    const int ID = input_shapes[2];
    const int IH = input_shapes[3];
    const int IW = input_shapes[4];
    const int OD = grid_shape[1];
    const int OH = grid_shape[2];
    const int OW = grid_shape[3];

#pragma omp parallel for schedule(static, omp_schedule(N *C))
    for (int n = 0; n < N; ++n) {
      const float *input = input_tensor.ptr + n * C * ID * IH * IW;
      const float *grid = grid_tensor.ptr + n * OD * OH * OW * 3;
      float *output = output_tensor.ptr + n * C * OD * OH * OW;
      for (int d = 0; d < OD; ++d) {
        for (int h = 0; h < OH; ++h) {
          for (int w = 0; w < OW; ++w) {
            auto fx = computeIndex(*grid, IW, padding_mode, align_corners);
            ++grid;
            auto fy = computeIndex(*grid, IH, padding_mode, align_corners);
            ++grid;
            auto fz = computeIndex(*grid, ID, padding_mode, align_corners);
            ++grid;
            switch (mode) {
            case GridSamplerBilinear: {
              int x = INT(std::floor(fx));
              int y = INT(std::floor(fy));
              int z = INT(std::floor(fz));
              float dx = fx - x;
              float dy = fy - y;
              float dz = fz - z;
              float tx = 1.f - dx;
              float ty = 1.f - dy;
              float tz = 1.f - dz;
              float txtytz = tx * ty * tz, txtydz = tx * ty * dz,
                    dxtytz = dx * ty * tz, dxtydz = dx * ty * dz;
              float txdytz = tx * dy * tz, txdydz = tx * dy * dz,
                    dxdytz = dx * dy * tz, dxdydz = dx * dy * dz;
              bool zBound_0 = z >= 0 && z < ID;
              bool zBound_1 = z + 1 >= 0 && z + 1 < ID;
              bool yBound_0 = y >= 0 && y < IH;
              bool yBound_1 = y + 1 >= 0 && y + 1 < IH;
              bool xBound_0 = x >= 0 && x < IW;
              bool xBound_1 = x + 1 >= 0 && x + 1 < IW;
              const float *iiter = input + z * IH * IW + y * IW + x;
              float *oiter = output;
              for (int c = 0; c < C; ++c) {
                *oiter = 0.f;
                if (zBound_0) {
                  if (yBound_0) {
                    if (xBound_0)
                      *oiter += iiter[0] * txtytz;
                    if (xBound_1)
                      *oiter += iiter[1] * dxtytz;
                  }
                  if (yBound_1) {
                    if (xBound_0)
                      *oiter += iiter[IW] * txdytz;
                    if (xBound_1)
                      *oiter += iiter[IW + 1] * dxdytz;
                  }
                }
                if (zBound_1) {
                  if (yBound_0) {
                    if (xBound_0)
                      *oiter += iiter[IH * IW + 0] * txtydz;
                    if (xBound_1)
                      *oiter += iiter[IH * IW + 1] * dxtydz;
                  }
                  if (yBound_1) {
                    if (xBound_0)
                      *oiter += iiter[IH * IW + IW] * txdydz;
                    if (xBound_1)
                      *oiter += iiter[IH * IW + IW + 1] * dxdydz;
                  }
                }
                iiter += ID * IH * IW;
                oiter += OD * OH * OW;
              }
            } break;
            case GridSamplerNearest: {
              int x = INT(std::round(fx));
              int y = INT(std::round(fy));
              int z = INT(std::round(fz));
              const float *iiter = input + z * IH * IW + y * IW + x;
              float *oiter = output;
              for (int c = 0; c < C; ++c) {
                *oiter =
                    z >= 0 && z < ID && y >= 0 && y < IH && x >= 0 && x < IW
                        ? *iiter
                        : 0.f;
                iiter += ID * IH * IW;
                oiter += OD * OH * OW;
              }
            } break;
            default:
              assert(0);
            }
            ++output;
          }
        }
      }
    }
    output_tensor.shape = {N, C, OD, OH, OW};
  }
  return;
}

DeformGatherFunc::DeformGatherFunc(DeformGatherParam &param) : param_(param) {}

void DeformGatherFunc::invoke() {
  std::vector<int64_t> input_shapes = param_.inputs[0].shape;
  deform_gather_attr_t attr;
  attr.n = input_shapes[0];
  attr.ic = input_shapes[1];
  attr.ih = input_shapes[2];
  attr.iw = input_shapes[3];
  attr.kh = param_.kh;
  attr.kw = param_.kw;
  attr.pht = param_.pad_t;
  attr.phb = param_.pad_b;
  attr.pwl = param_.pad_l;
  attr.pwr = param_.pad_r;
  attr.sh = param_.stride_h;
  attr.sw = param_.stride_w;
  attr.dh = param_.dilation_h;
  attr.dw = param_.dilation_w;
  const int conved_H =
      ((attr.ih - (attr.dh * (attr.kh - 1) + 1) + attr.pht + attr.phb) /
           attr.sh +
       1);
  const int conved_W =
      ((attr.iw - (attr.dw * (attr.kw - 1) + 1) + attr.pwl + attr.pwr) /
           attr.sw +
       1);
  attr.oc = attr.ic * attr.kh * attr.kw;
  attr.oh = conved_H;
  attr.ow = conved_W;
  attr.deform_groups = param_.deform_groups;
  attr.use_mask = param_.modulated;
  InferenceParameter p;
  for (int i = 0; i < param_.inputs.size(); ++i) {
    p.inputs.push_back(param_.inputs[i].ptr);
  }
  p.outputs.push_back(param_.output.ptr);
  processDeformGather(p, attr, param_.output.ptr, false);
}

CumSumFunc::CumSumFunc(CumSumParam &param) : param_(param) {}

void CumSumFunc::invoke() {
  std::vector<int64_t> in_shape = param_.inputs[0].shape;
  int64_t dim = param_.axis;
  int64_t num_dims = in_shape.size();
  assert(dim < in_shape.size());

  int64_t length = in_shape[dim];
  // stride
  int64_t stride = 1;
  for (int64_t i = dim + 1; i < num_dims; i++) {
    stride *= in_shape[i];
  }
  int64_t num_elements = param_.output.size;
  int64_t cur_index = 0;
  while (cur_index < num_elements) {
    for (int64_t l = 0; l < length; l++) {
      int64_t start = cur_index + l * stride;
      for (int64_t s = 0; s < stride; s++) {
        if (l == 0) {
          param_.output.ptr[start + s] = param_.inputs[0].ptr[start + s];
        } else {
          param_.output.ptr[start + s] = param_.inputs[0].ptr[start + s] +
                                         param_.output.ptr[start + s - stride];
        }
      }
    }
    cur_index += length * stride;
  }
}

} // namespace tpu_mlir
