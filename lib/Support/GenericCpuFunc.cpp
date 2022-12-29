//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <map>
#include <math.h>
#include <sstream>

namespace tpu_mlir {

static float softplus_activate(float x) {
  return std::log(std::exp(x) + 1);
}

static inline float tanh_activate (float x) {
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
                            std::vector<int64_t> grid_size, float *anchor,
                            std::vector<int64_t> yolo_size,
                            int64_t num_of_class, float obj_threshold) {
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
  int num_priors = param_.prior_shape[2] / 4;
  int num_loc_classes = param_.share_location ? 1 : param_.num_classes;
  float eta = 1.0;
  bool variance_encoded_in_target = false;
  std::vector<std::map<int, std::vector<std::pair<float, int>>>>
      all_conf_scores;
  GetConfidenceScores_opt(param_.conf_data, num, num_priors, param_.num_classes,
                          param_.confidence_threshold, &all_conf_scores);
  for (int i = 0; i < num; ++i) {
    for (int c = 0; c < param_.num_classes; ++c) {
      if (all_conf_scores[i].find(c) == all_conf_scores[i].end()) {
        continue;
      }
      std::vector<std::pair<float, int>> &scores =
          all_conf_scores[i].find(c)->second;

      if (param_.top_k < (int)scores.size()) {
        std::partial_sort(scores.begin(), scores.begin() + param_.top_k, scores.end(),
                          SortScoreCmp0);
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
        ptr = decode_keep_index + num_priors * param_.num_classes * i + num_priors * c;
      }
      if (c == param_.background_label_id) {
        // Ignore background class.
        continue;
      }

      if (all_conf_scores[i].find(c) == all_conf_scores[i].end())
        continue;
      std::vector<std::pair<float, int>> &scores =
          all_conf_scores[i].find(c)->second;
      int length = param_.top_k < (int)scores.size() ? param_.top_k : scores.size();
      for (int k = 0; k < length; ++k) {
        ptr[scores[k].second] = 1;
      }
    }
  }

  // Retrieve all location predictions.
  std::vector<LabelBBox_l> all_loc_preds;
  GetLocPredictions_opt(param_.loc_data, num, num_priors, num_loc_classes,
                        param_.share_location, decode_keep_index, &all_loc_preds);

  // Decode all loc predictions to bboxes.
  std::vector<LabelBBox_l> all_decode_bboxes;
  const bool clip_bbox = false;
  DecodeBBoxesAll_opt(all_loc_preds, num_priors, param_.prior_data, num,
                      param_.share_location, num_loc_classes, param_.background_label_id,
                      param_.code_type, variance_encoded_in_target, clip_bbox,
                      decode_keep_index, &all_decode_bboxes);
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
      ApplyNMSFast_opt(bboxes, aa, param_.confidence_threshold, param_.nms_threshold, eta,
                       param_.top_k, &(indices[c]));

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
  std::istringstream iss(param_.anchors);
  std::string s;
  while (std::getline(iss, s, ',')) {
    _anchors.push_back(atof(s.c_str()));
  }
  std::sort(param_.inputs.begin(), param_.inputs.end(),
            [](const tensor_list_t &a, const tensor_list_t &b) {
              return a.shape[3] > b.shape[3];
            });
  if (param_.tiny) {
    assert(param_.inputs.size() == 2);
    if (_anchors.size() == 0) {
      _anchors = {
          10, 14, 23,  27,  37,  58, // layer23-conv (26*26)
          81, 82, 135, 169, 344, 319 // layer16-conv (13*13)
      };
    }
  } else {
    assert(param_.inputs.size() == 3);
    if (_anchors.size() == 0) {
      if (param_.yolo_v4) {
        _anchors = {
            142, 110, 192, 243, 459, 401, // layer161-conv
            36,  75,  76,  55,  72,  146, // layer150-conv
            12,  16,  19,  36,  40,  28,  // layer139-conv
        };
      } else {
        // Yolov3 default anchors
        _anchors = {
            10,  13, 16,  30,  33,  23,  // layer106-conv (52*52)
            30,  61, 62,  45,  59,  119, // layer94-conv  (26*26)
            116, 90, 156, 198, 373, 326  // layer82-conv  (13*13)
        };
      }
    }
  }
}

void YoloDetectionFunc::invoke() {
  auto top_data = param_.output.ptr;
  memset(top_data, 0, param_.output.size);
  int batch = param_.output.shape[0];

  size_t bottom_count = param_.inputs.size();
  assert(_anchors.size() == bottom_count * 6);
  float(*anchors)[6] = (float(*)[6])_anchors.data();

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
                      param_.class_num, param_.obj_threshold);
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
  generate_anchors(param_.anchor_base_size, anchor_scale, anchor_ratio, anchor_boxes);
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

ROIPoolingFunc::ROIPoolingFunc(ROIPoolingParam &param) : param_(param) {

}

void ROIPoolingFunc::invoke() {
  assert(param_.inputs.size() == 2);
  float *input_data = param_.inputs[0].ptr;
  float *rois = param_.inputs[1].ptr;
  float *output_data = param_.output.ptr;
  auto input_shape =  param_.inputs[0].shape;
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

      float *batch_data =
          input_data + roi_batch_ind * channel * height * width;

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

FrcnDetctionFunc::FrcnDetctionFunc(FrcnDetParam &param) : param_(param) {

}

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

} // namespace tpu_mlir
