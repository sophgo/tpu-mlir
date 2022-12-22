//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <math.h>

namespace tpu_mlir {
#define MAX_DET 200
#define MAX_DET_RAW 500

enum Decode_CodeType {
  PriorBoxParameter_CodeType_CORNER = 1,
  PriorBoxParameter_CodeType_CENTER_SIZE = 2,
  PriorBoxParameter_CodeType_CORNER_SIZE = 3
};

class BBox_l {
public:
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  float size;

  void CalcSize() {
    if (xmax < xmin || ymax < ymin) {
      size = 0;
    } else {
      float width = xmax - xmin;
      float height = ymax - ymin;
      size = width * height;
    }
  }
};

typedef Decode_CodeType CodeType;
typedef std::map<int, std::vector<BBox_l>> LabelBBox_l;
struct box {
  float x, y, w, h;
};

struct tensor_list_t {
  float *ptr;
  size_t size;
  std::vector<int64_t> shape;
};

struct detection {
  box bbox;
  int cls;
  float score;
};

typedef struct {
  float x1, y1, x2, y2;
} coord;

typedef struct {
  coord bbox;
  int cls;
  float score;
} detections;

struct DetParam {
  std::vector<int64_t> loc_shape;
  std::vector<int64_t> conf_shape;
  std::vector<int64_t> prior_shape;
  float *loc_data;
  float *conf_data;
  float *prior_data;
  float *output_data;
  int64_t keep_top_k;
  int64_t top_k;
  int64_t num_classes;
  int64_t background_label_id;
  double nms_threshold;
  double confidence_threshold;
  bool share_location;
  Decode_CodeType code_type;
};

class DetectionOutputFunc {
public:
  DetectionOutputFunc(DetParam &param);
  void invoke();
private:
  DetParam param_;
};

struct YoloDetParam {
  std::string anchors;
  std::vector<tensor_list_t> inputs;
  tensor_list_t output;
  int64_t net_input_h;
  int64_t net_input_w;
  int64_t keep_topk;
  int64_t class_num;
  double nms_threshold;
  double obj_threshold;
  bool tiny;
  bool yolo_v4;
  bool spp_net;
};

class YoloDetectionFunc {
public:
  YoloDetectionFunc(YoloDetParam &param);
  void invoke();
private:
  YoloDetParam param_;
  std::vector<float> _anchors;
};

struct ProposalParam {
  std::vector<tensor_list_t> inputs;
  tensor_list_t output;
  int64_t net_input_h;
  int64_t net_input_w;
  int64_t feat_stride;
  int64_t anchor_base_size;
  double rpn_obj_threshold;
  double rpn_nms_threshold;
  int64_t rpn_nms_post_top_n;
};

class ProposalFunc {
public:
  ProposalFunc(ProposalParam &param);
  void invoke();
private:
  ProposalParam param_;
  std::vector<float> anchor_scale = {8, 16, 32};
  std::vector<float> anchor_ratio = {0.5, 1, 2};
  std::vector<float> anchor_boxes;
};

struct ROIPoolingParam {
  std::vector<tensor_list_t> inputs;
  tensor_list_t output;
  int64_t pooled_h;
  int64_t pooled_w;
  double spatial_scale;
};

class ROIPoolingFunc {
public:
  ROIPoolingFunc(ROIPoolingParam &param);
  void invoke();
private:
  ROIPoolingParam param_;
};

struct FrcnDetParam {
  std::vector<tensor_list_t> inputs;
  tensor_list_t output;
  int64_t class_num;
  double obj_threshold;
  double nms_threshold;
  int64_t keep_topk;
};

class FrcnDetctionFunc {
public:
  FrcnDetctionFunc(FrcnDetParam &param);
  void invoke();
private:
  FrcnDetParam param_;
};

float my_mish_activate(float x_val);

} // namespace tpu_mlir
