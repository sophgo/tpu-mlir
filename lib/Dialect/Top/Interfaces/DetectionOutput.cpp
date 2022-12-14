//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "detection-output"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

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
typedef std::map<int, std::vector<BBox_l> > LabelBBox_l;

static bool SortScoreCmp0 (const std::pair<float,int> &pair1,
    const std::pair<float,int> &pair2) {
  return pair1.first > pair2.first;
}

static bool SortScoreCmp1 (const std::pair<float, std::pair<int, int>>& pair1,
    const std::pair<float, std::pair<int, int>>& pair2) {
  return pair1.first > pair2.first;
}

static void GetConfidenceScores_opt (const float* conf_data, const int num,
    const int num_preds_per_class, const int num_classes, const float score_threshold,
    std::vector<std::map<int, std::vector<std::pair<float ,int>> > >* conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    std::map<int, std::vector<std::pair<float ,int>> >& label_scores = (*conf_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      for (int c = 0; c < num_classes; ++c) {
        if (conf_data[start_idx + c] > score_threshold) {
          label_scores[c].push_back(std::make_pair(conf_data[start_idx + c],p));
        }
      }
    }
    conf_data += num_preds_per_class * num_classes;
  }
}

static void GetLocPredictions_opt (const float* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, float *decode_index,
      std::vector<LabelBBox_l>* loc_preds) {
  loc_preds->clear();
  if (share_location) {
    assert(num_loc_classes==1);
  }
  loc_preds->resize(num);
  float * decode_pos = decode_index;
  for (int i = 0; i < num; ++i) {
    if (share_location) {
      decode_pos = decode_index + i*num_preds_per_class;
    }
    LabelBBox_l& label_bbox = (*loc_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_loc_classes * 4;
      for (int c = 0; c < num_loc_classes; ++c) {
        if (!share_location) {
          decode_pos = decode_index + num_preds_per_class*num_loc_classes*i + num_preds_per_class*c;
        }
        int label = share_location ? -1 : c;
        if (label_bbox.find(label) == label_bbox.end()) {
          label_bbox[label].resize(num_preds_per_class);
        }
        if (decode_pos[p]!=1) {
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

static void DecodeBBoxesAll_opt (const std::vector<LabelBBox_l>& all_loc_preds,
    int num_priors, const float* prior_data,
    const int num, const bool share_location,
    const int num_loc_classes, const int background_label_id,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip, float *decode_index ,
    std::vector<LabelBBox_l>* all_decode_bboxes) {
  assert(all_loc_preds.size() == (size_t)num);
  all_decode_bboxes->clear();
  all_decode_bboxes->resize(num);
  float * decode_pos = decode_index;
  for (int i = 0; i < num; ++i) {
    if (share_location) {
      decode_pos = decode_index + i*num_priors;
    }
    // Decode predictions into bboxes.
    for (int c = 0; c < num_loc_classes; ++c) {
      int label = share_location ? -1 : c;
      if (label == background_label_id) {
        // Ignore background class.
        continue;
      }
      if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
       llvm::errs() << "Could not find location predictions for label " << label;
      }
      const std::vector<BBox_l>& bboxes = all_loc_preds[i].find(label)->second;
      LabelBBox_l& decode_bboxes = (*all_decode_bboxes)[i];
      std::vector<BBox_l>* p = &(decode_bboxes[label]);
      p->clear();

      if (!share_location) {
        decode_pos = decode_index + num_priors*num_loc_classes*i + num_priors*c;
      }
      for (int k = 0; k < num_priors; ++k) {
        //NormalizedBBox decode_bbox;
        BBox_l decode_bbox;
        if (decode_pos[k] != 1) {
          p->push_back(decode_bbox);
          continue;
        }
        //opt CENTER_SIZE
        assert (code_type==PriorBoxParameter_CodeType_CENTER_SIZE);
        //prior_bboxes
        int start_idx = k * 4;
        const float *p0 = prior_data + start_idx;
        const float *p1 = prior_data + start_idx + 4*num_priors;
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
          // variance is encoded in bbox, we need to scale the offset accordingly.
          decode_bbox_center_x = p1[0] * bboxes[k].xmin * prior_width + prior_center_x;
          decode_bbox_center_y = p1[1] * bboxes[k].ymin * prior_height + prior_center_y;
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

static void ApplyNMSFast_opt (const std::vector<BBox_l>& bboxes,
    const std::vector<std::pair<float ,int>> & conf_score ,
    const float score_threshold, const float nms_threshold, const float eta, int top_k,
    std::vector<std::pair<float,int>>* indices) {
  // Do nms.
  float adaptive_threshold = nms_threshold;
  int i = 0;
  int length = (top_k < (int)conf_score.size()) ? top_k : conf_score.size();
    while (length != i) {
    bool keep = true;
    for (int k = 0; k < (int)indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k].second;
        const BBox_l & b1 = bboxes[conf_score[i].second];
        const BBox_l & b2 = bboxes[kept_idx];
        if (b2.xmin > b1.xmax || b2.xmax < b1.xmin ||
            b2.ymin > b1.ymax || b2.ymax < b1.ymin) {
          keep = true;
        }
        else {
          const float inter_xmin = std::max(b1.xmin, b2.xmin);
          const float inter_ymin = std::max(b1.ymin, b2.ymin);
          const float inter_xmax = std::min(b1.xmax, b2.xmax);
          const float inter_ymax = std::min(b1.ymax, b2.ymax);
          const float inter_width = inter_xmax - inter_xmin;
          const float inter_height = inter_ymax - inter_ymin;
          const float inter_size = inter_width * inter_height;
          const float total_size = b1.size + b2.size;
          keep = (inter_size*(adaptive_threshold+1) <= total_size*adaptive_threshold) ? true : false;
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

int64_t top::DetectionOutputOp::getFLOPs() {
  return Module::getNumElements(output());
}

LogicalResult top::DetectionOutputOp::init(InferenceParameter &p) {
  return success();
}
void top::DetectionOutputOp::deinit(InferenceParameter &p) {}

LogicalResult top::DetectionOutputOp::inference(InferenceParameter &p) {
  std::vector<int64_t> loc_shape;
  std::vector<int64_t> conf_shape;
  std::vector<int64_t> prior_shape;
  Module::getShapeVec(this->inputs()[0], loc_shape);
  Module::getShapeVec(this->inputs()[1], conf_shape);
  Module::getShapeVec(this->inputs()[2], prior_shape);
  int64_t keep_top_k = this->keep_top_k();
  double confidence_threshold = this->confidence_threshold().convertToDouble();
  double nms_threshold = this->nms_threshold().convertToDouble();
  int64_t top_k = this->top_k();
  int64_t num_classes = this->num_classes();
  bool share_location = this->share_location();
  int64_t background_label_id = this->background_label_id();

  std::string str_code_type = this->code_type().str();
  Decode_CodeType code_type;
  if (str_code_type == "CORNER") {
    code_type = PriorBoxParameter_CodeType_CORNER;
  } else if (str_code_type == "CENTER_SIZE") {
    code_type = PriorBoxParameter_CodeType_CENTER_SIZE;
  } else if (str_code_type == "CORNER_SIZE") {
    code_type = PriorBoxParameter_CodeType_CORNER_SIZE;
  } else {
    llvm_unreachable("code type wrong");
  }

  auto loc_data = p.inputs[0];
  auto conf_data = p.inputs[1];
  auto prior_data = p.inputs[2];
  auto output_data = p.outputs[0];

  int num = loc_shape[0];
  int num_priors = prior_shape[2] / 4;
  int num_loc_classes = share_location ? 1 : num_classes;
  float eta = 1.0;
  bool variance_encoded_in_target = false;
  std::vector<std::map<int, std::vector<std::pair<float, int>>>>
      all_conf_scores;
  GetConfidenceScores_opt(conf_data, num, num_priors, num_classes,
                          confidence_threshold, &all_conf_scores);
  for (int i = 0; i < num; ++i) {
    for (int c = 0; c < num_classes; ++c) {
      if (all_conf_scores[i].find(c) == all_conf_scores[i].end()) {
        LLVM_DEBUG(std::cout << "class with no score idx = %d," << c << "\n";);
        continue;
      }
      std::vector<std::pair<float, int>> &scores =
          all_conf_scores[i].find(c)->second;

      if (top_k < (int)scores.size()) {
        std::partial_sort(scores.begin(), scores.begin() + top_k, scores.end(),
                          SortScoreCmp0);
      } else {
        std::sort(scores.begin(), scores.end(), SortScoreCmp0);
      }
    }
  }

  // build keep for decode ,recode vilad index
  float *decode_keep_index;
  int buf_length = 0;
  if (share_location) {
    buf_length = num * num_priors;
  } else {
    buf_length = num * num_priors * num_classes;
  }
  decode_keep_index = new float[buf_length];
  memset(decode_keep_index, 0, buf_length * 4);
  float *ptr = decode_keep_index;
  for (int i = 0; i < num; ++i) {
    if (share_location) {
      ptr = decode_keep_index + num_priors * i;
    }
    for (int c = 0; c < num_classes; ++c) {
      if (!share_location) {
        ptr = decode_keep_index + num_priors * num_classes * i + num_priors * c;
      }
      if (c == background_label_id) {
        // Ignore background class.
        continue;
      }

      if (all_conf_scores[i].find(c) == all_conf_scores[i].end())
        continue;
      std::vector<std::pair<float, int>> &scores =
          all_conf_scores[i].find(c)->second;
      int length = top_k < (int)scores.size() ? top_k : scores.size();
      for (int k = 0; k < length; ++k) {
        ptr[scores[k].second] = 1;
      }
    }
  }

  // Retrieve all location predictions.
  std::vector<LabelBBox_l> all_loc_preds;
  GetLocPredictions_opt(loc_data, num, num_priors, num_loc_classes,
                        share_location, decode_keep_index, &all_loc_preds);

  // Decode all loc predictions to bboxes.
  std::vector<LabelBBox_l> all_decode_bboxes;
  const bool clip_bbox = false;
  DecodeBBoxesAll_opt(all_loc_preds, num_priors, prior_data, num,
                      share_location, num_loc_classes, background_label_id,
                      code_type, variance_encoded_in_target, clip_bbox,
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
    for (int c = 0; c < num_classes; ++c) {
      if (c == background_label_id) {
        // Ignore background class.
        continue;
      }
      if (conf_scores.find(c) == conf_scores.end())
        continue;
      int label = share_location ? -1 : c;
      if (decode_bboxes.find(label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        llvm::errs() << "Could not find location predictions for label "
                     << label;
        continue;
      }
      const std::vector<BBox_l> &bboxes = decode_bboxes.find(label)->second;
      const std::vector<std::pair<float, int>> &aa =
          conf_scores.find(c)->second;
      ApplyNMSFast_opt(bboxes, aa, confidence_threshold, nms_threshold, eta,
                       top_k, &(indices[c]));

      num_det += indices[c].size();
    }

    if (keep_top_k > -1 && num_det > keep_top_k) {
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
      score_index_pairs.resize(keep_top_k);
      // Store the new indices.
      std::map<int, std::vector<std::pair<float, int>>> new_indices;
      for (int j = 0; j < (int)score_index_pairs.size(); ++j) {

        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        float s = score_index_pairs[j].first;

        new_indices[label].push_back(std::make_pair(s, idx));
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  }
  float *top_data = (float *)output_data;

  int output_size = num * keep_top_k * 1 * 1 * 7;
  // init output buf
  for (int i = 0; i < output_size; ++i) {
    top_data[i] = -1;
  }

  if (num_kept == 0) {
    LLVM_DEBUG(llvm::errs() << "Couldn't find any detections";);
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
        int loc_label = share_location ? -1 : label;
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
  return success();
}
