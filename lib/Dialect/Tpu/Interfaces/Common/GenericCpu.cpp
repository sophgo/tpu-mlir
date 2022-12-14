//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::GenericCpuOp::init(InferenceParameter &p) {
  return success();
}
void tpu::GenericCpuOp::deinit(InferenceParameter &p) {}

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

static inline float coordinate_transform(float x_resized, float x_scale,
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

class InterpolationOpKernel {
public:
  InterpolationOpKernel(tpu::GenericCpuOp &op, InferenceParameter &p) {
    Module::getShapeVec(op.inputs()[0], this->input_shape);
    Module::getShapeVec(op.output(), this->output_shape);
    assert(input_shape.size() == 4);
    assert(output_shape.size() == 4);
    mlir::DictionaryAttr param = op.param().value();
    this->height = param.get("height").cast<IntegerAttr>().getInt();
    this->width = param.get("width").cast<IntegerAttr>().getInt();
    this->pad_beg = param.get("pad_beg").cast<IntegerAttr>().getInt();
    this->pad_beg = param.get("pad_end").cast<IntegerAttr>().getInt();
    this->shrink_factor =
        param.get("shrink_factor").cast<IntegerAttr>().getInt();
    this->zoom_factor = param.get("zoom_factor").cast<IntegerAttr>().getInt();
    this->coordinate_transformation_mode =
        param.get("coordinate_transformation_mode")
            .cast<StringAttr>()
            .getValue();
    // get tensors
    input_data = p.inputs[0];
    output_data = p.outputs[0];
  }
  void invoke() {
    int in = input_shape[0];
    int ic = input_shape[1];
    int ih = input_shape[2];
    int iw = input_shape[3];
    int oh = output_shape[2];
    int ow = output_shape[3];
    int height_in_ = ih;
    int width_in_ = iw;
    int height_in_eff_ = height_in_ + pad_beg + pad_end;
    int width_in_eff_ = width_in_ + pad_beg + pad_end;
    int height_out_ = -1;
    int width_out_ = -1;
    if (this->shrink_factor && !this->zoom_factor) {
      assert(shrink_factor >= 1 && "Shrink factor must be positive");
      height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
      width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
    } else if (this->zoom_factor && !this->shrink_factor) {
      assert(zoom_factor >= 1 && "Zoom factor must be positive");
      height_out_ = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor - 1);
      width_out_ = width_in_eff_ + (width_in_eff_ - 1) * (zoom_factor - 1);
    } else if (this->height && this->width) {
      height_out_ = this->height;
      width_out_ = this->width;
    } else if (this->zoom_factor && this->shrink_factor) {
      assert(shrink_factor >= 1 && "Shrink factor must be positive");
      assert(zoom_factor >= 1 && "Zoom factor must be positive");

      height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
      width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
      height_out_ = height_out_ + (height_out_ - 1) * (zoom_factor - 1);
      width_out_ = width_out_ + (width_out_ - 1) * (zoom_factor - 1);
    }
    if (coordinate_transformation_mode == "align_corners") {
      // TODO: verify pad_end_ > 0
      my_interp(in * ic, input_data, -pad_beg, -pad_beg, height_in_eff_,
                width_in_eff_, height_in_, width_in_, output_data, 0, 0,
                height_out_, width_out_, height_out_, width_out_);
    } else if (coordinate_transformation_mode == "half_pixel") {
      interp_linear(input_data, output_data, in, ic, ih, iw, oh, ow);
    } else if (coordinate_transformation_mode == "pytorch_half_pixel") {
      interp_linear(input_data, output_data, in, ic, ih, iw, oh, ow, true);
    } else if (coordinate_transformation_mode == "nearest_half_pixel") {
      interp_neast(input_data, output_data, in, ic, ih, iw, oh, ow, true);
    } else if (coordinate_transformation_mode == "nearest") {
      interp_neast(input_data, output_data, in, ic, ih, iw, oh, ow, false);
    } else if (coordinate_transformation_mode == "asymmetric") {
      interp_asymmetric(input_data, output_data, in, ic, ih, iw, oh, ow);
    } else {
      llvm_unreachable("coordinate_transformation_model not support");
    }
  }

private:
  float *input_data;
  float *output_data;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;

  // param
  int shrink_factor = 0;
  int zoom_factor = 0;
  int height = 0;
  int width = 0;
  int pad_beg;
  int pad_end;
  std::string coordinate_transformation_mode;
};

class EmbeddingOpKernel {
public:
  EmbeddingOpKernel(tpu::GenericCpuOp &op, InferenceParameter &p) {
    Module::getShapeVec(op.inputs()[0], this->input_shape);
    Module::getShapeVec(op.inputs()[1], this->table_shape);
    Module::getShapeVec(op.output(), this->output_shape);
    input_data = p.inputs[0];
    table_data = p.inputs[1];
    output_data = p.outputs[0];
  }

  void invoke() {
    auto feature_dim = table_shape.back();
    assert(output_shape.back() == feature_dim &&
           "must be the same feature dim");
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

private:
  float *input_data;
  float *table_data;
  float *output_data;
  // float *scale_data;
  //  SyncedData zeropoint_data;
  //  SyncedData output_data;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> table_shape;
  std::vector<int64_t> output_shape;
  bool mix_bf16;
};

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

class DetectionOutputOpKernel {
public:
  DetectionOutputOpKernel(tpu::GenericCpuOp &op, InferenceParameter &p) {
    Module::getShapeVec(op.inputs()[0], this->loc_shape);
    Module::getShapeVec(op.inputs()[1], this->conf_shape);
    Module::getShapeVec(op.inputs()[2], this->prior_shape);
    loc_data = p.inputs[0];
    conf_data = p.inputs[1];
    prior_data = p.inputs[2];
    output_data = p.outputs[0];
    mlir::DictionaryAttr param = op.param().value();
    this->keep_top_k = param.get("keep_top_k").cast<IntegerAttr>().getInt();
    this->top_k = param.get("top_k").cast<IntegerAttr>().getInt();
    this->num_classes = param.get("num_classes").cast<IntegerAttr>().getInt();
    this->background_label_id =
        param.get("background_label_id").cast<IntegerAttr>().getInt();
    this->share_location =
        param.get("share_location").cast<BoolAttr>().getValue();
    this->confidence_threshold =
        param.get("confidence_threshold").cast<FloatAttr>().getValueAsDouble();
    this->nms_threshold =
        param.get("nms_threshold").cast<FloatAttr>().getValueAsDouble();
    std::string str_code_type =
        param.get("code_type").cast<StringAttr>().getValue().str();
    if (str_code_type == "CORNER") {
      this->code_type = PriorBoxParameter_CodeType_CORNER;
    } else if (str_code_type == "CENTER_SIZE") {
      this->code_type = PriorBoxParameter_CodeType_CENTER_SIZE;
    } else if (str_code_type == "CORNER_SIZE") {
      this->code_type = PriorBoxParameter_CodeType_CORNER_SIZE;
    } else {
      llvm_unreachable("code type wrong");
    }
  }

  void invoke() {
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
          continue;
        }
        std::vector<std::pair<float, int>> &scores =
            all_conf_scores[i].find(c)->second;

        if (top_k < (int)scores.size()) {
          std::partial_sort(scores.begin(), scores.begin() + top_k,
                            scores.end(), SortScoreCmp0);
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
          ptr =
              decode_keep_index + num_priors * num_classes * i + num_priors * c;
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
          // Something bad happened if there are no predictions for current
          // label.
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
      // Generate fake results per image.
      for (int i = 0; i < num; ++i) {
        top_data[0] = i;
        top_data += 7;
      }
    } else {
      int count = 0;
      for (int i = 0; i < num; ++i) {
        const LabelBBox_l &decode_bboxes = all_decode_bboxes[i];
        for (auto it = all_indices[i].begin(); it != all_indices[i].end();
             ++it) {
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
  }

private:
  std::vector<int64_t> loc_shape;
  std::vector<int64_t> conf_shape;
  std::vector<int64_t> prior_shape;
  float *loc_data;
  float *conf_data;
  float *prior_data;
  float *output_data;
  double confidence_threshold;
  double nms_threshold;
  int64_t top_k;
  int64_t keep_top_k;
  int64_t num_classes;
  int64_t background_label_id;
  Decode_CodeType code_type;
  bool share_location;
};

LogicalResult tpu::GenericCpuOp::inference(InferenceParameter &p) {
  std::string func_name = operation_name().str();
  if (func_name == "quant") {
    assert(inputs().size() == 1);
    auto num_elem = Module::getNumElements(output());
    auto in_type = Module::getStorageType(inputs()[0]);
    auto out_type = Module::getStorageType(output());
    if (in_type.isF32() && out_type.isSignedInteger()) {
      auto qtype = Quant::getUniformQuantizedType(output());
      quantizeToInt8(p.inputs[0], p.outputs[0], num_elem,
                     1. / qtype.getScale());
    } else {
      llvm_unreachable("not supported!\n");
    }
  } else if (func_name == "interp") {
    InterpolationOpKernel interp_kernel(*this, p);
    interp_kernel.invoke();
  } else if (func_name == "embedding") {
    EmbeddingOpKernel embed_kernel(*this, p);
    embed_kernel.invoke();
  } else if (func_name == "detectionoutput") {
    DetectionOutputOpKernel det_kernel(*this, p);
    det_kernel.invoke();
  } else {
    llvm_unreachable("generic cpu func not supported!\n");
  }
  return success();
}
