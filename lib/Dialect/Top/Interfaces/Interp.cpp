//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

using namespace std;

#define MIN(x, y) (((x)) < ((y)) ? (x) : (y))
#define MAX(x, y) (((x)) > ((y)) ? (x) : (y))
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

static inline float calc_resize_scale(int in_size, int out_size,
                                      bool align_corners,
                                      PLATFORM_SUPPORT platform_sp) {

  int _in_size = in_size, _out_size = out_size;
  switch (platform_sp) {
  case TENSORFLOW_NEAREST:
  case TENSORFLOW_SUPPORT:
  case PYTORCH_SUPPORT:
  case ONNX_NEAREST: {
    if (!align_corners)
      break;
  }
  case CAFFE_NEAREST:
  case CAFFE_SUPPORT: {
    if (out_size <= 1)
      return 0.0f;
    --_in_size;
    --_out_size;
  } break;
  default:;
  }

  return _in_size / (float)_out_size;
}

static inline void backward_map(int out_x, float scale, int d, float &xs,
                                int &in_x, int align_corners,
                                int half_pixel_centers,
                                PLATFORM_SUPPORT platform_sp) {
  switch (platform_sp) {
  case TENSORFLOW_NEAREST: {
    xs = half_pixel_centers ? (out_x + 0.5f) * scale : out_x * scale;
    xs = MIN(MAX(xs, 0.0f), d - 1);
    in_x = align_corners ? round(xs) : floor(xs);
  } break;
  case PYTORCH_SUPPORT: {
    xs = align_corners ? out_x * scale : (out_x + 0.5f) * scale - 0.5f;
    xs = MIN(MAX(xs, 0.0f), d - 1);
    in_x = floor(xs);
  } break;
  case PYTORCH_NEAREST: {
    xs = out_x * scale;
    xs = MIN(MAX(xs, 0.0f), d - 1);
    in_x = floor(xs);
  } break;
  case ONNX_NEAREST: {
    if (half_pixel_centers) {
      xs = (out_x + 0.5f) * scale - 0.5f;
      xs = MIN(MAX(xs, 0.0f), d - 1);
      in_x = floor(xs);
    } else {
      xs = out_x * scale;
      xs = MIN(MAX(xs, 0.0f), d - 1);
      in_x = align_corners ? round(xs) : floor(xs);
    }
  } break;
  case CAFFE_SUPPORT:
  case CAFFE_NEAREST: {
    xs = out_x * scale;
    xs = MIN(MAX(xs, 0.0f), d - 1);
    in_x = floor(xs);
    break;
  } break;
  default: {
    assert(false);
  }
  }
}

template <typename T> T fp_cast(float x);

template <> float fp_cast<float>(float x) { return x; }

template <typename T>
void interp_core(const T *in, T *out, int input_h, int input_w, int output_h,
                 int output_w, int pad_bag, int pad_end, bool align_corners,
                 bool half_pixel_centers, PLATFORM_SUPPORT platform_sp) {
  assert(platform_sp == PYTORCH_SUPPORT || platform_sp == TENSORFLOW_NEAREST ||
         platform_sp == PYTORCH_NEAREST || platform_sp == ONNX_NEAREST ||
         platform_sp == CAFFE_NEAREST || platform_sp == CAFFE_SUPPORT);
  assert(pad_bag == 0 && pad_end == 0);

  bool bilinear =
      platform_sp == PYTORCH_SUPPORT || platform_sp == CAFFE_SUPPORT;

  const float h_scale =
      calc_resize_scale(input_h, output_h, align_corners, platform_sp);
  const float w_scale =
      calc_resize_scale(input_w, output_w, align_corners, platform_sp);

  std::vector<float> Ys(output_h);
  std::vector<float> Xs(output_w);
  std::vector<int> Yi(output_h);
  std::vector<int> Xi(output_w);

  for (int yo = 0; yo < output_h; ++yo)
    backward_map(yo, h_scale, input_h, Ys[yo], Yi[yo], align_corners,
                 half_pixel_centers, platform_sp);

  for (int xo = 0; xo < output_w; ++xo)
    backward_map(xo, w_scale, input_w, Xs[xo], Xi[xo], align_corners,
                 half_pixel_centers, platform_sp);

  std::vector<T> buf(output_h * input_w);
  if (bilinear) {
    for (int yo = 0; yo < output_h; ++yo) {
      if (Yi[yo] == input_h - 1) {
        for (int xf = 0; xf < input_w; ++xf) {
          buf[yo * input_w + xf] = in[(input_h - 1) * input_w + xf];
        }
      } else {
        float dy = Ys[yo] - Yi[yo];
        for (int xf = 0; xf < input_w; ++xf) {
          buf[yo * input_w + xf] =
              in[Yi[yo] * input_w + xf] +
              fp_cast<T>(dy) *
                  (in[(Yi[yo] + 1) * input_w + xf] - in[Yi[yo] * input_w + xf]);
        }
      }
    }

    for (int xo = 0; xo < output_w; ++xo) {
      if (Xi[xo] == input_w - 1) {
        for (int yo = 0; yo < output_h; ++yo) {
          out[yo * output_w + xo] = buf[yo * input_w + input_w - 1];
        }
      } else {
        float dx = Xs[xo] - Xi[xo];
        for (int yo = 0; yo < output_h; ++yo) {
          out[yo * output_w + xo] =
              buf[yo * input_w + Xi[xo]] +
              fp_cast<T>(dx) *
                  (buf[yo * input_w + Xi[xo] + 1] - buf[yo * input_w + Xi[xo]]);
        }
      }
    }
  } else {
    for (int xo = 0; xo < output_w; ++xo) {
      for (int yo = 0; yo < output_h; ++yo) {
        out[yo * output_w + xo] = in[Yi[yo] * input_w + Xi[xo]];
      }
    }
  }
}

int64_t top::InterpOp::getFLOPs() {
  // flops:
  // 1. bilinear: 2 * output_element_num
  // 2. nearest: 1 * output_element_num
  if (getMode() == "nearest")
    return module::getNumElements(getOutput()) * 1;
  else
    return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::InterpOp::init(InferenceParameter &p) { return success(); }
void top::InterpOp::deinit(InferenceParameter &p) {}

LogicalResult top::InterpOp::inference(InferenceParameter &p) {
  int64_t n, c, ih, iw, oh, ow;
  module::getNCHW(getInput(), n, c, ih, iw, false);
  module::getNCHW(getOutput(), n, c, oh, ow, false);

  // dynamic
  if (p.inputs[1]) {
    auto target_shape = module::getShape(getTargetShape());
    float *target_shape_ = p.inputs[1];
    std::vector<int64_t> out_shape;
    if(target_shape[0] == 3) {
      out_shape = {(int)target_shape_[0], (int)target_shape_[1],
            (int)target_shape_[2]};
      setScaleH(APFloat((double)out_shape[1] / ih));
      setScaleW(APFloat((double)out_shape[2] / iw));
      module::setShape(getOutput(), out_shape);
      oh = out_shape[1];
      ow = out_shape[2];

    } else {
      out_shape = {(int)target_shape_[0], (int)target_shape_[1],
              (int)target_shape_[2], (int)target_shape_[3]};
      setScaleH(APFloat((double)out_shape[2] / ih));
      setScaleW(APFloat((double)out_shape[3] / iw));
      module::setShape(getOutput(), out_shape);
      oh = out_shape[2];
      ow = out_shape[3];
    }

  }
  PLATFORM_SUPPORT platform_sp;
  int coord = 0;
  bool align_corners = (getCoordMode() == "align_corners");
  bool half_pixel = (getCoordMode() == "half_pixel");
  if (getCoordMode() == "half_pixel")
    coord = 0;
  else if (getCoordMode() == "pytorch_half_pixel")
    coord = 1;
  else if (getCoordMode() == "align_corners")
    coord = 2;
  else if (getCoordMode() == "asymmetric") {
    coord = 3;
  } else {
    llvm_unreachable("Unsupport coord mode.");
  }
  const int in_hw = ih * iw;
  const int out_hw = oh * ow;
  if (getMode() == "nearest") {
    auto platform = module::getPlatform();
    switch (platform) {
    case module::Platform::ONNX:
      platform_sp = ONNX_NEAREST;
      break;
    case module::Platform::CAFFE:
      platform_sp = CAFFE_NEAREST;
      break;
    case module::Platform::TORCH:
      platform_sp = PYTORCH_NEAREST;
      break;
    case module::Platform::TFLITE:
      platform_sp = TENSORFLOW_NEAREST;
      break;
    default:
      platform_sp = ONNX_NEAREST;
      break;
    }
    align_corners = true;
    half_pixel = false;
    if (coord == 3) {
      align_corners = false;
    }
  } else if (getMode() == "linear") {
    auto platform = module::getPlatform();
    switch (platform) {
    case module::Platform::TORCH:
      platform_sp = PYTORCH_SUPPORT;
      break;
    case module::Platform::CAFFE:
      platform_sp = CAFFE_SUPPORT;
      break;
    default:
      platform_sp = PYTORCH_SUPPORT;
      break;
    }
    align_corners = (coord == 2) ? 1 : 0;
    half_pixel = (coord == 0 || coord == 1) ? 1 : 0;
  }

#pragma omp parallel for schedule(static, omp_schedule(n *c))
  for (int i = 0; i < n * c; i++) {
    interp_core<float>(p.inputs[0] + i * in_hw, p.outputs[0] + i * out_hw, ih,
                       iw, oh, ow, 0, 0, align_corners, half_pixel,
                       platform_sp);
  }
  return success();
}

void top::InterpOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto target_shape_ = getTargetShape();
  auto scale_h_ = getScaleH().convertToDouble();
  auto scale_w_ = getScaleW().convertToDouble();
  auto nof_dims = in_shape.size();
  int32_t hidx = nof_dims >= 4 ? nof_dims - 2 : -1;
  int32_t widx = nof_dims - 1;
  std::vector<int64_t> out_shape(in_shape);
  if (getMode() == "nearest" || getMode() == "linear") {
    bool is_shape_flag = false;
    if (!module::isNone(target_shape_)) {
      if (auto c_op =
              dyn_cast_or_null<top::ConcatOp>(target_shape_.getDefiningOp())) {
        if (c_op.getNumOperands() == 2) {
          if (module::isShape(c_op.getInputs()[0])) {
            is_shape_flag = true;
          }
          if (isa<top::WeightOp>(c_op.getInputs()[1].getDefiningOp())) {
            target_shape_ = c_op.getInputs()[1];
          }
        }
      }
      if (auto shape_op =
              dyn_cast<top::WeightOp>(target_shape_.getDefiningOp())) {
        auto target_shape = shape_op.read<float>();
        if (nof_dims == 5) {
          ASSERT_THIS(target_shape->at(0) ==
                      in_shape[nof_dims - 3]); // upsample_nearest_3d only
                                               // support scale_d = 1
        }
        out_shape[widx] = (int)target_shape->at(nof_dims - 2 - 1);
        if (nof_dims >= 4) {
          out_shape[hidx] =
              nof_dims >= 4 ? (int)target_shape->at(nof_dims - 2 - 2) : 1;
          setScaleH(APFloat((double)out_shape[hidx] / in_shape[hidx]));
        } else {
          setScaleH(APFloat(1.0));
        }
        setScaleW(APFloat((double)out_shape[widx] / in_shape[widx]));
        if (!is_shape_flag) {
          setOperand(1, module::getNoneOp(getOperation()));
        }
      } else if (in_shape.size() == 4) {
        // TODO
        // Temporarily modify to use fixed shape for SAM
        out_shape = {in_shape[0], in_shape[1], 2000, 2000};
        setScaleH(APFloat((double)out_shape[hidx] / in_shape[hidx]));
        setScaleW(APFloat((double)out_shape[widx] / in_shape[widx]));
      } else {
        UNREACHABLE_THIS("Not Implemented");
      }

    } else if (nof_dims >= 4 && scale_h_ > 0 && scale_w_ > 0) {
      out_shape[hidx] = floor(out_shape[hidx] * scale_h_);
      out_shape[widx] = floor(out_shape[widx] * scale_w_);
    } else if (nof_dims >= 3 && scale_w_ > 0) {
      out_shape[widx] = floor(out_shape[widx] * scale_w_);
    } else {
      llvm::errs() << "You must specify either target shape or scale.\n";
      llvm_unreachable("You must specify either target shape or scale.\n");
    }
    module::setShapeOrVerify(getOutput(), out_shape);
  } else {
    llvm_unreachable("Unsupport Interp mode type.\n");
  }
}
