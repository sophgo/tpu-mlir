//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"
#include <llvm/Support/Debug.h>
#define DEBUG_TYPE "preprocess_inference"

namespace tpu_mlir {
namespace tpu {
template <typename T>
void swapInputChannelOfFilter(std::vector<T> &filter_data,
                              std::vector<T> &new_filter,
                              RankedTensorType &filter_type) {

  std::vector<int64_t> shape(filter_type.getShape());
  int64_t size = std::accumulate(std::begin(shape), std::end(shape), 1,
                                 std::multiplies<>());
  assert(filter_data.size() == size && "filter size should be equal");

  int64_t oc, ic, frame_size;
  int64_t index = shape.size();
  assert((index == 4 || index == 5) && "filter shape size should be 4 or 5");

  frame_size = shape[index - 1] * shape[index - 2];
  ic = shape[index - 3];
  oc = shape[index - 4];
  if (index == 5) {
    oc *= shape[index - 5];
  }
  std::vector<int> order{2, 1, 0};
  T *filter = (T *)filter_data.data();
  for (int i = 0; i < oc; ++i) {
    for (int j = 0; j < ic; ++j) {
      assert(order[j] < ic);
      T *in = filter + i * ic * frame_size + order[j] * frame_size;
      T *out = (T *)new_filter.data() + i * ic * frame_size + j * frame_size;
      memcpy(out, in, frame_size * sizeof(T));
    }
  }
}

template <typename T>
LogicalResult FoldSwapChannelOp(Operation *op, PatternRewriter &rewriter,
                                tpu::Conv2DOp &convOp,
                                RankedTensorType &filter_type) {
  assert(convOp.getNumOperands() == 3 && "Conv2D op should have 3 operands");
  // filter
  auto filterOp = cast<top::WeightOp>(convOp.getFilter().getDefiningOp());
  auto filter_data = *(filterOp.read<T>());
  auto filter_name = module::getName(filterOp.getOutput()).str();
  std::vector<T> new_filter_data(filter_data.size());
  swapInputChannelOfFilter<T>(filter_data, new_filter_data, filter_type);
  auto newFilterOp = top::WeightOp::create(
      convOp, filter_name + "_swap_channel", new_filter_data, filter_type);
  convOp.setOperand(1, newFilterOp);
  rewriter.replaceOp(op, op->getOperands());
  return success();
}

class ReplacePreprocess {
public:
  void replacePreprocess(PatternRewriter &rewriter, tpu::PreprocessOp op) {
    auto name = module::getName(op.getOutput()).str();
    auto resized_dims = module::getI64Array(op.getResizeDims());
    std::string channel_order = op.getChannelOrder().str();
    std::string pixel_format = op.getCustomizationFormat().str();
    this->mean = *(module::getF64Array(op.getMean()));
    this->scale = *(module::getF64Array(op.getScale()));
    this->sign = op.getSign();

    module::getNCHW(op.getResult(), n, c, h, w, false);
    if (resized_dims->size() == 2) {
      this->resize_h = resized_dims->at(0);
      this->resize_w = resized_dims->at(1);
    } else {
      this->resize_h = h;
      this->resize_w = w;
    }

    // in non-INT8 quant mode, a castOp will be inserted right after inputOp
    mlir::Value currentOut = op.getInput();
    auto castOp0 = dyn_cast_or_null<tpu::CastOp>(currentOut.getDefiningOp());
    double qscale = 0;
    // INT8 quant mode
    if (!castOp0) {
      auto uniform_type = module::getUniformQuantizedType(op.getResult());
      qscale = uniform_type.getScale();
    }
    // FP mode, insert permute and slice before castOp
    else {
      currentOut = castOp0.getInput();
      auto castOut = castOp0.getOutput();
      castOut.setType(
          RankedTensorType::get({n, c, h, w}, module::getStorageType(castOut)));
    }
    rewriter.setInsertionPointAfterValue(currentOut);

    auto eleType = module::getStorageType(op.getResult());
    this->isInt8 = eleType.isInteger(8);
    // in mix precision case, a castOp will be inserted right after PreprocessOp
    auto nextOp = module::getNextOp(op);
    auto castOp1 = dyn_cast_or_null<tpu::CastOp>(nextOp);
    if (castOp1) {
      nextOp = module::getNextOp(nextOp);
      eleType = module::getStorageType(castOp1.getResult());
      this->isInt8 = eleType.isInteger(8);
    }

    std::map<std::string, std::pair<std::string, std::string>> attributes_map =
        {{"RGB_PLANAR", {"rgb", "nchw"}},   {"RGB_PACKED", {"rgb", "nhwc"}},
         {"BGR_PLANAR", {"bgr", "nchw"}},   {"BGR_PACKED", {"bgr", "nhwc"}},
         {"GRAYSCALE", {"gray", "nchw"}},   {"YUV420_PLANAR", {"bgr", "nchw"}},
         {"YUV_NV21", {"bgr", "nchw"}},     {"YUV_NV12", {"bgr", "nchw"}},
         {"RGBA_PLANAR", {"rgba", "nchw"}}, {"GBRG_RAW", {"gbrg", "nchw"}},
         {"GRBG_RAW", {"grbg", "nchw"}},    {"BGGR_RAW", {"bggr", "nchw"}},
         {"RGGB_RAW", {"rggb", "nchw"}}};

    if (attributes_map.find(pixel_format) == attributes_map.end())
      llvm_unreachable("customization format is not supported yet.");
    auto color = std::get<0>(attributes_map[pixel_format]);
    auto layout = std::get<1>(attributes_map[pixel_format]);
    bool swap_channel = (color != channel_order);
    LLVM_DEBUG(llvm::dbgs()
                   << "pixel_format:" << pixel_format << ", color:" << color
                   << ", channel_order_attr:" << channel_order << ", layout:"
                   << layout << ", swap_channel:" << swap_channel << "\n";);
    if (module::isBM1684XFamily() && pixel_format.substr(0, 3) == "YUV") {
      // insert Yuv2RgbOp
      currentOut = this->insertYuv2RgbOp(rewriter, name, currentOut,
                                         pixel_format, channel_order);
      c = 3; // channel
      swap_channel = false;
      resize_h = (resize_h * 2) / 3;
    }

    // insert PackRawOp, no need other preprocessOp & castOp
    if (color == "gbrg" || color == "grbg" || color == "rggb" ||
        color == "bggr") {
      assert(c == 4 && "processed raw image should include 4 channel");
      // GBRG->(2, 3, 1, 0)->RGBG
      // GRBG->(1, 0, 2, 3)->RGBG
      // RGGB->(0, 1, 3, 2)->RGBG
      // BGGR->(3, 2, 0, 1)->RGBG
      if (color == "gbrg")
        this->channel_order = {2, 3, 1, 0};
      else if (color == "grbg")
        this->channel_order = {1, 0, 2, 3};
      else if (color == "rggb")
        this->channel_order = {0, 1, 3, 2};
      else if (color == "bggr")
        this->channel_order = {3, 2, 0, 1};
      else
        llvm_unreachable("raw format not support current type");

      int64_t zeropoint = 0;
      auto finaltype = module::getStorageType(op.getResult());
      if (castOp1)
        finaltype = module::getStorageType(castOp1.getResult());
      if (castOp0)
        module::getScaleAndZeroPoint(op.getOutput(), qscale, zeropoint,
                                     this->sign, module::isAsymmetric());
      currentOut =
          this->insertPackRawOp(rewriter, name, currentOut, qscale, finaltype);
      rewriter.setInsertionPointAfterValue(currentOut);
      if (castOp0)
        rewriter.replaceOp(castOp0, {currentOut});
      rewriter.replaceOp(op, {currentOut});
      if (castOp1)
        rewriter.replaceOp(castOp1, {currentOut});
      return;
    }

    // insert permuteOp
    if (layout == "nhwc") {
      currentOut = this->insertTransposeOp(rewriter, name, currentOut);
      rewriter.setInsertionPointAfterValue(currentOut);
    }

    // insert SliceOp
    if (resize_h != h || resize_w != w) {
      currentOut = this->insertSliceOp(rewriter, name, currentOut);
      rewriter.setInsertionPointAfterValue(currentOut);
    }

    // FP mode, insert permute and slice before castOp
    if (castOp0 && !this->isInt8) {
      castOp0.setOperand(currentOut);
      currentOut = castOp0.getOutput();
      rewriter.setInsertionPointAfterValue(currentOut);
    }

    // insert ScaleLutOp (INT8) or ScaleOp (FP)
    if (this->isInt8 || castOp1) {
      quant::UniformQuantizedType qtype;
      if (castOp0) { // FP --> INT8 mix precision case
        int64_t zeropoint = 0;
        module::getScaleAndZeroPoint(op.getOutput(), qscale, zeropoint,
                                     this->sign, module::isAsymmetric());
        int64_t qmin = this->sign ? -128 : 0, qmax = this->sign ? 127 : 255;
        auto ctx = op.getOutput().getContext();
        qtype = quant::UniformQuantizedType::get(
            (uint32_t)this->sign, IntegerType::get(ctx, 8),
            rewriter.getF32Type(), qscale, zeropoint, qmin, qmax);
      } else { // pure INT8 quant mode
        qtype = module::getUniformQuantizedType(op.getResult());
      }
      currentOut = this->insertScaleLutOpOrLutOp(rewriter, name, currentOut, op,
                                                 qscale, qtype, swap_channel);
    } else if (eleType.isF32()) {
      currentOut = this->insertDWConv<float>(rewriter, name, currentOut,
                                             eleType, swap_channel);
    } else {
      currentOut = this->insertDWConv<uint16_t>(rewriter, name, currentOut,
                                                eleType, swap_channel);
    }
    rewriter.setInsertionPointAfterValue(currentOut);

    // insert swapChannelOp
    if (swap_channel) {
      currentOut = this->insertSwapChannelOp(rewriter, name, currentOut);
    }

    // replace castOp0 in FP--> INT8 mix precision case
    if (castOp0 && this->isInt8) {
      rewriter.replaceOp(castOp0, {currentOut});
    }

    // replace PreprocessOp
    rewriter.replaceOp(op, {currentOut});

    // replace castOp1 in FP --> INT8 mix precision case
    if (castOp1 && this->isInt8) {
      rewriter.replaceOp(castOp1, {currentOut});
    }

    // merge swapChannelOp into the following Conv filter
    if (swap_channel) {
      Operation *curOp = currentOut.getDefiningOp();
      auto convOp = dyn_cast_or_null<tpu::Conv2DOp>(nextOp);
      if (convOp && convOp.getGroup() == 1) {
        auto filter_type =
            convOp.getFilter().getType().template cast<RankedTensorType>();
        auto filterType = filter_type.getElementType();
        if (filterType.isInteger(8)) {
          FoldSwapChannelOp<int8_t>(curOp, rewriter, convOp, filter_type);
        } else if (filterType.isBF16() || filterType.isF16()) {
          FoldSwapChannelOp<uint16_t>(curOp, rewriter, convOp, filter_type);
        } else if (filterType.isF32()) {
          FoldSwapChannelOp<float>(curOp, rewriter, convOp, filter_type);
        } else {
          llvm_unreachable(
              "FoldSwapChannelOp does not support current quantize type");
        }
      }
    }
  }

private:
  int64_t n, c, h, w;
  int64_t resize_h, resize_w;
  std::vector<double> mean, scale;
  std::vector<int64_t> channel_order;
  bool _asymmetric = module::isAsymmetric();
  bool sign;
  bool isInt8;

  Value insertYuv2RgbOp(PatternRewriter &rewriter, std::string &name, Value opd,
                        const std::string &pixel_format,
                        const std::string &target_order) {

    int32_t src_fmt;
    if (pixel_format == "YUV420_PLANAR")
      src_fmt = 0; // FORMAT_MAPPING_YUV420P_YU12
    else if (pixel_format == "YUV_NV21")
      src_fmt = 3; // FORMAT_MAPPING_NV21
    else if (pixel_format == "YUV_NV12")
      src_fmt = 2; // FORMAT_MAPPING_NV12
    else
      llvm_unreachable("BM1688 unsopprt yuv format");
    int32_t dst_fmt = (target_order == "rgb") ? 4 : 5; // RGB=4,BGR=5

    // attr
    mlir::MLIRContext *ctx = rewriter.getContext();
    auto image_format = ImageOutFormatAttr::get(ctx, ImageOutFormat::UINT8);
    auto formula_mode =
        Yuv2rgbFormulaAttr::get(ctx, Yuv2rgbFormula::_601_limited);
    std::vector<NamedAttribute> attrs = {
        rewriter.getNamedAttr("src_format",
                              rewriter.getUI32IntegerAttr(src_fmt)),
        rewriter.getNamedAttr("dst_format",
                              rewriter.getUI32IntegerAttr(dst_fmt)),
        rewriter.getNamedAttr("image_format", image_format),
        rewriter.getNamedAttr("formula_mode", formula_mode),
    };

    // [n, c, h, w] -> [n,3,h,w]
    auto in_shape = module::getShape(opd);
    int64_t n = in_shape[1];
    int64_t h = (in_shape[2] * 2) / 3; // YUV420 height convert
    int64_t w = in_shape[3];
    auto out_type = RankedTensorType::get({n, 3, h, w},
                                          module::getUniformQuantizedType(opd));
    return rewriter.create<tpu::Yuv2rgbFormulaOp>(
        NameLoc::get(rewriter.getStringAttr(name + "_yuv2rgb")), out_type,
        ValueRange{opd}, attrs);
  }

  Value insertTransposeOp(PatternRewriter &rewriter, std::string &name,
                          Value opd) {
    llvm::errs() << "Inserting PermuteOp.\n";
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_transpose"));
    std::vector<int64_t> order{0, 3, 1, 2};
    auto none = module::getNoneOp(opd.getDefiningOp());
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
    RankedTensorType type;

    // permute shape inference;
    std::vector<int64_t> in_shape = module::getShape(opd);
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < in_shape.size(); ++i) {
      out_shape.push_back(in_shape[order[i]]);
    }

    type =
        RankedTensorType::get(out_shape, module::getUniformQuantizedType(opd));
    auto newOp = rewriter.create<tpu::PermuteOp>(
        loc, type, ArrayRef<Value>{opd, none}, attrs);
    return newOp.getOutput();
  }

  Value insertSliceOp(PatternRewriter &rewriter, std::string &name, Value opd) {
    llvm::errs() << "Inserting SliceOp.\n";
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_slice"));
    int64_t start_h = resize_h / 2 - h / 2;
    int64_t start_w = resize_w / 2 - w / 2;
    std::vector<int64_t> slice_offset{0, 0, start_h, start_w};
    std::vector<int64_t> slice_step{1, 1, 1, 1};
    std::vector<int64_t> slice_ends{-1, -1, resize_h - start_h,
                                    resize_w - start_w};
    std::vector<int64_t> axes{2, 3};
    std::vector<NamedAttribute> attrs;
    auto none = module::getNoneOp(opd.getDefiningOp());
    attrs.emplace_back(rewriter.getNamedAttr(
        "offset", rewriter.getI64ArrayAttr(slice_offset)));
    attrs.emplace_back(
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(slice_step)));
    attrs.emplace_back(
        rewriter.getNamedAttr("ends", rewriter.getI64ArrayAttr(slice_ends)));
    attrs.emplace_back(rewriter.getNamedAttr("hasparamConvert_axes",
                                             rewriter.getI64ArrayAttr(axes)));
    RankedTensorType type;
    type = RankedTensorType::get({n, c, h, w},
                                 module::getUniformQuantizedType(opd));
    auto newOp = rewriter.create<tpu::SliceOp>(
        loc, type, ArrayRef<Value>{opd, none, none, none, none}, attrs);
    return newOp.getOutput();
  }

  Value insertLutOp(PatternRewriter &rewriter, double scales, double bias,
                    quant::UniformQuantizedType qtype, std::string &name,
                    Value opd) {
    llvm::errs() << "Inserting LutOp.\n";
    int table_h = 16;
    int table_w = 16;
    int table_hw = table_h * table_w;
    int table_size = table_hw;
    auto table_shape = std::vector<int64_t>{1, 1, table_h, table_w};
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_lut"));
    auto type = RankedTensorType::get({n, c, h, w}, qtype);

    auto none = module::getNoneOp(opd.getDefiningOp());
    auto newOp =
        rewriter.create<tpu::LutOp>(loc, type, ArrayRef<Value>{opd, none});

    if (!this->sign && !module::isCV18xx()) {
      std::vector<uint8_t> table(table_size, 0);
      auto table_type =
          RankedTensorType::get(table_shape, rewriter.getIntegerType(8, false));
      for (int idx = 0; idx < table_hw; ++idx) {
        table[idx] = to_uint8(idx * scales + bias, ROUNDING_HALF_UP);
      }
      auto table_op =
          top::WeightOp::create(newOp, name + "_table", table, table_type);
      newOp->setOperand(1, table_op);
    } else {
      std::vector<int8_t> table(table_size, 0);
      auto table_type =
          RankedTensorType::get(table_shape, rewriter.getI8Type());
      for (int idx = 0; idx < table_hw; ++idx) {
        table[idx] = to_int8(idx * scales + bias, ROUNDING_HALF_UP);
      }
      auto table_op =
          top::WeightOp::create(newOp, name + "_table", table, table_type);
      newOp->setOperand(1, table_op);
    }
    return newOp.getOutput();
  }

  Value insertScaleLutOp(PatternRewriter &rewriter, std::vector<double> scales,
                         std::vector<double> bias,
                         quant::UniformQuantizedType qtype, std::string &name,
                         Value opd) {
    llvm::errs() << "Inserting ScalelutOp.\n";
    int table_h = 16;
    int table_w = 16;
    int table_hw = table_h * table_w;
    int table_size = c * table_hw;
    auto table_shape = std::vector<int64_t>{1, c, table_h, table_w};
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_scale_lut"));

    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(
        rewriter.getNamedAttr("scale", rewriter.getF64ArrayAttr(scales)));
    attrs.emplace_back(
        rewriter.getNamedAttr("bias", rewriter.getF64ArrayAttr(bias)));
    attrs.emplace_back(
        rewriter.getNamedAttr("sign", rewriter.getBoolAttr(this->sign)));

    auto type = RankedTensorType::get({n, c, h, w}, qtype);

    auto none = module::getNoneOp(opd.getDefiningOp());
    auto newOp = rewriter.create<tpu::ScaleLutOp>(
        loc, type, ArrayRef<Value>{opd, none}, attrs);

    if (!this->sign && !module::isCV18xx()) {
      std::vector<uint8_t> table(table_size, 0);
      auto table_type =
          RankedTensorType::get(table_shape, rewriter.getIntegerType(8, false));
      for (int i = 0; i < c; i++) {
        for (int idx = 0; idx < table_hw; ++idx) {
          table[i * table_hw + idx] =
              to_uint8(idx * scales[i], ROUNDING_HALF_UP);
        }
      }
      auto table_op =
          top::WeightOp::create(newOp, name + "_table", table, table_type);
      newOp->setOperand(1, table_op);
    } else {
      std::vector<int8_t> table(table_size, 0);
      auto table_type =
          RankedTensorType::get(table_shape, rewriter.getI8Type());
      for (int i = 0; i < c; i++) {
        for (int idx = 0; idx < table_hw; ++idx) {
          table[i * table_hw + idx] =
              to_int8(idx * scales[i] + bias[i], ROUNDING_HALF_UP);
        }
      }
      auto table_op =
          top::WeightOp::create(newOp, name + "_table", table, table_type);
      newOp->setOperand(1, table_op);
    }
    return newOp.getOutput();
  }

  Value insertScaleLutOpOrLutOp(PatternRewriter &rewriter, std::string &name,
                                Value opd, tpu::PreprocessOp op, double qscale,
                                quant::UniformQuantizedType qtype,
                                bool swap_channel) {
    std::vector<double> scales;
    std::vector<double> bias;
    for (int i = 0; i < c; i++) {
      scales.push_back(this->scale[i]);
      bias.push_back(-1 * this->scale[i] * this->mean[i]);
    }
    if (swap_channel) {
      // keep order bgr
      std::swap(scales[0], scales[2]);
      std::swap(bias[0], bias[2]);
    }
    // quant
    bool in_out_equal = true;
    bool s_m_equal = true;
    for (int i = 0; i < c; i++) {
      scales[i] /= qscale;
      bias[i] /= qscale;
      if (scales[i] != scales[0] || bias[i] != bias[0]) {
        s_m_equal = false;
      }
      if (scales[i] >= 0.99 && scales[i] <= 1.01 && bias[i] == 0.0) {
        // in out will be same, no need do ScaleLut
      } else {
        in_out_equal = false;
      }
    }
    // no need to insert scalelut when scale = 1 and bias = 0
    if (in_out_equal && !this->sign) {
      // set qtype to previous op in case next op is cast
      opd.setType(RankedTensorType::get({n, c, h, w}, qtype));
      return opd;
    } else if (s_m_equal) {
      // scale and bias are the same, insert lutop
      return this->insertLutOp(rewriter, scales[0], bias[0], qtype, name, opd);
    } else {
      // scale and bias are not same, insert scalelutop
      return this->insertScaleLutOp(rewriter, scales, bias, qtype, name, opd);
    }
  }

  template <typename T>
  Value insertDWConv(PatternRewriter &rewriter, std::string &name, Value opd,
                     Type eleType, bool swap_channel) {
    llvm::errs() << "Inserting DWConvOp.\n";
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_dwconv"));
    auto none = module::getNoneOp(opd.getDefiningOp());
    std::vector<T> scales;
    std::vector<float> bias;
    for (int i = 0; i < c; i++) {
      if (eleType.isBF16()) {
        scales.push_back(f32_to_bf16(this->scale[i]));
      } else if (eleType.isF16()) {
        scales.push_back(f32_to_f16(this->scale[i]));
      } else {
        scales.push_back(this->scale[i]);
      }
      bias.push_back(-1 * this->scale[i] * this->mean[i]);
    }

    if (c == 3 && swap_channel) {
      std::swap(scales[0], scales[2]);
      std::swap(bias[0], bias[2]);
    }

    NamedAttrList attrs;
    attrs.set("kernel_shape", rewriter.getI64ArrayAttr({1, 1}));
    attrs.set("strides", rewriter.getI64ArrayAttr({1, 1}));
    attrs.set("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0}));
    attrs.set("group", rewriter.getI64IntegerAttr(c));
    attrs.set("with_bias", rewriter.getBoolAttr(true));

    auto type = RankedTensorType::get({n, c, h, w}, eleType);
    auto newOp = rewriter.create<tpu::Conv2DOp>(
        loc, type, ValueRange{opd, none, none}, attrs);

    auto filter_type = RankedTensorType::get({c, 1, 1, 1}, eleType);
    auto scale_weight =
        top::WeightOp::create(newOp, "weight", scales, filter_type);
    auto bias_type = RankedTensorType::get({c}, rewriter.getF32Type());
    auto bias_weight = top::WeightOp::create(newOp, "bias", bias, bias_type);

    newOp.setOperand(1, scale_weight);
    newOp.setOperand(2, bias_weight);

    return newOp.getOutput();
  }

  Value insertSwapChannelOp(PatternRewriter &rewriter, std::string &name,
                            Value opd) {
    llvm::errs() << "Inserting SwapChannelOp.\n";
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_swap_channel"));
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> orders{2, 1, 0};
    attrs.emplace_back(rewriter.getNamedAttr("channel_order",
                                             rewriter.getI64ArrayAttr(orders)));
    auto type = opd.getType();
    auto newOp = rewriter.create<tpu::SwapChannelOp>(
        loc, type, ArrayRef<Value>{opd}, attrs);
    return newOp.getOutput();
  }

  Value insertPackRawOp(PatternRewriter &rewriter, std::string &name, Value opd,
                        double threshold, Type qtype) {
    llvm::errs() << "Inserting PackRawOp.\n";
    float white_level = 4095.;
    float black_level = 112.;
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(rewriter.getNamedAttr(
        "white_level", rewriter.getF64FloatAttr(white_level)));
    attrs.emplace_back(rewriter.getNamedAttr(
        "black_level", rewriter.getF64FloatAttr(black_level)));
    attrs.emplace_back(rewriter.getNamedAttr(
        "threshold", rewriter.getF64FloatAttr(threshold)));
    attrs.emplace_back(rewriter.getNamedAttr(
        "channel_order", rewriter.getI64ArrayAttr(channel_order)));
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_pack_raw"));
    auto type = RankedTensorType::get({n, c, h, w}, qtype);
    auto none_op = module::getNoneOp(opd.getDefiningOp());
    auto newOp = rewriter.create<tpu::PackRawOp>(
        loc, type, ArrayRef<Value>{opd, none_op, none_op}, attrs);
    return newOp.getOutput();
  };
};

struct PreprocessFuse : public OpRewriterPatternEx<tpu::PreprocessOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  PreprocessFuse(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::PreprocessOp>(context, "PreprocessFuse") {}

  LogicalResult matchAndRewriteImpl(tpu::PreprocessOp op,
                                    PatternRewriter &rewriter) const override {
    ReplacePreprocess replacer = ReplacePreprocess();
    replacer.replacePreprocess(rewriter, op);
    return success();
  }
};

void tpu::PreprocessOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.insert<PreprocessFuse>(context);
}

} // namespace tpu
} // namespace tpu_mlir
