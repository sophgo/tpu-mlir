//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/FormatVariadic.h"
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "interp_top2tpu"

namespace tpu_mlir {
namespace cv18xx {

static void gen_half_pixel_scale(double scale, int64_t pad,
                                 std::vector<float> &scale_list) {
  // for example, scale = 2, scale_list = [0.75, 0.25, 0.25, 0.75]
  int64_t count = static_cast<int64_t>(scale);
  scale_list.resize(2 * count);
  for (int64_t i = 0; i < count; ++i) {
    int64_t idx = pad - i;
    if (idx < 0) {
      idx += count;
    }
    float distance = (0.5 + i) / scale + 0.5;
    distance -= static_cast<int>(distance);
    scale_list[idx] = 1 - distance;
    scale_list[count + idx] = distance;
  }
}

static void dot(std::vector<float> &lhs, std::vector<float> &rhs,
                std::vector<float> &rst) {
  size_t lhs_size = lhs.size();
  size_t rhs_size = rhs.size();
  rst.resize(lhs_size * rhs_size);
  for (auto i = 0; i < rhs_size; ++i) {
    for (auto j = 0; j < lhs_size; ++j) {
      rst[j * rhs_size + i] = lhs[j] * rhs[i];
    }
  }
}

static void resize_to_conv1(PatternRewriter &rewriter, top::InterpOp &op,
                            double scale_h, double scale_w) {
  std::vector<Value> operands;
  std::vector<int64_t> pads = {0, 0, 1, 1, 0, 0, 1, 1};
  double const_val = 0.0;

  std::vector<int64_t> shape_after_pad;
  auto input_shape = module::getShape(op.getInput());
  for (int i = 0; i < input_shape.size(); ++i) {
    shape_after_pad.emplace_back(pads[i] + input_shape[i] + pads[i + 4]);
  }

  // insert pad op
  rewriter.setInsertionPointAfterValue(op.getInput());
  std::string name = module::getName(op.getInput()).str() + "_pad_edge";
  auto loc = NameLoc::get(rewriter.getStringAttr(name));
  std::vector<NamedAttribute> pad_attrs;
  pad_attrs.emplace_back(
      rewriter.getNamedAttr("paddings", rewriter.getI64ArrayAttr(pads)));
  pad_attrs.emplace_back(
      rewriter.getNamedAttr("val", rewriter.getF64FloatAttr(const_val)));
  pad_attrs.push_back(
      rewriter.getNamedAttr("mode", rewriter.getStringAttr("edge")));
  auto pad_type = module::getTypeLike(op.getInput(), shape_after_pad);
  auto pad_op = rewriter.create<top::PadOp>(
      loc, pad_type, ValueRange{op.getInput(), module::getNoneOp(op)},
      pad_attrs);

  // insert conv op
  int64_t ic = shape_after_pad[1];
  int64_t pad_t = int64_t(scale_h / 2) - 1;
  int64_t pad_b = int64_t(scale_h) - pad_t - 2;
  int64_t pad_l = int64_t(scale_w / 2) - 1;
  int64_t pad_r = int64_t(scale_w) - pad_l - 2;
  std::vector<int64_t> conv_strides = {1, 1};
  std::vector<int64_t> conv_pads = {pad_t, pad_l, pad_b, pad_r};
  std::vector<int64_t> conv_kernel_shape = {static_cast<int64_t>(2 * scale_h),
                                            static_cast<int64_t>(2 * scale_w)};
  std::vector<int64_t> conv_dilation = {1, 1};
  std::vector<int64_t> conv_ins = {int64_t(scale_h - 1), int64_t(scale_w - 1)};
  int64_t group = ic;
  std::vector<float> factor_w;
  std::vector<float> factor_h;
  std::vector<float> factor;
  gen_half_pixel_scale(scale_w, pad_l, factor_w);
  gen_half_pixel_scale(scale_h, pad_t, factor_h);
  dot(factor_h, factor_w, factor);
  std::vector<float> weight;
  for (auto i = 0; i < ic; ++i) {
    std::copy(factor.begin(), factor.end(), std::back_inserter(weight));
  }

  // create conv kernel (weight)
  // weight_shape = [ic, 1, 1, kh, kw]
  std::vector<int64_t> weight_shape = {ic, 1, 1, conv_kernel_shape[0],
                                       conv_kernel_shape[1]};
  auto weight_type = RankedTensorType::get(weight_shape, rewriter.getF32Type());
  std::string weight_name =
      module::getName(op.getInput()).str() + "_add_weight";
  auto weight_operand =
      top::WeightOp::create(op, weight_name, weight, weight_type);

  // create conv op
  operands.emplace_back(pad_op.getOutput());
  operands.emplace_back(weight_operand);
  operands.emplace_back(module::getNoneOp(op));
  std::vector<NamedAttribute> conv_attrs;
  conv_attrs.emplace_back(rewriter.getNamedAttr(
      "kernel_shape", rewriter.getI64ArrayAttr(conv_kernel_shape)));
  conv_attrs.emplace_back(
      rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr(conv_strides)));
  conv_attrs.emplace_back(
      rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr(conv_pads)));
  conv_attrs.emplace_back(rewriter.getNamedAttr(
      "dilations", rewriter.getI64ArrayAttr(conv_dilation)));
  conv_attrs.emplace_back(
      rewriter.getNamedAttr("inserts", rewriter.getI64ArrayAttr(conv_ins)));
  conv_attrs.emplace_back(
      rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(group)));
  rewriter.replaceOpWithNewOp<top::ConvOp>(op, op.getType(), operands,
                                           conv_attrs);
}

static void resize_to_conv2(PatternRewriter &rewriter, top::InterpOp &op,
                            double scale_h, double scale_w) {
  std::vector<Value> operands;
  auto input_shape = module::getShape(op.getInput());
  int64_t ic = input_shape[1];
  std::vector<int64_t> conv_strides = {2, 2};
  std::vector<int64_t> conv_pads = {0, 0, 0, 0};
  std::vector<int64_t> conv_kernel_shape = {2, 2};
  std::vector<int64_t> conv_dilation = {1, 1};
  int64_t group = ic;
  std::vector<float> factor = {0.25, 0.25, 0.25, 0.25};
  std::vector<float> weight;
  for (auto i = 0; i < ic; ++i) {
    std::copy(factor.begin(), factor.end(), std::back_inserter(weight));
  }

  // create conv kernel (weight)
  // weight_shape = [ic, 1, 1, kh, kw]
  std::vector<int64_t> weight_shape = {ic, 1, 1, conv_kernel_shape[0],
                                       conv_kernel_shape[1]};
  auto weight_type = RankedTensorType::get(weight_shape, rewriter.getF32Type());
  std::string weight_name =
      module::getName(op.getInput()).str() + "_conv_filter";
  auto weight_operand =
      top::WeightOp::create(op, weight_name, weight, weight_type);

  // create conv op
  operands.emplace_back(op.getInput());
  operands.emplace_back(weight_operand);
  operands.emplace_back(module::getNoneOp(op));
  std::vector<NamedAttribute> conv_attrs;
  conv_attrs.emplace_back(rewriter.getNamedAttr(
      "kernel_shape", rewriter.getI64ArrayAttr(conv_kernel_shape)));
  conv_attrs.emplace_back(
      rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr(conv_strides)));
  conv_attrs.emplace_back(
      rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr(conv_pads)));
  conv_attrs.emplace_back(rewriter.getNamedAttr(
      "dilations", rewriter.getI64ArrayAttr(conv_dilation)));
  conv_attrs.emplace_back(
      rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(group)));
  rewriter.replaceOpWithNewOp<top::ConvOp>(op, op.getType(), operands,
                                           conv_attrs);
}

static void getInterpHWScale(int64_t oh, int64_t ow, int64_t ih, int64_t iw,
                             float &rheight, float &rwidth) {
  if (oh == 1) {
    rheight = ih;
  } else {
    rheight = static_cast<float>(ih - 1) / (oh - 1);
  }
  if (ow == 1) {
    rwidth = iw;
  } else {
    rwidth = static_cast<float>(iw - 1) / (ow - 1);
  }
}

// \floatDividend if gived, it should be find one divisor that the range
// should be in floatDividend < x < 2 * floatDividend e.g: getDivisors(32, 5)
// should be 4 * 8, 5< 8 < 10
static std::pair<std::vector<std::pair<int, int>>, int>
getDivisors(int n, int floatDividend = 0, bool isInsInConv = true) {
  std::vector<std::pair<int, int>> divisors;
  int insertMax = 14;
  if (!isInsInConv) {
    insertMax = n; // directly use dilate op
  }
  auto div = [&](int n) mutable -> int {
    // FIXME: depends by hw, 14 is the max size of insert number
    if (n < insertMax) {
      return n; // no need to slice
    }

    for (int i = sqrt(n); i > 1; i--) {
      if (n % i == 0 && i < insertMax) {
        return i;
      }
    }
    return 0;
  };

  int maxFloatDividend = 0;
  if (floatDividend) {
    // check possible divisors's range between
    // \floatDividend<x<2*\floatDividend
    int found = 0;
    int i;
    int floatDividendStart = std::min(2 * floatDividend - 1, insertMax);
    for (i = floatDividendStart; i > floatDividend; i--) {
      float is_disivible = n / (float)i;
      if ((ceilf(is_disivible) == is_disivible &&
           floorf(is_disivible) == is_disivible)) {
        found = 1;
        break;
      }
    }

    if (found) {
      n = n / i;
      maxFloatDividend = i;
    } else {
      return std::make_pair(divisors, maxFloatDividend);
    }
  }

  while (n != 1) {
    int d = div(n);
    if (!d) {
      divisors.clear();
      break;
    }

    divisors.push_back(std::make_pair(d, 1));
    n = n / d;
  }

  return std::make_pair(divisors, maxFloatDividend);
}

static bool resize_to_conv_deconv(PatternRewriter &rewriter, top::InterpOp &op,
                                  int64_t in, int64_t ic, int64_t ih,
                                  int64_t iw, int64_t on, int64_t oc,
                                  int64_t oh, int64_t ow) {
  bool is_shrink = true;
  float shrink_factor = 0.0f;
  if (oh > ih && ow > iw) {
    is_shrink = false;
  } else if (oh < ih && ow < iw) {
    float h_factor = static_cast<float>(ih - 1) / (oh - 1);
    float w_factor = static_cast<float>(iw - 1) / (ow - 1);
    if (std::abs(h_factor - w_factor) > 1e-5) {
      llvm::errs() << "resize convert not support ih/iw:" << ih << "/" << iw
                   << ", oh/ow:" << oh << "/" << ow;
      return false;
    }
    shrink_factor = h_factor;
  } else {
    llvm::errs() << "resize convert not support ih/iw:" << ih << "/" << iw
                 << ", oh/ow:" << oh << "/" << ow;
    return false;
  }

  if (is_shrink) {
    if (std::abs(std::ceil(shrink_factor) - std::floor(shrink_factor)) < 1e-5) {
      int64_t factor = shrink_factor;
      // replace with conv
      int filter_size = factor * factor;
      float filter_val = 1; // nearnest
      std::vector<int64_t> filter_shape = {ic, 1, factor, factor};
      std::vector<float> new_filter(filter_size * ic, 0);
      for (int i = 0; i < ic; ++i) {
        new_filter[i * filter_size] = filter_val; // nearest
      }
      // insert conv op
      std::vector<int64_t> conv_strides = {factor, factor};
      std::vector<int64_t> conv_pads = {0, 0, 1, 1};
      std::vector<int64_t> conv_kernel_shape = {factor, factor};
      int64_t group = ic;
      // create conv kernel (weight)
      // weight_shape = [ic, 1, 1, kh, kw]
      auto weight_type =
          RankedTensorType::get(filter_shape, rewriter.getF32Type());
      std::string weight_name =
          module::getName(op.getInput()).str() + "_conv_weight";
      auto weight_operand =
          top::WeightOp::create(op, weight_name, new_filter, weight_type);

      // create conv op
      std::vector<Value> operands;
      operands.emplace_back(op.getInput());
      operands.emplace_back(weight_operand);
      operands.emplace_back(module::getNoneOp(op));
      std::vector<NamedAttribute> conv_attrs;
      conv_attrs.emplace_back(rewriter.getNamedAttr(
          "kernel_shape", rewriter.getI64ArrayAttr(conv_kernel_shape)));
      conv_attrs.emplace_back(rewriter.getNamedAttr(
          "strides", rewriter.getI64ArrayAttr(conv_strides)));
      conv_attrs.emplace_back(
          rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr(conv_pads)));
      conv_attrs.emplace_back(
          rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(group)));
      rewriter.replaceOpWithNewOp<top::ConvOp>(op, op.getType(), operands,
                                               conv_attrs);
      return true;
    }
    return false;
  }

  float rwidth = 0.0f;
  float rheight = 0.0f;
  int rwidthInt = 0;
  int rheightInt = 0;

  int kh = -1;
  int kw = -1;

  // keep Dividend / Divisor for later non-divisable
  std::vector<std::pair<int, int>> maxInsertWAtOnce;
  std::vector<std::pair<int, int>> maxInsertHAtOnce;
  // seperate Dividend, Divisor as scale to deal with float case
  // scale[0] as h, scale[1] for w
  // pair is Dividend / Divisor
  SmallVector<std::pair<int, int>, 2> scale = {{0, 0}, {0, 0}};

  getInterpHWScale(ih, iw, oh, ow, rheight, rwidth);
  int floatDividend = 0;
  int maxFloatDividend = 0;
  auto loc = op.getLoc();

  // TODO: slice h/w under ins case
  bool isInsInConv = false;

  // deal with non divisable case
  // h
  if (ceilf(rheight) == floorf(rheight)) {
    // integer case
    rheightInt = int(rheight);
    std::tie(maxInsertHAtOnce, maxFloatDividend) = getDivisors(rheightInt);
  } else {
    // 2047 / 63 = 89 * 23 / 7 * 9
    // float case: e.g: 6->33 = 6 * (2/5)
    floatDividend = ih - 1;
    std::tie(maxInsertHAtOnce, maxFloatDividend) =
        getDivisors(oh - 1, floatDividend);
    if (!maxInsertHAtOnce.size()) {
      // TODO: merge info into scale
      std::vector<std::pair<int, int>> ohDivisors;
      std::tie(ohDivisors, maxFloatDividend) =
          getDivisors(oh - 1, 0, isInsInConv);
      if (!ohDivisors.size()) {
        ohDivisors.push_back(std::make_pair(oh - 1, 1));
      }
      int count = ohDivisors.size();
      for (int i = 0; i < count; i++) {
        int ohDivisor = ohDivisors[i].first;
        int ihDivisor = i < count - 1 ? 1 : ih - 1;
        maxInsertHAtOnce.push_back(std::make_pair(ohDivisor, ihDivisor));
      }
    } else {
      scale[0] = (std::make_pair(maxFloatDividend, floatDividend));
    }
  }

  // w
  if ((ceilf(rwidth) == rwidth && floorf(rwidth) == rwidth)) {
    // integer case
    rwidthInt = int(rwidth);
    std::tie(maxInsertWAtOnce, maxFloatDividend) = getDivisors(rwidthInt);
  } else {
    // float case: e.g: 6->33 = 6 * (2/5)
    // we seleprate integer part and float part
    // 6->33 = 32 / 5 = 4 * (8/5) = 4x . 8/5
    // 8/5 which means we insert (8-1) and stride = 5
    // NOTICE: float part SHOULD BE 1<x<2
    floatDividend = iw - 1;
    std::tie(maxInsertWAtOnce, maxFloatDividend) =
        getDivisors(ow - 1, floatDividend);
    if (!maxInsertWAtOnce.size()) {
      // TODO: seperate all divisor
      std::vector<std::pair<int, int>> owDivisors;
      std::tie(owDivisors, maxFloatDividend) =
          getDivisors(ow - 1, 0, isInsInConv);
      if (!owDivisors.size()) {
        owDivisors.push_back(std::make_pair(ow - 1, 1));
      }
      int count = owDivisors.size();
      for (int i = 0; i < count; i++) {
        int owDivisor = owDivisors[i].first;
        int iwDivisor = i < count - 1 ? 1 : iw - 1;
        maxInsertWAtOnce.push_back(std::make_pair(owDivisor, iwDivisor));
      }
    } else {
      scale[1] = (std::make_pair(maxFloatDividend, floatDividend));
    }
  }
  // construct conv with insert/padding
  auto input = op->getOperand(0);
  auto input_type = input.getType().cast<RankedTensorType>();
  auto input_shape = input_type.getShape();
  int g = input_shape[1]; // g == ic for depthwise

  auto NoneOp = module::getNoneOp(op);

  int _ic = ic;
  int _oc = oc;
  int _ih = ih;
  int _iw = iw;
  // NOTICE: 1x1 ALWAYS fill the same value
  int is1x1Input = ih == 1 && ih == iw;
  if (is1x1Input && (!maxInsertHAtOnce.size() || !maxInsertWAtOnce.size())) {
    // deeplabv3_mobilenetv2 case
    // 1x1->46x80 case that 46 seperate 2x23 and the limitation of dilate
    // is 15, replace with upsample case(tiu copy)
    std::vector<NamedAttribute> attrs;
    std::vector<Value> operands;
    attrs.push_back(
        rewriter.getNamedAttr("scale_h", rewriter.getI64IntegerAttr(oh)));
    attrs.push_back(
        rewriter.getNamedAttr("scale_w", rewriter.getI64IntegerAttr(ow)));
    operands.push_back(input);
    rewriter.replaceOpWithNewOp<top::UpsampleOp>(op, op.getType(), operands,
                                                 attrs);
    return true;
  }
  // TODO support
  // not support in tpu-mlir:convert in top2tpu will cause less of threshold
  // when convert one op to multiply ops.
  return false;

  // check for hw spec, ins/stride range is 0-15 in 1835
  for (auto h_ins_stride : maxInsertHAtOnce) {
    int stride, ins;
    std::tie(ins, stride) = h_ins_stride;
    if (ins > 15 || stride > 15) {
      LLVM_DEBUG(llvm::errs()
                 << "h-params over hardware limitation, leverage cpu,"
                 << "ins/stride is:" << ins << "/" << stride << "\n");
      return false;
    }
  }

  for (auto w_ins_stride : maxInsertWAtOnce) {
    int stride, ins;
    std::tie(ins, stride) = w_ins_stride;
    if (ins > 15 || stride > 15) {
      LLVM_DEBUG(llvm::errs()
                     << "w-params over hardware limitation, leverage cpu,"
                     << "ins/stride is:" << ins << "/" << stride << "\n";);
      return false;
    }
  }

  int loop = std::max(maxInsertHAtOnce.size(), maxInsertWAtOnce.size());

  auto calc_dilute_hw = [&](int h, int ins_h, int ins_h_l, int pad_h_b,
                            int pad_h_t) mutable -> int {
    return (h - 1) * (ins_h + 1) + ins_h_l + 1 + pad_h_t + pad_h_b;
  };

  auto calc_output_hw = [&](int hw, int khw, int stride) mutable -> int {
    return (hw - khw) / stride + 1;
  };

  auto createConvAttr =
      [&](std::vector<int64_t> kernel, std::vector<int64_t> stride,
          std::vector<int64_t> dilation, std::vector<int64_t> padding, int g,
          bool is_dw, bool with_bias,
          std::vector<int64_t> ins) mutable -> std::vector<NamedAttribute> {
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(rewriter.getNamedAttr("kernel_shape",
                                             rewriter.getI64ArrayAttr(kernel)));
    attrs.emplace_back(
        rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr(stride)));
    attrs.emplace_back(
        rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr(padding)));
    attrs.emplace_back(
        rewriter.getNamedAttr("dilations", rewriter.getI64ArrayAttr(dilation)));
    attrs.emplace_back(
        rewriter.getNamedAttr("inserts", rewriter.getI64ArrayAttr(ins)));
    attrs.emplace_back(
        rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(g)));
    return attrs;
  };

  auto createConv2D = [&](Value input, int d,
                          bool isNonDivisible = false) mutable
      -> std::tuple<std::vector<Value>, std::vector<NamedAttribute>,
                    RankedTensorType> {
    if (_ih == 1 || _iw == 1) {
      assert(_iw == _ih && "not support asymmetrical under _ih = 1 or _iw = 1");
    }

    rheightInt = 1;
    rwidthInt = 1;
    std::vector<int64_t> kernel(2), stride(2), dilation(2), padding(4);
    std::vector<int64_t> ins(2), _ins(2);

    // TODO: support d not integer case, e.g: d = 1.3
    stride[0] = 1; // sh
    stride[1] = 1; // sw

    if (isNonDivisible) {
      if (scale[0].first) {
        std::tie(rheightInt, stride[0]) = scale[0];
      }

      if (scale[1].first) {
        std::tie(rwidthInt, stride[1]) = scale[1];
      }
    } else {
      int divisor, dividend;
      if (d < (int)maxInsertHAtOnce.size()) { // star with 0
        std::tie(dividend, divisor) = maxInsertHAtOnce[d];
        float rheight = dividend / (float)divisor;
        if ((ceilf(rheight) == rheight && floorf(rheight) == rheight)) {
          rheightInt = rheight; // divisible
        } else {
          stride[0] = divisor;   // sh
          rheightInt = dividend; // hw ins_w
        }
      }

      if (d < (int)maxInsertWAtOnce.size()) { // star with 0
        std::tie(dividend, divisor) = maxInsertWAtOnce[d];
        float rwidth = dividend / (float)divisor;
        if ((ceilf(rwidth) == rwidth && floorf(rwidth) == rwidth)) {
          rwidthInt = rwidth; // divisible
        } else {
          stride[1] = divisor;
          rwidthInt = dividend; // hw ins_w
        }
      }
    }

    // init parameter
    kh = (rheightInt - 1) * 2 + 1;
    kw = (rwidthInt - 1) * 2 + 1;
    bool is_dw = true;
    bool with_bias = false;
    kernel[0] = kh;
    kernel[1] = kw;
    dilation[0] = dilation[1] = 1;

    ins[0] = rwidthInt - 1;  // hw ins_w
    ins[1] = rheightInt - 1; // hw ins_h
    _ins = ins;

    padding[0] = padding[1] = rheightInt - 1; // padding top/bottom
    padding[2] = padding[3] = rwidthInt - 1;  // padding left/right

    // depthwise case
    _oc = 1;
    _ic = 1;

    if (is1x1Input) {
      kh = rheightInt;
      kw = rwidthInt;
      stride[0] = kh;
      stride[1] = kw;
      padding[2] = padding[3] = padding[0] = padding[1] = 0;

      ins[0] = 0;
      ins[1] = 0;
    }

    // init filter
    int count = g * _ic * _oc * kh * kw; // depthewise, ic == oc
    std::vector<float> filter(count, 1);
    std::vector<int64_t> filter_shape;

    if (is1x1Input) {
      // default fill to 1
    } else {
      // fill filter from corner
      for (int i = 0; i < kh / 2 + 1; i++) {
        for (int j = 0; j < kw / 2 + 1; j++) {
          float f = (i + 1) * (j + 1) / float(rheightInt * rwidthInt);
          filter.data()[i * kw + j] = f;
          filter.data()[i * kw + (kw - 1) - j] = f;
          filter.data()[(kh - 1 - i) * kw + j] = f;
          filter.data()[(kh - 1 - i) * kw + (kw - 1) - j] = f;
        }
      }

      // duplicate to ic oc g
      int j = _ic * _oc * g;
      int khw = kh * kw;
      for (int i = 1; i < j; i++) {
        std::copy(filter.data(), filter.data() + khw, filter.data() + i * khw);
      }
    }

    if (g != 1) {
      filter_shape.push_back(g);
    }

    filter_shape.push_back(_oc);
    filter_shape.push_back(_ic);
    filter_shape.push_back(kh);
    filter_shape.push_back(kw);

    // prepare filter
    auto filter_type =
        RankedTensorType::get(filter_shape, rewriter.getF32Type());
    std::string filter_name = std::to_string(d) + "_filter";
    auto weight_operand =
        top::WeightOp::create(op, filter_name, filter, filter_type);

    // it could Dilated in activation in hw once `ins` set
    // the final output should be input->Dilated(ins_w/ins_h)->conv
    std::vector<int64_t> top_dim(2); // [0] is h, [1] is w
    _oc = ic;                        // depthwise case
    std::string prefix = llvm::formatv("_{0}_", std::to_string(d)).str();

    if (!is1x1Input) {
#if 0
// TODO support this
      // to check memory usage per lane
      // create fake op for check
      std::vector<Value> operands;
      operands.push_back(input);
      operands.push_back(weight_operand);
      operands.push_back(module::getNoneOp(op)); // bias
      kernel[0] = kh;
      kernel[1] = kw;
      std::vector<NamedAttribute> attrs = createConvAttr(
          kernel, stride, dilation, padding, g, is_dw, with_bias, ins);

      int ih_ext = calc_dilute_hw(_ih, ins[1], 0, padding[0], padding[1]);
      int iw_ext = calc_dilute_hw(_iw, ins[0], 0, padding[2], padding[3]);
      top_dim[0] = calc_output_hw(ih_ext, kh, stride[0]); // oh
      top_dim[1] = calc_output_hw(iw_ext, kw, stride[1]); // ow
      RankedTensorType dilateOutput = RankedTensorType::get(
          {in, _oc, top_dim[0], top_dim[1]}, input_type.getElementType());

      auto fakeOp = rewriter.create<tpu::Conv2DOp>(
          loc, dilateOutput, ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});

      // FIXME: no need init every travel
      uint64_t totalPerLane = SimpleConv2DMemoryUsageAnalysis(fakeOp, NULL);

      // depthwse with ins SHOULD not slice h/w
      // just slice ic, <n, ic, ih, iw> -> <n, 1, ih, iw>
      int chunkPerLane = (ic + MInfo::lane_num) / MInfo::lane_num;
      if (!isInsInConv || totalPerLane / chunkPerLane > MInfo::lmem_per_lane) {
        // if lmem not enough
        LLVM_DEBUG(llvm::errs()
                   << _interpOp.nameAttr().getValue()
                   << ", lmem not enough, dynamic add dilate op\n");

        // create dilateOp if need
        top_dim[0] = calc_dilute_hw(_ih, ins[1], 0, 0, 0);
        top_dim[1] = calc_dilute_hw(_iw, ins[0], 0, 0, 0);

        // init output
        RankedTensorType output = RankedTensorType::get(
            {in, _oc, top_dim[0], top_dim[1]}, input_type.getElementType());

        // init input
        operands.clear();
        operands.push_back(input);

        // init attr
        std::vector<NamedAttribute> attrs;
        attrs.push_back(rewriter.getNamedAttr(
            "ins", rewriter.getI32ArrayAttr(
                       ArrayRef<int32_t>({ins})))); // [0]ins_w/[1]ins_h
        attrs.push_back(rewriter.getNamedAttr(
            "name",
            rewriter.getStringAttr(prefix + "_dilate_" +
                                   _interpOp.nameAttr().getValue().str())));
        attrs.push_back(rewriter.getNamedAttr(
            "fill_constant",
            rewriter.getI32IntegerAttr(0))); // default insert 0
        attrs.push_back(
            rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

        auto dilateOp = rewriter.create<tpu::DilateOp>(
            loc, output, ArrayRef<Value>{operands},
            ArrayRef<NamedAttribute>{attrs});
        input = dilateOp.getResult();
        ins = {0, 0}; // no dilate in conv
      }
#else
      return std::make_tuple(std::vector<Value>(),
                             std::vector<NamedAttribute>(), RankedTensorType());
#endif
    }

    // prepare output operands
    std::vector<Value> operands;
    operands.push_back(input);
    operands.push_back(weight_operand);
    operands.push_back(NoneOp); // bias

    // prepare attr
    kernel[0] = kh;
    kernel[1] = kw;
    std::vector<NamedAttribute> attrs = createConvAttr(
        kernel, stride, dilation, padding, g, is_dw, with_bias, ins);

    if (loop - 1 == d) {
      // last one replace the interp name for compare
      prefix = "";
    }

    // prepare output shape
    if (is1x1Input) {
      // upsample
      top_dim[0] = _ih * stride[0];
      top_dim[1] = _iw * stride[1];
    } else {
      int ih_ext = calc_dilute_hw(_ih, _ins[1], 0, padding[0], padding[1]);
      int iw_ext = calc_dilute_hw(_iw, _ins[0], 0, padding[2], padding[3]);
      top_dim[0] = calc_output_hw(ih_ext, kh, stride[0]); // oh
      top_dim[1] = calc_output_hw(iw_ext, kw, stride[1]); // ow
    }

    auto input_type = input.getType().cast<RankedTensorType>();
    RankedTensorType output = RankedTensorType::get(
        {in, _oc, top_dim[0], top_dim[1]}, input_type.getElementType());

    return std::make_tuple(operands, attrs, output);
  };

  // recursive add
  top::DeconvOp deconv2d;
  top::ConvOp conv2d;
  int d_start = -1;
  if (scale[0].first != 0 || scale[1].first != 0) {
    d_start = loop;
    loop++; // put later for accuracy
  }

  for (int d = 0; d < loop; d++) {
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    RankedTensorType output;
    std::tie(operands, attrs, output) = createConv2D(input, d, d == d_start);
    if (operands.empty()) {
      return false;
    }

    if (is1x1Input) {
      // just deconv(upsample) it
      deconv2d =
          rewriter.create<top::DeconvOp>(loc, output, ArrayRef<Value>{operands},
                                         ArrayRef<NamedAttribute>{attrs});
      input = deconv2d.getResult();
    } else {
      conv2d =
          rewriter.create<top::ConvOp>(loc, output, ArrayRef<Value>{operands},
                                       ArrayRef<NamedAttribute>{attrs});
      input = conv2d.getResult();
    }

    // intpu as previous output
    auto input_type = input.getType().cast<RankedTensorType>();
    auto input_shape = input_type.getShape();
    _ih = input_shape[2]; // next input's shape
    _iw = input_shape[3];

    int hardwareHWMax = 4095 - 32;

    auto curr_output_shape = module::getShape(operands[0]);
    if (curr_output_shape[2] > hardwareHWMax ||
        curr_output_shape[3] > hardwareHWMax) {
      LLVM_DEBUG(llvm::errs()
                     << "hw over hardware limitation, leverage cpu, hw is:"
                     << curr_output_shape[2] << "/" << curr_output_shape[3]
                     << "\n";);
      return false;
    }
  }

  // interp's output SHOULD BE EQ with conv's
  int64_t conv_on, conv_oc, conv_oh, conv_ow;
  auto conv_output_shape = module::getShape(input);
  module::getNCHW(conv_output_shape, conv_on, conv_oc, conv_oh, conv_ow);
  assert((conv_on == in && conv_oc == oc && conv_oh == oh && conv_ow == ow) &&
         "Transformsed conv shape SHOULD be equal with interp");

  if (is1x1Input) {
    rewriter.replaceOp(op, deconv2d);
  } else {
    rewriter.replaceOp(op, conv2d);
  }
  return true;
}

template <typename T>
static void LoweringInterp(PatternRewriter &rewriter, top::InterpOp op,
                           bool asymmetric) {
  auto mode = tpu::symbolizeResizeMode(op.getMode());
  auto coord_mode = tpu::symbolizeResizeCoordMode(op.getCoordMode());
  assert(mode && coord_mode);
  std::string coordinate_transformation_mode;
  auto o_shape = module::getShape(op.getOutput());
  auto i_shape = module::getShape(op.getInput());
  assert(o_shape.size() >= 2);
  switch (coord_mode.value()) {
  case tpu::ResizeCoordMode::half_pixel:
    coordinate_transformation_mode = "half_pixel";
    break;
  case tpu::ResizeCoordMode::align_corners:
    coordinate_transformation_mode = "align_corners";
    break;
  case tpu::ResizeCoordMode::pytorch_half_pixel:
    coordinate_transformation_mode = "pytorch_half_pixel";
    break;
  case tpu::ResizeCoordMode::asymmetric:
    coordinate_transformation_mode = "asymmetric";
    break;
  default:
    llvm_unreachable("Unsupport interp coord type \n");
  }

  double scale_h = op.getScaleH().convertToDouble();
  double scale_w = op.getScaleW().convertToDouble();
  if (mode.value() == tpu::ResizeMode::linear) {
    if (o_shape[o_shape.size() - 1] > 1 && o_shape[o_shape.size() - 2] > 1 &&
        coordinate_transformation_mode == "pytorch_half_pixel") {
      coordinate_transformation_mode = "half_pixel";
    }
    // convert interp to conv/deconv ...
    if (coordinate_transformation_mode == "half_pixel") {
      if (std::ceil(scale_h) == std::floor(scale_h) &&
          std::ceil(scale_w) == std::floor(scale_w)) {
        // todo only support ins == 0
        if (scale_h != 1.0 && scale_w != 1.0) {
          return resize_to_conv1(rewriter, op, scale_h, scale_w);
        }
      }
      if (std::abs(scale_h - scale_w) < 1e-6 &&
          std::abs(scale_h - 0.5) < 1e-6) {
        return resize_to_conv2(rewriter, op, scale_h, scale_w);
      }
    }
  } else if (mode.value() == tpu::ResizeMode::nearest) {
    if (std::ceil(scale_h) == std::floor(scale_h) &&
        std::ceil(scale_w) == std::floor(scale_w)) {
      assert(0 && "it should be already converted in onnx_convert.\n");
    }
    if (coordinate_transformation_mode == "pytorch_half_pixel" ||
        coordinate_transformation_mode == "asymmetric") {
      // when pytorch use nearest method, coordinate_transformation_mode is
      // actually nearest.
      coordinate_transformation_mode = "nearest";
    } else {
      coordinate_transformation_mode = "nearest_half_pixel";
    }

  } else {
    llvm_unreachable("Unsupport interp mode type \n");
  }

  int64_t on, oc, oh, ow;
  int64_t in, ic, ih, iw;
  module::getNCHW(o_shape, on, oc, oh, ow);
  module::getNCHW(i_shape, in, ic, ih, iw);
  if (coordinate_transformation_mode == "align_corners") {
    if (resize_to_conv_deconv(rewriter, op, in, ic, ih, iw, on, oc, oh, ow)) {
      return;
    }
  }

  // lowering to cpu op
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> param;
  attrs.emplace_back(
      rewriter.getNamedAttr("cpu_op_name", rewriter.getStringAttr("interp")));
  param.emplace_back(rewriter.getNamedAttr(
      "width", rewriter.getI32IntegerAttr(o_shape[o_shape.size() - 1])));
  param.emplace_back(rewriter.getNamedAttr(
      "height", rewriter.getI32IntegerAttr(o_shape[o_shape.size() - 2])));
  param.emplace_back(
      rewriter.getNamedAttr("pad_beg", rewriter.getI32IntegerAttr(0)));
  param.emplace_back(
      rewriter.getNamedAttr("pad_end", rewriter.getI32IntegerAttr(0)));
  param.emplace_back(
      rewriter.getNamedAttr("shrink_factor", rewriter.getI32IntegerAttr(0)));
  param.emplace_back(
      rewriter.getNamedAttr("zoom_factor", rewriter.getI32IntegerAttr(0)));
  param.emplace_back(rewriter.getNamedAttr(
      "coordinate_transformation_mode",
      rewriter.getStringAttr(coordinate_transformation_mode)));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(param)));
  std::vector<Value> operands;
  operands.emplace_back(op.getInput());
  mlir::Type new_type;
  if (mode.value() == tpu::ResizeMode::nearest) {
    if constexpr (std::is_same_v<T, mlir::IntegerType>) {
      new_type = getQuantInt8Type(op.getOutput(), asymmetric);
    } else {
      new_type = getQuantBF16Type(op.getOutput());
    }
  } else {
    new_type = getQuantFloatType(op.getOutput());
  }
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, new_type, operands, attrs);
}

void InterpLowering::LoweringINT8(PatternRewriter &rewriter, top::InterpOp op,
                                  bool asymmetric) const {
  LoweringInterp<mlir::IntegerType>(rewriter, op, asymmetric);
}

void InterpLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::InterpOp op) const {
  LoweringInterp<mlir::BFloat16Type>(rewriter, op, false);
}
} // namespace cv18xx
} // namespace tpu_mlir
