#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/fp16_bf16.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;

Value top::ConvOp::lowering_int8_bm1686() {
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  operands.push_back(input());
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  auto filterOp = cast<top::WeightOp>(filter().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  Quant::getScaleAndZeroPoint(input(), in_scale, in_zp);
  Quant::getScaleAndZeroPoint(output(), out_scale, out_zp);

  std::shared_ptr<std::vector<int32_t>> bias_int32;
  std::shared_ptr<std::vector<float>> bias_fp32;
  auto filter_int8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  if (with_bias) {
    auto biasOp = cast<top::WeightOp>(bias().getDefiningOp());
    bias_fp32 = biasOp.read<float>();
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
  } else if (in_zp != 0) {
    bias_int32 = std::make_shared<std::vector<int32_t>>(oc);
  }

  std::vector<int64_t> rshift_v;
  std::vector<int64_t> multiplier_v;
  int int32_multiplier, shift;
  int dim = filter_f32->size() / oc;
  for (int c = 0; c < oc; c++) { // per-channel量化
    float *p_filter = filter_f32->data() + c * dim;
    float w_max = findMaxabs(p_filter, dim);
    w_max = std::max(1e-5f, w_max);
    double scale_w = w_max / 127.0;
    double scale_f = scale_w * in_scale / out_scale;
    get_scale_and_shift(scale_f, int32_multiplier, shift, 32);
    multiplier_v.push_back(int32_multiplier);
    rshift_v.push_back(shift);

    for (int t = 0; t < dim; t++) {
      filter_int8->data()[c * dim + t] = Quant::to_int8(p_filter[t] / scale_w);
    }

    double bias_w_xz = 0;
    if (in_zp) {
      for (int t = 0; t < dim; t++) {
        bias_w_xz += filter_int8->data()[c * dim + t] * in_zp;
      }
    }

    if (with_bias) {
      bias_int32->data()[c] =
          std::round(bias_fp32->data()[c] / (scale_w * in_scale) - bias_w_xz);
    } else if (in_zp) {
      bias_int32->data()[c] = std::round(-bias_w_xz);
    }
  }
  with_bias = with_bias || 0 != in_zp;

  auto filter_type = filter().getType().cast<RankedTensorType>();
  auto new_type =
      RankedTensorType::get(filter_type.getShape(), builder.getI8Type());
  auto new_filter = WeightOp::create(op, "filter_int8", *filter_int8, new_type);
  operands.push_back(new_filter);
  if (with_bias) {
    auto bias_type = bias().getType().cast<RankedTensorType>();
    auto new_type =
        RankedTensorType::get(bias_type.getShape(), builder.getI32Type());
    auto new_bias = WeightOp::create(op, "bias_int32", *bias_int32, new_type);
    operands.push_back(new_bias);
  } else {
    auto none = Module::getNoneOp(op);
    operands.push_back(none);
  }

  std::vector<NamedAttribute> attrs;

  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(builder.getNamedAttr(
      "rshift", builder.getI64ArrayAttr(ArrayRef<int64_t>{rshift_v})));
  attrs.push_back(builder.getNamedAttr(
      "multiplier", builder.getI64ArrayAttr(ArrayRef<int64_t>{multiplier_v})));
  attrs.push_back(
      builder.getNamedAttr("with_bias", builder.getBoolAttr(with_bias)));
  auto newOp = builder.create<tpu::ConvOp>(op->getLoc(), output().getType(),
                                           ArrayRef<Value>{operands},
                                           ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output(), true);
  return newOp.output();
}

Value top::ConvOp::lowering_fp(llvm::StringRef mode) {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);

  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  operands.push_back(input());
  if (mode == Quant::Type::F32) {
    operands.push_back(filter());
  } else {
    auto filterOp = cast<top::WeightOp>(filter().getDefiningOp());
    auto filter_f32 = filterOp.read<float>();
    auto filter_ui16 =
        std::make_shared<std::vector<uint16_t>>(filter_f32->size());
    unsigned int *filter_uint32 = (unsigned int *)filter_f32->data();
    for (int i = 0; i < filter_f32->size(); i++) {
      if (mode == Quant::Type::F16) {
        filter_ui16->data()[i] =
            float_to_fp16_uint16_nvidia(filter_f32->data()[i]);
      } else {
        filter_ui16->data()[i] =
            (uint16_t)((filter_uint32[i] & 0xFFFF0000) >> 16);
      }
      /*if (i < 128) {
        printf("filter i:%d, %f, 0x%x, 0x%x\n", i, filter_f32->data()[i],
               filter_ui16->data()[i],
               float_to_fp16_uint16(filter_f32->data()[i]));
      }*/
    }
    auto filter_type = filter().getType().cast<RankedTensorType>();
    Type elementType = (mode == Quant::Type::F16) ? builder.getF16Type()
                                                  : builder.getBF16Type();
    llvm::StringRef suffix =
        (mode == Quant::Type::F16) ? "filter_f16" : "filter_bf16";
    auto new_type = RankedTensorType::get(filter_type.getShape(), elementType);
    auto new_filter = WeightOp::create(op, suffix, *filter_ui16, new_type);
    operands.push_back(new_filter);
  }

  if (mode == Quant::Type::F32) {
    operands.push_back(bias());
  } else {
    if (with_bias) {
      auto biasOp = cast<top::WeightOp>(bias().getDefiningOp());
      auto bias_fp32 = biasOp.read<float>();
      auto bias_ui16 =
          std::make_shared<std::vector<uint16_t>>(bias_fp32->size());
      unsigned int *bias_uint32 = (unsigned int *)bias_fp32->data();
      for (int i = 0; i < bias_fp32->size(); i++) {
        if (mode == Quant::Type::F16) {
          bias_ui16->data()[i] =
              float_to_fp16_uint16_nvidia(bias_fp32->data()[i]);
        } else {
          bias_ui16->data()[i] =
              (uint16_t)((bias_uint32[i] & 0xFFFF0000) >> 16);
        }
        /*if (i < 128) {
          printf("bias i:%d, %f, 0x%x, 0x%x\n", i, bias_fp32->data()[i],
                 bias_ui16->data()[i],
                 float_to_fp16_uint16(bias_fp32->data()[i]));
        }*/
      }

      auto bias_type = bias().getType().cast<RankedTensorType>();
      Type elementType = (mode == Quant::Type::F16) ? builder.getF16Type()
                                                    : builder.getBF16Type();
      llvm::StringRef suffix =
          (mode == Quant::Type::F16) ? "bias_f16" : "bias_bf16";
      auto new_type = RankedTensorType::get(bias_type.getShape(), elementType);
      auto new_bias = WeightOp::create(op, suffix, *bias_ui16, new_type);
      operands.push_back(new_bias);
    } else {
      auto none = Module::getNoneOp(op);
      operands.push_back(none);
    }
  }

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(
      builder.getNamedAttr("with_bias", builder.getBoolAttr(with_bias)));
  auto newOp = builder.create<tpu::ConvOp>(op->getLoc(), output().getType(),
                                           ArrayRef<Value>{operands},
                                           ArrayRef<NamedAttribute>{attrs});
  return newOp.output();
}
