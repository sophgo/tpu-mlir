#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Quant.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;

Value top::ConvOp::quantize_int8_bm1686() {
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
  auto th_input_min = Quant::getMin(input());
  auto th_output_min = Quant::getMin(output());
  auto th_input_max = Quant::getMax(input());
  auto th_output_max = Quant::getMax(output());
  if (th_input_min > 0 && (pt > 0 || pb > 0 || pl > 0 || pr > 0)) {
    th_input_min = 0;
  }
  if (th_input_max < 0 && (pt > 0 || pb > 0 || pl > 0 || pr > 0)) {
    th_input_max = 0;
  }
  float input_scale = (127 - (-128)) / (th_input_max - th_input_min);
  int input_zeropoint = std::round(-th_input_min * input_scale);

  std::shared_ptr<std::vector<int32_t>> bias_int32;
  std::shared_ptr<std::vector<float>> bias_fp32;
  auto filter_int8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  auto filter_uint8 = std::make_shared<std::vector<uint8_t>>(filter_f32->size());
  bool weightsAllPositive = !with_bias; //有bias，认为权重始终是signed的
  if (with_bias) {
    auto biasOp = cast<top::WeightOp>(bias().getDefiningOp());
    bias_fp32 = biasOp.read<float>();
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
  } else {
    if (input_zeropoint) {
      bias_int32 = std::make_shared<std::vector<int32_t>>(oc);
    }

    for (int i = 0; i < filter_f32->size(); i++) {
      if (filter_f32->data()[i] < 0.0) {
        weightsAllPositive = false;
        break;
      }
    }
  }

  std::vector<int64_t> rshift_v;
  std::vector<int64_t> multiplier_v;
  int int32_multiplier, shift;
  int dim = filter_f32->size() / oc;
  for (int c = 0; c < oc; c++) { // per-channel量化
    float w_max = findMaxabs(filter_f32->data() + c * dim, dim);
    w_max = w_max < 1e-5 ? 1e-5 : w_max;
    // w_max = getRefinedThreshold(filter_f32->data() + c * dim, dim);
    float w_min = -w_max; //对称量化
    if (weightsAllPositive)
      w_min = 0.0;
    float scale_w = (127 - (-127)) / (w_max - w_min);
    float scale_f =(w_max - w_min)*(th_input_max - th_input_min) / ((th_output_max - th_output_min)*254);
    get_scale_and_shift(scale_f, int32_multiplier, shift, 32);
    multiplier_v.push_back(int32_multiplier);
    rshift_v.push_back(shift);
    int w_zeropoint = 0; //权重用int8对称量化或用uint8表示，零点始终为0

    for (int t = 0; t < dim; t++) {
      int quant_weight =
          std::round(filter_f32->data()[c * dim + t] * scale_w)+w_zeropoint;
      if (weightsAllPositive) {
        quant_weight = quant_weight > 254 ? 254 : quant_weight;
        filter_uint8->data()[c * dim + t] = (unsigned char)quant_weight;
      } else {
        quant_weight = quant_weight > 127    ? 127
                       : quant_weight < -127 ? -127
                                             : quant_weight;
        filter_int8->data()[c * dim + t] = (char)quant_weight;
      }
    }

    long long int bias_w_xz = 0;
    if (input_zeropoint) {
      for (int t = 0; t < dim; t++) {
        if (weightsAllPositive) {
          bias_w_xz += (long long int )filter_uint8->data()[c * dim + t] * input_zeropoint;
        } else {
          bias_w_xz += (long long int )filter_int8->data()[c * dim + t] * input_zeropoint;
        }
      }
    }

    if (with_bias) {
      bias_int32->data()[c] =
          std::round(bias_fp32->data()[c]*scale_w*input_scale) - bias_w_xz;
    } else if (input_zeropoint) {
      bias_int32->data()[c] = -bias_w_xz;
    }
  }

  auto filter_type = filter().getType().cast<RankedTensorType>();
  auto new_type = RankedTensorType::get(
      filter_type.getShape(), builder.getIntegerType(8, !weightsAllPositive));
  if (weightsAllPositive) {
    auto new_filter =
        WeightOp::create(op, "filter_uint8", *filter_uint8, new_type);
    operands.push_back(new_filter);
  } else {
    auto new_filter =
        WeightOp::create(op, "filter_int8", *filter_int8, new_type);
    operands.push_back(new_filter);
  }
  Value new_bias = bias();
  if (with_bias || input_zeropoint) {
    auto bias_type = bias().getType().cast<RankedTensorType>();
    auto new_type =
        RankedTensorType::get(bias_type.getShape(), builder.getIntegerType(32));
    new_bias = WeightOp::create(op, "bias_int32", *bias_int32, new_type);
  }

  std::vector<NamedAttribute> attrs;
  operands.push_back(new_bias);
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(builder.getNamedAttr(
      "rshift", builder.getI64ArrayAttr(ArrayRef<int64_t>{rshift_v})));
  attrs.push_back(builder.getNamedAttr(
      "multiplier", builder.getI64ArrayAttr(ArrayRef<int64_t>{multiplier_v})));
  if (input_zeropoint)
    with_bias = true;
  attrs.push_back(
      builder.getNamedAttr("with_bias", builder.getBoolAttr(with_bias)));
  auto newOp = builder.create<tpu::ConvOp>(op->getLoc(), output().getType(),
                                           ArrayRef<Value>{operands},
                                           ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output(), true, false);
  return newOp.output();
}

