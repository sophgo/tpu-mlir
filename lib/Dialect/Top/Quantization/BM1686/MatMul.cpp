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

Value top::MatMulOp::quantize_int8_bm1686() {
  // refer quantize_convlike_layer_int8
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  int64_t batch, M, K, N;
  bool with_bias, has_relu;
  parseParam(batch, M, K, N, with_bias, has_relu);
  assert(batch == 1); // only for fullyconnected now
  const int nInputs = op->getNumOperands();
  auto filterOp = cast<top::WeightOp>(right().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  auto th_input_min = Quant::getMin(input());
  auto th_output_min = Quant::getMin(output());
  auto th_input_max = Quant::getMax(input());
  auto th_output_max = Quant::getMax(output());
  float input_scale = (127 - (-128)) / (th_input_max - th_input_min);
  float output_scale = (127 - (-128)) / (th_output_max - th_output_min);
  int input_zeropoint = std::round(-th_input_min * input_scale) - 127;
  if (th_input_min >= 0) {
    input_zeropoint += 127;
  }
  double w_max = findMaxabs(filter_f32->data(), filter_f32->size());
  float w_min = -w_max;
  float scale_w = (127 - (-127)) / (w_max - w_min);
  int w_zeropoint = 0; //对称量化，固定为0
  auto filter_int8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  for (int t = 0; t < filter_f32->size(); t++) {
    filter_int8->data()[t] =
        (char)std::round((filter_f32->data()[t] - w_min) * scale_w - 127);
  }

  std::shared_ptr<std::vector<int32_t>> bias_int32;
  if (with_bias) {
    auto biasOp = cast<top::WeightOp>(bias().getDefiningOp());
    auto bias_f32 = biasOp.read<float>();
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_f32->size());

    for (int j = 0; j < N; j++) {
      int bias_w_xz = 0;
      // auto idtype =
      // input().getType().cast<RankedTensorType>().getElementType(); if
      // (idtype.isa<quant::UniformQuantizedType>() &&
      // idtype.cast<quant::UniformQuantizedType>().isSigned()) {
      for (int i = 0; i < M; i++) {
        bias_w_xz += ((int)filter_int8->data()[j * M + i] - w_zeropoint) *
                     input_zeropoint;
      }
      bias_int32->data()[j] =
          std::round(bias_f32->data()[j] * scale_w * input_scale);
      bias_int32->data()[j] -= bias_w_xz;
    }
  }

  int scale, shift;
  float scale_f = output_scale / (scale_w * input_scale);
  get_scale_and_shift(scale_f, scale, shift, 32);

  attrs.push_back(
      builder.getNamedAttr("rshift", builder.getI64IntegerAttr(shift)));
  attrs.push_back(
      builder.getNamedAttr("multiplier", builder.getI64IntegerAttr(scale)));
  auto filter_type = right().getType().cast<RankedTensorType>();
  auto new_type =
      RankedTensorType::get(filter_type.getShape(), builder.getI8Type());
  auto new_filter = WeightOp::create(op, "filter_int8", *filter_int8, new_type);
  operands.push_back(input());
  operands.push_back(new_filter);
  auto new_bias = bias();
  if (with_bias) {
    auto bias_type = bias().getType().cast<RankedTensorType>();
    auto new_type =
        RankedTensorType::get(bias_type.getShape(), builder.getIntegerType(32));
    new_bias = WeightOp::create(op, "bias_int32", *bias_int32, new_type);
  }
  operands.push_back(new_bias);
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newOp = builder.create<tpu::MatMulOp>(op->getLoc(), output().getType(),
                                             ArrayRef<Value>{operands},
                                             ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output(), true, !has_relu);
  return newOp.output();
}
