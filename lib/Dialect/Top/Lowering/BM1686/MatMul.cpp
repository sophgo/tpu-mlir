#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/fp16_bf16.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;

Value top::MatMulOp::lowering_int8_bm1686() {
  // refer quantize_convlike_layer_int8
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  int64_t batch, M, K, N;
  bool relu, with_bias;
  parseParam(batch, M, K, N, with_bias, relu);
  assert(batch == 1); // only for fullyconnected now
  const int nInputs = op->getNumOperands();
  auto filterOp = cast<top::WeightOp>(right().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  int64_t in_zp, out_zp;
  double in_scale, out_scale;
  Quant::getScaleAndZeroPoint(input(), in_scale, in_zp);
  Quant::getScaleAndZeroPoint(output(), out_scale, out_zp);

  double w_max = findMaxabs(filter_f32->data(), filter_f32->size());
  double w_scale = w_max / 127.0;
  auto filter_int8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  for (int t = 0; t < filter_f32->size(); t++) {
    filter_int8->data()[t] = Quant::to_int8(filter_f32->data()[t] / w_scale);
  }

  std::shared_ptr<std::vector<int32_t>> bias_int32;
  std::shared_ptr<std::vector<float>> bias_fp32;
  if (with_bias) {
    auto biasOp = cast<top::WeightOp>(bias().getDefiningOp());
    bias_fp32 = biasOp.read<float>();
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
  } else if (in_zp) {
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
  }

  for (int j = 0; j < N; j++) {
    int64_t bias_w_xz = 0;
    for (int i = 0; i < K; i++) {
      bias_w_xz += (int64_t)filter_int8->data()[i * N + j] * in_zp;
    }

    if (with_bias) {
      bias_int32->data()[j] =
          std::round(bias_fp32->data()[j] / (w_scale * in_scale) - bias_w_xz);
    } else {
      bias_int32->data()[j] = -bias_w_xz;
    }
  }
  with_bias = with_bias || in_zp != 0;
  int scale, shift;
  float scale_f = in_scale * w_scale / out_scale;
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
    std::vector<int64_t> shape = {N};
    auto new_type =
        RankedTensorType::get(shape, builder.getI32Type());
    new_bias = WeightOp::create(op, "bias_int32", *bias_int32, new_type);
    operands.push_back(new_bias);
  } else {
    auto none = Module::getNoneOp(op);
    operands.push_back(none);
  }

  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newOp = builder.create<tpu::MatMulOp>(op->getLoc(), output().getType(),
                                             ArrayRef<Value>{operands},
                                             ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output(), true);
  return newOp.output();
}

Value top::MatMulOp::lowering_fp(llvm::StringRef mode) {
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  int64_t batch, M, K, N;
  bool relu, with_bias;
  parseParam(batch, M, K, N, with_bias, relu);
  operands.push_back(input());

  if (mode == Quant::Type::F32) {
    operands.push_back(right());
    if (with_bias) {
      operands.push_back(bias());
    } else {
      auto none = Module::getNoneOp(op);
      operands.push_back(none);
    }
  } else {
    auto filterOp = cast<top::WeightOp>(right().getDefiningOp());
    auto filter_f32 = filterOp.read<float>();
    auto filter_ui16 = std::make_shared<std::vector<uint16_t>>(filter_f32->size());
    unsigned int* filter_uint32 = (unsigned int*)filter_f32->data();
    for (int i = 0; i < filter_f32->size(); i++) {
      if (mode == Quant::Type::F16) {
        filter_ui16->data()[i] = float_to_fp16_uint16_nvidia(filter_f32->data()[i]);
      } else {
        filter_ui16->data()[i] = (uint16_t)((filter_uint32[i] & 0xFFFF0000)>>16);
      }
    }
    auto filter_type = right().getType().cast<RankedTensorType>();
    Type elementType = (mode == Quant::Type::F16) ? builder.getF16Type()
                                                  : builder.getBF16Type();
    llvm::StringRef suffix =
        (mode == Quant::Type::F16) ? "filter_f16" : "filter_bf16";
    auto new_type =
        RankedTensorType::get(filter_type.getShape(), elementType);
    auto new_filter = WeightOp::create(op, suffix, *filter_ui16, new_type);
    operands.push_back(new_filter);
    if (with_bias) {
      auto biasOp = cast<top::WeightOp>(bias().getDefiningOp());
      auto bias_fp32 = biasOp.read<float>();
      auto bias_ui16 = std::make_shared<std::vector<uint16_t>>(bias_fp32->size());
      unsigned int* bias_uint32 = (unsigned int*)bias_fp32->data();
      for (int i = 0; i < bias_fp32->size(); i++) {
        bias_ui16->data()[i] = (uint16_t)((bias_uint32[i] & 0xFFFF0000)>>16);
      }

      auto bias_type = bias().getType().cast<RankedTensorType>();
      Type elementType = (mode == Quant::Type::F16) ? builder.getF16Type()
                                                    : builder.getBF16Type();
      llvm::StringRef suffix =
          (mode == Quant::Type::F16) ? "bias_f16" : "bias_bf16";
      auto new_type =
          RankedTensorType::get(bias_type.getShape(), elementType);
      auto new_bias = WeightOp::create(op, suffix, *bias_ui16, new_type);
      operands.push_back(new_bias);
    } else {
      auto none = Module::getNoneOp(op);
      operands.push_back(none);
    }
  }

  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newOp = builder.create<tpu::MatMulOp>(op->getLoc(), output().getType(),
                                             ArrayRef<Value>{operands},
                                             ArrayRef<NamedAttribute>{attrs});
  return newOp.output();
}
