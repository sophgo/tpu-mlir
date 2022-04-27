#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Quant.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;

Value top::AddOp::quantize_int8_bm1686() {
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  std::vector<int64_t> rshift_v(nInputs);
  std::vector<int64_t> multiplier_v(nInputs);
  std::vector<double> coeff_v(nInputs, 1.0);
  auto th_output_min = Quant::getMin(output());
  auto th_output_max = Quant::getMax(output());

  if (coeff().hasValue()) {
    int idx = 0;
    for (auto v : coeff().getValue()) {
      coeff_v[idx++] = v.cast<FloatAttr>().getValueAsDouble();
    }
  }

  float bias = 0;
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    operands.push_back(input);
    auto th_input_min = Quant::getMin(input);
    auto th_input_max = Quant::getMax(input);
    double scale = (th_input_max - th_input_min) / (127 - (-128));
    int64_t zeropoint = std::round(-th_input_min / scale) - 128;
    int scalei, shifti;
    float alpha = (th_input_max - th_input_min) /
                            (th_output_max - th_output_min);
    bias += alpha*zeropoint;
    get_scale_and_shift(coeff_v[i] * alpha,
                        scalei, shifti, 8);
    coeff_v[i] = (double)scalei;
    rshift_v[i] = shifti; //
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu", do_reluAttr()));
  attrs.push_back(
      builder.getNamedAttr("coeff", builder.getF64ArrayAttr(coeff_v)));
  attrs.push_back(
      builder.getNamedAttr("rshifts", builder.getI64ArrayAttr(rshift_v)));
  attrs.push_back(
      builder.getNamedAttr("rectified_bias", builder.getSI32IntegerAttr(std::round(bias))));
  auto newOp = builder.create<tpu::AddOp>(op->getLoc(), output().getType(),
                                          ArrayRef<Value>{operands},
                                          ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output(), true);
  return newOp.output();
}
