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
    for (auto v : coeff().getValue()) {
      coeff_v.push_back(v.cast<FloatAttr>().getValueAsDouble());
    }
  }

  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    operands.push_back(input);
    auto th_input_min = Quant::getMin(input);
    auto th_input_max = Quant::getMax(input);
    int scalei, shifti;
    get_scale_and_shift(coeff_v[i] * (th_input_max - th_input_min) /
                            (th_output_max - th_output_min),
                        scalei, shifti, 8);
    coeff_v[i] = (char)scalei;
    rshift_v[i] = (char)shifti;
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu", do_reluAttr()));
  attrs.push_back(
      builder.getNamedAttr("coeff", builder.getF64ArrayAttr(coeff_v)));
  attrs.push_back(
      builder.getNamedAttr("rshifts", builder.getI64ArrayAttr(rshift_v)));
  auto newOp = builder.create<tpu::AddOp>(op->getLoc(), output().getType(),
                                          ArrayRef<Value>{operands},
                                          ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output(), true, !do_relu());
  return newOp.output();
}
