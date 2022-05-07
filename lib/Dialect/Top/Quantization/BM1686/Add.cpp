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
  std::vector<double> coeff_param_v(nInputs, 1.0);
  std::vector<int64_t> coeff_v(nInputs, 1);
  auto th_output_min = Quant::getMin(output());
  auto th_output_max = Quant::getMax(output());

  if (coeff().hasValue()) {
    int idx = 0;
    for (auto v : coeff().getValue()) {
      coeff_param_v[idx++] = v.cast<FloatAttr>().getValueAsDouble();
    }
  }

  int alg_type = 0; //0:normal, 1:only rightshift the output //设为1略微提升精度，但会导致backend_api_eltwise_fixed_global assert失败, 因为对某分支的multiplier左移会导致coeff_v大于128
  double bias = 0;
  int max_shifti = -32;
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    operands.push_back(input);
    auto th_input_min = Quant::getMin(input);
    auto th_input_max = Quant::getMax(input);
    double scale = (th_input_max - th_input_min) / (127 - (-128));
    int64_t zeropoint = std::round(-th_input_min / scale) - 128;
    int scalei, shifti;
    double alpha = (th_input_max - th_input_min) /
                            (th_output_max - th_output_min);
    bias += alpha*zeropoint;
    get_scale_and_shift(coeff_param_v[i] * alpha,
                        scalei, shifti, 8); //改为32，cos反而略降
    coeff_v[i] = scalei;
    rshift_v[i] = shifti;
    if (shifti > max_shifti)
      max_shifti = shifti;
    //printf("i:%d scalei:%d, shifti:%d\n", i, scalei, shifti);
  }

  if (alg_type) {
    for (int i = 0; i < nInputs; i++) {
      coeff_v[i] = coeff_v[i]<<(max_shifti - rshift_v[i]); //这里左移会导致coeff_v大于128
      //printf("i:%d, max_shifti:%d - rshift_v:%d\n", i, max_shifti, rshift_v[i]);
      rshift_v[i] = 0;
    }
    rshift_v[0] = max_shifti;
    bias = bias*std::pow(2, max_shifti);
  }

  //printf("quantize rectified_bias:%f\n", bias);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu", do_reluAttr()));
  attrs.push_back(
      builder.getNamedAttr("multipliers", builder.getI64ArrayAttr(coeff_v)));
  attrs.push_back(
      builder.getNamedAttr("rshifts", builder.getI64ArrayAttr(rshift_v)));
  attrs.push_back(
      builder.getNamedAttr("rectified_bias", builder.getF64FloatAttr(bias)));
  auto newOp = builder.create<tpu::AddOp>(op->getLoc(), output().getType(),
                                          ArrayRef<Value>{operands},
                                          ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output(), true);
  return newOp.output();
}
