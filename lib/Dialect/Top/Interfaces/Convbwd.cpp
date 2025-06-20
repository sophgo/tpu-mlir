#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"

convbwd_attr_t top::ConvbwdOp::parseParam() {
  convbwd_attr_t p = {0};
  auto input_s = getInput().getType().cast<RankedTensorType>().getShape();
  auto gradout_s = getGradOut().getType().cast<RankedTensorType>().getShape();
  p.n = input_s[0];
  p.ic = input_s[1];
  p.ih = input_s.size() > 2 ? input_s[2] : 1;
  p.iw = input_s.size() > 3 ? input_s[3] : 1;
  p.oc = gradout_s[1];
  p.oh = gradout_s.size() > 2 ? gradout_s[2] : 1;
  p.ow = gradout_s.size() > 3 ? gradout_s[3] : 1;
  auto kernel = module::getI64Array(getKernelShape());
  p.kh = kernel->at(2);
  p.kw = kernel->at(3);
  auto pads_v = module::getI64Array(getPadding());
  p.pht = pads_v->at(0);
  p.pwl = pads_v->at(1);
  p.phb = pads_v->at(2);
  p.pwr = pads_v->at(3);
  auto strides_v = module::getI64Array(getStride());
  p.sh = strides_v->at(0);
  p.sw = strides_v->at(1);
  auto dhdw = module::getI64Array(getDilations(), 2, 1);
  p.dh = dhdw->at(0);
  p.dw = dhdw->at(1);
  auto ins = module::getI64Array(getInserts());
  p.insh = ins->at(0);
  p.insw = ins->at(1);
  p.groups = getGroups();
  return p;
}

int64_t top::ConvbwdOp::getFLOPs() {
  auto attr = parseParam();
  int64_t flops_per_instance =
      2 * attr.oc * attr.ic * attr.kh * attr.kw * attr.oh * attr.ow;
  int64_t total_flops = flops_per_instance * attr.n;
  return total_flops;
}

LogicalResult top::ConvbwdOp::init(InferenceParameter &p) { return success(); }
void top::ConvbwdOp::deinit(InferenceParameter &p) {}

LogicalResult top::ConvbwdOp::inference(InferenceParameter &p) {
  auto attr = parseParam();
  float *gradout = p.inputs[0];
  float *input = p.inputs[1];
  float *kernel = p.inputs[2];
  float *gradinput = p.outputs[0];
  float *gradweight = p.outputs[1];
  float *gradbias = p.outputs[2];
  auto cal_grad_input = getGradInputEnable();
  auto cal_grad_weight = getGradWeightEnable();
  auto cal_grad_bias = getGradBiasEnable();

  ConvBwd conv_bwd;
  conv_bwd.setup(input, kernel, gradout, gradinput, gradweight, gradbias,
                 attr.n, attr.ic, attr.oc, attr.ih, attr.iw, attr.kh, attr.kw,
                 attr.oh, attr.ow, attr.sh, attr.sw, attr.pht, attr.pwl,
                 attr.phb, attr.pwr, cal_grad_input, cal_grad_weight,
                 cal_grad_bias, attr.dh, attr.dw);
  conv_bwd.run();

  return success();
}

void top::ConvbwdOp::shape_inference() {
  auto kernel_shape = module::getShape(getOperand(2));
  auto grad_input_shape_attr = getInputShape().cast<ArrayAttr>();
  std::vector<int64_t> grad_input_shape;
  for (auto attr : grad_input_shape_attr) {
    grad_input_shape.push_back(attr.cast<IntegerAttr>().getInt());
  }
  module::setShapeOrVerify(getResult(0), grad_input_shape);
  module::setShapeOrVerify(getResult(1), kernel_shape);
  module::setShapeOrVerify(getResult(2), {kernel_shape[0]});
}
