#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Module.h"

convbwd_weight_attr_t tpu::ConvBwdWeightOp::parseParam() {
  convbwd_weight_attr_t p = {0};
  auto input_s = getInput().getType().cast<RankedTensorType>().getShape();
  auto gradout_s = getGradout().getType().cast<RankedTensorType>().getShape();
  p.has_bias = getGradBiasEnable();
  p.n = input_s[0];
  p.ic = input_s[1];
  p.ih = input_s.size() > 2 ? input_s[2] : 1;
  p.iw = input_s.size() > 3 ? input_s[3] : 1;
  p.oc = gradout_s[1];
  p.oh = gradout_s.size() > 2 ? gradout_s[2] : 1;
  p.ow = gradout_s.size() > 3 ? gradout_s[3] : 1;
  auto kernel = module::getI64Array(getKernelShape());
  p.kh = kernel->at(0);
  p.kw = kernel->at(1);
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
  p.groups = getGroups();
  return p;
}

LogicalResult tpu::ConvBwdWeightOp::init(InferenceParameter &p) {

  return success();
}

void tpu::ConvBwdWeightOp::deinit(InferenceParameter &p) {

}

LogicalResult tpu::ConvBwdWeightOp::inference(InferenceParameter &p) {

  return success();
}

uint32_t tpu::ConvBwdWeightOp::dyn_codegen_global_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}

int64_t tpu::ConvBwdWeightOp::dyn_codegen_global_bm1684x(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}

int64_t tpu::ConvBwdWeightOp::get_fw_type_bm1684() {
  return -1;
}

int64_t tpu::ConvBwdWeightOp::get_fw_type_bm1684x() {
  return -1;
}