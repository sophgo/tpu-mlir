#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

void top::ConvOp::parseParam(int64_t &n, int64_t &ic, int64_t &ih, int64_t &iw,
                             int64_t &oc, int64_t &oh, int64_t &ow, int64_t &g,
                             int64_t &kh, int64_t &kw, int64_t &ins_h,
                             int64_t &ins_w, int64_t &sh, int64_t &sw,
                             int64_t &pt, int64_t &pb, int64_t &pl, int64_t &pr,
                             int64_t &dh, int64_t &dw, bool &is_dw,
                             bool &with_bias, bool &do_relu) {
  auto i_s = input().getType().cast<RankedTensorType>().getShape();
  auto k_s = filter().getType().cast<RankedTensorType>().getShape();
  auto o_s = output().getType().cast<RankedTensorType>().getShape();
  do_relu = this->do_relu();
  with_bias = !bias().getType().isa<NoneType>();
  n = i_s[0];
  ic = i_s[1];
  ih = i_s[2];
  iw = i_s[3];
  oc = o_s[1];
  oh = o_s[2];
  ow = o_s[3];
  kh = kernel_shape().getValue()[0].cast<IntegerAttr>().getInt();
  kw = kernel_shape().getValue()[1].cast<IntegerAttr>().getInt();
  pt = pads().getValue()[0].cast<IntegerAttr>().getInt();
  pl = pads().getValue()[1].cast<IntegerAttr>().getInt();
  pb = pads().getValue()[2].cast<IntegerAttr>().getInt();
  pr = pads().getValue()[3].cast<IntegerAttr>().getInt();
  sh = strides().getValue()[0].cast<IntegerAttr>().getInt();
  sw = strides().getValue()[1].cast<IntegerAttr>().getInt();
  dh = dilations().getValue()[0].cast<IntegerAttr>().getInt();
  dw = dilations().getValue()[1].cast<IntegerAttr>().getInt();
  g = group();
  is_dw = (oc == ic && oc == g);
  return;
}

int64_t top::ConvOp::getFLOPs() {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, has_relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, has_relu);
  auto extra = with_bias ? 1 : 0 + has_relu ? 1 : 0;
  return Module::getNumElements(output()) * (kw * kw * ic / g * 2 + extra);
}

LogicalResult top::ConvOp::init(InferenceParameter &p) {
  auto conv = new Conv();
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  conv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], n, ic, ih,
              iw, oc, oh, ow, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr, g,
              do_relu());
  p.handle = (void *)conv;
  return success();
}

void top::ConvOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto conv = (Conv *)p.handle;
    delete conv;
    p.handle = nullptr;
  }
}

LogicalResult top::ConvOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto conv = (Conv *)p.handle;
  conv->run();
  return success();
}
