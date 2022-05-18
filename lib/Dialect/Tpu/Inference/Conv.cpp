#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/Helper/Quant.h"
using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult tpu::ConvOp::init(InferenceParameter &p) {
  auto conv = new Conv();
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  int izp = 0;
  if (Quant::isUniformQuantized(input())) {
    izp = Quant::getUniformQuantizedType(input()).getZeroPoint();
  }
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  conv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], n, ic, ih,
              iw, oc, oh, ow, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr, g,
              do_relu(), izp);
  p.handle = (void *)conv;
  return success();
}

void tpu::ConvOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto conv = (Conv *)p.handle;
    delete conv;
    p.handle = nullptr;
  }
}

LogicalResult tpu::ConvOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto conv = (Conv *)p.handle;
  conv->run();
  // requant
  if (Quant::isUniformQuantized(output())) {
    int64_t n, c, h, w;
    Module::getNCHW(output(), n, c, h, w);
    auto o_qtype = Quant::getUniformQuantizedType(output());
    auto rshift_v = Module::getI64Array(rshift().getValue());
    bool peraxis = rshift_v->size() == c;
    std::shared_ptr<std::vector<int64_t>> multiplier_v;
    if (multiplier().hasValue()) {
      multiplier_v = Module::getI64Array(multiplier().getValue());
    } else {
      multiplier_v = std::make_shared<std::vector<int64_t>>(rshift_v->size(), 1);
    }
  #pragma omp parallel for schedule(static, omp_schedule(c))
    for (int ic = 0; ic < c; ic++) {
      int64_t shift = peraxis ? rshift_v->at(ic) : rshift_v->at(0);
      int64_t multi = peraxis ? multiplier_v->at(ic) : multiplier_v->at(0);
      for (int in = 0; in < n; in++) {
        for (int hw = 0; hw < h * w; hw++) {
          int offset = (in * c + ic) * h * w + hw;
          auto v = (((int64_t)(p.outputs[0][offset] * multi)) >> shift) +
                  o_qtype.getZeroPoint();
          p.outputs[0][offset] = Quant::to_int8(v);
        }
      }
    }
  }
  return success();
}
