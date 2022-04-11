#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Interfaces/InferenceInterface.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult tpu::ConvOp::init(InferenceParameter &p) {
  auto conv = new Conv();
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);

  auto idt = getDnnlType(input());
  auto wdt = getDnnlType(filter());
  auto bdt = memory::data_type::f32;
  auto odt = memory::data_type::f32;
  if (with_bias) {
    bdt = getDnnlType(bias());
  }
  if (Quant::isUniformQuantized(output())) {
    odt = memory::data_type::s32;
  }

  /*auto module = Module::getModuleOp(this);
  if (Module::getChip(module) == Module::Chip::BM1686) {
    newValue = quantize_op.quantize_int8_bm1686();
  }*/

  int *p_rshift = nullptr;
  int *p_multipler = nullptr;
  std::vector<int> rshift_v;
  std::vector<int> multipler_v;
  bool per_channel = false;
  int size = multipler().hasValue() ? multipler().getValue().size() : 0;
  if (size > 1) {
    per_channel = true;
  }
  if (size > 0) {
    for (size_t i = 0; i < oc; i++) {
      rshift_v.push_back(rshift().getValue()[i].cast<IntegerAttr>().getInt());
      multipler_v.push_back(
          multipler().getValue()[i].cast<IntegerAttr>().getInt());
    }
    p_rshift = rshift_v.data();
    p_multipler = multipler_v.data();
  } else {
    rshift_v.push_back(rshift().getValue()[0].cast<IntegerAttr>().getInt());
    p_rshift = rshift_v.data();
  }

  conv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], n, ic,
              ih, // fixme p.inputs[2] maybe null???
              iw, oc, oh, ow, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr, g,
              do_relu(), p_rshift, p_multipler, idt, wdt, bdt, odt,
              per_channel);
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
  // llvm::errs() << "ConvOp inference:" << this->name() << "\n";
  for (int i = 0; i < 5; i++) {
    // printf("%d  %f x %d +%f = %f\n", i, p.inputs[0][i],
    // (int8_t)p.inputs[1][i], p.inputs[2][i], p.outputs[0][i]);
  }
  return success();
}
