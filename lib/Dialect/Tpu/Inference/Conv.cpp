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

  //llvm::errs() << "ConvOp setup:" << this->name() << "\n";
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

  int *p_rshift = nullptr;
  int *p_multiplier = nullptr;
  std::vector<int> rshift_v;
  std::vector<int> multiplier_v;
  bool per_channel = false;
  int size = multiplier().hasValue() ? multiplier().getValue().size() : 0;
  if (size > 1) {
    per_channel = true;
  }
  if (size > 0) {
    for (size_t i = 0; i < oc; i++) {
      rshift_v.push_back(rshift().getValue()[i].cast<IntegerAttr>().getInt());
      multiplier_v.push_back(
          multiplier().getValue()[i].cast<IntegerAttr>().getInt());
    }
    p_rshift = rshift_v.data();
    p_multiplier = multiplier_v.data();
  } else {
    rshift_v.push_back(rshift().getValue()[0].cast<IntegerAttr>().getInt());
    p_rshift = rshift_v.data();
  }

  int izp = 0;
  auto dtype = input().getType().cast<RankedTensorType>().getElementType();
  if (dtype.isa<quant::UniformQuantizedType>()) {
    izp = dtype.cast<quant::UniformQuantizedType>().getZeroPoint();
  }

  int ozp = 0;
  dtype = output().getType().cast<RankedTensorType>().getElementType();
  if (dtype.isa<quant::UniformQuantizedType>()) {
    ozp = dtype.cast<quant::UniformQuantizedType>().getZeroPoint();
  }

  auto module = Module::getModuleOp(getOperation());
  int chip = (Module::getChip(module) == Module::Chip::BM1686)?1:0;
  conv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], n, ic,
              ih, // fixme p.inputs[2] maybe null???
              iw, oc, oh, ow, kh, kw, sh, sw, dh, dw, pt, pb, pl, pr, g,
              do_relu(), izp, ozp, p_rshift, p_multiplier, idt, wdt, bdt, odt,
              per_channel, chip);
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
#ifdef DEBUG_TPU_INFER
  llvm::errs() << "ConvOp inference:" << this->name() << "\n";
  for (int i = 0; i < 5; i++) {
    printf("%d  %f x %d +%f = %f\n", i, p.inputs[0][i],
    (int8_t)p.inputs[1][i], p.inputs[2][i], p.outputs[0][i]);
  }
#endif
  return success();
}
