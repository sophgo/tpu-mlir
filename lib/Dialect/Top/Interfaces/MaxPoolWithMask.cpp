//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "float.h"
#include "tpu_mlir/Support/MathUtils.h"

pool_attr_t top::MaxPoolWithMaskOp::parseParam() {
  pool_attr_t p = {0};
  ASSERT_THIS(getKernelShape().size() == 2); // only support 2d now
  auto ishape = getInput().getType().dyn_cast<RankedTensorType>().getShape();
  auto oshape = getOutput().getType().dyn_cast<RankedTensorType>().getShape();
  auto kernel = module::getI64Array(getKernelShape());
  auto stride = module::getI64Array(getStrides());
  auto pad = module::getI64Array(getPads());
  p.id = 1;
  p.od = 1;
  p.kd = 1;
  p.sd = 1;
  module::getNCHW(ishape, p.n, p.c, p.ih, p.iw);
  module::getNCHW(oshape, p.n, p.c, p.oh, p.ow);
  p.kh = kernel->at(0);
  p.kw = kernel->at(1);
  p.sh = stride->at(0);
  p.sw = stride->at(1);
  p.pad_h = pad->at(0);
  p.pad_w = pad->at(1);
  p.pad_h_after = pad->at(2);
  p.pad_w_after = pad->at(3);
  p.pad_value = 0;
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.is_global = p.id == p.kd && p.ih == p.kh && p.iw == p.kw && p.od == 1 &&
                p.oh == 1 && p.ow == 1;
  p.count_include_pad = getCountIncludePad();
  return p;
}

int64_t top::MaxPoolWithMaskOp::getFLOPs() {
  auto attr = parseParam();
  auto extra = attr.do_relu ? 1 : 0;
  return module::getNumElements(getOutput()) *
         (attr.kd * attr.kh * attr.kw + extra);
}

LogicalResult top::MaxPoolWithMaskOp::init(InferenceParameter &p) {
  return success();
}

void top::MaxPoolWithMaskOp::deinit(InferenceParameter &p) { return; }

LogicalResult top::MaxPoolWithMaskOp::inference(InferenceParameter &p) {
  auto attr = parseParam();
  int64_t nc = attr.n * attr.c;
  auto num_elem = module::getNumElements(getOutput());
  std::fill_n(p.outputs[0], num_elem, (float)(-FLT_MAX));
#pragma omp parallel for schedule(static, omp_schedule(nc))
  for (int64_t idx = 0; idx < nc; ++idx) {
    auto bottom_data = p.inputs[0] + idx * attr.ih * attr.iw;
    auto top_data = p.outputs[0] + idx * attr.oh * attr.ow;
    auto top_mask = p.outputs[1] + idx * attr.oh * attr.ow;
    for (int64_t ph = 0; ph < attr.oh; ++ph) {
      for (int64_t pw = 0; pw < attr.ow; ++pw) {
        auto hstart = ph * attr.sh - attr.pad_h;
        auto wstart = pw * attr.sw - attr.pad_w;
        auto hend = std::min(hstart + attr.kh, attr.ih);
        auto wend = std::min(wstart + attr.kw, attr.iw);
        if (hstart < 0) {
          hstart = 0;
        }
        if (wstart < 0) {
          wstart = 0;
        }
        auto pool_index = ph * attr.ow + pw;
        for (int64_t h = hstart; h < hend; ++h) {
          for (int64_t w = wstart; w < wend; ++w) {
            auto index = h * attr.iw + w;
            if (bottom_data[index] > top_data[pool_index]) {
              top_data[pool_index] = bottom_data[index];
              top_mask[pool_index] = index;
            }
          }
        }
      }
    }
  }

  return success();
}

void top::MaxPoolWithMaskOp::shape_inference() {
  int64_t out_h, out_w;
  auto input_shape = module::getShape(getInput());
  auto kernel = module::getI64Array(getKernelShape());
  auto stride = module::getI64Array(getStrides());
  auto pad = module::getI64Array(getPads());

  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(input_shape[0]);
  out_shape.push_back(input_shape[1]);
  out_h =
      (ceil(input_shape[2] + 2 * pad->at(0) - kernel->at(0)) / stride->at(0)) +
      1;
  out_w =
      (ceil(input_shape[3] + 2 * pad->at(1) - kernel->at(1)) / stride->at(1)) +
      1;
  out_shape.push_back(out_h);
  out_shape.push_back(out_w);

  module::setShapeOrVerify(getOutput(), out_shape);
  module::setShapeOrVerify(getMask(), out_shape);
}
