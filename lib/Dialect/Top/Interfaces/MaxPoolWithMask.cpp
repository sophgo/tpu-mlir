//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "float.h"
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

pool_attr_t top::MaxPoolWithMaskOp::parseParam() {
  pool_attr_t p = {0};
  assert(kernel_shape().size() == 2); // only support 2d now
  auto ishape = input().getType().dyn_cast<RankedTensorType>().getShape();
  auto oshape = output().getType().dyn_cast<RankedTensorType>().getShape();
  auto kernel = Module::getI64Array(kernel_shape());
  auto stride = Module::getI64Array(strides());
  auto pad = Module::getI64Array(pads());
  p.id = 1;
  p.od = 1;
  p.kd = 1;
  p.sd = 1;
  Module::getNCHW(ishape, p.n, p.c, p.ih, p.iw);
  Module::getNCHW(oshape, p.n, p.c, p.oh, p.ow);
  p.kh = kernel->at(0);
  p.kw = kernel->at(1);
  p.sh = stride->at(0);
  p.sw = stride->at(1);
  p.pad_h = pad->at(0);
  p.pad_w = pad->at(1);
  p.pad_h_after = pad->at(2);
  p.pad_w_after = pad->at(3);
  p.pad_value = 0;
  p.do_relu = do_relu();
  p.relu_limit = relu_limit().convertToDouble();
  p.is_global = p.id == p.kd && p.ih == p.kh && p.iw == p.kw && p.od == 1 &&
                p.oh == 1 && p.ow == 1;
  p.count_include_pad = count_include_pad();
  return p;
}

int64_t top::MaxPoolWithMaskOp::getFLOPs() {
  auto attr = parseParam();
  auto extra = attr.do_relu ? 1 : 0;
  return Module::getNumElements(output()) *
         (attr.kd * attr.kh * attr.kw + extra);
}

LogicalResult top::MaxPoolWithMaskOp::init(InferenceParameter &p) {
  return success();
}

void top::MaxPoolWithMaskOp::deinit(InferenceParameter &p) { return; }

LogicalResult top::MaxPoolWithMaskOp::inference(InferenceParameter &p) {
  auto attr = parseParam();
  int64_t nc = attr.n * attr.c;
  auto num_elem = Module::getNumElements(output());
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
