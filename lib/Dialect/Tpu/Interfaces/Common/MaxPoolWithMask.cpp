//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Float16.h"
#include "float.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

void tpu::MaxPoolWithMaskOp::parseParam(void *param) {
  pool_attr_t *p = (pool_attr_t *)param;
  memset(p, 0, sizeof(pool_attr_t));
  p->id = 1;
  p->od = 1;
  p->kd = 1;
  p->sd = 1;
  auto ishape = input().getType().dyn_cast<RankedTensorType>().getShape();
  auto oshape = output().getType().dyn_cast<RankedTensorType>().getShape();
  Module::getNCHW(ishape, p->n, p->c, p->ih, p->iw);
  Module::getNCHW(oshape, p->n, p->c, p->oh, p->ow);

  auto kernel = Module::getI64Array(kernel_shape());
  p->kh = kernel->at(0);
  p->kw = kernel->at(1);
  auto stride = Module::getI64Array(strides());
  p->sh = stride->at(0);
  p->sw = stride->at(1);
  auto pad = Module::getI64Array(pads());
  p->pad_h = pad->at(0);
  p->pad_w = pad->at(1);
  p->pad_h_after = pad->at(2);
  p->pad_w_after = pad->at(3);
  p->pad_value = 0;
  p->do_relu = do_relu();
  p->relu_limit = relu_limit().convertToDouble();
  p->is_global = p->ih == p->kh && p->iw == p->kw && p->oh == 1 && p->ow == 1;
  p->count_include_pad = 0;
}

LogicalResult tpu::MaxPoolWithMaskOp::init(InferenceParameter &p) {
  return success();
}

void tpu::MaxPoolWithMaskOp::deinit(InferenceParameter &p) { return; }

LogicalResult tpu::MaxPoolWithMaskOp::inference(InferenceParameter &p) {
  pool_attr_t attrs;
  parseParam(&attrs);
  int64_t nc = attrs.n * attrs.c;
  auto num_elem = Module::getNumElements(output());
  std::fill_n(p.outputs[0], num_elem, (float)(-FLT_MAX));
#pragma omp parallel for schedule(static, omp_schedule(nc))
  for (int64_t idx = 0; idx < nc; ++idx) {
    auto bottom_data = p.inputs[0] + idx * attrs.ih * attrs.iw;
    auto top_data = p.outputs[0] + idx * attrs.oh * attrs.ow;
    auto top_mask = p.outputs[1] + idx * attrs.oh * attrs.ow;
    for (int64_t ph = 0; ph < attrs.oh; ++ph) {
      for (int64_t pw = 0; pw < attrs.ow; ++pw) {
        auto hstart = ph * attrs.sh - attrs.pad_h;
        auto wstart = pw * attrs.sw - attrs.pad_w;
        auto hend = std::min(hstart + attrs.kh, attrs.ih);
        auto wend = std::min(wstart + attrs.kw, attrs.iw);
        if (hstart < 0) {
          hstart = 0;
        }
        if (wstart < 0) {
          wstart = 0;
        }
        auto pool_index = ph * attrs.ow + pw;
        for (int64_t h = hstart; h < hend; ++h) {
          for (int64_t w = wstart; w < wend; ++w) {
            auto index = h * attrs.iw + w;
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

LogicalResult tpu::MaxPoolWithMaskOp::LocalGenSupport() {
  // auto stride = Module::getI64Array(strides());
  // if ((stride->at(0) > 15 || stride->at(1) > 15)) {
  //   return failure();
  // }
  // return success();
  return failure();
}

LogicalResult tpu::MaxPoolWithMaskOp::BackwardH(int64_t &in_idx,
                                                int64_t &in_slice,
                                                int64_t out_idx,
                                                int64_t out_slice) {
  pool_attr_t attrs;
  parseParam(&attrs);
  in_slice = (out_slice - 1) * attrs.sh + attrs.kh;
  in_idx = out_idx * attrs.sh - attrs.pad_h;
  bool is_last = (out_idx + out_slice == attrs.oh);
  LocalGenInterface::fixSlice(in_idx, in_slice, attrs.ih, is_last);
  return success();
}
