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
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

void tpu::DeconvOp::parseParam(void *param) {
  deconv_attr_t *p = (deconv_attr_t *)param;
  memset(p, 0, sizeof(deconv_attr_t));
  p->id = 1;
  p->od = 1;
  p->kd = 1;
  p->sd = 1;
  p->dd = 1;
  auto ishape = input().getType().dyn_cast<RankedTensorType>().getShape();
  auto oshape = output().getType().dyn_cast<RankedTensorType>().getShape();
  Module::getNCHW(ishape, p->n, p->ic, p->ih, p->iw);
  Module::getNCHW(oshape, p->n, p->oc, p->oh, p->ow);

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
  auto dilation = Module::getI64Array(dilations(), 2, 1);
  p->dh = dilation->at(0);
  p->dw = dilation->at(1);
  auto ins = Module::getI64Array(inserts(), 2, 0);
  p->ins_h = ins->at(0);
  p->ins_w = ins->at(1);
  p->g = group();
  p->do_relu = do_relu();
  p->relu_limit = relu_limit().convertToDouble();
  p->with_bias = with_bias();
  p->is_dw = (p->oc == p->ic && p->oc == p->g && p->g > 1);
  return;
}

LogicalResult tpu::DeconvOp::init(InferenceParameter &p) {
  auto deconv = new Deconv();
  deconv_attr_t attrs;
  parseParam(&attrs);
  int izp = 0;
  if (Quant::isUniformQuantized(input())) {
    izp = Quant::getUniformQuantizedType(input()).getZeroPoint();
  }
  deconv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], attrs,
                izp);
  p.handle = (void *)deconv;
  return success();
}

void tpu::DeconvOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto deconv = (Deconv *)p.handle;
    delete deconv;
    p.handle = nullptr;
  }
}

LogicalResult tpu::DeconvOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto deconv = (Deconv *)p.handle;
  deconv->run();
  auto chip = Module::getChip(getOperation());
  bool is_cv18xx = Module::isCV18xx(chip);
  // requant
  auto out_type = Module::getStorageType(output());
  auto num_elem = Module::getNumElements(output());
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem, is_cv18xx);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (is_cv18xx && Quant::isUniformQuantized(output())) {
    // apply multiplier && rshift inplace
    int64_t n, c, h, w;
    Module::getNCHW(output(), n, c, h, w);
    auto rshift_v = Module::getI64Array(rshift().value());
    auto multiplier_v = Module::getI64Array(multiplier().value());
    assert(rshift_v->size() == c && "CV18xx must be per_axis.");
#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int oc = 0; oc < c; oc++) {
      int64_t shift = rshift_v->at(oc);
      int64_t multi = multiplier_v->at(oc);
      for (int on = 0; on < n; on++) {
        for (int hw = 0; hw < h * w; hw++) {
          int offset = (on * c + oc) * h * w + hw;
          int64_t v = 0;
          v = applyMultiplierAndRShift(p.outputs[0][offset], multi, shift,
                                       CVI_QDM_QUANT);
          p.outputs[0][offset] = Quant::to_int8(v);
        }
      }
    }
  }

  return success();
}

LogicalResult tpu::DeconvOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  deconv_attr_t attrs;
  parseParam(&attrs);
  int kh_ext = (attrs.kh - 1) * attrs.dh + 1;
  int pad_h = kh_ext - attrs.pad_h - 1;
  in_idx = out_idx - attrs.pad_h;
  in_idx = in_idx <= 0 ? in_idx : std::ceil((float)in_idx / attrs.sh);
  if (in_idx <= 0) {
    pad_h = -in_idx;
    in_slice =
        std::ceil((out_slice - pad_h + kh_ext - 1) / (float)(attrs.sh)) + pad_h;
  } else {
    pad_h = in_idx * attrs.sh + pad_h - out_idx;
    in_slice = std::ceil((out_slice - pad_h + kh_ext - 1) / (float)attrs.sh);
  }
  bool is_last = (out_idx + out_slice == attrs.oh);
  LocalGenInterface::fixSlice(in_idx, in_slice, attrs.ih, is_last);
  if (in_slice == 0) {
    return failure();
  }
  return success();
}

mlir::Type tpu::DeconvOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return type_verify_case_i32(getOperation(), opd_idx, mode);
}
