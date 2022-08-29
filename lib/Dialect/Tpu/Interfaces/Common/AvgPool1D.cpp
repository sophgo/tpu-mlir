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

void tpu::AvgPool1DOp::parseParam(void *param) {
  pool_attr_t *p = (pool_attr_t *)param;
  memset(p, 0, sizeof(pool_attr_t));
  p->id = 1;
  p->od = 1;
  p->kd = 1;
  p->kw = 1;
  p->sd = 1;
  p->sw = 1;
  auto ishape = input().getType().dyn_cast<RankedTensorType>().getShape();
  auto oshape = output().getType().dyn_cast<RankedTensorType>().getShape();
  Module::getNCHW(ishape, p->n, p->c, p->ih, p->iw);
  Module::getNCHW(oshape, p->n, p->c, p->oh, p->ow);
  assert(p->iw == 1 && p->ow == 1);

  auto kernel = Module::getI64Array(kernel_shape());
  p->kh = kernel->at(0);
  auto stride = Module::getI64Array(strides());
  p->sh = stride->at(0);
  auto pad = Module::getI64Array(pads());
  p->pad_h = pad->at(0);
  p->pad_h_after = pad->at(1);
  p->pad_value = pad_value();
  p->do_relu = do_relu();
  p->is_global = p->ih == p->kh && p->iw == p->kw && p->oh == 1 && p->ow == 1;
  p->count_include_pad = count_include_pad();
}

LogicalResult tpu::AvgPool1DOp::init(InferenceParameter &p) {
  auto pooling = new Pooling();
  pool_attr_t attrs;
  parseParam(&attrs);

  int izp = 0;
  auto dtype = input().getType().cast<RankedTensorType>().getElementType();
  if (dtype.isa<quant::UniformQuantizedType>()) {
    izp = dtype.cast<quant::UniformQuantizedType>().getZeroPoint();
  }

  pooling->setup(p.inputs[0], p.outputs[0], attrs, true, izp);
  p.handle = (void *)pooling;
  return success();
}

void tpu::AvgPool1DOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto pooling = (Pooling *)p.handle;
    delete pooling;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::AvgPool1DOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto pooling = (Pooling *)p.handle;
  pooling->run();
  auto out_type = Module::getStorageType(output());
  auto num_elem = Module::getNumElements(output());
  if (out_type.isInteger(8)) {
    auto i_qtype = Quant::getUniformQuantizedType(input());
    auto o_qtype = Quant::getUniformQuantizedType(output());
    auto module = Module::getModuleOp(getOperation());

    if (Module::getAsymmetric(module) == false) {
      auto multi = multiplier().getValue();
      auto rs = rshift().getValue();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; ++i) {
        p.outputs[0][i] = applyMultiplierAndRShift(
            std::round(p.outputs[0][i] * pooling->kh), multi, rs);
        p.outputs[0][i] = out_type.isUnsignedInteger(8)
                              ? Quant::to_uint8(p.outputs[0][i])
                              : Quant::to_int8(p.outputs[0][i]);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; ++i) {
        p.outputs[0][i] = p.outputs[0][i] * pooling->kh *
                              scale().getValue().convertToDouble() +
                          offset().getValue().convertToDouble();
        p.outputs[0][i] = out_type.isUnsignedInteger(8)
                              ? Quant::to_uint8(p.outputs[0][i])
                              : Quant::to_int8(p.outputs[0][i]);
      }
    }
  } else if (out_type.isa<FloatType>()) {
    if (do_relu()) {
      auto limit = relu_limit().convertToDouble();
      function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
    }
    if (out_type.isBF16()) {
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  }

  return success();
}

LogicalResult tpu::AvgPool1DOp::LocalGenSupport() {
  auto stride = Module::getI64Array(strides());
  if (stride->at(0) > 15) {
    return failure();
  }
  return success();
}

LogicalResult tpu::AvgPool1DOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                          int64_t out_idx, int64_t out_slice) {
  pool_attr_t attrs;
  parseParam(&attrs);
  in_slice = (out_slice - 1) * attrs.sh + attrs.kh;
  in_idx = out_idx * attrs.sh - attrs.pad_h;
  LocalGenInterface::fixSlice(in_idx, in_slice, attrs.ih);
  return success();
}
