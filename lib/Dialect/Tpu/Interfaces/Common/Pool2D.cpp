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

void tpu::Pool2DOp::parseParam(void *param) {
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
  p->pad_value = pad_value();
  p->do_relu = do_relu();
  p->relu_limit = relu_limit().convertToDouble();
  p->is_global = p->ih == p->kh && p->iw == p->kw && p->oh == 1 && p->ow == 1;
  p->count_include_pad = count_include_pad();
}

LogicalResult tpu::Pool2DOp::init(InferenceParameter &p) {
  auto pooling = new Pooling();
  pool_attr_t attrs;
  parseParam(&attrs);

  int izp = 0;
  auto dtype = input().getType().cast<RankedTensorType>().getElementType();
  bool is_avg_pooling = pool_mode() == tpu::PoolMode::Avg;
  if (dtype.isa<quant::UniformQuantizedType>() && is_avg_pooling) {
    izp = dtype.cast<quant::UniformQuantizedType>().getZeroPoint();
  }
  pooling->setup(p.inputs[0], p.outputs[0], attrs, is_avg_pooling, izp);
  p.handle = (void *)pooling;
  return success();
}

void tpu::Pool2DOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto pooling = (Pooling *)p.handle;
    delete pooling;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::Pool2DOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto pooling = (Pooling *)p.handle;
  pooling->run();
  if (pool_mode() == tpu::PoolMode::Max) {
    if (do_relu()) {
      auto limit = relu_limit().convertToDouble();
      function_relu(p.outputs[0], p.outputs[0],
                    Module::getNumElements(output()), limit,
                    Module::getStorageType(output()));
    }
    return success();
  }
  // average pooling
  auto chip = Module::getChip(getOperation());
  bool is_cv18xx = Module::isCV18xx(chip);
  auto m_type = is_cv18xx ? CVI_QUANT : BM_QUANT;
  auto out_type = Module::getStorageType(output());
  auto num_elem = Module::getNumElements(output());
  if (out_type.isInteger(8)) {
    auto i_qtype = Quant::getUniformQuantizedType(input());
    auto o_qtype = Quant::getUniformQuantizedType(output());
    auto module = Module::getModuleOp(getOperation());

    if (Module::getAsymmetric(module) == false) {
      auto multi = multiplier().value();
      auto rs = rshift().value();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; ++i) {
        int64_t v = 0;
        if (is_cv18xx) {
          v = Quant::to_int(p.outputs[0][i] * pooling->kh * pooling->kw,
                            ROUNDING_HALF_UP);
        } else {
          v = std::round(p.outputs[0][i] * pooling->kh * pooling->kw);
        }
        p.outputs[0][i] = applyMultiplierAndRShift(v, multi, rs, m_type);
        p.outputs[0][i] = out_type.isUnsignedInteger(8)
                              ? Quant::to_uint8(p.outputs[0][i])
                              : Quant::to_int8(p.outputs[0][i]);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; ++i) {
        p.outputs[0][i] = p.outputs[0][i] * pooling->kh * pooling->kw *
                              scale().value().convertToDouble() +
                          offset().value().convertToDouble();
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
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem, is_cv18xx);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  }

  return success();
}

LogicalResult tpu::Pool2DOp::LocalGenSupport() {
  pool_attr_t attrs;
  parseParam(&attrs);
  auto stride = Module::getI64Array(strides());
  if ((stride->at(0) > 15 || stride->at(1) > 15)) {
    return failure();
  }
  if (attrs.is_global) {
    // TODO: bug, need to be fixed
    auto in_stype = Module::getStorageType(input());
    if (in_stype.isF16() || in_stype.isBF16()) {
      return failure();
    }
  }
  return success();
}

LogicalResult tpu::Pool2DOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  pool_attr_t attrs;
  parseParam(&attrs);
  if (attrs.is_global) {
    if (out_idx != 0 || out_slice != attrs.oh) {
      return failure();
    }
    in_idx = 0;
    in_slice = attrs.ih;
    return success();
  }
  in_slice = (out_slice - 1) * attrs.sh + attrs.kh;
  in_idx = out_idx * attrs.sh - attrs.pad_h;
  bool is_last = (out_idx + out_slice == attrs.oh);
  LocalGenInterface::fixSlice(in_idx, in_slice, attrs.ih, is_last);
  return success();
}
