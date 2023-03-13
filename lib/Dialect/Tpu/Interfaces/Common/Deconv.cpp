//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"

deconv_attr_t tpu::DeconvOp::parseParam() {
  deconv_attr_t p = {0};
  p.id = 1;
  p.od = 1;
  p.kd = 1;
  p.sd = 1;
  p.dd = 1;
  auto ishape = getInput().getType().dyn_cast<RankedTensorType>().getShape();
  auto oshape = getOutput().getType().dyn_cast<RankedTensorType>().getShape();
  module::getNCHW(ishape, p.n, p.ic, p.ih, p.iw);
  module::getNCHW(oshape, p.n, p.oc, p.oh, p.ow);

  auto kernel = module::getI64Array(getKernelShape());
  p.kh = kernel->at(0);
  p.kw = kernel->at(1);
  auto stride = module::getI64Array(getStrides());
  p.sh = stride->at(0);
  p.sw = stride->at(1);
  auto pad = module::getI64Array(getPads());
  p.pad_h = pad->at(0);
  p.pad_w = pad->at(1);
  p.pad_h_after = pad->at(2);
  p.pad_w_after = pad->at(3);
  auto dilation = module::getI64Array(getDilations(), 2, 1);
  p.dh = dilation->at(0);
  p.dw = dilation->at(1);
  p.g = getGroup();
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.with_bias = getWithBias();
  p.is_dw = (p.oc == p.ic && p.oc == p.g && p.g > 1);
  return p;
}

LogicalResult tpu::DeconvOp::init(InferenceParameter &p) {
  auto deconv = new Deconv();
  auto attr = parseParam();
  int izp = 0;
  if (module::isUniformQuantized(getInput())) {
    izp = module::getUniformQuantizedType(getInput()).getZeroPoint();
  }
  deconv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], attr, izp);
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
  // requant
  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (module::isUniformQuantized(getOutput())) {
    auto qmode = getQuantMode();
    // apply multiplier && rshift inplace
    int64_t n, c, h, w;
    module::getNCHW(getOutput(), n, c, h, w);
    auto rshift_v = module::getI64Array(getRshift().value());
    auto multiplier_v = module::getI64Array(getMultiplier().value());
    assert(rshift_v->size() == c);
#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int oc = 0; oc < c; oc++) {
      int64_t shift = rshift_v->at(oc);
      int64_t multi = multiplier_v->at(oc);
      for (int on = 0; on < n; on++) {
        for (int hw = 0; hw < h * w; hw++) {
          int offset = (on * c + oc) * h * w + hw;
          int64_t v = 0;
          v = applyMultiplierAndRShift(p.outputs[0][offset], multi, shift,
                                       qmode);
          p.outputs[0][offset] = to_int8(v);
        }
      }
    }
  }

  return success();
}

Optional<SmallVector<float, 4>>
tpu_mlir::DeconvSlice(int64_t out_idx, int64_t out_slice, int64_t stride,
                      int64_t filter, int64_t ih, int64_t pad) {
  // pad top with (kh_ext - pad_h - 1), ins with (stride - 1)
  // all cal here assume input is expanded(pad && ins)
  float max_real_in_idx = ih - 1;
  int pad_th = filter - pad - 1;
  int in_idx = out_idx;
  float real_in_idx = (in_idx - pad_th <= 0)
                          ? 0
                          : std::ceil((float)(in_idx - pad_th) / stride);

  int in_end_idx = out_idx + out_slice + filter - 2;
  float real_in_end_idx =
      (in_end_idx - pad_th <= 0)
          ? 0
          : std::floor((float)(in_end_idx - pad_th) / stride);
  real_in_idx = std::min(real_in_idx, max_real_in_idx);
  real_in_end_idx = std::min(real_in_end_idx, max_real_in_idx);

  float pad_t = (pad_th + real_in_idx * stride - in_idx);
  float pad_b = (in_end_idx - (pad_th + real_in_end_idx * stride));
  assert(pad_t >= 0 && pad_b >= 0 && pad_t < filter && pad_b < filter);
  return Optional<SmallVector<float, 4>>(
      {pad_t, pad_b, real_in_idx, real_in_end_idx - real_in_idx + 1});
}

LogicalResult tpu::DeconvOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto &attr = getDeconvParam(*this);
  int kh_ext = (attr.kh - 1) * attr.dh + 1;
  if (auto ret = DeconvSlice(out_idx, out_slice, attr.sh, kh_ext, attr.ih, attr.pad_h)) {
    in_idx = ret.value()[2];
    in_slice = ret.value()[3];
  } else {
    return failure();
  }

  bool is_last = (out_idx + out_slice == attr.oh);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.ih, is_last);
  if (in_slice == 0)
    return failure();
  return success();
}

mlir::Type tpu::DeconvOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return type_verify_case_i32(getOperation(), opd_idx, mode);
}

LogicalResult tpu::DeconvOp::LocalGenSupport() {
  auto attr = parseParam();
  int kh_ext = (attr.kh - 1) * attr.dh + 1;
  if (kh_ext < attr.sh) {
    return failure();
  }

  if (module::isCV18xx()) {
    if (attr.ic > MAX_TIU_CHL || attr.iw > MAX_TIU_CHL ||
        attr.ow > MAX_TIU_CHL) {
      return failure();
    }
  }
  return success();
}
