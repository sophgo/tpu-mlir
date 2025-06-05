//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"

deconv_attr_t tpu::Deconv3DOp::parseParam() {
  deconv_attr_t p = {0};
  module::getNCDHW(getInput(), p.n, p.ic, p.id, p.ih, p.iw, GROUP_3D);
  module::getNCDHW(getOutput(), p.n, p.oc, p.od, p.oh, p.ow, GROUP_3D);
  auto kernel = module::getI64Array(getKernelShape());
  p.kd = kernel->at(0);
  p.kh = kernel->at(1);
  p.kw = kernel->at(2);
  auto stride = module::getI64Array(getStrides());
  p.sd = stride->at(0);
  p.sh = stride->at(1);
  p.sw = stride->at(2);
  auto pad = module::getI64Array(getPads());
  p.pad_d = pad->at(0);
  p.pad_h = pad->at(1);
  p.pad_w = pad->at(2);
  p.pad_d_after = pad->at(3);
  p.pad_h_after = pad->at(4);
  p.pad_w_after = pad->at(5);
  auto dilation = module::getI64Array(getDilations(), 3, 1);
  p.dd = dilation->at(0);
  p.dh = dilation->at(1);
  p.dw = dilation->at(2);
  auto output_padding = module::getI64Array(getOutputPadding(), 3, 0);
  p.output_pad_d = output_padding->at(0);
  p.output_pad_h = output_padding->at(1);
  p.output_pad_w = output_padding->at(2);
  p.g = getGroup();
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.with_bias = getWithBias();
  return p;
}

LogicalResult tpu::Deconv3DOp::init(InferenceParameter &p) {
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

void tpu::Deconv3DOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto deconv = (Deconv *)p.handle;
    delete deconv;
    p.handle = nullptr;
  }
}

LogicalResult tpu::Deconv3DOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto deconv = (Deconv *)p.handle;
  deconv->run();
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
    bool is_tf = qmode == tpu::RequantMode::QDM ||
                 qmode == tpu::RequantMode::TFLite ||
                 qmode == tpu::RequantMode::TFLite_LShift;
    auto rmode = is_tf ? ROUNDING_HALF_AWAY_FROM_ZERO : ROUNDING_HALF_UP;
    // apply multiplier && rshift inplace
    auto output_shape = module::getShape(getOutput());
    int64_t n = output_shape[0], c = output_shape[1], d = output_shape[2],
            h = output_shape[3], w = output_shape[4];
    int64_t shift, multi;
    auto rshift_v = module::getI64Array(getRshift().value());
    auto multiplier_v = module::getI64Array(getMultiplier().value());
#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int oc = 0; oc < c; oc++) {
      if (c > rshift_v->size()) {
        shift = rshift_v->at(0);
        multi = multiplier_v->at(0);
      } else {
        shift = rshift_v->at(oc);
        multi = multiplier_v->at(oc);
      }
      for (int on = 0; on < n; on++) {
        for (int dhw = 0; dhw < d * h * w; dhw++) {
          int offset = (on * c + oc) * d * h * w + dhw;
          auto v = applyMultiplierAndRShift(p.outputs[0][offset], multi, shift,
                                            qmode, rmode);
          p.outputs[0][offset] = to_int8(v);
        }
      }
    }
  }
  return success();
}

LogicalResult tpu::Deconv3DOp::LocalGenSupport() { return failure(); }

void tpu::Deconv3DOp::assign_fw_param(void *param) {
  fw_deconv3d_layer_param_t deconv3d_param;
  auto p = parseParam();
  deconv3d_param.oc = p.oc;
  deconv3d_param.groups = p.g;
  deconv3d_param.kernel[0] = p.kd, deconv3d_param.kernel[1] = p.kh,
  deconv3d_param.kernel[2] = p.kw;
  deconv3d_param.dilation[0] = p.dd, deconv3d_param.dilation[1] = p.dh,
  deconv3d_param.dilation[2] = p.dw;
  deconv3d_param.pads[0] = p.pad_d, deconv3d_param.pads[1] = p.pad_d_after;
  deconv3d_param.pads[2] = p.pad_h, deconv3d_param.pads[3] = p.pad_h_after;
  deconv3d_param.pads[4] = p.pad_w, deconv3d_param.pads[5] = p.pad_w_after;
  deconv3d_param.stride[0] = p.sd, deconv3d_param.stride[1] = p.sh,
  deconv3d_param.stride[2] = p.sw;
  deconv3d_param.output_padding[0] = p.output_pad_d,
  deconv3d_param.output_padding[1] = p.output_pad_h,
  deconv3d_param.output_padding[2] = p.output_pad_w;
  deconv3d_param.using_bias = p.with_bias;
  deconv3d_param.if_relu = p.do_relu;
  deconv3d_param.relu_upper_limit = p.relu_limit;
  memcpy(param, &deconv3d_param, sizeof(fw_deconv3d_layer_param_t));
}

bool tpu::Deconv3DOp::support_multi_core() { return false; }

mlir::Type tpu::Deconv3DOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return type_verify_case_i32(getOperation(), opd_idx, mode);
}