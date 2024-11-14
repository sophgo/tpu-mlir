//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"

deconv_attr_t tpu::DeconvOp::parseParam() {
  deconv_attr_t p = {0};
  auto ishape = getInput().getType().cast<RankedTensorType>().getShape();
  auto oshape = getOutput().getType().cast<RankedTensorType>().getShape();
  auto kernel = module::getI64Array(getKernelShape());
  auto stride = module::getI64Array(getStrides());
  auto dilation =
      module::getI64Array(getDilations(), getKernelShape().size(), 1);
  auto pad = module::getI64Array(getPads());
  auto output_padding =
      module::getI64Array(getOutputPadding(), getKernelShape().size(), 0);
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.with_bias = getWithBias();
  ;
  p.g = getGroup();
  p.n = ishape[0];
  p.ic = ishape[1];
  p.oc = oshape[1];

  auto dims = ishape.size() - 2;
  if (dims == 3) {
    p.id = ishape[2];
    p.ih = ishape[3];
    p.iw = ishape[4];
    p.od = oshape[2];
    p.oh = oshape[3];
    p.ow = oshape[4];
    p.kd = kernel->at(0);
    p.kh = kernel->at(1);
    p.kw = kernel->at(2);
    p.sd = stride->at(0);
    p.sh = stride->at(1);
    p.sw = stride->at(2);
    p.dd = dilation->at(0);
    p.dh = dilation->at(1);
    p.dw = dilation->at(2);
    p.pad_d = pad->at(0);
    p.pad_h = pad->at(1);
    p.pad_w = pad->at(2);
    p.pad_d_after = pad->at(3);
    p.pad_h_after = pad->at(4);
    p.pad_w_after = pad->at(5);
    p.output_pad_d = output_padding->at(0);
    p.output_pad_h = output_padding->at(1);
    p.output_pad_w = output_padding->at(2);
  } else if (dims == 2) {
    p.ih = ishape[2];
    p.iw = ishape[3];
    p.oh = oshape[2];
    p.ow = oshape[3];
    p.kh = kernel->at(0);
    p.kw = kernel->at(1);
    p.sh = stride->at(0);
    p.sw = stride->at(1);
    p.dh = dilation->at(0);
    p.dw = dilation->at(1);
    p.pad_h = pad->at(0);
    p.pad_w = pad->at(1);
    p.pad_h_after = pad->at(2);
    p.pad_w_after = pad->at(3);
    p.output_pad_h = output_padding->at(0);
    p.output_pad_w = output_padding->at(1);
    p.id = 1;
    p.od = 1;
    p.kd = 1;
    p.sd = 1;
    p.dd = 1;
  } else if (dims == 1) {
    p.id = p.od = p.kd = p.dd = p.sd = 1;
    p.iw = p.ow = p.kw = p.dw = p.sw = 1;
    p.ih = ishape[2];
    p.oh = oshape[2];
    p.kh = kernel->at(0);
    p.pad_h = pad->at(0);
    p.pad_h_after = pad->size() > 2 ? pad->at(2) : pad->at(1);
    p.sh = stride->at(0);
    p.dh = dilation->at(0);
    p.output_pad_h = output_padding->at(0);
  }
  p.is_dw = (p.oc == p.ic && p.oc == p.g && p.g > 1);
  return p;
}

LogicalResult tpu::DeconvOp::init(InferenceParameter &p) {
  auto deconv = new Deconv();
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
  auto attr = parseParam();
  int izp = 0;
  if (module::isUniformQuantized(getInput())) {
    izp = module::getUniformQuantizedType(getInput()).getZeroPoint();
  }
  deconv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], attr, izp);
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
    auto o_qtype = module::getUniformQuantizedType(getOutput());
    auto qmode = getQuantMode();
    bool is_tf = qmode == tpu::RequantMode::QDM ||
                 qmode == tpu::RequantMode::TFLite ||
                 qmode == tpu::RequantMode::TFLite_LShift;
    auto rmode = is_tf ? ROUNDING_HALF_AWAY_FROM_ZERO : ROUNDING_HALF_UP;
    // apply multiplier && rshift inplace
    int64_t n, c, h, w;
    module::getNCHW(getOutput(), n, c, h, w);
    auto rshift_v = module::getI64Array(getRshift().value());
    auto multiplier_v = module::getI64Array(getMultiplier().value());
#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int oc = 0; oc < c; oc++) {
      int64_t shift;
      int64_t multi;
      if (c > rshift_v->size()) {
        shift = rshift_v->at(0);
        multi = multiplier_v->at(0);
      } else {
        shift = rshift_v->at(oc);
        multi = multiplier_v->at(oc);
      }
      for (int on = 0; on < n; on++) {
        for (int hw = 0; hw < h * w; hw++) {
          int offset = (on * c + oc) * h * w + hw;
          int64_t v = 0;
          v = applyMultiplierAndRShift(p.outputs[0][offset], multi, shift,
                                       qmode, rmode) +
              o_qtype.getZeroPoint();
          p.outputs[0][offset] = to_int8(v);
        }
      }
    }
  }

  return success();
}

std::optional<SmallVector<float, 4>>
tpu_mlir::DeconvSlice(int64_t out_idx, int64_t out_slice, int64_t stride,
                      int64_t filter, int64_t ih, int64_t pad) {
  // pad top with (kh_ext - pad_h - 1), ins with (stride - 1)
  // all cal here assume input is expanded(pad && ins)
  float max_real_in_idx = ih - 1;
  int pad_th = filter - pad - 1;
  int in_idx = out_idx;
  float real_in_idx =
      (in_idx - pad_th <= 0) ? 0 : std::ceil((float)(in_idx - pad_th) / stride);

  int in_end_idx = out_idx + out_slice + filter - 2;
  float real_in_end_idx =
      (in_end_idx - pad_th <= 0)
          ? 0
          : std::floor((float)(in_end_idx - pad_th) / stride);
  real_in_idx = std::min(real_in_idx, max_real_in_idx);
  real_in_end_idx = std::min(real_in_end_idx, max_real_in_idx);

  float pad_t = (pad_th + real_in_idx * stride - in_idx);
  float pad_b = (in_end_idx - (pad_th + real_in_end_idx * stride));
  assert(pad_t >= 0 && pad_b >= 0);
  return std::optional<SmallVector<float, 4>>(
      {pad_t, pad_b, real_in_idx, real_in_end_idx - real_in_idx + 1});
}

LogicalResult tpu::DeconvOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto &attr = getDeconvParam(*this);
  int kh_ext = (attr.kh - 1) * attr.dh + 1;
  if (auto ret = DeconvSlice(out_idx, out_slice, attr.sh, kh_ext, attr.ih,
                             attr.pad_h)) {
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

LogicalResult tpu::DeconvOp::BackwardW(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto &attr = getDeconvParam(*this);
  int kw_ext = (attr.kw - 1) * attr.dw + 1;
  if (auto ret = DeconvSlice(out_idx, out_slice, attr.sw, kw_ext, attr.iw,
                             attr.pad_w)) {
    in_idx = ret.value()[2];
    in_slice = ret.value()[3];
  } else {
    return failure();
  }

  bool is_last = (out_idx + out_slice == attr.ow);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.iw, is_last);
  if (in_slice == 0)
    return failure();
  return success();
}

mlir::Type tpu::DeconvOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return type_verify_case_i32(getOperation(), opd_idx, mode);
}

LogicalResult tpu::DeconvOp::DynBackwardH(int64_t &in_idx, int64_t &in_slice,
                                          int64_t out_idx, int64_t out_slice) {
  auto &attr = getDeconvParam(*this);
  int kh_ext = (attr.kh - 1) * attr.dh + 1;
  auto ret =
      DeconvSlice(out_idx, out_slice, attr.sh, kh_ext, attr.ih, attr.pad_h);
  in_idx = ret.value()[2];
  in_slice = ret.value()[3];
  return success();
}

LogicalResult tpu::DeconvOp::DynBackwardKh(int64_t &in_kh, int64_t out_kh) {
  /*auto &attr = getDeconvParam(*this);
  int kh_with_dh = (attr.kh - 1) * attr.dh + 1;
  int val = (out_kh - std::max(kh_with_dh, attr.sh)) > 0 ? (out_kh -
  std::max(kh_with_dh, attr.sh)) : 0; in_kh = std::ceil(val / attr.sh) + 1;*/
  in_kh = out_kh;
  return success();
}

LogicalResult tpu::DeconvOp::DynBackwardStrideH(int64_t &in_stride_h,
                                                int64_t out_stride_h) {
  /*auto &attr = getDeconvParam(*this);
  in_stride_h = std::ceil( out_stride_h / attr.sh);*/
  in_stride_h = out_stride_h;
  return success();
}

LogicalResult tpu::DeconvOp::DynBackwardUpPadH(int64_t &in_up_pad_h,
                                               int64_t out_up_pad_h) {
  /*auto &attr = getDeconvParam(*this);
  in_up_pad_h = out_up_pad_h / attr.sh + attr.pad_h;*/
  in_up_pad_h = out_up_pad_h;
  return success();
}

LogicalResult tpu::DeconvOp::DynBackwardDownPadH(int64_t &in_down_pad_h,
                                                 int64_t out_down_pad_h) {
  /*auto &attr = getDeconvParam(*this);
  in_down_pad_h = out_down_pad_h / attr.sh + attr.pad_h_after;*/
  in_down_pad_h = out_down_pad_h;
  return success();
}

int64_t tpu::DeconvOp::DynForwardHeight(int64_t in_height) {
  auto &attr = getDeconvParam(*this);
  int out_height = 0;
  int kh_with_dh = (attr.kh - 1) * attr.dh + 1;
  out_height = (in_height - 1) * attr.sh + kh_with_dh - attr.pad_h -
               attr.pad_h_after + attr.output_pad_h;
  return out_height;
}

LogicalResult tpu::DeconvOp::LocalGenSupport() {
  auto attr = parseParam();
  int kh_ext = (attr.kh - 1) * attr.dh + 1;
  if (kh_ext < attr.sh) {
    return failure();
  }

  if (module::isCV18xx()) {
    if (attr.ic > MAX_TIU_CHL || attr.oc > MAX_TIU_CHL ||
        attr.iw > MAX_TIU_CHL || attr.ow > MAX_TIU_CHL) {
      return failure();
    }
  }
  return success();
}

void tpu::DeconvOp::assign_fw_param(void *param) {
  fw_deconv_layer_param_t fw_deconv_layer_param = {0};
  auto p = parseParam();
  fw_deconv_layer_param.output_dtype = BM168x::getDataType(getOutput());
  fw_deconv_layer_param.ic_oc = ((uint32_t)p.ic << 16) | (uint32_t)p.oc;
  fw_deconv_layer_param.groups = p.g;
  fw_deconv_layer_param.kh_kw = ((uint32_t)p.kh << 16) | (uint32_t)p.kw;
  fw_deconv_layer_param.dh = p.dh;
  fw_deconv_layer_param.dw = p.dw;
  fw_deconv_layer_param.pad_h = p.pad_h;
  fw_deconv_layer_param.pad_h_after = p.pad_h_after;
  fw_deconv_layer_param.pad_w = p.pad_w;
  fw_deconv_layer_param.pad_w_after = p.pad_w_after;
  fw_deconv_layer_param.stride_h = p.sh;
  fw_deconv_layer_param.stride_w = p.sw;
  fw_deconv_layer_param.using_bias = p.with_bias;
  fw_deconv_layer_param.if_relu = p.do_relu;
  fw_deconv_layer_param.relu_upper_limit = p.relu_limit;
  if (module::isUniformQuantized(getInput())) {
    fw_deconv_layer_param.rshift_num =
        module::getI64Array(getRshift(), 1, 0)->at(0);
    fw_deconv_layer_param.opd0_sign = module::isSign(getInput());
    fw_deconv_layer_param.opd1_sign = module::isSign(getFilter());
    if (p.with_bias)
      fw_deconv_layer_param.opd2_sign = module::isSign(getBias());
  }
  fw_deconv_layer_param.output_padding_h = p.output_pad_h;
  fw_deconv_layer_param.output_padding_w = p.output_pad_w;
  fw_deconv_layer_param.using_depthwise = 1;
  memcpy(param, &fw_deconv_layer_param, sizeof(fw_deconv_layer_param_t));
}

LogicalResult tpu::DeconvOp::AllowDataSplit(int64_t axis,
                                            group_type_t group_type) {
  /* because backward slice_h 、stride_h、kh, pad_up, pad_down calc formula
    is very complex, don;t split when dynamic mode */
  if (getRunMode(getOperation()) == RunMode::TPU_DYNAMIC) {
    return failure();
  } else {
    return success();
  }
}

bool tpu::DeconvOp::support_multi_core() { return false; }
