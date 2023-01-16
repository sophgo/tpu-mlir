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
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Interfaces/LocalGenInterface.h"
#include "tpu_mlir/Support/MathUtils.h"

conv_attr_t tpu::Conv2DOp::parseParam() {
  conv_attr_t p = {0};
  p.id = p.od = p.kd = p.sd = p.dd = 1;
  auto i_s = getInput().getType().cast<RankedTensorType>().getShape();
  auto o_s = getOutput().getType().cast<RankedTensorType>().getShape();
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.has_bias = getWithBias();
  p.n = i_s[0];
  p.ic = i_s[1];
  p.ih = i_s.size() > 2 ? i_s[2] : 1;
  p.iw = i_s.size() > 3 ? i_s[3] : 1;
  p.oc = o_s[1];
  p.oh = o_s.size() > 2 ? o_s[2] : 1;
  p.ow = o_s.size() > 3 ? o_s[3] : 1;
  auto kernel = module::getI64Array(getKernelShape());
  p.kh = kernel->at(0);
  p.kw = kernel->at(1);
  auto pads_v = module::getI64Array(getPads());
  p.pht = pads_v->at(0);
  p.pwl = pads_v->at(1);
  p.phb = pads_v->at(2);
  p.pwr = pads_v->at(3);
  if (module::isUniformQuantized(getInput())) {
    p.pad_value = module::getUniformQuantizedType(getInput()).getZeroPoint();
  }
  p.kernel_zp = getKernelZp();
  auto strides_v = module::getI64Array(getStrides());
  p.sh = strides_v->at(0);
  p.sw = strides_v->at(1);
  auto dhdw = module::getI64Array(getDilations(), 2, 1);
  p.dh = dhdw->at(0);
  p.dw = dhdw->at(1);
  auto ins = module::getI64Array(getInserts(), 2, 0);
  p.ins_h = ins->at(0);
  p.ins_w = ins->at(1);
  assert(p.ins_h == 0 && p.ins_w == 0);
  p.groups = getGroup();
  p.is_dw = (p.oc == p.ic && p.oc == p.groups && p.groups > 1);
  return p;
}

LogicalResult tpu::Conv2DOp::init(InferenceParameter &p) {
  auto conv = new Conv();
  auto attr = parseParam();
  if (module::isUniformQuantized(getOutput()) && attr.has_bias) {
    attr.do_relu = false;
    for (int i = 0; i < attr.oc; i++) {
      p.inputs[2][i] = 0.f;
    }
  }
  conv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], attr);
  p.handle = (void *)conv;
  return success();
}

void tpu::Conv2DOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto conv = (Conv *)p.handle;
    delete conv;
    p.handle = nullptr;
  }
}

LogicalResult tpu::Conv2DOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto conv = (Conv *)p.handle;
  conv->run();
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
    int64_t n, c, h, w;
    auto sType = module::getStorageType(getOutput());
    module::getNCHW(getOutput(), n, c, h, w);
    auto o_qtype = module::getUniformQuantizedType(getOutput());
    auto rshift_v = module::getI64Array(getRshift().value());
    auto multiplier_v =
        module::getI64Array(getMultiplier(), rshift_v->size(), 1);
    bool per_axis = rshift_v->size() == c;
    // do bias after conv prevent precision issue
    auto bias_i32 = std::make_shared<std::vector<int32_t>>(c, 0);
    bool do_relu = getDoRelu();
    if (getWithBias()) {
      auto biasOp = cast<top::WeightOp>(getBias().getDefiningOp());
      bias_i32 = biasOp.read_as_int32();
    }
    auto mode = getQuantMode();
    QuantMode q_mode;
    if (module::isCV18xx()) {
      q_mode = CVI_QUANT_QDM;
    } else if (module::isBM1684Family()) {
      q_mode = BM_QUANT_ONLY_SHIFT;
    } else {
      if (mode == tpu::RequantMode::TFlite_Lshift ||
          mode == tpu::RequantMode::TFlite) {
        q_mode = BM_QUANT_TFLITE;
      } else {
        q_mode = BM_QUANT_NORMAL;
      }
    }

#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int ic = 0; ic < c; ic++) {
      int64_t shift = per_axis ? rshift_v->at(ic) : rshift_v->at(0);
      int64_t multi = 1;
      if (q_mode != BM_QUANT_ONLY_SHIFT) {
        multi = per_axis ? multiplier_v->at(ic) : multiplier_v->at(0);
      }
      int32_t bias = bias_i32->at(ic);
      for (int in = 0; in < n; in++) {
        for (int hw = 0; hw < h * w; hw++) {
          int offset = (in * c + ic) * h * w + hw;
          int64_t v = 0;
          int64_t tmp = p.outputs[0][offset] + bias;
          v = applyMultiplierAndRShift(tmp, multi, shift, q_mode) +
              o_qtype.getZeroPoint();
          if (do_relu && (v < 0)) {
            v = 0;
          }
          p.outputs[0][offset] = saturate(v, out_type);
        }
      }
    }
  }

  return success();
}

LogicalResult tpu::Conv2DOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto &attr = getConv2DParam(*this);
  int kh_with_dh = (attr.kh - 1) * attr.dh + 1;
  in_slice = (out_slice - 1) * attr.sh +
             (kh_with_dh >= attr.sh ? kh_with_dh : attr.sh);
  in_idx = out_idx * attr.sh - attr.pht;
  bool is_last = (out_idx + out_slice == attr.oh);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.ih, is_last);
  return success();
}

mlir::Type tpu::Conv2DOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return type_verify_case_i32(getOperation(), opd_idx, mode);
}

LogicalResult tpu::Conv2DOp::DynBackwardH(int64_t &in_idx, int64_t &in_slice,
                                          int64_t out_idx, int64_t out_slice) {
  auto &attr = getConv2DParam(*this);
  int kh_with_dh = (attr.kh - 1) * attr.dh + 1;
  in_slice = (out_slice - 1) * attr.sh +
             (kh_with_dh >= attr.sh ? kh_with_dh : attr.sh);
  in_idx = out_idx * attr.sh - attr.pht;
  return success();
}

LogicalResult tpu::Conv2DOp::DynBackwardKh(int64_t &in_kh, int64_t out_kh) {
  auto &attr = getConv2DParam(*this);
  int kh_with_dh = (attr.kh - 1) * attr.dh + 1;
  in_kh =
      (out_kh - 1) * attr.sh + (kh_with_dh >= attr.sh ? kh_with_dh : attr.sh);
  return success();
}

LogicalResult tpu::Conv2DOp::DynBackwardStrideH(int64_t &in_stride_h,
                                                int64_t out_stride_h) {
  auto &attr = getConv2DParam(*this);
  in_stride_h = out_stride_h * attr.sh;
  return success();
}

LogicalResult tpu::Conv2DOp::DynBackwardUpPadH(int64_t &in_up_pad_h,
                                               int64_t out_up_pad_h) {
  auto &attr = getConv2DParam(*this);
  in_up_pad_h = out_up_pad_h * attr.sh + attr.pht;
  return success();
}

LogicalResult tpu::Conv2DOp::DynBackwardDownPadH(int64_t &in_down_pad_h,
                                                 int64_t out_down_pad_h) {
  auto &attr = getConv2DParam(*this);
  in_down_pad_h = out_down_pad_h * attr.sh + attr.phb;
  return success();
}
