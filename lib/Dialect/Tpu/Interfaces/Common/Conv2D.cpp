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

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Interfaces/LocalGenInterface.h"



conv_attr_t tpu::Conv2DOp::parseParam() {
  conv_attr_t p = {0};
  p.id = p.od = p.kd = p.sd = p.dd = 1;
  auto i_s = input().getType().cast<RankedTensorType>().getShape();
  auto o_s = output().getType().cast<RankedTensorType>().getShape();
  p.do_relu = do_relu();
  p.relu_limit = relu_limit().convertToDouble();
  p.has_bias = with_bias();
  p.n = i_s[0];
  p.ic = i_s[1];
  p.ih = i_s.size() > 2 ? i_s[2] : 1;
  p.iw = i_s.size() > 3 ? i_s[3] : 1;
  p.oc = o_s[1];
  p.oh = o_s.size() > 2 ? o_s[2] : 1;
  p.ow = o_s.size() > 3 ? o_s[3] : 1;
  auto kernel = module::getI64Array(kernel_shape());
  p.kh = kernel->at(0);
  p.kw = kernel->at(1);
  auto pads_v = module::getI64Array(pads());
  p.pht = pads_v->at(0);
  p.pwl = pads_v->at(1);
  p.phb = pads_v->at(2);
  p.pwr = pads_v->at(3);
  if (module::isUniformQuantized(input())) {
    p.pad_value = module::getUniformQuantizedType(input()).getZeroPoint();
  }
  p.kernel_zp = kernel_zp();
  auto strides_v = module::getI64Array(strides());
  p.sh = strides_v->at(0);
  p.sw = strides_v->at(1);
  auto dhdw = module::getI64Array(dilations(), 2, 1);
  p.dh = dhdw->at(0);
  p.dw = dhdw->at(1);
  auto ins = module::getI64Array(inserts(), 2, 0);
  p.ins_h = ins->at(0);
  p.ins_w = ins->at(1);
  assert(p.ins_h == 0 && p.ins_w == 0);
  p.groups = group();
  p.is_dw = (p.oc == p.ic && p.oc == p.groups && p.groups > 1);
  return p;
}

LogicalResult tpu::Conv2DOp::init(InferenceParameter &p) {
  auto conv = new Conv();
  auto attr = parseParam();

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
  bool is_cv18xx = module::isCV18xx();
  // requant
  auto out_type = module::getStorageType(output());
  auto num_elem = module::getNumElements(output());
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (module::isUniformQuantized(output())) {
    int64_t n, c, h, w;
    auto sType = module::getStorageType(output());
    module::getNCHW(output(), n, c, h, w);
    auto o_qtype = module::getUniformQuantizedType(output());
    auto rshift_v = module::getI64Array(rshift().value());
    auto multiplier_v = module::getI64Array(multiplier(), rshift_v->size(), 1);
    bool per_axis = rshift_v->size() == c;
    auto mode = quant_mode();
    MultiplierType m_type;
    if (is_cv18xx) {
      m_type = CVI_QDM_QUANT;
    } else {
      if (mode == tpu::RequantMode::TFlite_Lshift ||
          mode == tpu::RequantMode::TFlite) {
        m_type = BM_TFLITE_QUANT;
      } else {
        m_type = BM_QUANT;
      }
    }

#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int ic = 0; ic < c; ic++) {
      int64_t shift = per_axis ? rshift_v->at(ic) : rshift_v->at(0);
      int64_t multi = per_axis ? multiplier_v->at(ic) : multiplier_v->at(0);
      for (int in = 0; in < n; in++) {
        for (int hw = 0; hw < h * w; hw++) {
          int offset = (in * c + ic) * h * w + hw;
          int64_t v = 0;
          v = applyMultiplierAndRShift(p.outputs[0][offset], multi, shift,
                                       m_type) +
              o_qtype.getZeroPoint();
          if (sType.isInteger(8)) {
            p.outputs[0][offset] =
                sType.isUnsignedInteger(8) ? to_uint8(v) : to_int8(v);
          } else {
            p.outputs[0][offset] =
                sType.isUnsignedInteger(4) ? to_uint4(v) : to_int4(v);
          }
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
