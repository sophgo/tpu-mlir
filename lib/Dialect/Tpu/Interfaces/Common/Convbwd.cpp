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

convbwd_attr_t tpu::ConvbwdOp::parseParam() {
  convbwd_attr_t p = {0};
  auto input_s = getInput().getType().cast<RankedTensorType>().getShape();
  auto gradout_s = getGradOut().getType().cast<RankedTensorType>().getShape();
  p.n = input_s[0];
  p.ic = input_s[1];
  p.ih = input_s.size() > 2 ? input_s[2] : 1;
  p.iw = input_s.size() > 3 ? input_s[3] : 1;
  p.oc = gradout_s[1];
  p.oh = gradout_s.size() > 2 ? gradout_s[2] : 1;
  p.ow = gradout_s.size() > 3 ? gradout_s[3] : 1;
  auto kernel = module::getI64Array(getKernelShape());
  p.kh = kernel->at(2);
  p.kw = kernel->at(3);
  auto pads_v = module::getI64Array(getPadding());
  p.pht = pads_v->at(0);
  p.pwl = pads_v->at(1);
  p.phb = pads_v->at(2);
  p.pwr = pads_v->at(3);
  auto strides_v = module::getI64Array(getStride());
  p.sh = strides_v->at(0);
  p.sw = strides_v->at(1);
  auto dhdw = module::getI64Array(getDilations(), 2, 1);
  p.dh = dhdw->at(0);
  p.dw = dhdw->at(1);
  auto ins = module::getI64Array(getInserts());
  p.insh = ins->at(0);
  p.insw = ins->at(1);
  p.groups = getGroups();
  return p;
}

LogicalResult tpu::ConvbwdOp::init(InferenceParameter &p) { return success(); }

void tpu::ConvbwdOp::deinit(InferenceParameter &p) {}

#define DIV_UP(x, y) (((x) + (y)-1) / (y))

static void weight_reorder_i32(float *src, int oc, int ic, int kh, int kw) {
  auto ptr_src = std::make_unique<float[]>(oc * DIV_UP(ic, 32) * kh * kw * 32);
  float *src_data = ptr_src.get();
  memcpy(src_data, src, oc * ic * kh * kw * sizeof(float));
  // clean src data
  memset(src, 0, 1 * oc * DIV_UP(ic, 32) * kh * kw * 32 * sizeof(float));
  // copy src_data to src with 32ic layout
  for (int i = 0; i < oc; i++) {
    for (int j = 0; j < ic; j++) {
      // 32ic
      int ic32 = j / 32;
      int ic32_offset = j % 32;
      for (int k = 0; k < kh; k++) {
        for (int l = 0; l < kw; l++) {
          auto src_data_idx = i * ic * kh * kw + j * kh * kw + k * kw + l;
          auto src_idx = i * DIV_UP(ic, 32) * kh * kw * 32 +
                         ic32 * kh * kw * 32 + k * kw * 32 + l * 32 +
                         ic32_offset;
          src[src_idx] = src_data[src_data_idx];
        }
      }
    }
  }
}

LogicalResult tpu::ConvbwdOp::inference(InferenceParameter &p) {
  auto attr = parseParam();
  float *gradout = p.inputs[0];
  float *input = p.inputs[1];
  float *kernel = p.inputs[2];
  float *gradinput = p.outputs[0];
  float *gradweight = p.outputs[1];
  float *gradbias = p.outputs[2];
  auto cal_grad_input = getGradInputEnable();
  auto cal_grad_weight = getGradWeightEnable();
  auto cal_grad_bias = getGradBiasEnable();
  // auto is_half = module::getStorageType(getInput()).isBF16() ||
  // module::getStorageType(getInput()).isF16();
  tpu_mlir::ConvBwd conv_bwd;
  conv_bwd.setup(input, kernel, gradout, gradinput, gradweight, gradbias,
                 attr.n, attr.ic, attr.oc, attr.ih, attr.iw, attr.kh, attr.kw,
                 attr.oh, attr.ow, attr.sh, attr.sw, attr.pht, attr.pwl,
                 attr.phb, attr.pwr, cal_grad_input, cal_grad_weight,
                 cal_grad_bias, attr.dh, attr.dw);
  conv_bwd.run();
  // if(is_half && cal_grad_weight)
  // {
  //   weight_reorder_i32(gradweight, attr.oc, attr.ic, attr.kh, attr.kw);
  // }
  return success();
}

mlir::Type tpu::ConvbwdOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  auto op = getOperation();
  auto opd = op->getOperand(opd_idx);
  auto in_op = opd.getDefiningOp();
  if (in_op != nullptr && isa<top::WeightOp, top::NoneOp>(in_op)) {
    return do_nothing(mode);
  }
  auto stype = module::getStorageType(opd);
  if (stype.isF16()) {
    return do_nothing(mode);
  }
  mode = TypeCastMode::DO_CAST;
  return Builder(op).getF16Type();
}

uint32_t tpu::ConvbwdOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::ConvbwdOp::get_fw_type_bm1684() { return -1; }

bool tpu::ConvbwdOp::support_multi_core() { return true; }
