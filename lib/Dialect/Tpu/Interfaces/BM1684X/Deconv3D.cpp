//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

using namespace tpu_mlir::backend;

// ======================================
// GlobalGenInterface
// ======================================

void tpu::Deconv3DOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (getKernelShape().size() == 1) {
    BM168x::fix_shape(input_spec->at(0),
                      {attr.n, attr.ic, attr.id, attr.ih, attr.iw});
    BM168x::fix_shape(output_spec->at(0),
                      {attr.n, attr.oc, attr.od, attr.oh, attr.ow});
  }
  deconv3d_global_param_t param = {0};
  param.input_global_addr = module::getAddress(getInput());
  param.weight_global_addr = module::getAddress(getFilter());
  param.bias_global_addr = module::getAddress(getBias());
  param.output_global_addr = module::getAddress(getOutput());
  param.input_shape[0] = attr.n;
  param.input_shape[1] = attr.ic;
  param.input_shape[2] = attr.id;
  param.input_shape[3] = attr.ih;
  param.input_shape[4] = attr.iw;
  param.groups = attr.g;
  param.output_c = attr.oc;
  param.kernel[0] = attr.kd;
  param.kernel[1] = attr.kh;
  param.kernel[2] = attr.kw;
  param.stride[0] = attr.sd;
  param.stride[1] = attr.sh;
  param.stride[2] = attr.sw;
  param.dilation[0] = attr.dd;
  param.dilation[1] = attr.dh;
  param.dilation[2] = attr.dw;
  param.pad[0] = attr.pad_d;
  param.pad[1] = attr.pad_d_after;
  param.pad[2] = attr.pad_h;
  param.pad[3] = attr.pad_h_after;
  param.pad[4] = attr.pad_w;
  param.pad[5] = attr.pad_w_after;
  param.output_pad[0] = attr.output_pad_d;
  param.output_pad[0] = attr.output_pad_h;
  param.output_pad[1] = attr.output_pad_w;
  param.has_bias = attr.with_bias;
  param.input_dtype = BM168x::getDataType(getInput());
  param.weight_dtype = BM168x::getDataType(getFilter());
  if (getBias().getType().isa<RankedTensorType>()) {
    param.bias_dtype = BM168x::getDataType(getBias());
  }
  param.output_dtype = BM168x::getDataType(getOutput());
  param.if_relu = attr.do_relu;
  param.upper_limit = attr.relu_limit;
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    param.kzp_global_addr = 0;
    param.pad_insert_global_addr = 0;
    param.kzp_is_const = true;
    param.pad_insert_is_const = true;
    param.kzp_val = 0;
    param.pad_val = in_qtype.getZeroPoint();
    param.insert_val = in_qtype.getZeroPoint();
    param.kzp_dtype = param.input_dtype;
  }
  BM168x::call_global_func("backend_api_deconv3d_global", &param,
                           sizeof(param));
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::Deconv3DOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  // TODO
  return 0;
}

void tpu::Deconv3DOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                            int64_t h_step, int64_t d_step,
                                            int64_t w_step,
                                            group_type_t group_type,
                                            local_sec_info_t &sec_info) {
  UNREACHABLE_THIS("Not Implemented");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::Deconv3DOp::dyn_codegen_global_bm1684x(void *buffer) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::Deconv3DOp::get_fw_type_bm1684x() { return -1; }
