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
// using namespace tpu_mlir::bm1684x;

// ======================================
// GlobalGenInterface
// ======================================

void tpu::DeformGatherOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  deform_gather_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.deform_groups = attr.deform_groups;
  spec.kh = attr.kh;
  spec.kw = attr.kw;
  spec.stride_h = attr.sh;
  spec.stride_w = attr.sw;
  spec.dilation_h = attr.dh;
  spec.dilation_w = attr.dw;
  spec.pad_h = attr.phb;
  spec.pad_h_after = attr.pht;
  spec.pad_w = attr.pwl;
  spec.pad_w_after = attr.pwr;
  spec.modulated = attr.use_mask == true;
  spec.mode = 2; // torchvision
  spec.offset_interleave = 1;
  spec.buffer_addr = module::getAddress(getBuffer());
  BM168x::call_global_func("backend_api_deform_gather_global", &spec,
                           sizeof(spec), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::DeformGatherOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(deform_gather_global_spec_t);
  auto op = getOperation();
  auto attr = parseParam();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  deform_gather_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.deform_groups = attr.deform_groups;
  spec.kh = attr.kh;
  spec.kw = attr.kw;
  spec.stride_h = attr.sh;
  spec.stride_w = attr.sw;
  spec.dilation_h = attr.dh;
  spec.dilation_w = attr.dw;
  spec.pad_h = attr.phb;
  spec.pad_h_after = attr.pht;
  spec.pad_w = attr.pwl;
  spec.pad_w_after = attr.pwr;
  spec.modulated = attr.use_mask == true;
  spec.mode = 2; // torchvision
  spec.offset_interleave = 1;
  spec.buffer_addr = module::getAddress(getBuffer());
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::DeformGatherOp::get_fw_type_bm1684x() {
  return FW_BMNET_DEFORM_GATHER;
}
