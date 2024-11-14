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

// =========================================
// GlobalGenInterface
// =========================================
void tpu::TileOp::codegen_global_bm1684x() {
  // llvm_unreachable("Only support dynamic codegen");
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto &in = input_spec->at(0);
  auto &out = output_spec->at(0);
  if (in.dims < out.dims) {
    const int more_dims = out.dims - in.dims;
    memmove(in.shape + more_dims, in.shape, in.dims * sizeof(int));
    for (int i = 0; i < more_dims; ++i) {
      in.shape[i] = 1;
    }
    in.dims = out.dims;
  }
  tile_global_spec_t spec = {0};
  for (int i = 0; i < out.dims; ++i) {
    spec.common.tile_coeff[i] = out.shape[i] / in.shape[i];
    ;
  }
  spec.buffer_addr = module::getAddress(getBuffer());
  spec.common.type = 0;
  BM168x::call_global_func("backend_api_tile_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface not supported
// =========================================

int64_t tpu::TileOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::TileOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                        int64_t h_step, int64_t d_step,
                                        int64_t w_step, group_type_t group_type,
                                        local_sec_info_t &sec_info) {
  // auto op = getOperation();
  // auto input_spec = BM168x::get_input_spec(op);
  // auto output_spec = BM168x::get_output_spec(op);
  // auto &in = input_spec->at(0);
  // auto &out = output_spec->at(0);
  // tile_local_spec_t spec = {0};
  // for (int i = 0; i < in.dims; ++i) {
  //   spec.common.tile_coeff[i] = out.shape[i] / in.shape[i];
  // }
  // spec.common.type = 0;

  // BM168x::call_local_func("backend_api_tile_local", &spec, sizeof(spec),
  //                         &sec_info, input_spec->data(),
  //                         output_spec->data());
  llvm_unreachable("Not supported now");
}

// dynamic codegen
int64_t tpu::TileOp::dyn_codegen_local_bm1684x(void *buffer) {
  // if (!buffer)
  //   return sizeof(tile_local_spec_t);
  // auto op = getOperation();
  // auto input_spec = BM168x::get_input_spec(op);
  // auto output_spec = BM168x::get_output_spec(op);
  // auto &in = input_spec->at(0);
  // auto &out = output_spec->at(0);
  // tile_local_spec_t spec = {0};
  // for (int i = 0; i < in.dims; ++i) {
  //   spec.common.tile_coeff[i] = out.shape[i] / in.shape[i];
  // }
  // spec.common.type = 0;
  // return BM168x::dynamic_spec_to_buffer(buffer, spec);
  llvm_unreachable("Not supported now");
  return 0;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::TileOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) {
    return sizeof(tile_global_param_t);
  }
  tile_global_param_t param = {0};
  param.spec.common.type = 0;

  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);

  auto input_num = (*input_spec).size();
  if (input_num == 1 || (input_num == 2 && !module::isNone(getBuffer()))) {
    param.coeff_is_fixed = true;
    auto tile_vec = *module::getI64Array(getTile());
    auto tile_vec_size = tile_vec.size();
    for (int i = 0; i < tile_vec_size; ++i) {
      param.spec.common.tile_coeff[i] = tile_vec[i];
    }
    for (int i = tile_vec_size; i < MAX_SHAPE_DIMS; ++i) {
      param.spec.common.tile_coeff[i] = 1;
    }
  } else {
    param.coeff_is_fixed = false;
  }

  param.spec.buffer_addr = module::getAddress(getBuffer());
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::TileOp::get_fw_type_bm1684x() { return FW_BMNET_TILE; }
