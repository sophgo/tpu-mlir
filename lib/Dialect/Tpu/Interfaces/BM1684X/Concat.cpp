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

void tpu::ConcatOp::codegen_global_bm1684x() {
  auto op = getOperation();
  int num_input = getInputs().size();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (getOnlyMerge() && input_spec->at(0).addr == output_spec->at(0).addr) {
    bool is_inplace = true;
    int64_t begin_addr = module::getAddress(getOutput());
    int64_t end_addr = begin_addr + module::getBytes(getOutput());
    for (Value input : getInputs()) {
      if (module::getAddress(input) < begin_addr ||
          module::getAddress(input) >= end_addr) {
        is_inplace = false;
        break;
      }
    }
    if (is_inplace)
      return;
  }
  concat_global_spec_t spec = {0};
  spec.common.input_num = num_input;
  spec.common.concat_axis = getAxis();
  SmallVector<int> is_st_concat_way(num_input, 0);
  spec.is_st_concat_way = is_st_concat_way.data();

  BM168x::call_global_func("backend_api_concat_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ConcatOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::ConcatOp::codegen_local_bm1684x_kernel(
    std::vector<group_info_t> &in_group_infos,
    std::vector<group_info_t> &out_group_infos, local_sec_info_t &sec_info,
    std::shared_ptr<std::vector<tensor_spec_t>> input_spec,
    std::shared_ptr<std::vector<tensor_spec_t>> output_spec) {
  concat_local_spec_t spec = {0};
  int num_input = getInputs().size();
  SmallVector<int> is_st_concat_way(num_input, 0);
  spec.is_st_concat_way = is_st_concat_way.data();
  spec.common.input_num = num_input;
  spec.common.concat_axis = getAxis();

  auto gi = out_group_infos[0];
  for (int i = 0; i < num_input; i++) {
    auto in_gi = in_group_infos[i];
    setHWMargins(input_spec->at(i).hw_margins, in_gi, gi);
  }

  BM168x::call_local_func("backend_api_concat_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::ConcatOp::dyn_codegen_local_bm1684x(void *buffer) {
  int input_num = getInputs().size();
  if (buffer) {
    concat_common_spec_t common;
    memset(&common, 0, sizeof(common));
    common.input_num = input_num;
    common.concat_axis = getAxis();
    auto p = static_cast<char *>(buffer);
    memcpy(p, &common, sizeof(common));
    p += sizeof(common);
    int size = p - static_cast<char *>(buffer);
    buffer = (char *)buffer + size;
    SmallVector<int> is_st_concat_way(input_num, 0);
    p = static_cast<char *>(buffer);
    memcpy(p, is_st_concat_way.data(), sizeof(is_st_concat_way[0]) * input_num);
    p += sizeof(is_st_concat_way[0]) * input_num;
  }
  return sizeof(concat_common_spec_t) + input_num * sizeof(int);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ConcatOp::dyn_codegen_global_bm1684x(void *buffer) {
  int input_num = getInputs().size();
  if (buffer) {
    concat_common_spec_t common;
    memset(&common, 0, sizeof(common));
    common.input_num = input_num;
    common.concat_axis = getAxis();
    auto p = static_cast<char *>(buffer);
    memcpy(p, &common, sizeof(common));
    p += sizeof(common);
    int size = p - static_cast<char *>(buffer);
    buffer = (char *)buffer + size;
    SmallVector<int> is_st_concat_way(input_num, 0);
    p = static_cast<char *>(buffer);
    memcpy(p, is_st_concat_way.data(), sizeof(is_st_concat_way[0]) * input_num);
    p += sizeof(is_st_concat_way[0]) * input_num;
  }
  return sizeof(concat_common_spec_t) + input_num * sizeof(int);
}

int64_t tpu::ConcatOp::get_fw_type_bm1684x() { return FW_BMNET_CONCAT; }
