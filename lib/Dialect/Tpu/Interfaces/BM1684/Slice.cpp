//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::SliceOp::codegen_global_bm1684() {
  auto p = parseParam();
  auto input = getInput();
  auto output = getOutput();
  auto input_addr = module::getAddress(input);
  auto output_addr = module::getAddress(output);
  int begin_mask = 0, end_mask = 0;
  int shape_dim = p.is_4.size();
  // malloc
  int *input_shape = new int[MAX_SHAPE_DIMS];
  int *begin_index = new int[MAX_SHAPE_DIMS];
  int *end_index = new int[MAX_SHAPE_DIMS];
  int *stride = new int[MAX_SHAPE_DIMS];
  // assign param and call func
  for (int i = 0; i < shape_dim; ++i) {
    input_shape[i] = p.is_4[i];
    begin_index[i] =
        p.offset_4[i] < 0 ? p.offset_4[i] + p.is_4[i] : p.offset_4[i];
    end_index[i] = p.os_4[i] * p.step_4[i] + p.offset_4[i];
    stride[i] = p.step_4[i];
  }
  auto input_dtype = BM1684::getDataType(input);
  if (input_dtype == DTYPE_FP32 || input_dtype == DTYPE_INT32) {
    BM1684::instance().dl_nodechip_stride_slice_md(
        input_addr, output_addr, input_shape, shape_dim, begin_mask, end_mask,
        begin_index, end_index, stride,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else if (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8) {
    auto buffer_addr = module::getAddress(getBuffer());
    uint64_t input_size = ceiling_func(input_shape[0], 4) * 4 * input_shape[1] *
                          input_shape[2] * input_shape[3];
    uint64_t imm_buffer_addr = buffer_addr + input_size;
    BM1684::instance().dl_nodechip_stride_slice_fix8b(
        input_addr, output_addr, buffer_addr, imm_buffer_addr, NULL,
        input_shape, shape_dim, STORE_MODE_4N, STORE_MODE_4N, begin_mask,
        end_mask, begin_index, end_index, stride, 0,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    llvm_unreachable("Not Implemented.");
  }
  // release
  delete[] input_shape;
  delete[] begin_index;
  delete[] end_index;
  delete[] stride;
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::SliceOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  return 0;
}

void tpu::SliceOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                        local_sec_info_t &sec_info) {
  auto out_g_info = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in_g_info = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  int begin_mask = 0, end_mask = 0;
  std::vector<int64_t> in_shape = module::getShape(getInput());
  auto input_dtype = BM1684::getDataType(getInput());
  auto output_shape = module::getShape(getOutput());
  auto offset = module::getI64Array(getOffset());
  auto steps = module::getI64Array(getSteps());
  int num_dims = output_shape.size();
  // melloc
  int *input_shape = new int[MAX_SHAPE_DIMS];
  int *begin_index = new int[MAX_SHAPE_DIMS];
  int *end_index = new int[MAX_SHAPE_DIMS];
  int *strides = new int[MAX_SHAPE_DIMS];

  // slice local gen only support dim 4 in bm1684
  int idx[4] = {(int)out_g_info.n_idx, (int)out_g_info.c_idx,
                (int)out_g_info.h_idx, (int)out_g_info.w_idx};

  int slice[4] = {(int)out_g_info.n_slice, (int)output_shape[1],
                  (int)out_g_info.h_slice, (int)output_shape[3]};

  int in_slice[4] = {(int)in_g_info.n_slice, (int)in_g_info.c_slice,
                     (int)in_g_info.h_slice, (int)in_g_info.w_slice};

  for (int i = 0; i < 4; i++) {
    idx[i] = idx[i] >= 0 ? idx[i] : (input_shape[i] + idx[i]);
    slice[i] = slice[i] >= 0 ? slice[i] : (input_shape[i] + slice[i]);
  }

  for (int i = 0; i < num_dims; ++i) {
    if (offset->at(i) < 0) {
      offset->at(i) += in_shape[i];
    }
    // ====== calculate begin_index and end_index ======
    begin_index[i] =
        (offset->at(i) >= idx[i] && offset->at(i) <= idx[i] + in_slice[i])
            ? (offset->at(i) - idx[i])
            : 0;
    strides[i] = steps->at(i);
    end_index[i] = begin_index[i] + slice[i] * strides[i];
  }
  if (num_dims < 4) {
    for (int i = num_dims; i < 4; i++) {
      input_shape[i] = 1;
      begin_index[i] = 0;
      strides[i] = 1;
      end_index[i] = 1;
    }
    num_dims = 4;
  }
  module::getLocalShape(getInput(), n_step, h_step, input_shape);
  if (input_dtype == DTYPE_FP32 || input_dtype == DTYPE_INT32) {
    BM1684::instance().dl_nodechip_stride_slice_forward_local(
        in_g_info.out_addr, out_g_info.out_addr, input_shape, num_dims,
        begin_mask, end_mask, begin_index, end_index, strides,
        (CMD_ID_NODE *)BM1684::instance()->bdc_node);
  } else if (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8) {
    BM1684::instance().dl_nodechip_stride_slice_forward_local_fix8b(
        in_g_info.out_addr, out_g_info.out_addr, input_shape, num_dims,
        begin_mask, end_mask, begin_index, end_index, strides,
        (CMD_ID_NODE *)BM1684::instance()->bdc_node);
  } else {
    llvm_unreachable("Not Implemented.");
  }
  // release
  delete[] input_shape;
  delete[] begin_index;
  delete[] end_index;
  delete[] strides;
}

uint32_t tpu::SliceOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  GLOBAL_IR_COMMON(stride_slice);
}

int64_t tpu::SliceOp::get_fw_type_bm1684() { return FW_BMNET_STRIDESLICE; }

int32_t tpu::SliceOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  int fw_ir_length = 0;
  IR_PARAM_COMMON(stride_slice);
  // input tensor
  dynamic_push_back_local_tensor(layer_info->ir_tensor_info_v, getInput());
  // output
  dynamic_push_back_local_tensor(layer_info->ir_tensor_info_v, getOutput());
  // compute fw ir info length for crop input and output
  fw_ir_length += 2 * (2 * sizeof(uint32_t)) + sizeof(uint32_t);
  return fw_ir_length;
}
