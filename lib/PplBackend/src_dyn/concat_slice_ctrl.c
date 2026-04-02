//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "concat_slice.c"
#include "ppl_dyn_fw.h"

extern int tpu_core_num();

// global
void dynamic_glb_concat_slice_ctrl(void *ctx, void *param,
                                   global_tensor_spec_t *input_spec,
                                   global_tensor_spec_t *output_spec) {
  int32_t data_type = input_spec[0].dtype;
  output_spec[0].dtype = data_type;
  memcpy(output_spec[0].shape, input_spec[0].shape,
         input_spec[0].dims * sizeof(int));
  output_spec[0].dims = input_spec[0].dims;
  output_spec[0].elem_num = input_spec[0].elem_num;

  // Retrieve axis from the original param (concat_slice_spec_t)
  // The dynamic framework passes param as the generated kernel struct;
  // however the first field is ptr_out and axis is embedded differently.
  // We need to compute outer_size, axis_size_0, axis_size_1, inner_size
  // from input shapes and update the kernel struct fields.

  if (data_type == FW_DTYPE_FP16) {
    tpu_kernel_api_concat_slice_f16_t *param_ =
        (tpu_kernel_api_concat_slice_f16_t *)param;
    param_->ptr_out = output_spec[0].addr;
    param_->ptr_in0 = input_spec[0].addr;
    param_->ptr_in1 = input_spec[1].addr;
    if (param_->core_num > tpu_core_num()) {
      param_->core_num = tpu_core_num();
    }
    concat_slice_f16_entry(param_);
  } else if (data_type == FW_DTYPE_BFP16) {
    tpu_kernel_api_concat_slice_bf16_t *param_ =
        (tpu_kernel_api_concat_slice_bf16_t *)param;
    param_->ptr_out = output_spec[0].addr;
    param_->ptr_in0 = input_spec[0].addr;
    param_->ptr_in1 = input_spec[1].addr;
    if (param_->core_num > tpu_core_num()) {
      param_->core_num = tpu_core_num();
    }
    concat_slice_bf16_entry(param_);
  }
}

REGISTER_PPL_DYN_OP(PPL_FW_CONCAT_SLICE, dynamic_glb_concat_slice_ctrl, 0);
