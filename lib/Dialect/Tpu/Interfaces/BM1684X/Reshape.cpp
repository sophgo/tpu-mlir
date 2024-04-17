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

void tpu::ReshapeOp::codegen_global_bm1684x() {
  // do nothing
}

// ======================================
// LocalGenInterface
// ======================================
int64_t tpu::ReshapeOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::ReshapeOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                           int64_t h_step, int64_t d_step,
                                           int64_t w_step,
                                           group_type_t group_type,
                                           local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  if (input_spec->at(0).addr == output_spec->at(0).addr) {
    return;
  }

  auto shape = module::getShape(getOutput());
  reshape_spec_t spec;
  spec.dims = shape.size();
  for (size_t i = 0; i < shape.size(); ++i) {
    spec.shape[i] = shape[i];
  }
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  spec.eu_align = gi.eu_align;
  BM168x::call_local_func("backend_api_reshape_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::ReshapeOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(reshape_spec_t);
  reshape_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  // auto out_shape = module::getShape(getOutput());
  auto shape = module::getI64Array(getShape());
  spec.dims = shape->size();
  for (int i = 0; i < spec.dims; i++) {
    spec.shape[i] = shape->at(i);
  }

  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ReshapeOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(reshape_spec_t);
  reshape_spec_t spec;
#define FLAG_VAL 0x0a0a0a0a
#define MLIR_RESHAPE_FLAG 0x08000000
  memset(&spec, 0, sizeof(spec));
  memset(spec.shape, FLAG_VAL, sizeof(spec.shape));
  auto out_shape = module::getShape(getOutput());
  auto in_shape = module::getShape(getInput());
  spec.dims = out_shape.size() | MLIR_RESHAPE_FLAG;

  /* because reshapeop's 2nd tensor(shape tensor) is the same
   as the output tensor, it's infomation is useless, it need to
   inference the axis indicator accoring to input/output tensor,
   pls refer to mxnet/src/operator/tensor/matrix_op.cc:L174
   0: copy this dimension from the input to the output shape, Example::
      - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
      - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)
   -1: infers the dimension of the output shape by using the remainder
    of the input dimensions keeping the size of the new array same as
    that of the input array. At most one dimension of shape can be -1. Example::
    - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
    - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
    - input shape = (2,3,4), shape=(-1,), output shape = (24,)*/
  if (out_shape.size() == 1) {
    // case: input shape = (2,3,4), output shape = (24)
    spec.shape[0] = -1;
  } else {
    for (int32_t i = 0; i < out_shape.size(); i++) {
      for (int32_t j = 0; j < in_shape.size(); j++) {
        if (out_shape[i] == in_shape[j]) {
          spec.shape[i] = j;
          break;
        }
      }

      if (spec.shape[i] == FLAG_VAL) {
        /* case: input shape = (4,4,4), output shape = (2,2,16)
          for to distinguish between real shape and axis indicator
          with MLIR_RESHAPE_FLAG */
        spec.shape[i] = out_shape[i] | MLIR_RESHAPE_FLAG;
      }
    }
  }
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::ReshapeOp::get_fw_type_bm1684x() { return FW_BMNET_RESHAPE; }
