//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/DynCompileCommon.hpp"
using namespace tpu_mlir::backend;

void tpu::ReshapeOp::codegen_global_bm1684x() {
  // do nothing
}

//dynamic codegen
int64_t tpu::ReshapeOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer) return sizeof(reshape_spec_t);
  reshape_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  auto out_shape = module::getShape(getOutput());
  spec.dims = out_shape.size();
  for (int i = 0; i < spec.dims; i++) {
    spec.shape[i] = out_shape[i];
  }

  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ReshapeOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) return sizeof(reshape_spec_t);
  reshape_spec_t spec;
#define FLAG_VAL 0x0a0a0a0a
#define MLIR_RESHAPE_FLAG 0x08000000
  memset(&spec, 0, sizeof(spec));
  memset(spec.shape, FLAG_VAL, sizeof(spec.shape));
  auto out_shape = module::getShape(getOutput());
  auto in_shape = module::getShape(getInput());
  spec.dims = out_shape.size() | MLIR_RESHAPE_FLAG;
  std::vector<int32_t> bitmap(in_shape.size(), 0);

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
    //case: input shape = (2,3,4), output shape = (24)
    spec.shape[0] = -1;
  } else {
    for (int32_t i = 0; i < out_shape.size(); i++) {
      if (i < in_shape.size() && out_shape.size() == in_shape.size()
          && out_shape[i] == in_shape[i] && !bitmap[i]) {
        spec.shape[i] = 0; //remain the same
        bitmap[i] = 1;
      } else {
        for (int32_t j = 0; j < in_shape.size(); j++) {
          if (out_shape[i] == in_shape[j] && !bitmap[j]) {
            spec.shape[i] = j;
            bitmap[j] = 1;
            break;
          } else if (out_shape[i] == 1) {
            //case: input shape = (2,3,4), output shape = (2, 12, 1)
            spec.shape[i] = 1 | MLIR_RESHAPE_FLAG;
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

    int32_t flag = 0;
    //config the remain axis
    for (int32_t i = 0; i < out_shape.size(); i++) {
      if (spec.shape[i] == FLAG_VAL && !flag) {
        //case: input shape = (4,4,4), output shape = (2,2,16) -> (2,2,-1)
        spec.shape[i] = -1;
        flag = 1;
      } else if (spec.shape[i] == FLAG_VAL) {
        //other case, need to improve
        assert(0);
      }
    }
  }
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::ReshapeOp::get_layer_type() {
  return FW_BMNET_RESHAPE;
}
