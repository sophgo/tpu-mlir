//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::SliceOp::getFLOPs() {
  return 0;
}

LogicalResult top::SliceOp::init(InferenceParameter &p) { return success(); }
void top::SliceOp::deinit(InferenceParameter &p) {}

LogicalResult top::SliceOp::inference(InferenceParameter &p) {
  auto out_num_elem = Module::getNumElements(output());
  auto offset_v = Module::getI64Array(offset());
  auto steps_v = Module::getI64Array(steps());
  auto out_shape = Module::getShape(output());
  auto in_shape = Module::getShape(input());
  auto in_dims = in_shape.size();
  auto out_dims = out_shape.size();
  //just support the dims of input & input is equal and slice at one axis now.
  assert(in_dims == out_dims);

  if (in_dims == 2) {
    #pragma omp parallel for schedule(static, omp_schedule(out_num_elem))
    for (int i = 0 ; i < out_shape[0]; i++) {
      for (int j = 0; j < out_shape[1]; j++) {
        memcpy(p.outputs[0] + i * out_shape[1] + j,
                p.inputs[0] + (offset_v->at(0) + i * steps_v->at(0)) * in_shape[1] + (offset_v->at(1) + j * steps_v->at(1)),
                  sizeof(float));
      }
    }
  } else if (in_dims == 3) {
    #pragma omp parallel for schedule(static, omp_schedule(out_num_elem))
    for (int i = 0; i < out_shape[0]; i++) {
      for (int j = 0; j < out_shape[1]; j++) {
        for (int k = 0; k < out_shape[2]; k++) {
          memcpy(p.outputs[0] + i * out_shape[1] * out_shape[2] + j * out_shape[2] + k,
                p.inputs[0] + (offset_v->at(0) + i * steps_v->at(0)) * in_shape[1] * in_shape[2]
                  + (offset_v->at(1) + j * steps_v->at(1)) * in_shape[2]
                   + (offset_v->at(2) + k * steps_v->at(2)),
                sizeof(float));
        }
      }
    }
  } else if (in_dims == 4) {
    #pragma omp parallel for schedule(static, omp_schedule(out_num_elem))
    for (int i = 0; i < out_shape[0]; i++) {
      for (int j = 0; j < out_shape[1]; j++) {
        for (int k = 0; k < out_shape[2]; k++) {
          for (int z = 0; z < out_shape[3]; z++) {
            memcpy(p.outputs[0] + i * out_shape[1] * out_shape[2] * out_shape[3]
                     + j * out_shape[2] * out_shape[3] + k * out_shape[3] + z,
                    p.inputs[0] + (offset_v->at(0) + i * steps_v->at(0)) * in_shape[1] * in_shape[2] * in_shape[3]
                      + (offset_v->at(1) + j * steps_v->at(1)) * in_shape[2] * in_shape[3]
                      + (offset_v->at(2) + k * steps_v->at(2)) * in_shape[3]
                      + (offset_v->at(3) + z * steps_v->at(3)),
                    sizeof(float));
          }
        }
      }
    }
  }

  return success();
}
