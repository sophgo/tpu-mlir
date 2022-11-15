//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

template<typename T>
static int remove_value(std::vector<T> & v, T value) {
  int idx = 0;
  for(auto iter = v.begin(); iter != v.end(); iter++, idx++) {
    if (*iter == value) {
      v.erase(iter);
      return idx;
    }
  }
  return -1;
}

void tpu::SliceOp::parseParam(std::vector<int64_t> &is_4,
                    std::vector<int64_t> &os_4, std::vector<int> &offset_4,
                    std::vector<int> &step_4, bool &fusible) {
  auto is = input().getType().cast<RankedTensorType>().getShape().vec();
  auto os = output().getType().cast<RankedTensorType>().getShape().vec();
  int num_dims = is.size();
  auto crop_offset = Module::getI64Array(offset());
  auto crop_steps = Module::getI64Array(steps());

  assert(crop_offset->size() == crop_steps->size());
  assert(is.size() == crop_steps->size());
  assert(is.size() == os.size());

  if (num_dims > 4) {
    // remove dims = 1
    while (num_dims > 4) {
      int idx = remove_value<int64_t>(is, 1);
      if (idx < 0) {
        break;
      }
      crop_offset->erase(crop_offset->begin() + idx);
      crop_steps->erase(crop_steps->begin() + idx);
      os.erase(os.begin() + idx);
      num_dims--;
    }
    // remove continous
    while (num_dims > 4) {
      bool done = false;
      for (int i = 0; i < num_dims - 1; i++) {
        if (is[i] == os[i] && is[i + 1] == os[i + 1]) {
          is[i] *= is[i + 1];
          os[i] *= os[i + 1];
          is.erase(is.begin() + i + 1);
          os.erase(os.begin() + i + 1);
          crop_steps->erase(crop_steps->begin() + i + 1);
          crop_offset->erase(crop_offset->begin() + i + 1);
          num_dims--;
          done = true;
          break;
        }
      }
      if (done == false) {
        break;
      }
    }
    if (num_dims > 4) {
      llvm_unreachable("permute shape not support");
    }
  }
  is_4 = {1, 1, 1, 1};
  os_4 = {1, 1, 1, 1};
  step_4 = {1, 1, 1, 1};
  offset_4 = {0, 0, 0, 0};
  std::vector<int>real_axes;
  bool no_step = true;
  for (int idx = 0; idx < num_dims; idx++) {
    is_4[idx] = is[idx];
    os_4[idx] = os[idx];
    step_4[idx] = crop_steps->at(idx);
    offset_4[idx] = crop_offset->at(idx);
    if (no_step && crop_steps->at(idx) != 1) {
      no_step = false;
    }
    if (is_4[idx] != os_4[idx]) {
      real_axes.push_back(idx);
    }
  }
  fusible = false;
  if (no_step && real_axes.size() == 1) {
    int axis = real_axes[0];
    int outer_dim = std::accumulate(is_4.begin(), is_4.begin() + axis, 1, std::multiplies<int64_t>());
    if (outer_dim == 1) {
      fusible = true;
    }
  }
}

LogicalResult tpu::SliceOp::init(InferenceParameter &p) { return success(); }
void tpu::SliceOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SliceOp::inference(InferenceParameter &p) {
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
