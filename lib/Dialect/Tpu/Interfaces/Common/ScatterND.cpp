//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

LogicalResult tpu::ScatterNDOp::init(InferenceParameter &p) {
  return success();
}
void tpu::ScatterNDOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ScatterNDOp::inference(InferenceParameter &p) {
  const float *data = p.inputs[0];
  const float *indices = p.inputs[1];
  const float *updates = p.inputs[2];
  float *dst = p.outputs[0];

  auto data_shape = module::getShape(getInputData());
  auto indices_shape = module::getShape(getIndices());
  auto updates_shape = module::getShape(getUpdates());
  auto dtype_size = sizeof(float);
  int r = data_shape.size();
  int q = indices_shape.size();
  int k = indices_shape[q - 1];
  int updates_dims = updates_shape.size();
  assert(updates_dims == q + r - k - 1);
  int updates_elems = 1;
  int slice_elems = 1;
  for (int i = 0; i < q - 1; ++i) {
    assert(updates_shape[i] == indices_shape[i]);
    updates_elems *= indices_shape[i];
  }
  for (int j = k; j < r; ++j) {
    assert(updates_shape[j] == data_shape[j]);
    slice_elems *= updates_shape[j];
  }
  auto data_elems = module::getNumElements(getOutput());

  memcpy(dst, data, data_elems * dtype_size);

  int data_strides[k];
  for (int dim = k - 1; dim >= 0; --dim) {
    if (dim == k - 1) {
      data_strides[dim] = slice_elems;
    } else {
      data_strides[dim] = data_strides[dim + 1] * data_shape[dim + 1];
    }
  }
  int64_t idx = 0;
  if (r == k) {
    // #pragma omp parallel for schedule(static, omp_schedule(updates_elems))
    for (int64_t loc = 0; loc < updates_elems; ++loc) {
      idx = 0;
      for (int64_t i = 0; i < k; ++i) {
        idx += indices[loc * k + i] * data_strides[i];
      }
      dst[idx] = updates[loc];
    }
  } else if (k < r) {
    // #pragma omp parallel for schedule(static, omp_schedule(updates_elems))
    for (int64_t loc = 0; loc < updates_elems; ++loc) {
      idx = 0;
      for (int64_t i = 0; i < k; ++i) {
        idx += indices[loc * k + i] * data_strides[i];
      }
      memcpy(dst + idx, updates + loc * slice_elems, slice_elems * dtype_size);
    }
  }

  return success();
}

mlir::Type tpu::ScatterNDOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  auto op = getOperation();
  if (opd_idx == 1) {
    // indices
    auto opd = op->getOperand(1);
    auto in_op = opd.getDefiningOp();
    if (in_op != nullptr && isa<top::WeightOp, top::NoneOp>(in_op)) {
      return do_nothing(mode);
    }
    auto stype = module::getStorageType(opd);
    if (stype.isIntOrIndex()) {
      return do_nothing(mode);
    }
    mode = TypeCastMode::DO_CAST;
    return Builder(op).getIntegerType(32);
  }
  return type_verify_case_same(op, opd_idx, mode);
}
