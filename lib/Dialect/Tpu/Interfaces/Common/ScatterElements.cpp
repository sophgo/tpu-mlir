//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::ScatterElementsOp::init(InferenceParameter &p) {
  return success();
}
void tpu::ScatterElementsOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ScatterElementsOp::inference(InferenceParameter &p) {
  const float *input = p.inputs[0];
  const float *indices = p.inputs[1];
  const float *updates = p.inputs[2];
  float *output = p.outputs[0];

  const auto input_shape = module::getShape(getInput());
  const auto indices_shape = module::getShape(getIndices());
  const auto updates_shape = module::getShape(getUpdates());
  const int r = input_shape.size();
  const int _axis = getAxis();
  const int replace_add = getReduction();
  const int axis = _axis < 0 ? _axis + r : _axis;
  assert(0 <= axis && axis < r);

  for (int i = 0; i < r; ++i) {
    if (i != axis) {
      assert(input_shape[i] >= indices_shape[i]);
      assert(input_shape[i] >= updates_shape[i]);
    } else {
      assert(indices_shape[i] == updates_shape[i]);
    }
  }

  auto all_num_elem = module::getNumElements(getInput());
  auto upd_num_elem = module::getNumElements(getUpdates());
  memcpy(output, input, all_num_elem * sizeof(float));
  const int64_t s = input_shape[axis];

  std::vector<int64_t> in_stride;
  get_stride(input_shape, in_stride);
#pragma omp parallel for schedule(static, omp_schedule(upd_num_elem))
  for (int n = 0; n < upd_num_elem; ++n) {
    std::vector<int64_t> list_(r);
    idx_to_list(n, updates_shape, list_);
    const int64_t p = (int64_t)indices[n];
    assert(-s <= p && p < s);
    list_[axis] = p;
    int64_t in_idx = list_to_idx(list_, in_stride);
    if (replace_add) {
      output[in_idx] += updates[n];
    } else {
      output[in_idx] = updates[n];
    }
  }

  return success();
}

mlir::Type tpu::ScatterElementsOp::type_verify(uint64_t opd_idx,
                                               TypeCastMode &mode) {
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
    auto bitwidth = 32;
    return Builder(op).getIntegerType(bitwidth);
  }
  return type_verify_case_same(op, opd_idx, mode);
}

bool tpu::ScatterElementsOp::support_multi_core() { return false; }

ArrayAttr tpu::ScatterElementsOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  MLIRContext *ctx = getContext();
  auto num_dims = shape.size();

  int axis = (int)getAxis();
  axis = axis < 0 ? axis + num_dims : axis;
  auto nc_can_split = (bool)getNcCanSplit();
  if (axis != num_dims - 1 || not nc_can_split || num_dims <= 1) {
    return Builder(ctx).getAffineMapArrayAttr({});
  }
  auto empty_map = AffineMap::get(1, 0, ctx);
  auto input_map = AffineMap::getMultiDimIdentityMap(num_dims - 1, ctx);
  SmallVector<AffineMap> indexingMaps{input_map, input_map, input_map,
                                      empty_map, empty_map, input_map};
  return Builder(ctx).getAffineMapArrayAttr(indexingMaps);
}