//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::WhereOp::init(InferenceParameter &p) { return success(); }
void tpu::WhereOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::WhereOp::inference(InferenceParameter &p) {
  const auto num_element = module::getNumElements(getOutput());
  auto out_shape = module::getShape(getOutput());
  auto out_dim = out_shape.size();
  auto in0_shape = shape_expand_dim(module::getShape(getCond()), out_dim);
  const auto cond_element = module::getNumElements(getCond());
  auto x_const = getXIsConst();
  auto y_const = getYIsConst();

  std::vector<int64_t> in0_stride, in1_stride, in2_stride;
  if (x_const && y_const == false) {
    auto const_val = getXConstVal().convertToDouble();
    auto in1_shape = shape_expand_dim(module::getShape(getFbrn()), out_dim);
    const auto fbrn_element = module::getNumElements(getFbrn());
    if (num_element == cond_element && num_element == fbrn_element) {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        p.outputs[0][i] = p.inputs[0][i] ? const_val : p.inputs[2][i];
      }
    } else {
      get_stride(in0_shape, in0_stride);
      get_stride(in1_shape, in1_stride);
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        std::vector<int64_t> list_(out_dim);
        idx_to_list(i, out_shape, list_);
        int64_t cond_idx = list_to_idx(list_, in0_stride);
        int64_t fbrn_idx = list_to_idx(list_, in1_stride);
        p.outputs[0][i] =
            p.inputs[0][cond_idx] ? const_val : p.inputs[2][fbrn_idx];
      }
    }
  } else if (y_const && x_const == false) {
    auto const_val = getYConstVal().convertToDouble();
    auto in1_shape = shape_expand_dim(module::getShape(getTbrn()), out_dim);
    const auto tbrn_element = module::getNumElements(getTbrn());
    if (num_element == cond_element && num_element == tbrn_element) {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        p.outputs[0][i] = p.inputs[0][i] ? p.inputs[1][i] : const_val;
      }
    } else {
      get_stride(in0_shape, in0_stride);
      get_stride(in1_shape, in1_stride);
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        std::vector<int64_t> list_(out_dim);
        idx_to_list(i, out_shape, list_);
        int64_t cond_idx = list_to_idx(list_, in0_stride);
        int64_t tbrn_idx = list_to_idx(list_, in1_stride);
        p.outputs[0][i] =
            p.inputs[0][cond_idx] ? p.inputs[1][tbrn_idx] : const_val;
      }
    }
  } else if (y_const && x_const) {
    auto x_const_val_ = getXConstVal().convertToDouble();
    auto y_const_val_ = getYConstVal().convertToDouble();
    if (num_element == cond_element) {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        p.outputs[0][i] = p.inputs[0][i] ? x_const_val_ : y_const_val_;
      }
    } else {
      get_stride(in0_shape, in0_stride);
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        std::vector<int64_t> list_(out_dim);
        idx_to_list(i, out_shape, list_);
        int64_t cond_idx = list_to_idx(list_, in0_stride);
        p.outputs[0][i] = p.inputs[0][cond_idx] ? x_const_val_ : y_const_val_;
      }
    }
  } else {
    auto in1_shape = shape_expand_dim(module::getShape(getTbrn()), out_dim);
    auto in2_shape = shape_expand_dim(module::getShape(getFbrn()), out_dim);
    const auto tbrn_element = module::getNumElements(getTbrn());
    const auto fbrn_element = module::getNumElements(getFbrn());
    if (num_element == cond_element && num_element == tbrn_element &&
        num_element == fbrn_element) {
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        p.outputs[0][i] = p.inputs[0][i] ? p.inputs[1][i] : p.inputs[2][i];
      }
    } else {
      get_stride(in0_shape, in0_stride);
      get_stride(in1_shape, in1_stride);
      get_stride(in2_shape, in2_stride);
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t i = 0; i < num_element; ++i) {
        std::vector<int64_t> list_(out_dim);
        idx_to_list(i, out_shape, list_);
        int64_t cond_idx = list_to_idx(list_, in0_stride);
        int64_t tbrn_idx = list_to_idx(list_, in1_stride);
        int64_t fbrn_idx = list_to_idx(list_, in2_stride);
        p.outputs[0][i] = p.inputs[0][cond_idx] ? p.inputs[1][tbrn_idx]
                                                : p.inputs[2][fbrn_idx];
      }
    }
  }
  return success();
}

LogicalResult tpu::WhereOp::LocalGenSupport() {
  if (module::isBM1684Family()) {
    llvm::errs() << "1684 Where not support local\n";
    return failure();
  }
  return success();
}

ArrayAttr tpu::WhereOp::getIndexingMaps() {
  MLIRContext *ctx = getContext();
  auto out_shape = module::getShape(getOutput());
  auto num_dims = out_shape.size();
  auto num_operands = getNumOperands();
  // shape < 4 not support
  if (num_dims < 4) {
    return Builder(ctx).getAffineMapArrayAttr({});
  }
  SmallVector<AffineMap> indexingMaps;
  auto out_map = AffineMap::getMultiDimIdentityMap(num_dims, ctx);
  auto empty_map = AffineMap::get(num_dims, 0, ctx);
  for (int i = 0; i < num_operands - 1; i++) {
    if (!isa<top::NoneOp>(getOperand(i).getDefiningOp())) {
      indexingMaps.push_back(out_map);
    } else {
      indexingMaps.push_back(empty_map);
    }
  }
  indexingMaps.push_back(empty_map);
  indexingMaps.push_back(out_map);
  return Builder(ctx).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::WhereOp::support_multi_core() { return false; }
