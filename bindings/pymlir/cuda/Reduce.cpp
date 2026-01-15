//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

// Enum for reduction modes
enum ReductionMode {
    REDUCE_SUM = 0,
    REDUCE_MEAN,
    REDUCE_MAX,
    REDUCE_MIN,
    REDUCE_L2_NORM,
    REDUCE_L1_NORM,
    REDUCE_PROD,     // Product
    REDUCE_VAR,      // Variance
    REDUCE_STD,      // Standard deviation
    REDUCE_ANY,      // Logical OR (for boolean)
    REDUCE_ALL       // Logical AND (for boolean)
};

enum ReductionMode getReductionMode(std::string type_val) {
    if (type_val == "ReduceSum") return REDUCE_SUM;
    else if (type_val == "ReduceMean") return REDUCE_MEAN;
    else if (type_val == "ReduceMax") return REDUCE_MAX;
    else if (type_val == "ReduceMin") return REDUCE_MIN;
    else if (type_val == "ReduceL2") return REDUCE_L2_NORM;
    else if (type_val == "ReduceL1") return REDUCE_L1_NORM;
    else if (type_val == "ReduceProd") return REDUCE_PROD;
    else if (type_val == "VAR") return REDUCE_VAR; // not supported yet
    else if (type_val == "STD") return REDUCE_STD;
    else if (type_val == "ANY") return REDUCE_ANY;
    else if (type_val == "ALL") return REDUCE_ALL;
    else {
        throw std::invalid_argument("Unsupported reduction type: " + type_val);
    }
}

void py_cuda::cudaReduceOp(tpu::ReduceOp op) {
  std::string type_val = std::string(op.getMode().str());
  std::vector<int32_t> axes;
  auto axes_ = module::getI64Array(op.getAxes());
  axes.assign(axes_->begin(), axes_->end());
  auto input_shape = std::vector<int64_t>(module::getShape(op.getInput()));
  int num_dims = input_shape.size();
  if (num_dims > 8) {
    UNREACHABLE_OP("ReduceOp only support up to 8D tensor", op);
  }
  int num_axes = axes.size();
  for (auto a:axes) {
    if (a < 0) {
      a += num_dims;
    }
  }
  std::vector<int32_t> reduce_mask(8, 0);
  for (int i = 0; i < num_axes; i++) {
    reduce_mask[axes[i]] = 1;
  }
  std::vector<int32_t> _in_shape(8,1);
  for (int i=0;i<num_dims;i++) {
    _in_shape[i] = static_cast<int32_t>(input_shape[i]);
  }
  if (module::getStorageType(op.getInput()).isF32()) {
    auto in_f32 = getCudaData(op.getInput());
    auto out_f32 = getCudaData(op.getOutput());
    cuda::bmReduce(in_f32, out_f32, num_dims, (void *)_in_shape.data(), (void *)reduce_mask.data(), (int)getReductionMode(type_val));
  } {
    auto in_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    auto out_f32 = newCudaData(op.getOutput(), cuda::DT_F32);
    cuda::bmReduce(in_f32.get(), out_f32.get(), num_dims, (void *)_in_shape.data(), (void *)reduce_mask.data(), (int)getReductionMode(type_val));
    cuda::convertType(out_f32.get(), getCudaData(op.getOutput()), module::getNumElements(op.getOutput()), cuda::DT_F32,
                      getCudaType(op.getOutput()));
    in_f32.reset();
    out_f32.reset();
  }
}

void py_cuda::cudaReduceOp(top::ReduceOp op) {
  std::string type_val = std::string(op.getMode().str());
  std::vector<int32_t> axes;
  auto axes_ = module::getI64Array(op.getAxes());
  axes.assign(axes_->begin(), axes_->end());
  auto input_shape = std::vector<int64_t>(module::getShape(op.getInput()));
  int num_dims = input_shape.size();
  if (num_dims > 8) {
    UNREACHABLE_OP("ReduceOp only support up to 8D tensor", op);
  }
  int num_axes = axes.size();
  auto in_f32 = getCudaData(op.getInput());
  auto out_f32 = getCudaData(op.getOutput());
  for (auto a:axes) {
    if (a < 0) {
      a += num_dims;
    }
  }
  std::vector<int32_t> reduce_mask(8, 0);
  for (int i = 0; i < num_axes; i++) {
    reduce_mask[axes[i]] = 1;
  }
  std::vector<int32_t> _in_shape(8,1);
  for (int i=0;i<num_dims;i++) {
    _in_shape[i] = static_cast<int32_t>(input_shape[i]);
  }

  cuda::bmReduce(in_f32, out_f32, num_dims, (void *)_in_shape.data(), (void *)reduce_mask.data(), (int)getReductionMode(type_val));
}
