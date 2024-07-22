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

using namespace tpu_mlir::cuda;

void *py_cuda::getCudaData(mlir::Value v) {
  auto name = module::getName(v).str();
  if (module::isWeight(v)) {
    if (weight_map_.find(name) != weight_map_.end()) {
      return weight_map_[name].get();
    }
    UNREACHABLE_OP("Can't find weight data", v.getDefiningOp());
  } else {
    if (activation_map_.find(name) != activation_map_.end()) {
      return activation_map_[name].get();
    }
    UNREACHABLE_OP("Can't find activation data", v.getDefiningOp());
  }
  return nullptr;
}

data_type_t py_cuda::getCudaType(mlir::Value v) {
  auto stype = module::getStorageType(v);
  if (stype.isUnsignedInteger(8)) {
    return DT_UINT8;
  } else if (stype.isInteger(8)) {
    return DT_INT8;
  } else if (stype.isF32()) {
    return DT_F32;
  } else if (stype.isUnsignedInteger(32)) {
    return DT_UINT32;
  } else if (stype.isInteger(32)) {
    return DT_INT32;
  } else if (stype.isBF16()) {
    return DT_BF16;
  } else if (stype.isF16()) {
    return DT_F16;
  } else if (stype.isUnsignedInteger(16)) {
    return DT_UINT16;
  } else if (stype.isInteger(16)) {
    return DT_INT16;
  }
  v.dump();
  llvm_unreachable("Not Supported");
  return DT_F32;
}

cuda_ptr py_cuda::newCudaData(void *data, size_t num, data_type_t src_type,
                              data_type_t dst_type) {
  if (src_type == dst_type) {
    llvm_unreachable("Same type shouldn't convert");
  }
  void *newData;
  CHECK_CUDA(cudaMalloc(&newData, num * cuda::get_dtype_bytes(dst_type)));
  CHECK_CUDA(convertType(data, newData, num, src_type, dst_type));
  cuda_ptr wrapper(newData);
  return std::move(wrapper);
}

cuda_ptr py_cuda::newCudaData(mlir::Value v, data_type_t dst_type) {
  auto src_type = getCudaType(v);
  auto data = getCudaData(v);
  size_t num = module::getNumElements(v);
  return std::move(newCudaData(data, num, src_type, dst_type));
}
