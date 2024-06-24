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

void *py_cuda::getCudaData(Value v) {
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

cudnnDataType_t py_cuda::getCudnnType(Value v) {
  auto stype = module::getStorageType(v);
  if (stype.isUnsignedInteger(8)) {
    return CUDNN_DATA_UINT8;
  } else if (stype.isInteger(8)) {
    return CUDNN_DATA_INT8;
  } else if (stype.isF32()) {
    return CUDNN_DATA_FLOAT;
  } else if (stype.isSignlessInteger(32) || stype.isSignedInteger(32)) {
    return CUDNN_DATA_INT32;
  }
  v.dump();
  llvm_unreachable("Not supported");
  return CUDNN_DATA_FLOAT;
}

size_t py_cuda::getCudnnTypeBytes(cudnnDataType_t type) {
  switch (type) {
  case CUDNN_DATA_FLOAT:
    return sizeof(float);
  case CUDNN_DATA_DOUBLE:
    return sizeof(double);
  case CUDNN_DATA_HALF:
    return sizeof(short);
  case CUDNN_DATA_INT8:
  case CUDNN_DATA_UINT8:
    return 1;
  case CUDNN_DATA_INT32:
    return sizeof(int);
  case CUDNN_DATA_INT8x4:
    return 4;
  default:
    llvm_unreachable("Unknown type");
    return 0;
  }
}

cuda_ptr py_cuda::newCudaData(void *data, size_t num, cudnnDataType_t src_type,
                              cudnnDataType_t dst_type) {
  if (src_type == dst_type) {
    llvm_unreachable("Same type shouldn't convert");
  }
  void *newData;
  CHECK_CUDA(cudaMalloc(&newData, num * getCudnnTypeBytes(dst_type)));
  CHECK_CUDA(cudaTransform(data, newData, num, src_type, dst_type));
  cuda_ptr wrapper(newData);
  return std::move(wrapper);
}

cuda_ptr py_cuda::newCudaData(Value v, cudnnDataType_t dst_type) {
  auto src_type = getCudnnType(v);
  auto data = getCudaData(v);
  size_t num = module::getNumElements(v);
  return std::move(newCudaData(data, num, src_type, dst_type));
}
