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
#include "tpu_mlir/Support/Helper/Quant.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include <numeric>

using namespace mlir;
using namespace tpu_mlir::tpu;

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/IR/TpuOpsDialect.cpp.inc"

void TpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tpu_mlir/Dialect/Tpu/IR/TpuAttr.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tpu Operator Definitions.
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/IR/TpuEnum.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tpu_mlir/Dialect/Tpu/IR/TpuAttr.cpp.inc"

#define GET_OP_CLASSES
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.cpp.inc"

namespace tpu_mlir {
static std::map<Operation *, conv_attr_t> group_conv_attrs;
static std::map<Operation *, pool_attr_t> group_pool_attrs;
static std::map<Operation *, deconv_attr_t> group_deconv_attrs;

template <typename OpTy, typename AttrTy>
const AttrTy &getOpParam(OpTy &op, std::map<Operation *, AttrTy> &map) {
  auto op_ = op.getOperation();
  auto iter = map.find(op_);
  if (iter != map.end()) {
    return iter->second;
  }
  map[op_] = op.parseParam();
  return map[op_];
}

const conv_attr_t &getConv2DParam(tpu::Conv2DOp &op) {
  return getOpParam<tpu::Conv2DOp, conv_attr_t>(op, group_conv_attrs);
}

const deconv_attr_t &getDeconvParam(tpu::DeconvOp &op) {
  return getOpParam<tpu::DeconvOp, deconv_attr_t>(op, group_deconv_attrs);
}

const pool_attr_t &getPool2DParam(tpu::Pool2DOp &op) {
  return getOpParam<tpu::Pool2DOp, pool_attr_t>(op, group_pool_attrs);
}
} // namespace tpu_mlir
