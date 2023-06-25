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

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include <numeric>


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
namespace tpu {
static std::map<Operation *, conv_attr_t> group_conv_attrs;
static std::map<Operation *, pool_attr_t> group_pool_attrs;
static std::map<Operation *, deconv_attr_t> group_deconv_attrs;
static std::map<Operation *, slice_attr_t> group_slice_attrs;

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

const slice_attr_t &getSliceParam(tpu::SliceOp &op) {
  return getOpParam<tpu::SliceOp, slice_attr_t>(op, group_slice_attrs);
}

RunMode getRunMode(FuncOp func) {
  return func->getAttrOfType<tpu::RunModeAttr>("mode").getValue();
}

RunMode getRunMode(Operation *op) {
  FuncOp funcOp;
  if (isa<FuncOp>(op)) {
    funcOp = cast<FuncOp>(op);
  } else {
    funcOp = cast<FuncOp>(op->getParentOp());
  }
  return getRunMode(funcOp);
}

void IfOp::getSuccessorRegions(std::optional<unsigned> index,
                               ArrayRef<Attribute> operands,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (index) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
}

  // Don't consider the else region if it is empty.
  Region *elseRegion = &this->getElseBranch();
  if (elseRegion->empty())
    elseRegion = nullptr;

  // Otherwise, the successor is dependent on the condition.
  bool condition;
  if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
    condition = condAttr.getValue().isOne();
  } else {
    // If the condition isn't constant, both regions may be executed.
    regions.push_back(RegionSuccessor(&getThenBranch()));
    // If the else region does not exist, it is not a viable successor.
    if (elseRegion)
      regions.push_back(RegionSuccessor(elseRegion));
    return;
  }

  // Add the successor regions using the condition.
  regions.push_back(RegionSuccessor(condition ? &getThenBranch() : elseRegion));
}
} // namespace tpu
} // namespace tpu_mlir
