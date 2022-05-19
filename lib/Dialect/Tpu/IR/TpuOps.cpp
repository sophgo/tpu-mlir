#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Quant.h"
#include <numeric>

using namespace mlir;
using namespace sophgo::tpu;

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
#include "sophgo/Dialect/Tpu/IR/TpuOpsDialect.cpp.inc"
#include "sophgo/Dialect/Tpu/IR/TpuAttr.cpp.inc"

void TpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "sophgo/Dialect/Tpu/IR/TpuOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tpu Operator Definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "sophgo/Dialect/Tpu/IR/TpuOps.cpp.inc"
