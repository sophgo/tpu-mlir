#ifndef TPUDIALECT_H
#define TPUDIALECT_H

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include <cstdint>


#include "TPUOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TPUOps.h.inc"

#endif
