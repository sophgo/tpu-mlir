//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

// Template for mlir-src-sharder — NOT compiled directly. Shard files include
// this with #define GET_OP_DEFS_<index> to select op definitions by shard.

#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

#include "tpu_mlir/Dialect/Top/IR/TopOps.cpp.inc"
