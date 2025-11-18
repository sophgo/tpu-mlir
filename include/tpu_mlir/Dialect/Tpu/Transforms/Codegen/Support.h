//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

namespace tpu_mlir {
namespace tpu {

struct CodegenAttr {
  static constexpr llvm::StringRef CORE_ID = "core_id";
  static constexpr llvm::StringRef SYNC_ALL_BEGIN = "sync_all_begin";
  static constexpr llvm::StringRef SYNC_ALL_END = "sync_all_end";
  static constexpr llvm::StringRef ADDR_JOIN_START = "addr_join_start";
  static constexpr llvm::StringRef ADDR_JOIN_NEXT = "addr_join_next";
};

void DoPatternsForDynamic(ModuleOp m);

} // namespace tpu
} // namespace tpu_mlir
