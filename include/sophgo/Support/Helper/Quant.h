#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace sophgo {
namespace helper {
struct Quant {
  struct Type {
    static constexpr llvm::StringRef INT8 = "INT8";
    static constexpr llvm::StringRef BF16 = "BF16";
    static constexpr llvm::StringRef FP16 = "FP16";
    static constexpr llvm::StringRef FP32 = "FP32";
  };
};
} // namespace helper
} // namespace sophgo
