/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace acc {
::llvm::StringRef stringifyClauseDefaultValue(ClauseDefaultValue val) {
  switch (val) {
    case ClauseDefaultValue::Present: return "present";
    case ClauseDefaultValue::None: return "none";
  }
  return "";
}

::llvm::Optional<ClauseDefaultValue> symbolizeClauseDefaultValue(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ClauseDefaultValue>>(str)
      .Case("present", ClauseDefaultValue::Present)
      .Case("none", ClauseDefaultValue::None)
      .Default(::llvm::None);
}
::llvm::Optional<ClauseDefaultValue> symbolizeClauseDefaultValue(uint32_t value) {
  switch (value) {
  case 0: return ClauseDefaultValue::Present;
  case 1: return ClauseDefaultValue::None;
  default: return ::llvm::None;
  }
}

} // namespace acc
} // namespace mlir

namespace mlir {
namespace acc {
::llvm::StringRef stringifyReductionOp(ReductionOp val) {
  switch (val) {
    case ReductionOp::redop_add: return "redop_add";
    case ReductionOp::redop_mul: return "redop_mul";
    case ReductionOp::redop_max: return "redop_max";
    case ReductionOp::redop_min: return "redop_min";
    case ReductionOp::redop_and: return "redop_and";
    case ReductionOp::redop_or: return "redop_or";
    case ReductionOp::redop_xor: return "redop_xor";
    case ReductionOp::redop_leqv: return "redop_leqv";
    case ReductionOp::redop_lneqv: return "redop_lneqv";
    case ReductionOp::redop_land: return "redop_land";
    case ReductionOp::redop_lor: return "redop_lor";
  }
  return "";
}

::llvm::Optional<ReductionOp> symbolizeReductionOp(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ReductionOp>>(str)
      .Case("redop_add", ReductionOp::redop_add)
      .Case("redop_mul", ReductionOp::redop_mul)
      .Case("redop_max", ReductionOp::redop_max)
      .Case("redop_min", ReductionOp::redop_min)
      .Case("redop_and", ReductionOp::redop_and)
      .Case("redop_or", ReductionOp::redop_or)
      .Case("redop_xor", ReductionOp::redop_xor)
      .Case("redop_leqv", ReductionOp::redop_leqv)
      .Case("redop_lneqv", ReductionOp::redop_lneqv)
      .Case("redop_land", ReductionOp::redop_land)
      .Case("redop_lor", ReductionOp::redop_lor)
      .Default(::llvm::None);
}
::llvm::Optional<ReductionOp> symbolizeReductionOp(uint32_t value) {
  switch (value) {
  case 0: return ReductionOp::redop_add;
  case 1: return ReductionOp::redop_mul;
  case 2: return ReductionOp::redop_max;
  case 3: return ReductionOp::redop_min;
  case 4: return ReductionOp::redop_and;
  case 5: return ReductionOp::redop_or;
  case 6: return ReductionOp::redop_xor;
  case 7: return ReductionOp::redop_leqv;
  case 8: return ReductionOp::redop_lneqv;
  case 9: return ReductionOp::redop_land;
  case 10: return ReductionOp::redop_lor;
  default: return ::llvm::None;
  }
}

} // namespace acc
} // namespace mlir

