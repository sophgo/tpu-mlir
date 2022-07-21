/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyAllReduceOperation(AllReduceOperation val) {
  switch (val) {
    case AllReduceOperation::ADD: return "add";
    case AllReduceOperation::AND: return "and";
    case AllReduceOperation::MAX: return "max";
    case AllReduceOperation::MIN: return "min";
    case AllReduceOperation::MUL: return "mul";
    case AllReduceOperation::OR: return "or";
    case AllReduceOperation::XOR: return "xor";
  }
  return "";
}

::llvm::Optional<AllReduceOperation> symbolizeAllReduceOperation(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<AllReduceOperation>>(str)
      .Case("add", AllReduceOperation::ADD)
      .Case("and", AllReduceOperation::AND)
      .Case("max", AllReduceOperation::MAX)
      .Case("min", AllReduceOperation::MIN)
      .Case("mul", AllReduceOperation::MUL)
      .Case("or", AllReduceOperation::OR)
      .Case("xor", AllReduceOperation::XOR)
      .Default(::llvm::None);
}
::llvm::Optional<AllReduceOperation> symbolizeAllReduceOperation(uint32_t value) {
  switch (value) {
  case 0: return AllReduceOperation::ADD;
  case 1: return AllReduceOperation::AND;
  case 2: return AllReduceOperation::MAX;
  case 3: return AllReduceOperation::MIN;
  case 4: return AllReduceOperation::MUL;
  case 5: return AllReduceOperation::OR;
  case 6: return AllReduceOperation::XOR;
  default: return ::llvm::None;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyDimension(Dimension val) {
  switch (val) {
    case Dimension::x: return "x";
    case Dimension::y: return "y";
    case Dimension::z: return "z";
  }
  return "";
}

::llvm::Optional<Dimension> symbolizeDimension(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<Dimension>>(str)
      .Case("x", Dimension::x)
      .Case("y", Dimension::y)
      .Case("z", Dimension::z)
      .Default(::llvm::None);
}
::llvm::Optional<Dimension> symbolizeDimension(uint32_t value) {
  switch (value) {
  case 0: return Dimension::x;
  case 1: return Dimension::y;
  case 2: return Dimension::z;
  default: return ::llvm::None;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyShuffleMode(ShuffleMode val) {
  switch (val) {
    case ShuffleMode::XOR: return "xor";
    case ShuffleMode::UP: return "up";
    case ShuffleMode::DOWN: return "down";
    case ShuffleMode::IDX: return "idx";
  }
  return "";
}

::llvm::Optional<ShuffleMode> symbolizeShuffleMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ShuffleMode>>(str)
      .Case("xor", ShuffleMode::XOR)
      .Case("up", ShuffleMode::UP)
      .Case("down", ShuffleMode::DOWN)
      .Case("idx", ShuffleMode::IDX)
      .Default(::llvm::None);
}
::llvm::Optional<ShuffleMode> symbolizeShuffleMode(uint32_t value) {
  switch (value) {
  case 0: return ShuffleMode::XOR;
  case 2: return ShuffleMode::UP;
  case 1: return ShuffleMode::DOWN;
  case 3: return ShuffleMode::IDX;
  default: return ::llvm::None;
  }
}

} // namespace gpu
} // namespace mlir

namespace mlir {
namespace gpu {
::llvm::StringRef stringifyMMAElementwiseOp(MMAElementwiseOp val) {
  switch (val) {
    case MMAElementwiseOp::ADDF: return "addf";
    case MMAElementwiseOp::MULF: return "mulf";
    case MMAElementwiseOp::MAXF: return "maxf";
    case MMAElementwiseOp::MINF: return "minf";
    case MMAElementwiseOp::DIVF: return "divf";
  }
  return "";
}

::llvm::Optional<MMAElementwiseOp> symbolizeMMAElementwiseOp(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<MMAElementwiseOp>>(str)
      .Case("addf", MMAElementwiseOp::ADDF)
      .Case("mulf", MMAElementwiseOp::MULF)
      .Case("maxf", MMAElementwiseOp::MAXF)
      .Case("minf", MMAElementwiseOp::MINF)
      .Case("divf", MMAElementwiseOp::DIVF)
      .Default(::llvm::None);
}
::llvm::Optional<MMAElementwiseOp> symbolizeMMAElementwiseOp(uint32_t value) {
  switch (value) {
  case 0: return MMAElementwiseOp::ADDF;
  case 1: return MMAElementwiseOp::MULF;
  case 2: return MMAElementwiseOp::MAXF;
  case 3: return MMAElementwiseOp::MINF;
  case 4: return MMAElementwiseOp::DIVF;
  default: return ::llvm::None;
  }
}

} // namespace gpu
} // namespace mlir

