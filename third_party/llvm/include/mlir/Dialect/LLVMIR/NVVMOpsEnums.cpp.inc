/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace NVVM {
::llvm::StringRef stringifyMMAFrag(MMAFrag val) {
  switch (val) {
    case MMAFrag::a: return "a";
    case MMAFrag::b: return "b";
    case MMAFrag::c: return "c";
  }
  return "";
}

::llvm::Optional<MMAFrag> symbolizeMMAFrag(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<MMAFrag>>(str)
      .Case("a", MMAFrag::a)
      .Case("b", MMAFrag::b)
      .Case("c", MMAFrag::c)
      .Default(::llvm::None);
}
::llvm::Optional<MMAFrag> symbolizeMMAFrag(uint32_t value) {
  switch (value) {
  case 0: return MMAFrag::a;
  case 1: return MMAFrag::b;
  case 2: return MMAFrag::c;
  default: return ::llvm::None;
  }
}

} // namespace NVVM
} // namespace mlir

namespace mlir {
namespace NVVM {
::llvm::StringRef stringifyMMALayout(MMALayout val) {
  switch (val) {
    case MMALayout::row: return "row";
    case MMALayout::col: return "col";
  }
  return "";
}

::llvm::Optional<MMALayout> symbolizeMMALayout(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<MMALayout>>(str)
      .Case("row", MMALayout::row)
      .Case("col", MMALayout::col)
      .Default(::llvm::None);
}
::llvm::Optional<MMALayout> symbolizeMMALayout(uint32_t value) {
  switch (value) {
  case 0: return MMALayout::row;
  case 1: return MMALayout::col;
  default: return ::llvm::None;
  }
}

} // namespace NVVM
} // namespace mlir

namespace mlir {
namespace NVVM {
::llvm::StringRef stringifyMMATypes(MMATypes val) {
  switch (val) {
    case MMATypes::f16: return "f16";
    case MMATypes::f32: return "f32";
    case MMATypes::tf32: return "tf32";
  }
  return "";
}

::llvm::Optional<MMATypes> symbolizeMMATypes(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<MMATypes>>(str)
      .Case("f16", MMATypes::f16)
      .Case("f32", MMATypes::f32)
      .Case("tf32", MMATypes::tf32)
      .Default(::llvm::None);
}
::llvm::Optional<MMATypes> symbolizeMMATypes(uint32_t value) {
  switch (value) {
  case 0: return MMATypes::f16;
  case 1: return MMATypes::f32;
  case 2: return MMATypes::tf32;
  default: return ::llvm::None;
  }
}

} // namespace NVVM
} // namespace mlir

namespace mlir {
namespace NVVM {
::llvm::StringRef stringifyShflKind(ShflKind val) {
  switch (val) {
    case ShflKind::bfly: return "bfly";
    case ShflKind::up: return "up";
    case ShflKind::down: return "down";
    case ShflKind::idx: return "idx";
  }
  return "";
}

::llvm::Optional<ShflKind> symbolizeShflKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<ShflKind>>(str)
      .Case("bfly", ShflKind::bfly)
      .Case("up", ShflKind::up)
      .Case("down", ShflKind::down)
      .Case("idx", ShflKind::idx)
      .Default(::llvm::None);
}
::llvm::Optional<ShflKind> symbolizeShflKind(uint32_t value) {
  switch (value) {
  case 0: return ShflKind::bfly;
  case 1: return ShflKind::up;
  case 2: return ShflKind::down;
  case 3: return ShflKind::idx;
  default: return ::llvm::None;
  }
}

} // namespace NVVM
} // namespace mlir

