/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace linalg {
::llvm::StringRef stringifyBinaryFn(BinaryFn val) {
  switch (val) {
    case BinaryFn::add: return "add";
    case BinaryFn::sub: return "sub";
    case BinaryFn::mul: return "mul";
    case BinaryFn::max_signed: return "max_signed";
    case BinaryFn::min_signed: return "min_signed";
    case BinaryFn::max_unsigned: return "max_unsigned";
    case BinaryFn::min_unsigned: return "min_unsigned";
  }
  return "";
}

::std::optional<BinaryFn> symbolizeBinaryFn(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<BinaryFn>>(str)
      .Case("add", BinaryFn::add)
      .Case("sub", BinaryFn::sub)
      .Case("mul", BinaryFn::mul)
      .Case("max_signed", BinaryFn::max_signed)
      .Case("min_signed", BinaryFn::min_signed)
      .Case("max_unsigned", BinaryFn::max_unsigned)
      .Case("min_unsigned", BinaryFn::min_unsigned)
      .Default(::std::nullopt);
}
::std::optional<BinaryFn> symbolizeBinaryFn(uint32_t value) {
  switch (value) {
  case 0: return BinaryFn::add;
  case 1: return BinaryFn::sub;
  case 2: return BinaryFn::mul;
  case 3: return BinaryFn::max_signed;
  case 4: return BinaryFn::min_signed;
  case 5: return BinaryFn::max_unsigned;
  case 6: return BinaryFn::min_unsigned;
  default: return ::std::nullopt;
  }
}

} // namespace linalg
} // namespace mlir

namespace mlir {
namespace linalg {
::llvm::StringRef stringifyTypeFn(TypeFn val) {
  switch (val) {
    case TypeFn::cast_signed: return "cast_signed";
    case TypeFn::cast_unsigned: return "cast_unsigned";
  }
  return "";
}

::std::optional<TypeFn> symbolizeTypeFn(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TypeFn>>(str)
      .Case("cast_signed", TypeFn::cast_signed)
      .Case("cast_unsigned", TypeFn::cast_unsigned)
      .Default(::std::nullopt);
}
::std::optional<TypeFn> symbolizeTypeFn(uint32_t value) {
  switch (value) {
  case 0: return TypeFn::cast_signed;
  case 1: return TypeFn::cast_unsigned;
  default: return ::std::nullopt;
  }
}

} // namespace linalg
} // namespace mlir

namespace mlir {
namespace linalg {
::llvm::StringRef stringifyUnaryFn(UnaryFn val) {
  switch (val) {
    case UnaryFn::exp: return "exp";
    case UnaryFn::log: return "log";
    case UnaryFn::abs: return "abs";
    case UnaryFn::ceil: return "ceil";
    case UnaryFn::floor: return "floor";
    case UnaryFn::negf: return "negf";
  }
  return "";
}

::std::optional<UnaryFn> symbolizeUnaryFn(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<UnaryFn>>(str)
      .Case("exp", UnaryFn::exp)
      .Case("log", UnaryFn::log)
      .Case("abs", UnaryFn::abs)
      .Case("ceil", UnaryFn::ceil)
      .Case("floor", UnaryFn::floor)
      .Case("negf", UnaryFn::negf)
      .Default(::std::nullopt);
}
::std::optional<UnaryFn> symbolizeUnaryFn(uint32_t value) {
  switch (value) {
  case 0: return UnaryFn::exp;
  case 1: return UnaryFn::log;
  case 2: return UnaryFn::abs;
  case 3: return UnaryFn::ceil;
  case 4: return UnaryFn::floor;
  case 5: return UnaryFn::negf;
  default: return ::std::nullopt;
  }
}

} // namespace linalg
} // namespace mlir

