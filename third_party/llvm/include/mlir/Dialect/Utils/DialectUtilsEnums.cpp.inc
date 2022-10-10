/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace utils {
::llvm::StringRef stringifyIteratorType(IteratorType val) {
  switch (val) {
    case IteratorType::parallel: return "parallel";
    case IteratorType::reduction: return "reduction";
  }
  return "";
}

::llvm::Optional<IteratorType> symbolizeIteratorType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<IteratorType>>(str)
      .Case("parallel", IteratorType::parallel)
      .Case("reduction", IteratorType::reduction)
      .Default(::llvm::None);
}
::llvm::Optional<IteratorType> symbolizeIteratorType(uint32_t value) {
  switch (value) {
  case 0: return IteratorType::parallel;
  case 1: return IteratorType::reduction;
  default: return ::llvm::None;
  }
}

} // namespace utils
} // namespace mlir

