/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace tosa {
::llvm::StringRef stringifyTosaProfileEnum(TosaProfileEnum val) {
  switch (val) {
    case TosaProfileEnum::BaseInference: return "bi";
    case TosaProfileEnum::MainInference: return "mi";
    case TosaProfileEnum::MainTraining: return "mt";
    case TosaProfileEnum::Undefined: return "Undefined";
  }
  return "";
}

::std::optional<TosaProfileEnum> symbolizeTosaProfileEnum(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TosaProfileEnum>>(str)
      .Case("bi", TosaProfileEnum::BaseInference)
      .Case("mi", TosaProfileEnum::MainInference)
      .Case("mt", TosaProfileEnum::MainTraining)
      .Case("Undefined", TosaProfileEnum::Undefined)
      .Default(::std::nullopt);
}
::std::optional<TosaProfileEnum> symbolizeTosaProfileEnum(uint32_t value) {
  switch (value) {
  case 0: return TosaProfileEnum::BaseInference;
  case 1: return TosaProfileEnum::MainInference;
  case 2: return TosaProfileEnum::MainTraining;
  case 3: return TosaProfileEnum::Undefined;
  default: return ::std::nullopt;
  }
}

bool TosaProfileEnumAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 3)));
}
TosaProfileEnumAttr TosaProfileEnumAttr::get(::mlir::MLIRContext *context, TosaProfileEnum val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<TosaProfileEnumAttr>();
}
TosaProfileEnum TosaProfileEnumAttr::getValue() const {
  return static_cast<TosaProfileEnum>(::mlir::IntegerAttr::getInt());
}
} // namespace tosa
} // namespace mlir

