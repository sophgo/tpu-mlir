/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace transform {
::llvm::StringRef stringifyMatchInterfaceEnum(MatchInterfaceEnum val) {
  switch (val) {
    case MatchInterfaceEnum::LinalgOp: return "LinalgOp";
    case MatchInterfaceEnum::TilingInterface: return "TilingInterface";
  }
  return "";
}

::llvm::Optional<MatchInterfaceEnum> symbolizeMatchInterfaceEnum(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<MatchInterfaceEnum>>(str)
      .Case("LinalgOp", MatchInterfaceEnum::LinalgOp)
      .Case("TilingInterface", MatchInterfaceEnum::TilingInterface)
      .Default(::llvm::None);
}
::llvm::Optional<MatchInterfaceEnum> symbolizeMatchInterfaceEnum(uint32_t value) {
  switch (value) {
  case 0: return MatchInterfaceEnum::LinalgOp;
  case 1: return MatchInterfaceEnum::TilingInterface;
  default: return ::llvm::None;
  }
}

bool MatchInterfaceEnumAttr::classof(::mlir::Attribute attr) {
  return (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && (((attr.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((attr.cast<::mlir::IntegerAttr>().getInt() == 1)));
}
MatchInterfaceEnumAttr MatchInterfaceEnumAttr::get(::mlir::MLIRContext *context, MatchInterfaceEnum val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return baseAttr.cast<MatchInterfaceEnumAttr>();
}
MatchInterfaceEnum MatchInterfaceEnumAttr::getValue() const {
  return static_cast<MatchInterfaceEnum>(::mlir::IntegerAttr::getInt());
}
} // namespace transform
} // namespace mlir

