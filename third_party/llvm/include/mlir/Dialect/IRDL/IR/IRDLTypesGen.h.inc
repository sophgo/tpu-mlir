/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace irdl {
class AttributeType;
class AttributeType : public ::mlir::Type::TypeBase<AttributeType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"attribute"};
  }

};
} // namespace irdl
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::irdl::AttributeType)

#endif  // GET_TYPEDEF_CLASSES

