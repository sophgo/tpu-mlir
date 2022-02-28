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
namespace pdl {
class AttributeType;
class OperationType;
class RangeType;
class TypeType;
class ValueType;
class AttributeType : public ::mlir::Type::TypeBase<AttributeType, ::mlir::pdl::PDLType, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"attribute"};
  }

};
class OperationType : public ::mlir::Type::TypeBase<OperationType, ::mlir::pdl::PDLType, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"operation"};
  }

};
namespace detail {
struct RangeTypeStorage;
} // namespace detail
class RangeType : public ::mlir::Type::TypeBase<RangeType, ::mlir::pdl::PDLType, detail::RangeTypeStorage> {
public:
  using Base::Base;
  using Base::getChecked;
public:
  static RangeType get(Type elementType);
  static RangeType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType);
  static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"range"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getElementType() const;
};
class TypeType : public ::mlir::Type::TypeBase<TypeType, ::mlir::pdl::PDLType, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"type"};
  }

};
class ValueType : public ::mlir::Type::TypeBase<ValueType, ::mlir::pdl::PDLType, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"value"};
  }

};
} // namespace pdl
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::pdl::AttributeType)
DECLARE_EXPLICIT_TYPE_ID(::mlir::pdl::OperationType)
DECLARE_EXPLICIT_TYPE_ID(::mlir::pdl::RangeType)
DECLARE_EXPLICIT_TYPE_ID(::mlir::pdl::TypeType)
DECLARE_EXPLICIT_TYPE_ID(::mlir::pdl::ValueType)

#endif  // GET_TYPEDEF_CLASSES

