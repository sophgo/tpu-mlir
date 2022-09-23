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
namespace shape {
class ShapeType;
class SizeType;
class ValueShapeType;
class WitnessType;
class ShapeType : public ::mlir::Type::TypeBase<ShapeType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shape"};
  }

};
class SizeType : public ::mlir::Type::TypeBase<SizeType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"size"};
  }

};
class ValueShapeType : public ::mlir::Type::TypeBase<ValueShapeType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"value_shape"};
  }

};
class WitnessType : public ::mlir::Type::TypeBase<WitnessType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
public:
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"witness"};
  }

};
} // namespace shape
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::shape::ShapeType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::shape::SizeType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::shape::ValueShapeType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::shape::WitnessType)

#endif  // GET_TYPEDEF_CLASSES

