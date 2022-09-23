/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace vector {
class CombiningKindAttr;
namespace detail {
struct CombiningKindAttrStorage;
} // namespace detail
class CombiningKindAttr : public ::mlir::Attribute::AttrBase<CombiningKindAttr, ::mlir::Attribute, detail::CombiningKindAttrStorage> {
public:
  using Base::Base;
public:
  static CombiningKindAttr get(::mlir::MLIRContext *context, ::mlir::vector::CombiningKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"kind"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::vector::CombiningKind getValue() const;
};
} // namespace vector
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::vector::CombiningKindAttr)

#endif  // GET_ATTRDEF_CLASSES

