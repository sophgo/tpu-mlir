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
namespace arith {
class FastMathFlagsAttr;
namespace detail {
struct FastMathFlagsAttrStorage;
} // namespace detail
class FastMathFlagsAttr : public ::mlir::Attribute::AttrBase<FastMathFlagsAttr, ::mlir::Attribute, detail::FastMathFlagsAttrStorage> {
public:
  using Base::Base;
  static FastMathFlagsAttr get(::mlir::MLIRContext *context, ::mlir::arith::FastMathFlags value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"fastmath"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::arith::FastMathFlags getValue() const;
};
} // namespace arith
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::arith::FastMathFlagsAttr)

#endif  // GET_ATTRDEF_CLASSES

