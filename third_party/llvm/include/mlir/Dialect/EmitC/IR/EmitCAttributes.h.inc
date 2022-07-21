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
namespace emitc {
class OpaqueAttr;
namespace detail {
struct OpaqueAttrStorage;
} // namespace detail
class OpaqueAttr : public ::mlir::Attribute::AttrBase<OpaqueAttr, ::mlir::Attribute, detail::OpaqueAttrStorage> {
public:
  using Base::Base;
public:
  static OpaqueAttr get(::mlir::MLIRContext *context, ::llvm::StringRef value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"opaque"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::StringRef getValue() const;
};
} // namespace emitc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::emitc::OpaqueAttr)

#endif  // GET_ATTRDEF_CLASSES

