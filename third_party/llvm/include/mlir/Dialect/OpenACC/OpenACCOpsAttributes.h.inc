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
namespace acc {
class ClauseDefaultValueAttr;
class ReductionOpAttr;
namespace detail {
struct ClauseDefaultValueAttrStorage;
} // namespace detail
class ClauseDefaultValueAttr : public ::mlir::Attribute::AttrBase<ClauseDefaultValueAttr, ::mlir::Attribute, detail::ClauseDefaultValueAttrStorage> {
public:
  using Base::Base;
public:
  static ClauseDefaultValueAttr get(::mlir::MLIRContext *context, ::mlir::acc::ClauseDefaultValue value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"defaultvalue"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::acc::ClauseDefaultValue getValue() const;
};
namespace detail {
struct ReductionOpAttrStorage;
} // namespace detail
class ReductionOpAttr : public ::mlir::Attribute::AttrBase<ReductionOpAttr, ::mlir::Attribute, detail::ReductionOpAttrStorage> {
public:
  using Base::Base;
public:
  static ReductionOpAttr get(::mlir::MLIRContext *context, ::mlir::acc::ReductionOp value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"reduction_op"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::acc::ReductionOp getValue() const;
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ClauseDefaultValueAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ReductionOpAttr)

#endif  // GET_ATTRDEF_CLASSES

