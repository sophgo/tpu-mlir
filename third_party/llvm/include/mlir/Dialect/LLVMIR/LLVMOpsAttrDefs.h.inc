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
namespace LLVM {
class FMFAttr;
class LinkageAttr;
class LoopOptionsAttr;
namespace detail {
struct FMFAttrStorage;
} // namespace detail
class FMFAttr : public ::mlir::Attribute::AttrBase<FMFAttr, ::mlir::Attribute, detail::FMFAttrStorage> {
public:
  using Base::Base;
public:
  static FMFAttr get(::mlir::MLIRContext *context, FastmathFlags flags);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"fastmath"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  FastmathFlags getFlags() const;
};
namespace detail {
struct LinkageAttrStorage;
} // namespace detail
class LinkageAttr : public ::mlir::Attribute::AttrBase<LinkageAttr, ::mlir::Attribute, detail::LinkageAttrStorage> {
public:
  using Base::Base;
public:
  static LinkageAttr get(::mlir::MLIRContext *context, linkage::Linkage linkage);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"linkage"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  linkage::Linkage getLinkage() const;
};
namespace detail {
struct LoopOptionsAttrStorage;
} // namespace detail
class LoopOptionsAttr : public ::mlir::Attribute::AttrBase<LoopOptionsAttr, ::mlir::Attribute, detail::LoopOptionsAttrStorage> {
public:
  using Base::Base;
  using OptionValuePair = std::pair<LoopOptionCase, int64_t>;
  using OptionsArray = ArrayRef<std::pair<LoopOptionCase, int64_t>>;
  Optional<bool> disableUnroll();
  Optional<bool> disableLICM();
  Optional<int64_t> interleaveCount();
public:
  static LoopOptionsAttr get(::mlir::MLIRContext *context, ArrayRef<std::pair<LoopOptionCase, int64_t>> sortedOptions);
  static LoopOptionsAttr get(::mlir::MLIRContext *context, LoopOptionsAttrBuilder &optionBuilders);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"loopopts"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<std::pair<LoopOptionCase, int64_t>> getOptions() const;
};
} // namespace LLVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::LLVM::FMFAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::LLVM::LinkageAttr)
DECLARE_EXPLICIT_TYPE_ID(::mlir::LLVM::LoopOptionsAttr)

#endif  // GET_ATTRDEF_CLASSES

