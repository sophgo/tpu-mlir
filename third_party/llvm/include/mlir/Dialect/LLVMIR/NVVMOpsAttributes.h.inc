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
namespace NVVM {
class MMAB1OpAttr;
class MMAFragAttr;
class MMAIntOverflowAttr;
class MMALayoutAttr;
class MMATypesAttr;
class MMAShapeAttr;
class ShflKindAttr;
namespace detail {
struct MMAB1OpAttrStorage;
} // namespace detail
class MMAB1OpAttr : public ::mlir::Attribute::AttrBase<MMAB1OpAttr, ::mlir::Attribute, detail::MMAB1OpAttrStorage> {
public:
  using Base::Base;
public:
  static MMAB1OpAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::MMAB1Op value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_b1op"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::MMAB1Op getValue() const;
};
namespace detail {
struct MMAFragAttrStorage;
} // namespace detail
class MMAFragAttr : public ::mlir::Attribute::AttrBase<MMAFragAttr, ::mlir::Attribute, detail::MMAFragAttrStorage> {
public:
  using Base::Base;
public:
  static MMAFragAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::MMAFrag value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_frag"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::MMAFrag getValue() const;
};
namespace detail {
struct MMAIntOverflowAttrStorage;
} // namespace detail
class MMAIntOverflowAttr : public ::mlir::Attribute::AttrBase<MMAIntOverflowAttr, ::mlir::Attribute, detail::MMAIntOverflowAttrStorage> {
public:
  using Base::Base;
public:
  static MMAIntOverflowAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::MMAIntOverflow value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_int_overflow"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::MMAIntOverflow getValue() const;
};
namespace detail {
struct MMALayoutAttrStorage;
} // namespace detail
class MMALayoutAttr : public ::mlir::Attribute::AttrBase<MMALayoutAttr, ::mlir::Attribute, detail::MMALayoutAttrStorage> {
public:
  using Base::Base;
public:
  static MMALayoutAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::MMALayout value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_layout"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::MMALayout getValue() const;
};
namespace detail {
struct MMATypesAttrStorage;
} // namespace detail
class MMATypesAttr : public ::mlir::Attribute::AttrBase<MMATypesAttr, ::mlir::Attribute, detail::MMATypesAttrStorage> {
public:
  using Base::Base;
public:
  static MMATypesAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::MMATypes value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::MMATypes getValue() const;
};
namespace detail {
struct MMAShapeAttrStorage;
} // namespace detail
class MMAShapeAttr : public ::mlir::Attribute::AttrBase<MMAShapeAttr, ::mlir::Attribute, detail::MMAShapeAttrStorage> {
public:
  using Base::Base;
public:
  static MMAShapeAttr get(::mlir::MLIRContext *context, int m, int n, int k);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shape"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int getM() const;
  int getN() const;
  int getK() const;
};
namespace detail {
struct ShflKindAttrStorage;
} // namespace detail
class ShflKindAttr : public ::mlir::Attribute::AttrBase<ShflKindAttr, ::mlir::Attribute, detail::ShflKindAttrStorage> {
public:
  using Base::Base;
public:
  static ShflKindAttr get(::mlir::MLIRContext *context, ::mlir::NVVM::ShflKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shfl_kind"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::NVVM::ShflKind getValue() const;
};
} // namespace NVVM
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMAB1OpAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMAFragAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMAIntOverflowAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMALayoutAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMATypesAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMAShapeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::ShflKindAttr)

#endif  // GET_ATTRDEF_CLASSES

