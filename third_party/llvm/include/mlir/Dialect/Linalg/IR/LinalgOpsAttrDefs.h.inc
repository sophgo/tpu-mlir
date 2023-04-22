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
namespace linalg {
class BinaryFnAttr;
class IteratorTypeAttr;
class TypeFnAttr;
class UnaryFnAttr;
namespace detail {
struct BinaryFnAttrStorage;
} // namespace detail
class BinaryFnAttr : public ::mlir::Attribute::AttrBase<BinaryFnAttr, ::mlir::Attribute, detail::BinaryFnAttrStorage> {
public:
  using Base::Base;
  static BinaryFnAttr get(::mlir::MLIRContext *context, ::mlir::linalg::BinaryFn value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"binary_fn"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::linalg::BinaryFn getValue() const;
};
namespace detail {
struct IteratorTypeAttrStorage;
} // namespace detail
class IteratorTypeAttr : public ::mlir::Attribute::AttrBase<IteratorTypeAttr, ::mlir::Attribute, detail::IteratorTypeAttrStorage> {
public:
  using Base::Base;
  static IteratorTypeAttr get(::mlir::MLIRContext *context, ::mlir::utils::IteratorType value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"iterator_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::utils::IteratorType getValue() const;
};
namespace detail {
struct TypeFnAttrStorage;
} // namespace detail
class TypeFnAttr : public ::mlir::Attribute::AttrBase<TypeFnAttr, ::mlir::Attribute, detail::TypeFnAttrStorage> {
public:
  using Base::Base;
  static TypeFnAttr get(::mlir::MLIRContext *context, ::mlir::linalg::TypeFn value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"type_fn"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::linalg::TypeFn getValue() const;
};
namespace detail {
struct UnaryFnAttrStorage;
} // namespace detail
class UnaryFnAttr : public ::mlir::Attribute::AttrBase<UnaryFnAttr, ::mlir::Attribute, detail::UnaryFnAttrStorage> {
public:
  using Base::Base;
  static UnaryFnAttr get(::mlir::MLIRContext *context, ::mlir::linalg::UnaryFn value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"unary_fn"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::linalg::UnaryFn getValue() const;
};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::BinaryFnAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::IteratorTypeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::TypeFnAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::UnaryFnAttr)

#endif  // GET_ATTRDEF_CLASSES

