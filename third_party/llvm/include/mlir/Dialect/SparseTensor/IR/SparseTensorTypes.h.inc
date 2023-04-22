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
namespace sparse_tensor {
class StorageSpecifierType;
namespace detail {
struct StorageSpecifierTypeStorage;
} // namespace detail
class StorageSpecifierType : public ::mlir::Type::TypeBase<StorageSpecifierType, ::mlir::Type, detail::StorageSpecifierTypeStorage> {
public:
  using Base::Base;
  static StorageSpecifierType get(::mlir::MLIRContext *context, SparseTensorEncodingAttr encoding);
  static StorageSpecifierType get(SparseTensorEncodingAttr encoding);
  static StorageSpecifierType get(Type type);
  static StorageSpecifierType get(Value tensor);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"storage_specifier"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::sparse_tensor::SparseTensorEncodingAttr getEncoding() const;
};
} // namespace sparse_tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::StorageSpecifierType)

#endif  // GET_TYPEDEF_CLASSES

