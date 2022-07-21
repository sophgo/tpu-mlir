/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::sparse_tensor::SparseTensorEncodingAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  if (mnemonic == ::mlir::sparse_tensor::SparseTensorEncodingAttr::getMnemonic()) {
    value = ::mlir::sparse_tensor::SparseTensorEncodingAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  return {};
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::sparse_tensor::SparseTensorEncodingAttr>([&](auto t) {
      printer << ::mlir::sparse_tensor::SparseTensorEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace sparse_tensor {
namespace detail {
struct SparseTensorEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType>, AffineMap, unsigned, unsigned>;
  SparseTensorEncodingAttrStorage(::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dimLevelType, AffineMap dimOrdering, unsigned pointerBitWidth, unsigned indexBitWidth) : ::mlir::AttributeStorage(), dimLevelType(dimLevelType), dimOrdering(dimOrdering), pointerBitWidth(pointerBitWidth), indexBitWidth(indexBitWidth) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (dimLevelType == std::get<0>(tblgenKey)) && (dimOrdering == std::get<1>(tblgenKey)) && (pointerBitWidth == std::get<2>(tblgenKey)) && (indexBitWidth == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static SparseTensorEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto dimLevelType = std::get<0>(tblgenKey);
    auto dimOrdering = std::get<1>(tblgenKey);
    auto pointerBitWidth = std::get<2>(tblgenKey);
    auto indexBitWidth = std::get<3>(tblgenKey);
    dimLevelType = allocator.copyInto(dimLevelType);
    return new (allocator.allocate<SparseTensorEncodingAttrStorage>()) SparseTensorEncodingAttrStorage(dimLevelType, dimOrdering, pointerBitWidth, indexBitWidth);
  }

  ::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dimLevelType;
  AffineMap dimOrdering;
  unsigned pointerBitWidth;
  unsigned indexBitWidth;
};
} // namespace detail
SparseTensorEncodingAttr SparseTensorEncodingAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dimLevelType, AffineMap dimOrdering, unsigned pointerBitWidth, unsigned indexBitWidth) {
  return Base::get(context, dimLevelType, dimOrdering, pointerBitWidth, indexBitWidth);
}

SparseTensorEncodingAttr SparseTensorEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> dimLevelType, AffineMap dimOrdering, unsigned pointerBitWidth, unsigned indexBitWidth) {
  return Base::getChecked(emitError, context, dimLevelType, dimOrdering, pointerBitWidth, indexBitWidth);
}

::llvm::ArrayRef<SparseTensorEncodingAttr::DimLevelType> SparseTensorEncodingAttr::getDimLevelType() const {
  return getImpl()->dimLevelType;
}

AffineMap SparseTensorEncodingAttr::getDimOrdering() const {
  return getImpl()->dimOrdering;
}

unsigned SparseTensorEncodingAttr::getPointerBitWidth() const {
  return getImpl()->pointerBitWidth;
}

unsigned SparseTensorEncodingAttr::getIndexBitWidth() const {
  return getImpl()->indexBitWidth;
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorEncodingAttr)
namespace mlir {
namespace sparse_tensor {

/// Parse an attribute registered to this dialect.
::mlir::Attribute SparseTensorDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  if (::mlir::failed(parser.parseKeyword(&attrTag)))
    return {};
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, attrTag, type, attr);
    if (parseResult.hasValue())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void SparseTensorDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace sparse_tensor
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

