/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::sparse_tensor::StorageSpecifierType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::sparse_tensor::StorageSpecifierType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::sparse_tensor::StorageSpecifierType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::mlir::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(def)    .Case<::mlir::sparse_tensor::StorageSpecifierType>([&](auto t) {
      printer << ::mlir::sparse_tensor::StorageSpecifierType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace sparse_tensor {
namespace detail {
struct StorageSpecifierTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::mlir::sparse_tensor::SparseTensorEncodingAttr>;
  StorageSpecifierTypeStorage(::mlir::sparse_tensor::SparseTensorEncodingAttr encoding) : encoding(encoding) {}

  KeyTy getAsKey() const {
    return KeyTy(encoding);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (encoding == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static StorageSpecifierTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto encoding = std::get<0>(tblgenKey);
    return new (allocator.allocate<StorageSpecifierTypeStorage>()) StorageSpecifierTypeStorage(encoding);
  }

  ::mlir::sparse_tensor::SparseTensorEncodingAttr encoding;
};
} // namespace detail
StorageSpecifierType StorageSpecifierType::get(SparseTensorEncodingAttr encoding) {
  return get(encoding.getContext(), encoding);
}

StorageSpecifierType StorageSpecifierType::get(Type type) {
  return get(getSparseTensorEncoding(type));
}

StorageSpecifierType StorageSpecifierType::get(Value tensor) {
  return get(tensor.getType());
}

::mlir::Type StorageSpecifierType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::sparse_tensor::SparseTensorEncodingAttr> _result_encoding;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'encoding'
  _result_encoding = ::mlir::FieldParser<::mlir::sparse_tensor::SparseTensorEncodingAttr>::parse(odsParser);
  if (::mlir::failed(_result_encoding)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SparseTensor_StorageSpecifier parameter 'encoding' which is to be a `::mlir::sparse_tensor::SparseTensorEncodingAttr`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_encoding));
  return StorageSpecifierType::get(odsParser.getContext(),
      ::mlir::sparse_tensor::SparseTensorEncodingAttr((*_result_encoding).cast<::mlir::sparse_tensor::SparseTensorEncodingAttr>()));
}

void StorageSpecifierType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << getEncoding();
  odsPrinter << ">";
}

::mlir::sparse_tensor::SparseTensorEncodingAttr StorageSpecifierType::getEncoding() const {
  return getImpl()->encoding;
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::StorageSpecifierType)
namespace mlir {
namespace sparse_tensor {

/// Parse a type registered to this dialect.
::mlir::Type SparseTensorDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void SparseTensorDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}
} // namespace sparse_tensor
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

