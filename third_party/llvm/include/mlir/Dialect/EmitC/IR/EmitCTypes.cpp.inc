/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::emitc::OpaqueType,
::mlir::emitc::PointerType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef mnemonic, ::mlir::Type &value) {
  if (mnemonic == ::mlir::emitc::OpaqueType::getMnemonic()) {
    value = ::mlir::emitc::OpaqueType::parse(parser);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::emitc::PointerType::getMnemonic()) {
    value = ::mlir::emitc::PointerType::parse(parser);
    return ::mlir::success(!!value);
  }
  return {};
}

static ::mlir::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(def)    .Case<::mlir::emitc::OpaqueType>([&](auto t) {
      printer << ::mlir::emitc::OpaqueType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::emitc::PointerType>([&](auto t) {
      printer << ::mlir::emitc::PointerType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace emitc {
namespace detail {
struct OpaqueTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::StringRef>;
  OpaqueTypeStorage(::llvm::StringRef value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static OpaqueTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    value = allocator.copyInto(value);
    return new (allocator.allocate<OpaqueTypeStorage>()) OpaqueTypeStorage(value);
  }

  ::llvm::StringRef value;
};
} // namespace detail
OpaqueType OpaqueType::get(::mlir::MLIRContext *context, ::llvm::StringRef value) {
  return Base::get(context, value);
}

::llvm::StringRef OpaqueType::getValue() const {
  return getImpl()->value;
}

} // namespace emitc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::OpaqueType)
namespace mlir {
namespace emitc {
namespace detail {
struct PointerTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type>;
  PointerTypeStorage(Type pointee) : pointee(pointee) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (pointee == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static PointerTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto pointee = std::get<0>(tblgenKey);
    return new (allocator.allocate<PointerTypeStorage>()) PointerTypeStorage(pointee);
  }

  Type pointee;
};
} // namespace detail
PointerType PointerType::get(::mlir::MLIRContext *context, Type pointee) {
  return Base::get(context, pointee);
}

PointerType PointerType::get(Type pointee) {
  return Base::get(pointee.getContext(), pointee);
}

::mlir::Type PointerType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::FailureOr<Type> _result_pointee;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'pointee'
  _result_pointee = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_pointee)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse EmitC_PointerType parameter 'pointee' which is to be a `Type`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_pointee));
  return PointerType::get(odsParser.getContext(),
      *_result_pointee);
}

void PointerType::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  odsPrinter << getPointee();
  odsPrinter << ">";
}

Type PointerType::getPointee() const {
  return getImpl()->pointee;
}

} // namespace emitc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::PointerType)
namespace mlir {
namespace emitc {

/// Parse a type registered to this dialect.
::mlir::Type EmitCDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return ::mlir::Type();
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, mnemonic, genType);
  if (parseResult.hasValue())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void EmitCDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}
} // namespace emitc
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

