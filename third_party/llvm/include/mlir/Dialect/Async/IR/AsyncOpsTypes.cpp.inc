/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::async::CoroHandleType,
::mlir::async::CoroIdType,
::mlir::async::CoroStateType,
::mlir::async::GroupType,
::mlir::async::TokenType,
::mlir::async::ValueType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef mnemonic, ::mlir::Type &value) {
  if (mnemonic == ::mlir::async::CoroHandleType::getMnemonic()) {
    value = ::mlir::async::CoroHandleType::get(parser.getContext());
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::async::CoroIdType::getMnemonic()) {
    value = ::mlir::async::CoroIdType::get(parser.getContext());
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::async::CoroStateType::getMnemonic()) {
    value = ::mlir::async::CoroStateType::get(parser.getContext());
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::async::GroupType::getMnemonic()) {
    value = ::mlir::async::GroupType::get(parser.getContext());
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::async::TokenType::getMnemonic()) {
    value = ::mlir::async::TokenType::get(parser.getContext());
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::async::ValueType::getMnemonic()) {
    value = ::mlir::async::ValueType::parse(parser);
    return ::mlir::success(!!value);
  }
  return {};
}

static ::mlir::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::mlir::LogicalResult>(def)    .Case<::mlir::async::CoroHandleType>([&](auto t) {
      printer << ::mlir::async::CoroHandleType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::async::CoroIdType>([&](auto t) {
      printer << ::mlir::async::CoroIdType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::async::CoroStateType>([&](auto t) {
      printer << ::mlir::async::CoroStateType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::async::GroupType>([&](auto t) {
      printer << ::mlir::async::GroupType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::async::TokenType>([&](auto t) {
      printer << ::mlir::async::TokenType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::async::ValueType>([&](auto t) {
      printer << ::mlir::async::ValueType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace async {
} // namespace async
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::async::CoroHandleType)
namespace mlir {
namespace async {
} // namespace async
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::async::CoroIdType)
namespace mlir {
namespace async {
} // namespace async
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::async::CoroStateType)
namespace mlir {
namespace async {
} // namespace async
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::async::GroupType)
namespace mlir {
namespace async {
} // namespace async
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::async::TokenType)
namespace mlir {
namespace async {
namespace detail {
struct ValueTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type>;
  ValueTypeStorage(Type valueType) : valueType(valueType) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (valueType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ValueTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto valueType = std::get<0>(tblgenKey);
    return new (allocator.allocate<ValueTypeStorage>()) ValueTypeStorage(valueType);
  }

  Type valueType;
};
} // namespace detail
ValueType ValueType::get(Type valueType) {
  return Base::get(valueType.getContext(), valueType);
}

Type ValueType::getValueType() const {
  return getImpl()->valueType;
}

} // namespace async
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::async::ValueType)
namespace mlir {
namespace async {

/// Parse a type registered to this dialect.
::mlir::Type AsyncDialect::parseType(::mlir::DialectAsmParser &parser) const {
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
void AsyncDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}
} // namespace async
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

