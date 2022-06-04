/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::emitc::OpaqueAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  if (mnemonic == ::mlir::emitc::OpaqueAttr::getMnemonic()) {
    value = ::mlir::emitc::OpaqueAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  return {};
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::emitc::OpaqueAttr>([&](auto t) {
      printer << ::mlir::emitc::OpaqueAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace emitc {
namespace detail {
struct OpaqueAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::StringRef>;
  OpaqueAttrStorage(::llvm::StringRef value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static OpaqueAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    value = allocator.copyInto(value);
    return new (allocator.allocate<OpaqueAttrStorage>()) OpaqueAttrStorage(value);
  }

  ::llvm::StringRef value;
};
} // namespace detail
OpaqueAttr OpaqueAttr::get(::mlir::MLIRContext *context, ::llvm::StringRef value) {
  return Base::get(context, value);
}

::llvm::StringRef OpaqueAttr::getValue() const {
  return getImpl()->value;
}

} // namespace emitc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::OpaqueAttr)
namespace mlir {
namespace emitc {

/// Parse an attribute registered to this dialect.
::mlir::Attribute EmitCDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void EmitCDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace emitc
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

