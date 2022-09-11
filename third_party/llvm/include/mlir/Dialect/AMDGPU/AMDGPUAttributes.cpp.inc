/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::amdgpu::MFMAPermBAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::amdgpu::MFMAPermBAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::amdgpu::MFMAPermBAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return llvm::None;
    });
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::amdgpu::MFMAPermBAttr>([&](auto t) {
      printer << ::mlir::amdgpu::MFMAPermBAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace amdgpu {
namespace detail {
struct MFMAPermBAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::amdgpu::MFMAPermB>;
  MFMAPermBAttrStorage(::mlir::amdgpu::MFMAPermB value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MFMAPermBAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<MFMAPermBAttrStorage>()) MFMAPermBAttrStorage(value);
  }

  ::mlir::amdgpu::MFMAPermB value;
};
} // namespace detail
MFMAPermBAttr MFMAPermBAttr::get(::mlir::MLIRContext *context, ::mlir::amdgpu::MFMAPermB value) {
  return Base::get(context, value);
}

::mlir::Attribute MFMAPermBAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::mlir::FailureOr<::mlir::amdgpu::MFMAPermB> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::amdgpu::MFMAPermB> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::amdgpu::symbolizeMFMAPermB(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::amdgpu::MFMAPermB" << " to be one of: " << "none" << ", " << "bcast_first_32" << ", " << "bcast_second_32" << ", " << "rotate_16_right" << ", " << "bcast_first_16" << ", " << "bcast_second_16" << ", " << "bcast_third_16" << ", " << "bcast_fourth_16")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse AMDGPU_MFMAPermBAttr parameter 'value' which is to be a `::mlir::amdgpu::MFMAPermB`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return MFMAPermBAttr::get(odsParser.getContext(),
      ::mlir::amdgpu::MFMAPermB((*_result_value)));
}

void MFMAPermBAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyMFMAPermB(getValue());
}

::mlir::amdgpu::MFMAPermB MFMAPermBAttr::getValue() const {
  return getImpl()->value;
}

} // namespace amdgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::amdgpu::MFMAPermBAttr)
namespace mlir {
namespace amdgpu {

/// Parse an attribute registered to this dialect.
::mlir::Attribute AMDGPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
    if (parseResult.has_value())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void AMDGPUDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace amdgpu
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

