/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::vector::CombiningKindAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::vector::CombiningKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vector::CombiningKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return llvm::None;
    });
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::vector::CombiningKindAttr>([&](auto t) {
      printer << ::mlir::vector::CombiningKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace vector {
namespace detail {
struct CombiningKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::vector::CombiningKind>;
  CombiningKindAttrStorage(::mlir::vector::CombiningKind value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static CombiningKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<CombiningKindAttrStorage>()) CombiningKindAttrStorage(value);
  }

  ::mlir::vector::CombiningKind value;
};
} // namespace detail
CombiningKindAttr CombiningKindAttr::get(::mlir::MLIRContext *context, ::mlir::vector::CombiningKind value) {
  return Base::get(context, value);
}

::mlir::Attribute CombiningKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::mlir::FailureOr<::mlir::vector::CombiningKind> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::vector::CombiningKind> {
      ::mlir::vector::CombiningKind flags = {};
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      do {
        if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
          return ::mlir::failure();
        auto maybeEnum = ::mlir::vector::symbolizeCombiningKind(enumKeyword);
        if (!maybeEnum) {
            return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::vector::CombiningKind" << " to be one of: " << "add" << ", " << "mul" << ", " << "minui" << ", " << "minsi" << ", " << "minf" << ", " << "maxui" << ", " << "maxsi" << ", " << "maxf" << ", " << "and" << ", " << "or" << ", " << "xor")};
        }
        flags = flags | *maybeEnum;
      } while(::mlir::succeeded(odsParser.parseOptionalVerticalBar()));
      return flags;
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Vector_CombiningKindAttr parameter 'value' which is to be a `::mlir::vector::CombiningKind`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return CombiningKindAttr::get(odsParser.getContext(),
      ::mlir::vector::CombiningKind((*_result_value)));
}

void CombiningKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyCombiningKind(getValue());
  odsPrinter << ">";
}

::mlir::vector::CombiningKind CombiningKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace vector
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::CombiningKindAttr)
namespace mlir {
namespace vector {

/// Parse an attribute registered to this dialect.
::mlir::Attribute VectorDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void VectorDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace vector
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

