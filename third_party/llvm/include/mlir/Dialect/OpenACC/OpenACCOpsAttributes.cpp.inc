/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::acc::ClauseDefaultValueAttr,
::mlir::acc::ReductionOpAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  if (mnemonic == ::mlir::acc::ClauseDefaultValueAttr::getMnemonic()) {
    value = ::mlir::acc::ClauseDefaultValueAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::acc::ReductionOpAttr::getMnemonic()) {
    value = ::mlir::acc::ReductionOpAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  return {};
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::acc::ClauseDefaultValueAttr>([&](auto t) {
      printer << ::mlir::acc::ClauseDefaultValueAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::ReductionOpAttr>([&](auto t) {
      printer << ::mlir::acc::ReductionOpAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace acc {
namespace detail {
struct ClauseDefaultValueAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::acc::ClauseDefaultValue>;
  ClauseDefaultValueAttrStorage(::mlir::acc::ClauseDefaultValue value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseDefaultValueAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseDefaultValueAttrStorage>()) ClauseDefaultValueAttrStorage(value);
  }

  ::mlir::acc::ClauseDefaultValue value;
};
} // namespace detail
ClauseDefaultValueAttr ClauseDefaultValueAttr::get(::mlir::MLIRContext *context, ::mlir::acc::ClauseDefaultValue value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseDefaultValueAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::FailureOr<::mlir::acc::ClauseDefaultValue> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::acc::ClauseDefaultValue> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::acc::symbolizeClauseDefaultValue(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)odsParser.emitError(loc, "expected ::mlir::acc::ClauseDefaultValue to be one of: present, none")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DefaultValueAttr parameter 'value' which is to be a `::mlir::acc::ClauseDefaultValue`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseDefaultValueAttr::get(odsParser.getContext(),
      *_result_value);
}

void ClauseDefaultValueAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << ' ';
  odsPrinter << stringifyClauseDefaultValue(getValue());
}

::mlir::acc::ClauseDefaultValue ClauseDefaultValueAttr::getValue() const {
  return getImpl()->value;
}

} // namespace acc
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::ClauseDefaultValueAttr)
namespace mlir {
namespace acc {
namespace detail {
struct ReductionOpAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::acc::ReductionOp>;
  ReductionOpAttrStorage(::mlir::acc::ReductionOp value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ReductionOpAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ReductionOpAttrStorage>()) ReductionOpAttrStorage(value);
  }

  ::mlir::acc::ReductionOp value;
};
} // namespace detail
ReductionOpAttr ReductionOpAttr::get(::mlir::MLIRContext *context, ::mlir::acc::ReductionOp value) {
  return Base::get(context, value);
}

::mlir::Attribute ReductionOpAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::FailureOr<::mlir::acc::ReductionOp> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::acc::ReductionOp> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::acc::symbolizeReductionOp(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)odsParser.emitError(loc, "expected ::mlir::acc::ReductionOp to be one of: redop_add, redop_mul, redop_max, redop_min, redop_and, redop_or, redop_xor, redop_leqv, redop_lneqv, redop_land, redop_lor")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OpenACC_ReductionOpAttr parameter 'value' which is to be a `::mlir::acc::ReductionOp`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ReductionOpAttr::get(odsParser.getContext(),
      *_result_value);
}

void ReductionOpAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << ' ';
  odsPrinter << stringifyReductionOp(getValue());
}

::mlir::acc::ReductionOp ReductionOpAttr::getValue() const {
  return getImpl()->value;
}

} // namespace acc
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::ReductionOpAttr)
namespace mlir {
namespace acc {

/// Parse an attribute registered to this dialect.
::mlir::Attribute OpenACCDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void OpenACCDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
}
} // namespace acc
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

