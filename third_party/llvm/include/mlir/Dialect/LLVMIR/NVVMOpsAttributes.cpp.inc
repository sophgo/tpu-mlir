/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::NVVM::MMAFragAttr,
::mlir::NVVM::MMALayoutAttr,
::mlir::NVVM::MMATypesAttr,
::mlir::NVVM::ShflKindAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  if (mnemonic == ::mlir::NVVM::MMAFragAttr::getMnemonic()) {
    value = ::mlir::NVVM::MMAFragAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::NVVM::MMALayoutAttr::getMnemonic()) {
    value = ::mlir::NVVM::MMALayoutAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::NVVM::MMATypesAttr::getMnemonic()) {
    value = ::mlir::NVVM::MMATypesAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::NVVM::ShflKindAttr::getMnemonic()) {
    value = ::mlir::NVVM::ShflKindAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  return {};
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::NVVM::MMAFragAttr>([&](auto t) {
      printer << ::mlir::NVVM::MMAFragAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::NVVM::MMALayoutAttr>([&](auto t) {
      printer << ::mlir::NVVM::MMALayoutAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::NVVM::MMATypesAttr>([&](auto t) {
      printer << ::mlir::NVVM::MMATypesAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::NVVM::ShflKindAttr>([&](auto t) {
      printer << ::mlir::NVVM::ShflKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace NVVM {
namespace detail {
struct MMAFragAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::NVVM::MMAFrag>;
  MMAFragAttrStorage(::mlir::NVVM::MMAFrag value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MMAFragAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<MMAFragAttrStorage>()) MMAFragAttrStorage(value);
  }

  ::mlir::NVVM::MMAFrag value;
};
} // namespace detail
MMAFragAttr MMAFragAttr::get(::mlir::MLIRContext *context, ::mlir::NVVM::MMAFrag value) {
  return Base::get(context, value);
}

::mlir::Attribute MMAFragAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::FailureOr<::mlir::NVVM::MMAFrag> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::NVVM::MMAFrag> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::NVVM::symbolizeMMAFrag(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)odsParser.emitError(loc, "expected ::mlir::NVVM::MMAFrag to be one of: a, b, c")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MMAFragAttr parameter 'value' which is to be a `::mlir::NVVM::MMAFrag`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return MMAFragAttr::get(odsParser.getContext(),
      *_result_value);
}

void MMAFragAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  odsPrinter << stringifyMMAFrag(getValue());
  odsPrinter << ">";
}

::mlir::NVVM::MMAFrag MMAFragAttr::getValue() const {
  return getImpl()->value;
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMAFragAttr)
namespace mlir {
namespace NVVM {
namespace detail {
struct MMALayoutAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::NVVM::MMALayout>;
  MMALayoutAttrStorage(::mlir::NVVM::MMALayout value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MMALayoutAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<MMALayoutAttrStorage>()) MMALayoutAttrStorage(value);
  }

  ::mlir::NVVM::MMALayout value;
};
} // namespace detail
MMALayoutAttr MMALayoutAttr::get(::mlir::MLIRContext *context, ::mlir::NVVM::MMALayout value) {
  return Base::get(context, value);
}

::mlir::Attribute MMALayoutAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::FailureOr<::mlir::NVVM::MMALayout> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::NVVM::MMALayout> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::NVVM::symbolizeMMALayout(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)odsParser.emitError(loc, "expected ::mlir::NVVM::MMALayout to be one of: row, col")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MMALayoutAttr parameter 'value' which is to be a `::mlir::NVVM::MMALayout`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return MMALayoutAttr::get(odsParser.getContext(),
      *_result_value);
}

void MMALayoutAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  odsPrinter << stringifyMMALayout(getValue());
  odsPrinter << ">";
}

::mlir::NVVM::MMALayout MMALayoutAttr::getValue() const {
  return getImpl()->value;
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMALayoutAttr)
namespace mlir {
namespace NVVM {
namespace detail {
struct MMATypesAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::NVVM::MMATypes>;
  MMATypesAttrStorage(::mlir::NVVM::MMATypes value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MMATypesAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<MMATypesAttrStorage>()) MMATypesAttrStorage(value);
  }

  ::mlir::NVVM::MMATypes value;
};
} // namespace detail
MMATypesAttr MMATypesAttr::get(::mlir::MLIRContext *context, ::mlir::NVVM::MMATypes value) {
  return Base::get(context, value);
}

::mlir::Attribute MMATypesAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::FailureOr<::mlir::NVVM::MMATypes> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::NVVM::MMATypes> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::NVVM::symbolizeMMATypes(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)odsParser.emitError(loc, "expected ::mlir::NVVM::MMATypes to be one of: f16, f32, tf32")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MMATypesAttr parameter 'value' which is to be a `::mlir::NVVM::MMATypes`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return MMATypesAttr::get(odsParser.getContext(),
      *_result_value);
}

void MMATypesAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  odsPrinter << stringifyMMATypes(getValue());
  odsPrinter << ">";
}

::mlir::NVVM::MMATypes MMATypesAttr::getValue() const {
  return getImpl()->value;
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMATypesAttr)
namespace mlir {
namespace NVVM {
namespace detail {
struct ShflKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::NVVM::ShflKind>;
  ShflKindAttrStorage(::mlir::NVVM::ShflKind value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ShflKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ShflKindAttrStorage>()) ShflKindAttrStorage(value);
  }

  ::mlir::NVVM::ShflKind value;
};
} // namespace detail
ShflKindAttr ShflKindAttr::get(::mlir::MLIRContext *context, ::mlir::NVVM::ShflKind value) {
  return Base::get(context, value);
}

::mlir::Attribute ShflKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::FailureOr<::mlir::NVVM::ShflKind> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::NVVM::ShflKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::NVVM::symbolizeShflKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)odsParser.emitError(loc, "expected ::mlir::NVVM::ShflKind to be one of: bfly, up, down, idx")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ShflKindAttr parameter 'value' which is to be a `::mlir::NVVM::ShflKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ShflKindAttr::get(odsParser.getContext(),
      *_result_value);
}

void ShflKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << ' ';
  odsPrinter << stringifyShflKind(getValue());
}

::mlir::NVVM::ShflKind ShflKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace NVVM
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::ShflKindAttr)
namespace mlir {
namespace NVVM {

/// Parse an attribute registered to this dialect.
::mlir::Attribute NVVMDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void NVVMDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
}
} // namespace NVVM
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

