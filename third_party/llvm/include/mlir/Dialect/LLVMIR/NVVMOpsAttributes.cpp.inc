/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::NVVM::MMAB1OpAttr,
::mlir::NVVM::MMAFragAttr,
::mlir::NVVM::MMAIntOverflowAttr,
::mlir::NVVM::MMALayoutAttr,
::mlir::NVVM::MMATypesAttr,
::mlir::NVVM::MMAShapeAttr,
::mlir::NVVM::ShflKindAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::NVVM::MMAB1OpAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::NVVM::MMAB1OpAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::NVVM::MMAFragAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::NVVM::MMAFragAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::NVVM::MMAIntOverflowAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::NVVM::MMAIntOverflowAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::NVVM::MMALayoutAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::NVVM::MMALayoutAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::NVVM::MMATypesAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::NVVM::MMATypesAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::NVVM::MMAShapeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::NVVM::MMAShapeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::NVVM::ShflKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::NVVM::ShflKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return llvm::None;
    });
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::NVVM::MMAB1OpAttr>([&](auto t) {
      printer << ::mlir::NVVM::MMAB1OpAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::NVVM::MMAFragAttr>([&](auto t) {
      printer << ::mlir::NVVM::MMAFragAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::NVVM::MMAIntOverflowAttr>([&](auto t) {
      printer << ::mlir::NVVM::MMAIntOverflowAttr::getMnemonic();
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
    .Case<::mlir::NVVM::MMAShapeAttr>([&](auto t) {
      printer << ::mlir::NVVM::MMAShapeAttr::getMnemonic();
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
struct MMAB1OpAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::NVVM::MMAB1Op>;
  MMAB1OpAttrStorage(::mlir::NVVM::MMAB1Op value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MMAB1OpAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<MMAB1OpAttrStorage>()) MMAB1OpAttrStorage(value);
  }

  ::mlir::NVVM::MMAB1Op value;
};
} // namespace detail
MMAB1OpAttr MMAB1OpAttr::get(::mlir::MLIRContext *context, ::mlir::NVVM::MMAB1Op value) {
  return Base::get(context, value);
}

::mlir::Attribute MMAB1OpAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::mlir::FailureOr<::mlir::NVVM::MMAB1Op> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::NVVM::MMAB1Op> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::NVVM::symbolizeMMAB1Op(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::NVVM::MMAB1Op" << " to be one of: " << "none" << ", " << "xor_popc" << ", " << "and_popc")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MMAB1OpAttr parameter 'value' which is to be a `::mlir::NVVM::MMAB1Op`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return MMAB1OpAttr::get(odsParser.getContext(),
      ::mlir::NVVM::MMAB1Op((*_result_value)));
}

void MMAB1OpAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyMMAB1Op(getValue());
  odsPrinter << ">";
}

::mlir::NVVM::MMAB1Op MMAB1OpAttr::getValue() const {
  return getImpl()->value;
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMAB1OpAttr)
namespace mlir {
namespace NVVM {
namespace detail {
struct MMAFragAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::NVVM::MMAFrag>;
  MMAFragAttrStorage(::mlir::NVVM::MMAFrag value) : value(value) {}

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
  ::mlir::Builder odsBuilder(odsParser.getContext());
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
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::NVVM::MMAFrag" << " to be one of: " << "a" << ", " << "b" << ", " << "c")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MMAFragAttr parameter 'value' which is to be a `::mlir::NVVM::MMAFrag`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return MMAFragAttr::get(odsParser.getContext(),
      ::mlir::NVVM::MMAFrag((*_result_value)));
}

void MMAFragAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyMMAFrag(getValue());
  odsPrinter << ">";
}

::mlir::NVVM::MMAFrag MMAFragAttr::getValue() const {
  return getImpl()->value;
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMAFragAttr)
namespace mlir {
namespace NVVM {
namespace detail {
struct MMAIntOverflowAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::NVVM::MMAIntOverflow>;
  MMAIntOverflowAttrStorage(::mlir::NVVM::MMAIntOverflow value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MMAIntOverflowAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<MMAIntOverflowAttrStorage>()) MMAIntOverflowAttrStorage(value);
  }

  ::mlir::NVVM::MMAIntOverflow value;
};
} // namespace detail
MMAIntOverflowAttr MMAIntOverflowAttr::get(::mlir::MLIRContext *context, ::mlir::NVVM::MMAIntOverflow value) {
  return Base::get(context, value);
}

::mlir::Attribute MMAIntOverflowAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::mlir::FailureOr<::mlir::NVVM::MMAIntOverflow> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::NVVM::MMAIntOverflow> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::NVVM::symbolizeMMAIntOverflow(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::NVVM::MMAIntOverflow" << " to be one of: " << "satfinite" << ", " << "wrapped")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MMAIntOverflowAttr parameter 'value' which is to be a `::mlir::NVVM::MMAIntOverflow`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return MMAIntOverflowAttr::get(odsParser.getContext(),
      ::mlir::NVVM::MMAIntOverflow((*_result_value)));
}

void MMAIntOverflowAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyMMAIntOverflow(getValue());
  odsPrinter << ">";
}

::mlir::NVVM::MMAIntOverflow MMAIntOverflowAttr::getValue() const {
  return getImpl()->value;
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMAIntOverflowAttr)
namespace mlir {
namespace NVVM {
namespace detail {
struct MMALayoutAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::NVVM::MMALayout>;
  MMALayoutAttrStorage(::mlir::NVVM::MMALayout value) : value(value) {}

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
  ::mlir::Builder odsBuilder(odsParser.getContext());
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
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::NVVM::MMALayout" << " to be one of: " << "row" << ", " << "col")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MMALayoutAttr parameter 'value' which is to be a `::mlir::NVVM::MMALayout`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return MMALayoutAttr::get(odsParser.getContext(),
      ::mlir::NVVM::MMALayout((*_result_value)));
}

void MMALayoutAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyMMALayout(getValue());
  odsPrinter << ">";
}

::mlir::NVVM::MMALayout MMALayoutAttr::getValue() const {
  return getImpl()->value;
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMALayoutAttr)
namespace mlir {
namespace NVVM {
namespace detail {
struct MMATypesAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::NVVM::MMATypes>;
  MMATypesAttrStorage(::mlir::NVVM::MMATypes value) : value(value) {}

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
  ::mlir::Builder odsBuilder(odsParser.getContext());
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
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::NVVM::MMATypes" << " to be one of: " << "f16" << ", " << "f32" << ", " << "tf32" << ", " << "bf16" << ", " << "s8" << ", " << "u8" << ", " << "s32" << ", " << "s4" << ", " << "u4" << ", " << "b1" << ", " << "f64")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MMATypesAttr parameter 'value' which is to be a `::mlir::NVVM::MMATypes`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return MMATypesAttr::get(odsParser.getContext(),
      ::mlir::NVVM::MMATypes((*_result_value)));
}

void MMATypesAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyMMATypes(getValue());
  odsPrinter << ">";
}

::mlir::NVVM::MMATypes MMATypesAttr::getValue() const {
  return getImpl()->value;
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMATypesAttr)
namespace mlir {
namespace NVVM {
namespace detail {
struct MMAShapeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int, int, int>;
  MMAShapeAttrStorage(int m, int n, int k) : m(m), n(n), k(k) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (m == std::get<0>(tblgenKey)) && (n == std::get<1>(tblgenKey)) && (k == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static MMAShapeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto m = std::get<0>(tblgenKey);
    auto n = std::get<1>(tblgenKey);
    auto k = std::get<2>(tblgenKey);
    return new (allocator.allocate<MMAShapeAttrStorage>()) MMAShapeAttrStorage(m, n, k);
  }

  int m;
  int n;
  int k;
};
} // namespace detail
MMAShapeAttr MMAShapeAttr::get(::mlir::MLIRContext *context, int m, int n, int k) {
  return Base::get(context, m, n, k);
}

::mlir::Attribute MMAShapeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::mlir::FailureOr<int> _result_m;
  ::mlir::FailureOr<int> _result_n;
  ::mlir::FailureOr<int> _result_k;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_m = false;
  bool _seen_n = false;
  bool _seen_k = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_m && _paramKey == "m") {
        _seen_m = true;

        // Parse variable 'm'
        _result_m = ::mlir::FieldParser<int>::parse(odsParser);
        if (::mlir::failed(_result_m)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVVM_MMAShapeAttr parameter 'm' which is to be a `int`");
          return {};
        }
      } else if (!_seen_n && _paramKey == "n") {
        _seen_n = true;

        // Parse variable 'n'
        _result_n = ::mlir::FieldParser<int>::parse(odsParser);
        if (::mlir::failed(_result_n)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVVM_MMAShapeAttr parameter 'n' which is to be a `int`");
          return {};
        }
      } else if (!_seen_k && _paramKey == "k") {
        _seen_k = true;

        // Parse variable 'k'
        _result_k = ::mlir::FieldParser<int>::parse(odsParser);
        if (::mlir::failed(_result_k)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVVM_MMAShapeAttr parameter 'k' which is to be a `int`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 3; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 3 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_m));
  assert(::mlir::succeeded(_result_n));
  assert(::mlir::succeeded(_result_k));
  return MMAShapeAttr::get(odsParser.getContext(),
      int((*_result_m)),
      int((*_result_n)),
      int((*_result_k)));
}

void MMAShapeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "m = ";
    odsPrinter.printStrippedAttrOrType(getM());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "n = ";
    odsPrinter.printStrippedAttrOrType(getN());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "k = ";
    odsPrinter.printStrippedAttrOrType(getK());
  }
  odsPrinter << ">";
}

int MMAShapeAttr::getM() const {
  return getImpl()->m;
}

int MMAShapeAttr::getN() const {
  return getImpl()->n;
}

int MMAShapeAttr::getK() const {
  return getImpl()->k;
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::MMAShapeAttr)
namespace mlir {
namespace NVVM {
namespace detail {
struct ShflKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::NVVM::ShflKind>;
  ShflKindAttrStorage(::mlir::NVVM::ShflKind value) : value(value) {}

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
  ::mlir::Builder odsBuilder(odsParser.getContext());
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
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::NVVM::ShflKind" << " to be one of: " << "bfly" << ", " << "up" << ", " << "down" << ", " << "idx")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ShflKindAttr parameter 'value' which is to be a `::mlir::NVVM::ShflKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ShflKindAttr::get(odsParser.getContext(),
      ::mlir::NVVM::ShflKind((*_result_value)));
}

void ShflKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyShflKind(getValue());
}

::mlir::NVVM::ShflKind ShflKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::ShflKindAttr)
namespace mlir {
namespace NVVM {

/// Parse an attribute registered to this dialect.
::mlir::Attribute NVVMDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void NVVMDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace NVVM
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

