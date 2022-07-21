/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::omp::ClauseCancellationConstructTypeAttr,
::mlir::omp::ClauseDependAttr,
::mlir::omp::ClauseMemoryOrderKindAttr,
::mlir::omp::ClauseOrderKindAttr,
::mlir::omp::ClauseProcBindKindAttr,
::mlir::omp::ClauseScheduleKindAttr,
::mlir::omp::ScheduleModifierAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  if (mnemonic == ::mlir::omp::ClauseCancellationConstructTypeAttr::getMnemonic()) {
    value = ::mlir::omp::ClauseCancellationConstructTypeAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::omp::ClauseDependAttr::getMnemonic()) {
    value = ::mlir::omp::ClauseDependAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::omp::ClauseMemoryOrderKindAttr::getMnemonic()) {
    value = ::mlir::omp::ClauseMemoryOrderKindAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::omp::ClauseOrderKindAttr::getMnemonic()) {
    value = ::mlir::omp::ClauseOrderKindAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::omp::ClauseProcBindKindAttr::getMnemonic()) {
    value = ::mlir::omp::ClauseProcBindKindAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::omp::ClauseScheduleKindAttr::getMnemonic()) {
    value = ::mlir::omp::ClauseScheduleKindAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::omp::ScheduleModifierAttr::getMnemonic()) {
    value = ::mlir::omp::ScheduleModifierAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  return {};
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::omp::ClauseCancellationConstructTypeAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseCancellationConstructTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseDependAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseDependAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseMemoryOrderKindAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseMemoryOrderKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseOrderKindAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseOrderKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseProcBindKindAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseProcBindKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ClauseScheduleKindAttr>([&](auto t) {
      printer << ::mlir::omp::ClauseScheduleKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::omp::ScheduleModifierAttr>([&](auto t) {
      printer << ::mlir::omp::ScheduleModifierAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace omp {
namespace detail {
struct ClauseCancellationConstructTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseCancellationConstructType>;
  ClauseCancellationConstructTypeAttrStorage(::mlir::omp::ClauseCancellationConstructType value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseCancellationConstructTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseCancellationConstructTypeAttrStorage>()) ClauseCancellationConstructTypeAttrStorage(value);
  }

  ::mlir::omp::ClauseCancellationConstructType value;
};
} // namespace detail
ClauseCancellationConstructTypeAttr ClauseCancellationConstructTypeAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseCancellationConstructType value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseCancellationConstructTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::FailureOr<::mlir::omp::ClauseCancellationConstructType> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseCancellationConstructType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseCancellationConstructType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseCancellationConstructType" << " to be one of: " << "parallel" << ", " << "loop" << ", " << "sections" << ", " << "taskgroup")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse CancellationConstructTypeAttr parameter 'value' which is to be a `::mlir::omp::ClauseCancellationConstructType`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseCancellationConstructTypeAttr::get(odsParser.getContext(),
      *_result_value);
}

void ClauseCancellationConstructTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << ' ';
  odsPrinter << stringifyClauseCancellationConstructType(getValue());
}

::mlir::omp::ClauseCancellationConstructType ClauseCancellationConstructTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseCancellationConstructTypeAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseDependAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseDepend>;
  ClauseDependAttrStorage(::mlir::omp::ClauseDepend value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseDependAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseDependAttrStorage>()) ClauseDependAttrStorage(value);
  }

  ::mlir::omp::ClauseDepend value;
};
} // namespace detail
ClauseDependAttr ClauseDependAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseDepend value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseDependAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::FailureOr<::mlir::omp::ClauseDepend> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  // Parse literal '('
  if (odsParser.parseLParen()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseDepend> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseDepend(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseDepend" << " to be one of: " << "dependsource" << ", " << "dependsink")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ClauseDependAttr parameter 'value' which is to be a `::mlir::omp::ClauseDepend`");
    return {};
  }
  // Parse literal ')'
  if (odsParser.parseRParen()) return {};
  assert(::mlir::succeeded(_result_value));
  return ClauseDependAttr::get(odsParser.getContext(),
      *_result_value);
}

void ClauseDependAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "(";
  odsPrinter << stringifyClauseDepend(getValue());
  odsPrinter << ")";
}

::mlir::omp::ClauseDepend ClauseDependAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseDependAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseMemoryOrderKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseMemoryOrderKind>;
  ClauseMemoryOrderKindAttrStorage(::mlir::omp::ClauseMemoryOrderKind value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseMemoryOrderKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseMemoryOrderKindAttrStorage>()) ClauseMemoryOrderKindAttrStorage(value);
  }

  ::mlir::omp::ClauseMemoryOrderKind value;
};
} // namespace detail
ClauseMemoryOrderKindAttr ClauseMemoryOrderKindAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseMemoryOrderKind value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseMemoryOrderKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::FailureOr<::mlir::omp::ClauseMemoryOrderKind> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseMemoryOrderKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseMemoryOrderKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseMemoryOrderKind" << " to be one of: " << "seq_cst" << ", " << "acq_rel" << ", " << "acquire" << ", " << "release" << ", " << "relaxed")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MemoryOrderKindAttr parameter 'value' which is to be a `::mlir::omp::ClauseMemoryOrderKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseMemoryOrderKindAttr::get(odsParser.getContext(),
      *_result_value);
}

void ClauseMemoryOrderKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << ' ';
  odsPrinter << stringifyClauseMemoryOrderKind(getValue());
}

::mlir::omp::ClauseMemoryOrderKind ClauseMemoryOrderKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseMemoryOrderKindAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseOrderKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseOrderKind>;
  ClauseOrderKindAttrStorage(::mlir::omp::ClauseOrderKind value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseOrderKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseOrderKindAttrStorage>()) ClauseOrderKindAttrStorage(value);
  }

  ::mlir::omp::ClauseOrderKind value;
};
} // namespace detail
ClauseOrderKindAttr ClauseOrderKindAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseOrderKind value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseOrderKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::FailureOr<::mlir::omp::ClauseOrderKind> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseOrderKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseOrderKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseOrderKind" << " to be one of: " << "concurrent")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OrderKindAttr parameter 'value' which is to be a `::mlir::omp::ClauseOrderKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseOrderKindAttr::get(odsParser.getContext(),
      *_result_value);
}

void ClauseOrderKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << ' ';
  odsPrinter << stringifyClauseOrderKind(getValue());
}

::mlir::omp::ClauseOrderKind ClauseOrderKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseOrderKindAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseProcBindKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseProcBindKind>;
  ClauseProcBindKindAttrStorage(::mlir::omp::ClauseProcBindKind value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseProcBindKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseProcBindKindAttrStorage>()) ClauseProcBindKindAttrStorage(value);
  }

  ::mlir::omp::ClauseProcBindKind value;
};
} // namespace detail
ClauseProcBindKindAttr ClauseProcBindKindAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseProcBindKind value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseProcBindKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::FailureOr<::mlir::omp::ClauseProcBindKind> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseProcBindKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseProcBindKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseProcBindKind" << " to be one of: " << "primary" << ", " << "master" << ", " << "close" << ", " << "spread")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ProcBindKindAttr parameter 'value' which is to be a `::mlir::omp::ClauseProcBindKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseProcBindKindAttr::get(odsParser.getContext(),
      *_result_value);
}

void ClauseProcBindKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << ' ';
  odsPrinter << stringifyClauseProcBindKind(getValue());
}

::mlir::omp::ClauseProcBindKind ClauseProcBindKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseProcBindKindAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ClauseScheduleKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ClauseScheduleKind>;
  ClauseScheduleKindAttrStorage(::mlir::omp::ClauseScheduleKind value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseScheduleKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ClauseScheduleKindAttrStorage>()) ClauseScheduleKindAttrStorage(value);
  }

  ::mlir::omp::ClauseScheduleKind value;
};
} // namespace detail
ClauseScheduleKindAttr ClauseScheduleKindAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ClauseScheduleKind value) {
  return Base::get(context, value);
}

::mlir::Attribute ClauseScheduleKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::FailureOr<::mlir::omp::ClauseScheduleKind> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ClauseScheduleKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeClauseScheduleKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ClauseScheduleKind" << " to be one of: " << "static" << ", " << "dynamic" << ", " << "guided" << ", " << "auto" << ", " << "runtime")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ScheduleKindAttr parameter 'value' which is to be a `::mlir::omp::ClauseScheduleKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseScheduleKindAttr::get(odsParser.getContext(),
      *_result_value);
}

void ClauseScheduleKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << ' ';
  odsPrinter << stringifyClauseScheduleKind(getValue());
}

::mlir::omp::ClauseScheduleKind ClauseScheduleKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ClauseScheduleKindAttr)
namespace mlir {
namespace omp {
namespace detail {
struct ScheduleModifierAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::omp::ScheduleModifier>;
  ScheduleModifierAttrStorage(::mlir::omp::ScheduleModifier value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ScheduleModifierAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ScheduleModifierAttrStorage>()) ScheduleModifierAttrStorage(value);
  }

  ::mlir::omp::ScheduleModifier value;
};
} // namespace detail
ScheduleModifierAttr ScheduleModifierAttr::get(::mlir::MLIRContext *context, ::mlir::omp::ScheduleModifier value) {
  return Base::get(context, value);
}

::mlir::Attribute ScheduleModifierAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::FailureOr<::mlir::omp::ScheduleModifier> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::omp::ScheduleModifier> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::omp::symbolizeScheduleModifier(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::omp::ScheduleModifier" << " to be one of: " << "none" << ", " << "monotonic" << ", " << "nonmonotonic" << ", " << "simd")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ScheduleModifierAttr parameter 'value' which is to be a `::mlir::omp::ScheduleModifier`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ScheduleModifierAttr::get(odsParser.getContext(),
      *_result_value);
}

void ScheduleModifierAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << ' ';
  odsPrinter << stringifyScheduleModifier(getValue());
}

::mlir::omp::ScheduleModifier ScheduleModifierAttr::getValue() const {
  return getImpl()->value;
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ScheduleModifierAttr)
namespace mlir {
namespace omp {

/// Parse an attribute registered to this dialect.
::mlir::Attribute OpenMPDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void OpenMPDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace omp
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

