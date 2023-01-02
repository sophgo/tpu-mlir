/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

mlir::tosa::ConvOpQuantizationAttr,
mlir::tosa::MatMulOpQuantizationAttr,
mlir::tosa::PadOpQuantizationAttr,
mlir::tosa::UnaryOpQuantizationAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(mlir::tosa::ConvOpQuantizationAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::ConvOpQuantizationAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::MatMulOpQuantizationAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::MatMulOpQuantizationAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::PadOpQuantizationAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::PadOpQuantizationAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::UnaryOpQuantizationAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::UnaryOpQuantizationAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<mlir::tosa::ConvOpQuantizationAttr>([&](auto t) {
      printer << mlir::tosa::ConvOpQuantizationAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::MatMulOpQuantizationAttr>([&](auto t) {
      printer << mlir::tosa::MatMulOpQuantizationAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::PadOpQuantizationAttr>([&](auto t) {
      printer << mlir::tosa::PadOpQuantizationAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::UnaryOpQuantizationAttr>([&](auto t) {
      printer << mlir::tosa::UnaryOpQuantizationAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace tosa {
namespace detail {
struct ConvOpQuantizationAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, int64_t>;
  ConvOpQuantizationAttrStorage(int64_t input_zp, int64_t weight_zp) : input_zp(input_zp), weight_zp(weight_zp) {}

  KeyTy getAsKey() const {
    return KeyTy(input_zp, weight_zp);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (input_zp == std::get<0>(tblgenKey)) && (weight_zp == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static ConvOpQuantizationAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto input_zp = std::get<0>(tblgenKey);
    auto weight_zp = std::get<1>(tblgenKey);
    return new (allocator.allocate<ConvOpQuantizationAttrStorage>()) ConvOpQuantizationAttrStorage(input_zp, weight_zp);
  }

  int64_t input_zp;
  int64_t weight_zp;
};
} // namespace detail
ConvOpQuantizationAttr ConvOpQuantizationAttr::get(::mlir::MLIRContext *context, int64_t input_zp, int64_t weight_zp) {
  return Base::get(context, input_zp, weight_zp);
}

::mlir::Attribute ConvOpQuantizationAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int64_t> _result_input_zp;
  ::mlir::FailureOr<int64_t> _result_weight_zp;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_input_zp = false;
  bool _seen_weight_zp = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_input_zp && _paramKey == "input_zp") {
        _seen_input_zp = true;

        // Parse variable 'input_zp'
        _result_input_zp = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_input_zp)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_ConvOpQuantizationAttr parameter 'input_zp' which is to be a `int64_t`");
          return {};
        }
      } else if (!_seen_weight_zp && _paramKey == "weight_zp") {
        _seen_weight_zp = true;

        // Parse variable 'weight_zp'
        _result_weight_zp = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_weight_zp)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_ConvOpQuantizationAttr parameter 'weight_zp' which is to be a `int64_t`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 2; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 2 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_input_zp));
  assert(::mlir::succeeded(_result_weight_zp));
  return ConvOpQuantizationAttr::get(odsParser.getContext(),
      int64_t((*_result_input_zp)),
      int64_t((*_result_weight_zp)));
}

void ConvOpQuantizationAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "input_zp = ";
    odsPrinter.printStrippedAttrOrType(getInputZp());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "weight_zp = ";
    odsPrinter.printStrippedAttrOrType(getWeightZp());
  }
  odsPrinter << ">";
}

int64_t ConvOpQuantizationAttr::getInputZp() const {
  return getImpl()->input_zp;
}

int64_t ConvOpQuantizationAttr::getWeightZp() const {
  return getImpl()->weight_zp;
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ConvOpQuantizationAttr)
namespace mlir {
namespace tosa {
namespace detail {
struct MatMulOpQuantizationAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, int64_t>;
  MatMulOpQuantizationAttrStorage(int64_t a_zp, int64_t b_zp) : a_zp(a_zp), b_zp(b_zp) {}

  KeyTy getAsKey() const {
    return KeyTy(a_zp, b_zp);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (a_zp == std::get<0>(tblgenKey)) && (b_zp == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static MatMulOpQuantizationAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto a_zp = std::get<0>(tblgenKey);
    auto b_zp = std::get<1>(tblgenKey);
    return new (allocator.allocate<MatMulOpQuantizationAttrStorage>()) MatMulOpQuantizationAttrStorage(a_zp, b_zp);
  }

  int64_t a_zp;
  int64_t b_zp;
};
} // namespace detail
MatMulOpQuantizationAttr MatMulOpQuantizationAttr::get(::mlir::MLIRContext *context, int64_t a_zp, int64_t b_zp) {
  return Base::get(context, a_zp, b_zp);
}

::mlir::Attribute MatMulOpQuantizationAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int64_t> _result_a_zp;
  ::mlir::FailureOr<int64_t> _result_b_zp;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_a_zp = false;
  bool _seen_b_zp = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_a_zp && _paramKey == "a_zp") {
        _seen_a_zp = true;

        // Parse variable 'a_zp'
        _result_a_zp = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_a_zp)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_MatMulOpQuantizationAttr parameter 'a_zp' which is to be a `int64_t`");
          return {};
        }
      } else if (!_seen_b_zp && _paramKey == "b_zp") {
        _seen_b_zp = true;

        // Parse variable 'b_zp'
        _result_b_zp = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_b_zp)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_MatMulOpQuantizationAttr parameter 'b_zp' which is to be a `int64_t`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 2; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 2 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_a_zp));
  assert(::mlir::succeeded(_result_b_zp));
  return MatMulOpQuantizationAttr::get(odsParser.getContext(),
      int64_t((*_result_a_zp)),
      int64_t((*_result_b_zp)));
}

void MatMulOpQuantizationAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "a_zp = ";
    odsPrinter.printStrippedAttrOrType(getAZp());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "b_zp = ";
    odsPrinter.printStrippedAttrOrType(getBZp());
  }
  odsPrinter << ">";
}

int64_t MatMulOpQuantizationAttr::getAZp() const {
  return getImpl()->a_zp;
}

int64_t MatMulOpQuantizationAttr::getBZp() const {
  return getImpl()->b_zp;
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::MatMulOpQuantizationAttr)
namespace mlir {
namespace tosa {
namespace detail {
struct PadOpQuantizationAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t>;
  PadOpQuantizationAttrStorage(int64_t input_zp) : input_zp(input_zp) {}

  KeyTy getAsKey() const {
    return KeyTy(input_zp);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (input_zp == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static PadOpQuantizationAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto input_zp = std::get<0>(tblgenKey);
    return new (allocator.allocate<PadOpQuantizationAttrStorage>()) PadOpQuantizationAttrStorage(input_zp);
  }

  int64_t input_zp;
};
} // namespace detail
PadOpQuantizationAttr PadOpQuantizationAttr::get(::mlir::MLIRContext *context, int64_t input_zp) {
  return Base::get(context, input_zp);
}

::mlir::Attribute PadOpQuantizationAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int64_t> _result_input_zp;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_input_zp = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_input_zp && _paramKey == "input_zp") {
        _seen_input_zp = true;

        // Parse variable 'input_zp'
        _result_input_zp = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_input_zp)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_PadOpQuantizationAttr parameter 'input_zp' which is to be a `int64_t`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 1; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 1 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_input_zp));
  return PadOpQuantizationAttr::get(odsParser.getContext(),
      int64_t((*_result_input_zp)));
}

void PadOpQuantizationAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "input_zp = ";
    odsPrinter.printStrippedAttrOrType(getInputZp());
  }
  odsPrinter << ">";
}

int64_t PadOpQuantizationAttr::getInputZp() const {
  return getImpl()->input_zp;
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::PadOpQuantizationAttr)
namespace mlir {
namespace tosa {
namespace detail {
struct UnaryOpQuantizationAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, int64_t>;
  UnaryOpQuantizationAttrStorage(int64_t input_zp, int64_t output_zp) : input_zp(input_zp), output_zp(output_zp) {}

  KeyTy getAsKey() const {
    return KeyTy(input_zp, output_zp);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (input_zp == std::get<0>(tblgenKey)) && (output_zp == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static UnaryOpQuantizationAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto input_zp = std::get<0>(tblgenKey);
    auto output_zp = std::get<1>(tblgenKey);
    return new (allocator.allocate<UnaryOpQuantizationAttrStorage>()) UnaryOpQuantizationAttrStorage(input_zp, output_zp);
  }

  int64_t input_zp;
  int64_t output_zp;
};
} // namespace detail
UnaryOpQuantizationAttr UnaryOpQuantizationAttr::get(::mlir::MLIRContext *context, int64_t input_zp, int64_t output_zp) {
  return Base::get(context, input_zp, output_zp);
}

::mlir::Attribute UnaryOpQuantizationAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int64_t> _result_input_zp;
  ::mlir::FailureOr<int64_t> _result_output_zp;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_input_zp = false;
  bool _seen_output_zp = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_input_zp && _paramKey == "input_zp") {
        _seen_input_zp = true;

        // Parse variable 'input_zp'
        _result_input_zp = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_input_zp)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_UnaryOpQuantizationAttr parameter 'input_zp' which is to be a `int64_t`");
          return {};
        }
      } else if (!_seen_output_zp && _paramKey == "output_zp") {
        _seen_output_zp = true;

        // Parse variable 'output_zp'
        _result_output_zp = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_output_zp)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_UnaryOpQuantizationAttr parameter 'output_zp' which is to be a `int64_t`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 2; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 2 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_input_zp));
  assert(::mlir::succeeded(_result_output_zp));
  return UnaryOpQuantizationAttr::get(odsParser.getContext(),
      int64_t((*_result_input_zp)),
      int64_t((*_result_output_zp)));
}

void UnaryOpQuantizationAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "input_zp = ";
    odsPrinter.printStrippedAttrOrType(getInputZp());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "output_zp = ";
    odsPrinter.printStrippedAttrOrType(getOutputZp());
  }
  odsPrinter << ">";
}

int64_t UnaryOpQuantizationAttr::getInputZp() const {
  return getImpl()->input_zp;
}

int64_t UnaryOpQuantizationAttr::getOutputZp() const {
  return getImpl()->output_zp;
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::UnaryOpQuantizationAttr)
namespace mlir {
namespace tosa {

/// Parse an attribute registered to this dialect.
::mlir::Attribute TosaDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void TosaDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace tosa
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

