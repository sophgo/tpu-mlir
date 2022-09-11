/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::gpu::AllReduceOperationAttr,
::mlir::gpu::DimensionAttr,
::mlir::gpu::ShuffleModeAttr,
::mlir::gpu::MMAElementwiseOpAttr,
::mlir::gpu::ParallelLoopDimMappingAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::gpu::AllReduceOperationAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::AllReduceOperationAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::DimensionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::DimensionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::ShuffleModeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::ShuffleModeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::MMAElementwiseOpAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::MMAElementwiseOpAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::gpu::ParallelLoopDimMappingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::gpu::ParallelLoopDimMappingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return llvm::None;
    });
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::gpu::AllReduceOperationAttr>([&](auto t) {
      printer << ::mlir::gpu::AllReduceOperationAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::DimensionAttr>([&](auto t) {
      printer << ::mlir::gpu::DimensionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::ShuffleModeAttr>([&](auto t) {
      printer << ::mlir::gpu::ShuffleModeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::MMAElementwiseOpAttr>([&](auto t) {
      printer << ::mlir::gpu::MMAElementwiseOpAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::gpu::ParallelLoopDimMappingAttr>([&](auto t) {
      printer << ::mlir::gpu::ParallelLoopDimMappingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace gpu {
namespace detail {
struct AllReduceOperationAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::AllReduceOperation>;
  AllReduceOperationAttrStorage(::mlir::gpu::AllReduceOperation value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static AllReduceOperationAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<AllReduceOperationAttrStorage>()) AllReduceOperationAttrStorage(value);
  }

  ::mlir::gpu::AllReduceOperation value;
};
} // namespace detail
AllReduceOperationAttr AllReduceOperationAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::AllReduceOperation value) {
  return Base::get(context, value);
}

::mlir::Attribute AllReduceOperationAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::mlir::FailureOr<::mlir::gpu::AllReduceOperation> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::AllReduceOperation> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeAllReduceOperation(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::AllReduceOperation" << " to be one of: " << "add" << ", " << "and" << ", " << "max" << ", " << "min" << ", " << "mul" << ", " << "or" << ", " << "xor")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_AllReduceOperationAttr parameter 'value' which is to be a `::mlir::gpu::AllReduceOperation`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return AllReduceOperationAttr::get(odsParser.getContext(),
      ::mlir::gpu::AllReduceOperation((*_result_value)));
}

void AllReduceOperationAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyAllReduceOperation(getValue());
}

::mlir::gpu::AllReduceOperation AllReduceOperationAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::AllReduceOperationAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct DimensionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::Dimension>;
  DimensionAttrStorage(::mlir::gpu::Dimension value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static DimensionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<DimensionAttrStorage>()) DimensionAttrStorage(value);
  }

  ::mlir::gpu::Dimension value;
};
} // namespace detail
DimensionAttr DimensionAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::Dimension value) {
  return Base::get(context, value);
}

::mlir::Attribute DimensionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::mlir::FailureOr<::mlir::gpu::Dimension> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::Dimension> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeDimension(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::Dimension" << " to be one of: " << "x" << ", " << "y" << ", " << "z")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_DimensionAttr parameter 'value' which is to be a `::mlir::gpu::Dimension`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return DimensionAttr::get(odsParser.getContext(),
      ::mlir::gpu::Dimension((*_result_value)));
}

void DimensionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyDimension(getValue());
}

::mlir::gpu::Dimension DimensionAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::DimensionAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct ShuffleModeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::ShuffleMode>;
  ShuffleModeAttrStorage(::mlir::gpu::ShuffleMode value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ShuffleModeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<ShuffleModeAttrStorage>()) ShuffleModeAttrStorage(value);
  }

  ::mlir::gpu::ShuffleMode value;
};
} // namespace detail
ShuffleModeAttr ShuffleModeAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::ShuffleMode value) {
  return Base::get(context, value);
}

::mlir::Attribute ShuffleModeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::mlir::FailureOr<::mlir::gpu::ShuffleMode> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::ShuffleMode> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeShuffleMode(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::ShuffleMode" << " to be one of: " << "xor" << ", " << "up" << ", " << "down" << ", " << "idx")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_ShuffleModeAttr parameter 'value' which is to be a `::mlir::gpu::ShuffleMode`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ShuffleModeAttr::get(odsParser.getContext(),
      ::mlir::gpu::ShuffleMode((*_result_value)));
}

void ShuffleModeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyShuffleMode(getValue());
}

::mlir::gpu::ShuffleMode ShuffleModeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::ShuffleModeAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct MMAElementwiseOpAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::MMAElementwiseOp>;
  MMAElementwiseOpAttrStorage(::mlir::gpu::MMAElementwiseOp value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MMAElementwiseOpAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<MMAElementwiseOpAttrStorage>()) MMAElementwiseOpAttrStorage(value);
  }

  ::mlir::gpu::MMAElementwiseOp value;
};
} // namespace detail
MMAElementwiseOpAttr MMAElementwiseOpAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::MMAElementwiseOp value) {
  return Base::get(context, value);
}

::mlir::Attribute MMAElementwiseOpAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::mlir::FailureOr<::mlir::gpu::MMAElementwiseOp> _result_value;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::gpu::MMAElementwiseOp> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::gpu::symbolizeMMAElementwiseOp(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::MMAElementwiseOp" << " to be one of: " << "addf" << ", " << "mulf" << ", " << "maxf" << ", " << "minf" << ", " << "divf")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MMAElementWiseAttr parameter 'value' which is to be a `::mlir::gpu::MMAElementwiseOp`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return MMAElementwiseOpAttr::get(odsParser.getContext(),
      ::mlir::gpu::MMAElementwiseOp((*_result_value)));
}

void MMAElementwiseOpAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyMMAElementwiseOp(getValue());
}

::mlir::gpu::MMAElementwiseOp MMAElementwiseOpAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::MMAElementwiseOpAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct ParallelLoopDimMappingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::Processor, AffineMap, AffineMap>;
  ParallelLoopDimMappingAttrStorage(::mlir::gpu::Processor processor, AffineMap map, AffineMap bound) : processor(processor), map(map), bound(bound) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (processor == std::get<0>(tblgenKey)) && (map == std::get<1>(tblgenKey)) && (bound == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static ParallelLoopDimMappingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto processor = std::get<0>(tblgenKey);
    auto map = std::get<1>(tblgenKey);
    auto bound = std::get<2>(tblgenKey);
    return new (allocator.allocate<ParallelLoopDimMappingAttrStorage>()) ParallelLoopDimMappingAttrStorage(processor, map, bound);
  }

  ::mlir::gpu::Processor processor;
  AffineMap map;
  AffineMap bound;
};
} // namespace detail
ParallelLoopDimMappingAttr ParallelLoopDimMappingAttr::get(::mlir::MLIRContext *context, ::mlir::gpu::Processor processor, AffineMap map, AffineMap bound) {
  return Base::get(context, processor, map, bound);
}

::mlir::Attribute ParallelLoopDimMappingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::mlir::FailureOr<::mlir::gpu::Processor> _result_processor;
  ::mlir::FailureOr<AffineMap> _result_map;
  ::mlir::FailureOr<AffineMap> _result_bound;
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_processor = false;
  bool _seen_map = false;
  bool _seen_bound = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_processor && _paramKey == "processor") {
        _seen_processor = true;

        // Parse variable 'processor'
        _result_processor = [&]() -> ::mlir::FailureOr<::mlir::gpu::Processor> {
            auto loc = odsParser.getCurrentLocation();
            ::llvm::StringRef enumKeyword;
            if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
              return ::mlir::failure();
            auto maybeEnum = ::mlir::gpu::symbolizeProcessor(enumKeyword);
            if (maybeEnum)
              return *maybeEnum;
            return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::gpu::Processor" << " to be one of: " << "block_x" << ", " << "block_y" << ", " << "block_z" << ", " << "thread_x" << ", " << "thread_y" << ", " << "thread_z" << ", " << "sequential")};
          }();
        if (::mlir::failed(_result_processor)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ParallelLoopDimMappingAttr parameter 'processor' which is to be a `::mlir::gpu::Processor`");
          return {};
        }
      } else if (!_seen_map && _paramKey == "map") {
        _seen_map = true;

        // Parse variable 'map'
        _result_map = ::mlir::FieldParser<AffineMap>::parse(odsParser);
        if (::mlir::failed(_result_map)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ParallelLoopDimMappingAttr parameter 'map' which is to be a `AffineMap`");
          return {};
        }
      } else if (!_seen_bound && _paramKey == "bound") {
        _seen_bound = true;

        // Parse variable 'bound'
        _result_bound = ::mlir::FieldParser<AffineMap>::parse(odsParser);
        if (::mlir::failed(_result_bound)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse ParallelLoopDimMappingAttr parameter 'bound' which is to be a `AffineMap`");
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
  assert(::mlir::succeeded(_result_processor));
  assert(::mlir::succeeded(_result_map));
  assert(::mlir::succeeded(_result_bound));
  return ParallelLoopDimMappingAttr::get(odsParser.getContext(),
      ::mlir::gpu::Processor((*_result_processor)),
      AffineMap((*_result_map)),
      AffineMap((*_result_bound)));
}

void ParallelLoopDimMappingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "processor = ";
    odsPrinter << stringifyProcessor(getProcessor());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "map = ";
    odsPrinter.printStrippedAttrOrType(getMap());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "bound = ";
    odsPrinter.printStrippedAttrOrType(getBound());
  }
  odsPrinter << ">";
}

::mlir::gpu::Processor ParallelLoopDimMappingAttr::getProcessor() const {
  return getImpl()->processor;
}

AffineMap ParallelLoopDimMappingAttr::getMap() const {
  return getImpl()->map;
}

AffineMap ParallelLoopDimMappingAttr::getBound() const {
  return getImpl()->bound;
}

} // namespace gpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::ParallelLoopDimMappingAttr)
namespace mlir {
namespace gpu {

/// Parse an attribute registered to this dialect.
::mlir::Attribute GPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void GPUDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace gpu
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

