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
::mlir::gpu::MMAElementwiseOpAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  if (mnemonic == ::mlir::gpu::AllReduceOperationAttr::getMnemonic()) {
    value = ::mlir::gpu::AllReduceOperationAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::gpu::DimensionAttr::getMnemonic()) {
    value = ::mlir::gpu::DimensionAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::gpu::ShuffleModeAttr::getMnemonic()) {
    value = ::mlir::gpu::ShuffleModeAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  if (mnemonic == ::mlir::gpu::MMAElementwiseOpAttr::getMnemonic()) {
    value = ::mlir::gpu::MMAElementwiseOpAttr::parse(parser, type);
    return ::mlir::success(!!value);
  }
  return {};
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
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace gpu {
namespace detail {
struct AllReduceOperationAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::AllReduceOperation>;
  AllReduceOperationAttrStorage(::mlir::gpu::AllReduceOperation value) : ::mlir::AttributeStorage(), value(value) {}

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
      return {(::mlir::LogicalResult)odsParser.emitError(loc, "expected ::mlir::gpu::AllReduceOperation to be one of: add, and, max, min, mul, or, xor")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_AllReduceOperationAttr parameter 'value' which is to be a `::mlir::gpu::AllReduceOperation`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return AllReduceOperationAttr::get(odsParser.getContext(),
      *_result_value);
}

void AllReduceOperationAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << ' ';
  odsPrinter << stringifyAllReduceOperation(getValue());
}

::mlir::gpu::AllReduceOperation AllReduceOperationAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::AllReduceOperationAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct DimensionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::Dimension>;
  DimensionAttrStorage(::mlir::gpu::Dimension value) : ::mlir::AttributeStorage(), value(value) {}

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
      return {(::mlir::LogicalResult)odsParser.emitError(loc, "expected ::mlir::gpu::Dimension to be one of: x, y, z")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_DimensionAttr parameter 'value' which is to be a `::mlir::gpu::Dimension`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return DimensionAttr::get(odsParser.getContext(),
      *_result_value);
}

void DimensionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << ' ';
  odsPrinter << stringifyDimension(getValue());
}

::mlir::gpu::Dimension DimensionAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::DimensionAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct ShuffleModeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::ShuffleMode>;
  ShuffleModeAttrStorage(::mlir::gpu::ShuffleMode value) : ::mlir::AttributeStorage(), value(value) {}

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
      return {(::mlir::LogicalResult)odsParser.emitError(loc, "expected ::mlir::gpu::ShuffleMode to be one of: xor, up, down, idx")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse GPU_ShuffleModeAttr parameter 'value' which is to be a `::mlir::gpu::ShuffleMode`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ShuffleModeAttr::get(odsParser.getContext(),
      *_result_value);
}

void ShuffleModeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << ' ';
  odsPrinter << stringifyShuffleMode(getValue());
}

::mlir::gpu::ShuffleMode ShuffleModeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::ShuffleModeAttr)
namespace mlir {
namespace gpu {
namespace detail {
struct MMAElementwiseOpAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::gpu::MMAElementwiseOp>;
  MMAElementwiseOpAttrStorage(::mlir::gpu::MMAElementwiseOp value) : ::mlir::AttributeStorage(), value(value) {}

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
      return {(::mlir::LogicalResult)odsParser.emitError(loc, "expected ::mlir::gpu::MMAElementwiseOp to be one of: addf, mulf, maxf, minf, divf")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse MMAElementWiseAttr parameter 'value' which is to be a `::mlir::gpu::MMAElementwiseOp`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return MMAElementwiseOpAttr::get(odsParser.getContext(),
      *_result_value);
}

void MMAElementwiseOpAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << ' ';
  odsPrinter << stringifyMMAElementwiseOp(getValue());
}

::mlir::gpu::MMAElementwiseOp MMAElementwiseOpAttr::getValue() const {
  return getImpl()->value;
}

} // namespace gpu
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::MMAElementwiseOpAttr)
namespace mlir {
namespace gpu {

/// Parse an attribute registered to this dialect.
::mlir::Attribute GPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void GPUDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
}
} // namespace gpu
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

