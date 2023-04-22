/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::sparse_tensor::SparseTensorDimSliceAttr,
::mlir::sparse_tensor::SparseTensorEncodingAttr,
::mlir::sparse_tensor::SparseTensorSortKindAttr,
::mlir::sparse_tensor::StorageSpecifierKindAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::sparse_tensor::SparseTensorDimSliceAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::sparse_tensor::SparseTensorDimSliceAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::sparse_tensor::SparseTensorEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::sparse_tensor::SparseTensorEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::sparse_tensor::SparseTensorSortKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::sparse_tensor::SparseTensorSortKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::sparse_tensor::StorageSpecifierKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::sparse_tensor::StorageSpecifierKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::sparse_tensor::SparseTensorDimSliceAttr>([&](auto t) {
      printer << ::mlir::sparse_tensor::SparseTensorDimSliceAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::sparse_tensor::SparseTensorEncodingAttr>([&](auto t) {
      printer << ::mlir::sparse_tensor::SparseTensorEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::sparse_tensor::SparseTensorSortKindAttr>([&](auto t) {
      printer << ::mlir::sparse_tensor::SparseTensorSortKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::sparse_tensor::StorageSpecifierKindAttr>([&](auto t) {
      printer << ::mlir::sparse_tensor::StorageSpecifierKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace sparse_tensor {
namespace detail {
struct SparseTensorDimSliceAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, int64_t, int64_t>;
  SparseTensorDimSliceAttrStorage(int64_t offset, int64_t size, int64_t stride) : offset(offset), size(size), stride(stride) {}

  KeyTy getAsKey() const {
    return KeyTy(offset, size, stride);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (offset == std::get<0>(tblgenKey)) && (size == std::get<1>(tblgenKey)) && (stride == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static SparseTensorDimSliceAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto offset = std::get<0>(tblgenKey);
    auto size = std::get<1>(tblgenKey);
    auto stride = std::get<2>(tblgenKey);
    return new (allocator.allocate<SparseTensorDimSliceAttrStorage>()) SparseTensorDimSliceAttrStorage(offset, size, stride);
  }

  int64_t offset;
  int64_t size;
  int64_t stride;
};
} // namespace detail
SparseTensorDimSliceAttr SparseTensorDimSliceAttr::get(::mlir::MLIRContext *context, int64_t offset, int64_t size, int64_t stride) {
  return Base::get(context, offset, size, stride);
}

SparseTensorDimSliceAttr SparseTensorDimSliceAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int64_t offset, int64_t size, int64_t stride) {
  return Base::getChecked(emitError, context, offset, size, stride);
}

int64_t SparseTensorDimSliceAttr::getOffset() const {
  return getImpl()->offset;
}

int64_t SparseTensorDimSliceAttr::getSize() const {
  return getImpl()->size;
}

int64_t SparseTensorDimSliceAttr::getStride() const {
  return getImpl()->stride;
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorDimSliceAttr)
namespace mlir {
namespace sparse_tensor {
namespace detail {
struct SparseTensorEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<::mlir::sparse_tensor::DimLevelType>, AffineMap, AffineMap, unsigned, unsigned, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr>>;
  SparseTensorEncodingAttrStorage(::llvm::ArrayRef<::mlir::sparse_tensor::DimLevelType> dimLevelType, AffineMap dimOrdering, AffineMap higherOrdering, unsigned posWidth, unsigned crdWidth, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices) : dimLevelType(dimLevelType), dimOrdering(dimOrdering), higherOrdering(higherOrdering), posWidth(posWidth), crdWidth(crdWidth), dimSlices(dimSlices) {}

  KeyTy getAsKey() const {
    return KeyTy(dimLevelType, dimOrdering, higherOrdering, posWidth, crdWidth, dimSlices);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (dimLevelType == std::get<0>(tblgenKey)) && (dimOrdering == std::get<1>(tblgenKey)) && (higherOrdering == std::get<2>(tblgenKey)) && (posWidth == std::get<3>(tblgenKey)) && (crdWidth == std::get<4>(tblgenKey)) && (dimSlices == std::get<5>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey));
  }

  static SparseTensorEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto dimLevelType = std::get<0>(tblgenKey);
    auto dimOrdering = std::get<1>(tblgenKey);
    auto higherOrdering = std::get<2>(tblgenKey);
    auto posWidth = std::get<3>(tblgenKey);
    auto crdWidth = std::get<4>(tblgenKey);
    auto dimSlices = std::get<5>(tblgenKey);
    dimLevelType = allocator.copyInto(dimLevelType);
    dimSlices = allocator.copyInto(dimSlices);
    return new (allocator.allocate<SparseTensorEncodingAttrStorage>()) SparseTensorEncodingAttrStorage(dimLevelType, dimOrdering, higherOrdering, posWidth, crdWidth, dimSlices);
  }

  ::llvm::ArrayRef<::mlir::sparse_tensor::DimLevelType> dimLevelType;
  AffineMap dimOrdering;
  AffineMap higherOrdering;
  unsigned posWidth;
  unsigned crdWidth;
  ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices;
};
} // namespace detail
SparseTensorEncodingAttr SparseTensorEncodingAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::sparse_tensor::DimLevelType> dimLevelType, AffineMap dimOrdering, AffineMap higherOrdering, unsigned posWidth, unsigned crdWidth, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices) {
  return Base::get(context, dimLevelType, dimOrdering, higherOrdering, posWidth, crdWidth, dimSlices);
}

SparseTensorEncodingAttr SparseTensorEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::sparse_tensor::DimLevelType> dimLevelType, AffineMap dimOrdering, AffineMap higherOrdering, unsigned posWidth, unsigned crdWidth, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices) {
  return Base::getChecked(emitError, context, dimLevelType, dimOrdering, higherOrdering, posWidth, crdWidth, dimSlices);
}

SparseTensorEncodingAttr SparseTensorEncodingAttr::get(::mlir::MLIRContext *context, ArrayRef<::mlir::sparse_tensor::DimLevelType> dimLevelType, AffineMap dimOrdering, AffineMap higherOrdering, unsigned posWidth, unsigned crdWidth) {
  return Base::get(context, dimLevelType,
                     dimOrdering,
                     higherOrdering,
                     posWidth,
                     crdWidth,
                     ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr>{});
}

SparseTensorEncodingAttr SparseTensorEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ArrayRef<::mlir::sparse_tensor::DimLevelType> dimLevelType, AffineMap dimOrdering, AffineMap higherOrdering, unsigned posWidth, unsigned crdWidth) {
  return Base::getChecked(emitError, context, dimLevelType,
                     dimOrdering,
                     higherOrdering,
                     posWidth,
                     crdWidth,
                     ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr>{});
}

::llvm::ArrayRef<::mlir::sparse_tensor::DimLevelType> SparseTensorEncodingAttr::getDimLevelType() const {
  return getImpl()->dimLevelType;
}

AffineMap SparseTensorEncodingAttr::getDimOrdering() const {
  return getImpl()->dimOrdering;
}

AffineMap SparseTensorEncodingAttr::getHigherOrdering() const {
  return getImpl()->higherOrdering;
}

unsigned SparseTensorEncodingAttr::getPosWidth() const {
  return getImpl()->posWidth;
}

unsigned SparseTensorEncodingAttr::getCrdWidth() const {
  return getImpl()->crdWidth;
}

::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> SparseTensorEncodingAttr::getDimSlices() const {
  return getImpl()->dimSlices;
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorEncodingAttr)
namespace mlir {
namespace sparse_tensor {
namespace detail {
struct SparseTensorSortKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::sparse_tensor::SparseTensorSortKind>;
  SparseTensorSortKindAttrStorage(::mlir::sparse_tensor::SparseTensorSortKind value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static SparseTensorSortKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<SparseTensorSortKindAttrStorage>()) SparseTensorSortKindAttrStorage(value);
  }

  ::mlir::sparse_tensor::SparseTensorSortKind value;
};
} // namespace detail
SparseTensorSortKindAttr SparseTensorSortKindAttr::get(::mlir::MLIRContext *context, ::mlir::sparse_tensor::SparseTensorSortKind value) {
  return Base::get(context, value);
}

::mlir::Attribute SparseTensorSortKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::sparse_tensor::SparseTensorSortKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::sparse_tensor::SparseTensorSortKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::sparse_tensor::symbolizeSparseTensorSortKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::sparse_tensor::SparseTensorSortKind" << " to be one of: " << "hybrid_quick_sort" << ", " << "insertion_sort_stable" << ", " << "quick_sort" << ", " << "heap_sort")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SparseTensorSortKindAttr parameter 'value' which is to be a `::mlir::sparse_tensor::SparseTensorSortKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return SparseTensorSortKindAttr::get(odsParser.getContext(),
      ::mlir::sparse_tensor::SparseTensorSortKind((*_result_value)));
}

void SparseTensorSortKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifySparseTensorSortKind(getValue());
}

::mlir::sparse_tensor::SparseTensorSortKind SparseTensorSortKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorSortKindAttr)
namespace mlir {
namespace sparse_tensor {
namespace detail {
struct StorageSpecifierKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::sparse_tensor::StorageSpecifierKind>;
  StorageSpecifierKindAttrStorage(::mlir::sparse_tensor::StorageSpecifierKind value) : value(value) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static StorageSpecifierKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<StorageSpecifierKindAttrStorage>()) StorageSpecifierKindAttrStorage(value);
  }

  ::mlir::sparse_tensor::StorageSpecifierKind value;
};
} // namespace detail
StorageSpecifierKindAttr StorageSpecifierKindAttr::get(::mlir::MLIRContext *context, ::mlir::sparse_tensor::StorageSpecifierKind value) {
  return Base::get(context, value);
}

::mlir::Attribute StorageSpecifierKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::sparse_tensor::StorageSpecifierKind> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::sparse_tensor::StorageSpecifierKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::sparse_tensor::symbolizeStorageSpecifierKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::mlir::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::sparse_tensor::StorageSpecifierKind" << " to be one of: " << "lvl_sz" << ", " << "pos_mem_sz" << ", " << "crd_mem_sz" << ", " << "val_mem_sz" << ", " << "dim_offset" << ", " << "dim_stride")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SparseTensorStorageSpecifierKindAttr parameter 'value' which is to be a `::mlir::sparse_tensor::StorageSpecifierKind`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return StorageSpecifierKindAttr::get(odsParser.getContext(),
      ::mlir::sparse_tensor::StorageSpecifierKind((*_result_value)));
}

void StorageSpecifierKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyStorageSpecifierKind(getValue());
}

::mlir::sparse_tensor::StorageSpecifierKind StorageSpecifierKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::StorageSpecifierKindAttr)
namespace mlir {
namespace sparse_tensor {

/// Parse an attribute registered to this dialect.
::mlir::Attribute SparseTensorDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void SparseTensorDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace sparse_tensor
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

