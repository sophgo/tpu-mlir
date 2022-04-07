/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::AffineMapAttr,
::mlir::ArrayAttr,
::mlir::DenseIntOrFPElementsAttr,
::mlir::DenseStringElementsAttr,
::mlir::DictionaryAttr,
::mlir::FloatAttr,
::mlir::IntegerAttr,
::mlir::IntegerSetAttr,
::mlir::OpaqueAttr,
::mlir::OpaqueElementsAttr,
::mlir::SparseElementsAttr,
::mlir::StringAttr,
::mlir::SymbolRefAttr,
::mlir::TypeAttr,
::mlir::UnitAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

namespace mlir {
namespace detail {
struct AffineMapAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<AffineMap>;
  AffineMapAttrStorage(AffineMap value) : ::mlir::AttributeStorage(IndexType::get(value.getContext())), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static AffineMapAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<AffineMapAttrStorage>()) AffineMapAttrStorage(value);
  }

  AffineMap value;
};
} // namespace detail
AffineMapAttr AffineMapAttr::get(AffineMap value) {
  return Base::get(value.getContext(), value);
}

AffineMap AffineMapAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::AffineMapAttr)
namespace mlir {
namespace detail {
struct ArrayAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<Attribute>>;
  ArrayAttrStorage(::llvm::ArrayRef<Attribute> value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ArrayAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    value = allocator.copyInto(value);
    return new (allocator.allocate<ArrayAttrStorage>()) ArrayAttrStorage(value);
  }

  ::llvm::ArrayRef<Attribute> value;
};
} // namespace detail
ArrayAttr ArrayAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<Attribute> value) {
  return Base::get(context, value);
}

::llvm::ArrayRef<Attribute> ArrayAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ArrayAttr)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DenseIntOrFPElementsAttr)
namespace mlir {
DenseStringElementsAttr DenseStringElementsAttr::get(ShapedType type, ArrayRef<StringRef> values) {
  return Base::get(type.getContext(), type, values,
               /* isSplat */(values.size() == 1));
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DenseStringElementsAttr)
namespace mlir {
namespace detail {
struct DictionaryAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<NamedAttribute>>;
  DictionaryAttrStorage(::llvm::ArrayRef<NamedAttribute> value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static DictionaryAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    value = allocator.copyInto(value);
    return new (allocator.allocate<DictionaryAttrStorage>()) DictionaryAttrStorage(value);
  }

  ::llvm::ArrayRef<NamedAttribute> value;
};
} // namespace detail
::llvm::ArrayRef<NamedAttribute> DictionaryAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DictionaryAttr)
namespace mlir {
namespace detail {
struct FloatAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::Type, ::llvm::APFloat>;
  FloatAttrStorage(::mlir::Type type, ::llvm::APFloat value) : ::mlir::AttributeStorage(type), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (getType() == std::get<0>(tblgenKey)) && (value.bitwiseIsEqual(std::get<1>(tblgenKey)));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static FloatAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto type = std::get<0>(tblgenKey);
    auto value = std::get<1>(tblgenKey);
    return new (allocator.allocate<FloatAttrStorage>()) FloatAttrStorage(type, value);
  }

  ::llvm::APFloat value;
};
} // namespace detail
FloatAttr FloatAttr::get(Type type, const APFloat &value) {
  return Base::get(type.getContext(), type, value);
}

FloatAttr FloatAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type type, const APFloat &value) {
  return Base::getChecked(emitError, type.getContext(), type, value);
}

FloatAttr FloatAttr::get(Type type, double value) {
  if (type.isF64() || !type.isa<FloatType>())
    return Base::get(type.getContext(), type, APFloat(value));

  // This handles, e.g., F16 because there is no APFloat constructor for it.
  bool unused;
  APFloat val(value);
  val.convert(type.cast<FloatType>().getFloatSemantics(),
              APFloat::rmNearestTiesToEven, &unused);
  return Base::get(type.getContext(), type, val);
}

FloatAttr FloatAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type type, double value) {
  if (type.isF64() || !type.isa<FloatType>())
    return Base::getChecked(emitError, type.getContext(), type, APFloat(value));

  // This handles, e.g., F16 because there is no APFloat constructor for it.
  bool unused;
  APFloat val(value);
  val.convert(type.cast<FloatType>().getFloatSemantics(),
              APFloat::rmNearestTiesToEven, &unused);
  return Base::getChecked(emitError, type.getContext(), type, val);
}

::mlir::Type FloatAttr::getType() const {
  return getImpl()->getType().cast<::mlir::Type>();
}

::llvm::APFloat FloatAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::FloatAttr)
namespace mlir {
namespace detail {
struct IntegerAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::Type, APInt>;
  IntegerAttrStorage(::mlir::Type type, APInt value) : ::mlir::AttributeStorage(type), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (getType() == std::get<0>(tblgenKey)) && (value == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static IntegerAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto type = std::get<0>(tblgenKey);
    auto value = std::get<1>(tblgenKey);
    return new (allocator.allocate<IntegerAttrStorage>()) IntegerAttrStorage(type, value);
  }

  APInt value;
};
} // namespace detail
IntegerAttr IntegerAttr::get(Type type, const APInt &value) {
  if (type.isSignlessInteger(1))
    return BoolAttr::get(type.getContext(), value.getBoolValue());
  return Base::get(type.getContext(), type, value);
}

IntegerAttr IntegerAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type type, const APInt &value) {
  if (type.isSignlessInteger(1))
    return BoolAttr::get(type.getContext(), value.getBoolValue());
  return Base::getChecked(emitError, type.getContext(), type, value);
}

IntegerAttr IntegerAttr::get(::mlir::MLIRContext *context, const APSInt &value) {
  auto signedness = value.isSigned() ?
    IntegerType::Signed : IntegerType::Unsigned;
  auto type = IntegerType::get(context, value.getBitWidth(), signedness);
  return Base::get(type.getContext(), type, value);
}

IntegerAttr IntegerAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, const APSInt &value) {
  auto signedness = value.isSigned() ?
    IntegerType::Signed : IntegerType::Unsigned;
  auto type = IntegerType::get(context, value.getBitWidth(), signedness);
  return Base::getChecked(emitError, type.getContext(), type, value);
}

IntegerAttr IntegerAttr::get(Type type, int64_t value) {
  // `index` has a defined internal storage width.
  if (type.isIndex()) {
    APInt apValue(IndexType::kInternalStorageBitWidth, value);
    return Base::get(type.getContext(), type, apValue);
  }

  IntegerType intTy = type.cast<IntegerType>();
  APInt apValue(intTy.getWidth(), value, intTy.isSignedInteger());
  return Base::get(type.getContext(), type, apValue);
}

IntegerAttr IntegerAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type type, int64_t value) {
  // `index` has a defined internal storage width.
  if (type.isIndex()) {
    APInt apValue(IndexType::kInternalStorageBitWidth, value);
    return Base::getChecked(emitError, type.getContext(), type, apValue);
  }

  IntegerType intTy = type.cast<IntegerType>();
  APInt apValue(intTy.getWidth(), value, intTy.isSignedInteger());
  return Base::getChecked(emitError, type.getContext(), type, apValue);
}

::mlir::Type IntegerAttr::getType() const {
  return getImpl()->getType().cast<::mlir::Type>();
}

APInt IntegerAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::IntegerAttr)
namespace mlir {
namespace detail {
struct IntegerSetAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<IntegerSet>;
  IntegerSetAttrStorage(IntegerSet value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static IntegerSetAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<IntegerSetAttrStorage>()) IntegerSetAttrStorage(value);
  }

  IntegerSet value;
};
} // namespace detail
IntegerSetAttr IntegerSetAttr::get(IntegerSet value) {
  return Base::get(value.getContext(), value);
}

IntegerSet IntegerSetAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::IntegerSetAttr)
namespace mlir {
namespace detail {
struct OpaqueAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<StringAttr, ::llvm::StringRef, ::mlir::Type>;
  OpaqueAttrStorage(StringAttr dialectNamespace, ::llvm::StringRef attrData, ::mlir::Type type) : ::mlir::AttributeStorage(type), dialectNamespace(dialectNamespace), attrData(attrData) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (dialectNamespace == std::get<0>(tblgenKey)) && (attrData == std::get<1>(tblgenKey)) && (getType() == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static OpaqueAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto dialectNamespace = std::get<0>(tblgenKey);
    auto attrData = std::get<1>(tblgenKey);
    auto type = std::get<2>(tblgenKey);
    attrData = allocator.copyInto(attrData);
    return new (allocator.allocate<OpaqueAttrStorage>()) OpaqueAttrStorage(dialectNamespace, attrData, type);
  }

  StringAttr dialectNamespace;
  ::llvm::StringRef attrData;
};
} // namespace detail
OpaqueAttr OpaqueAttr::get(StringAttr dialect, StringRef attrData, Type type) {
  return Base::get(dialect.getContext(), dialect, attrData, type);
}

OpaqueAttr OpaqueAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr dialect, StringRef attrData, Type type) {
  return Base::getChecked(emitError, dialect.getContext(), dialect, attrData, type);
}

StringAttr OpaqueAttr::getDialectNamespace() const {
  return getImpl()->dialectNamespace;
}

::llvm::StringRef OpaqueAttr::getAttrData() const {
  return getImpl()->attrData;
}

::mlir::Type OpaqueAttr::getType() const {
  return getImpl()->getType().cast<::mlir::Type>();
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::OpaqueAttr)
namespace mlir {
namespace detail {
struct OpaqueElementsAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<StringAttr, ::llvm::StringRef, ShapedType>;
  OpaqueElementsAttrStorage(StringAttr dialect, ::llvm::StringRef value, ShapedType type) : ::mlir::AttributeStorage(type), dialect(dialect), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (dialect == std::get<0>(tblgenKey)) && (value == std::get<1>(tblgenKey)) && (getType() == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static OpaqueElementsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto dialect = std::get<0>(tblgenKey);
    auto value = std::get<1>(tblgenKey);
    auto type = std::get<2>(tblgenKey);
    value = allocator.copyInto(value);
    return new (allocator.allocate<OpaqueElementsAttrStorage>()) OpaqueElementsAttrStorage(dialect, value, type);
  }

  StringAttr dialect;
  ::llvm::StringRef value;
};
} // namespace detail
OpaqueElementsAttr OpaqueElementsAttr::get(StringAttr dialect, ShapedType type, StringRef value) {
  return Base::get(dialect.getContext(), dialect, value, type);
}

OpaqueElementsAttr OpaqueElementsAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr dialect, ShapedType type, StringRef value) {
  return Base::getChecked(emitError, dialect.getContext(), dialect, value, type);
}

OpaqueElementsAttr OpaqueElementsAttr::get(Dialect *dialect, ShapedType type, StringRef value) {
  MLIRContext *ctxt = dialect->getContext();
  StringAttr dialectName = StringAttr::get(ctxt, dialect->getNamespace());
  return Base::get(ctxt, dialectName, value, type);
}

OpaqueElementsAttr OpaqueElementsAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Dialect *dialect, ShapedType type, StringRef value) {
  MLIRContext *ctxt = dialect->getContext();
  StringAttr dialectName = StringAttr::get(ctxt, dialect->getNamespace());
  return Base::getChecked(emitError, ctxt, dialectName, value, type);
}

StringAttr OpaqueElementsAttr::getDialect() const {
  return getImpl()->dialect;
}

::llvm::StringRef OpaqueElementsAttr::getValue() const {
  return getImpl()->value;
}

ShapedType OpaqueElementsAttr::getType() const {
  return getImpl()->getType().cast<ShapedType>();
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::OpaqueElementsAttr)
namespace mlir {
namespace detail {
struct SparseElementsAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<ShapedType, DenseIntElementsAttr, DenseElementsAttr>;
  SparseElementsAttrStorage(ShapedType type, DenseIntElementsAttr indices, DenseElementsAttr values) : ::mlir::AttributeStorage(type), indices(indices), values(values) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (getType() == std::get<0>(tblgenKey)) && (indices == std::get<1>(tblgenKey)) && (values == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static SparseElementsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto type = std::get<0>(tblgenKey);
    auto indices = std::get<1>(tblgenKey);
    auto values = std::get<2>(tblgenKey);
    return new (allocator.allocate<SparseElementsAttrStorage>()) SparseElementsAttrStorage(type, indices, values);
  }

  DenseIntElementsAttr indices;
  DenseElementsAttr values;
};
} // namespace detail
SparseElementsAttr SparseElementsAttr::get(ShapedType type, DenseElementsAttr indices, DenseElementsAttr values) {
  assert(indices.getType().getElementType().isInteger(64) &&
         "expected sparse indices to be 64-bit integer values");
  assert((type.isa<RankedTensorType, VectorType>()) &&
         "type must be ranked tensor or vector");
  assert(type.hasStaticShape() && "type must have static shape");
  return Base::get(type.getContext(), type,
               indices.cast<DenseIntElementsAttr>(), values);
}

SparseElementsAttr SparseElementsAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ShapedType type, DenseElementsAttr indices, DenseElementsAttr values) {
  assert(indices.getType().getElementType().isInteger(64) &&
         "expected sparse indices to be 64-bit integer values");
  assert((type.isa<RankedTensorType, VectorType>()) &&
         "type must be ranked tensor or vector");
  assert(type.hasStaticShape() && "type must have static shape");
  return Base::getChecked(emitError, type.getContext(), type,
               indices.cast<DenseIntElementsAttr>(), values);
}

ShapedType SparseElementsAttr::getType() const {
  return getImpl()->getType().cast<ShapedType>();
}

DenseIntElementsAttr SparseElementsAttr::getIndices() const {
  return getImpl()->indices;
}

DenseElementsAttr SparseElementsAttr::getValues() const {
  return getImpl()->values;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::SparseElementsAttr)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::StringAttr)
namespace mlir {
namespace detail {
struct SymbolRefAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<StringAttr, ::llvm::ArrayRef<FlatSymbolRefAttr>>;
  SymbolRefAttrStorage(StringAttr rootReference, ::llvm::ArrayRef<FlatSymbolRefAttr> nestedReferences) : ::mlir::AttributeStorage(), rootReference(rootReference), nestedReferences(nestedReferences) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (rootReference == std::get<0>(tblgenKey)) && (nestedReferences == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static SymbolRefAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto rootReference = std::get<0>(tblgenKey);
    auto nestedReferences = std::get<1>(tblgenKey);
    nestedReferences = allocator.copyInto(nestedReferences);
    return new (allocator.allocate<SymbolRefAttrStorage>()) SymbolRefAttrStorage(rootReference, nestedReferences);
  }

  StringAttr rootReference;
  ::llvm::ArrayRef<FlatSymbolRefAttr> nestedReferences;
};
} // namespace detail
SymbolRefAttr SymbolRefAttr::get(StringAttr rootReference, ArrayRef<FlatSymbolRefAttr> nestedReferences) {
  return Base::get(rootReference.getContext(), rootReference, nestedReferences);
}

StringAttr SymbolRefAttr::getRootReference() const {
  return getImpl()->rootReference;
}

::llvm::ArrayRef<FlatSymbolRefAttr> SymbolRefAttr::getNestedReferences() const {
  return getImpl()->nestedReferences;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::SymbolRefAttr)
namespace mlir {
namespace detail {
struct TypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<Type>;
  TypeAttrStorage(Type value) : ::mlir::AttributeStorage(), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    return new (allocator.allocate<TypeAttrStorage>()) TypeAttrStorage(value);
  }

  Type value;
};
} // namespace detail
TypeAttr TypeAttr::get(Type type) {
  return Base::get(type.getContext(), type);
}

Type TypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::TypeAttr)
namespace mlir {
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::UnitAttr)

#endif  // GET_ATTRDEF_CLASSES

