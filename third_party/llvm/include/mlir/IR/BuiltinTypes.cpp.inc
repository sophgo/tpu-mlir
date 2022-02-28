/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::BFloat16Type,
::mlir::ComplexType,
::mlir::Float128Type,
::mlir::Float16Type,
::mlir::Float32Type,
::mlir::Float64Type,
::mlir::Float80Type,
::mlir::FunctionType,
::mlir::IndexType,
::mlir::IntegerType,
::mlir::MemRefType,
::mlir::NoneType,
::mlir::OpaqueType,
::mlir::RankedTensorType,
::mlir::TupleType,
::mlir::UnrankedMemRefType,
::mlir::UnrankedTensorType,
::mlir::VectorType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

namespace mlir {
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::BFloat16Type)
namespace mlir {
namespace detail {
struct ComplexTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type>;
  ComplexTypeStorage(Type elementType) : elementType(elementType) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ComplexTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto elementType = std::get<0>(tblgenKey);
    return new (allocator.allocate<ComplexTypeStorage>()) ComplexTypeStorage(elementType);
  }

  Type elementType;
};
} // namespace detail
ComplexType ComplexType::get(Type elementType) {
  return Base::get(elementType.getContext(), elementType);
}

ComplexType ComplexType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType) {
  return Base::getChecked(emitError, elementType.getContext(), elementType);
}

Type ComplexType::getElementType() const {
  return getImpl()->elementType;
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::ComplexType)
namespace mlir {
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::Float128Type)
namespace mlir {
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::Float16Type)
namespace mlir {
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::Float32Type)
namespace mlir {
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::Float64Type)
namespace mlir {
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::Float80Type)
namespace mlir {
FunctionType FunctionType::get(::mlir::MLIRContext *context, TypeRange inputs, TypeRange results) {
  return Base::get(context, inputs, results);
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::FunctionType)
namespace mlir {
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::IndexType)
namespace mlir {
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::IntegerType)
namespace mlir {
namespace detail {
struct MemRefTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, Type, MemRefLayoutAttrInterface, Attribute>;
  MemRefTypeStorage(::llvm::ArrayRef<int64_t> shape, Type elementType, MemRefLayoutAttrInterface layout, Attribute memorySpace) : shape(shape), elementType(elementType), layout(layout), memorySpace(memorySpace) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey)) && (layout == std::get<2>(tblgenKey)) && (memorySpace == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static MemRefTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto shape = std::get<0>(tblgenKey);
    auto elementType = std::get<1>(tblgenKey);
    auto layout = std::get<2>(tblgenKey);
    auto memorySpace = std::get<3>(tblgenKey);
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<MemRefTypeStorage>()) MemRefTypeStorage(shape, elementType, layout, memorySpace);
  }

  ::llvm::ArrayRef<int64_t> shape;
  Type elementType;
  MemRefLayoutAttrInterface layout;
  Attribute memorySpace;
};
} // namespace detail
::llvm::ArrayRef<int64_t> MemRefType::getShape() const {
  return getImpl()->shape;
}

Type MemRefType::getElementType() const {
  return getImpl()->elementType;
}

MemRefLayoutAttrInterface MemRefType::getLayout() const {
  return getImpl()->layout;
}

Attribute MemRefType::getMemorySpace() const {
  return getImpl()->memorySpace;
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::MemRefType)
namespace mlir {
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::NoneType)
namespace mlir {
namespace detail {
struct OpaqueTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<StringAttr, ::llvm::StringRef>;
  OpaqueTypeStorage(StringAttr dialectNamespace, ::llvm::StringRef typeData) : dialectNamespace(dialectNamespace), typeData(typeData) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (dialectNamespace == std::get<0>(tblgenKey)) && (typeData == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static OpaqueTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto dialectNamespace = std::get<0>(tblgenKey);
    auto typeData = std::get<1>(tblgenKey);
    typeData = allocator.copyInto(typeData);
    return new (allocator.allocate<OpaqueTypeStorage>()) OpaqueTypeStorage(dialectNamespace, typeData);
  }

  StringAttr dialectNamespace;
  ::llvm::StringRef typeData;
};
} // namespace detail
OpaqueType OpaqueType::get(StringAttr dialectNamespace, StringRef typeData) {
  return Base::get(dialectNamespace.getContext(), dialectNamespace, typeData);
}

OpaqueType OpaqueType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, StringAttr dialectNamespace, StringRef typeData) {
  return Base::getChecked(emitError, dialectNamespace.getContext(), dialectNamespace, typeData);
}

StringAttr OpaqueType::getDialectNamespace() const {
  return getImpl()->dialectNamespace;
}

::llvm::StringRef OpaqueType::getTypeData() const {
  return getImpl()->typeData;
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::OpaqueType)
namespace mlir {
namespace detail {
struct RankedTensorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, Type, Attribute>;
  RankedTensorTypeStorage(::llvm::ArrayRef<int64_t> shape, Type elementType, Attribute encoding) : shape(shape), elementType(elementType), encoding(encoding) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey)) && (encoding == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static RankedTensorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto shape = std::get<0>(tblgenKey);
    auto elementType = std::get<1>(tblgenKey);
    auto encoding = std::get<2>(tblgenKey);
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<RankedTensorTypeStorage>()) RankedTensorTypeStorage(shape, elementType, encoding);
  }

  ::llvm::ArrayRef<int64_t> shape;
  Type elementType;
  Attribute encoding;
};
} // namespace detail
RankedTensorType RankedTensorType::get(ArrayRef<int64_t> shape, Type elementType, Attribute encoding) {
  return Base::get(elementType.getContext(), shape, elementType, encoding);
}

RankedTensorType RankedTensorType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape, Type elementType, Attribute encoding) {
  return Base::getChecked(emitError, elementType.getContext(), shape, elementType, encoding);
}

::llvm::ArrayRef<int64_t> RankedTensorType::getShape() const {
  return getImpl()->shape;
}

Type RankedTensorType::getElementType() const {
  return getImpl()->elementType;
}

Attribute RankedTensorType::getEncoding() const {
  return getImpl()->encoding;
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::RankedTensorType)
namespace mlir {
TupleType TupleType::get(::mlir::MLIRContext *context, TypeRange elementTypes) {
  return Base::get(context, elementTypes);
}

TupleType TupleType::get(::mlir::MLIRContext *context) {
  return Base::get(context, TypeRange());
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::TupleType)
namespace mlir {
namespace detail {
struct UnrankedMemRefTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, Attribute>;
  UnrankedMemRefTypeStorage(Type elementType, Attribute memorySpace) : elementType(elementType), memorySpace(memorySpace) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey)) && (memorySpace == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static UnrankedMemRefTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto elementType = std::get<0>(tblgenKey);
    auto memorySpace = std::get<1>(tblgenKey);
    return new (allocator.allocate<UnrankedMemRefTypeStorage>()) UnrankedMemRefTypeStorage(elementType, memorySpace);
  }

  Type elementType;
  Attribute memorySpace;
};
} // namespace detail
UnrankedMemRefType UnrankedMemRefType::get(Type elementType, Attribute memorySpace) {
  // Drop default memory space value and replace it with empty attribute.
  Attribute nonDefaultMemorySpace = skipDefaultMemorySpace(memorySpace);
  return Base::get(elementType.getContext(), elementType, nonDefaultMemorySpace);
}

UnrankedMemRefType UnrankedMemRefType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, Attribute memorySpace) {
  // Drop default memory space value and replace it with empty attribute.
  Attribute nonDefaultMemorySpace = skipDefaultMemorySpace(memorySpace);
  return Base::getChecked(emitError, elementType.getContext(), elementType, nonDefaultMemorySpace);
}

UnrankedMemRefType UnrankedMemRefType::get(Type elementType, unsigned memorySpace) {
  // Convert deprecated integer-like memory space to Attribute.
  Attribute memorySpaceAttr =
      wrapIntegerMemorySpace(memorySpace, elementType.getContext());
  return UnrankedMemRefType::get(elementType, memorySpaceAttr);
}

UnrankedMemRefType UnrankedMemRefType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, unsigned memorySpace) {
  // Convert deprecated integer-like memory space to Attribute.
  Attribute memorySpaceAttr =
      wrapIntegerMemorySpace(memorySpace, elementType.getContext());
  return UnrankedMemRefType::get(elementType, memorySpaceAttr);
}

Type UnrankedMemRefType::getElementType() const {
  return getImpl()->elementType;
}

Attribute UnrankedMemRefType::getMemorySpace() const {
  return getImpl()->memorySpace;
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::UnrankedMemRefType)
namespace mlir {
namespace detail {
struct UnrankedTensorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type>;
  UnrankedTensorTypeStorage(Type elementType) : elementType(elementType) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static UnrankedTensorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto elementType = std::get<0>(tblgenKey);
    return new (allocator.allocate<UnrankedTensorTypeStorage>()) UnrankedTensorTypeStorage(elementType);
  }

  Type elementType;
};
} // namespace detail
UnrankedTensorType UnrankedTensorType::get(Type elementType) {
  return Base::get(elementType.getContext(), elementType);
}

UnrankedTensorType UnrankedTensorType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType) {
  return Base::getChecked(emitError, elementType.getContext(), elementType);
}

Type UnrankedTensorType::getElementType() const {
  return getImpl()->elementType;
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::UnrankedTensorType)
namespace mlir {
namespace detail {
struct VectorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, Type, unsigned>;
  VectorTypeStorage(::llvm::ArrayRef<int64_t> shape, Type elementType, unsigned numScalableDims) : shape(shape), elementType(elementType), numScalableDims(numScalableDims) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey)) && (numScalableDims == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static VectorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto shape = std::get<0>(tblgenKey);
    auto elementType = std::get<1>(tblgenKey);
    auto numScalableDims = std::get<2>(tblgenKey);
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<VectorTypeStorage>()) VectorTypeStorage(shape, elementType, numScalableDims);
  }

  ::llvm::ArrayRef<int64_t> shape;
  Type elementType;
  unsigned numScalableDims;
};
} // namespace detail
VectorType VectorType::get(ArrayRef<int64_t> shape, Type elementType, unsigned numScalableDims) {
  return Base::get(elementType.getContext(), shape, elementType,
               numScalableDims);
}

VectorType VectorType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape, Type elementType, unsigned numScalableDims) {
  return Base::getChecked(emitError, elementType.getContext(), shape, elementType,
               numScalableDims);
}

::llvm::ArrayRef<int64_t> VectorType::getShape() const {
  return getImpl()->shape;
}

Type VectorType::getElementType() const {
  return getImpl()->elementType;
}

unsigned VectorType::getNumScalableDims() const {
  return getImpl()->numScalableDims;
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::VectorType)

#endif  // GET_TYPEDEF_CLASSES

