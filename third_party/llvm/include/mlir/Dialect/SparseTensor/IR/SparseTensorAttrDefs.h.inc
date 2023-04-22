/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace sparse_tensor {
class SparseTensorDimSliceAttr;
class SparseTensorEncodingAttr;
class SparseTensorSortKindAttr;
class StorageSpecifierKindAttr;
namespace detail {
struct SparseTensorDimSliceAttrStorage;
} // namespace detail
class SparseTensorDimSliceAttr : public ::mlir::Attribute::AttrBase<SparseTensorDimSliceAttr, ::mlir::Attribute, detail::SparseTensorDimSliceAttrStorage> {
public:
  using Base::Base;
  /// Special value for dynamic offset/size/stride.
  static constexpr int64_t kDynamic = -1;

  static bool isDynamic(int64_t v) {
    return v == kDynamic;
  }

  std::optional<uint64_t> getStaticOffset() const {
    if (isDynamic(getOffset()))
      return std::nullopt;
    return static_cast<uint64_t>(getOffset());
  };

  std::optional<uint64_t> getStaticStride() const {
    if (isDynamic(getStride()))
      return std::nullopt;
    return static_cast<uint64_t>(getStride());
  }

  std::optional<uint64_t> getStaticSize() const {
    if (isDynamic(getSize()))
      return std::nullopt;
    return static_cast<uint64_t>(getSize());
  }

  bool isCompletelyDynamic() const {
    return isDynamic(getOffset()) && isDynamic(getStride()) && isDynamic(getSize());
  };
  using Base::getChecked;
  static SparseTensorDimSliceAttr get(::mlir::MLIRContext *context, int64_t offset, int64_t size, int64_t stride);
  static SparseTensorDimSliceAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int64_t offset, int64_t size, int64_t stride);
  static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int64_t offset, int64_t size, int64_t stride);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"slice"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getOffset() const;
  int64_t getSize() const;
  int64_t getStride() const;
};
namespace detail {
struct SparseTensorEncodingAttrStorage;
} // namespace detail
class SparseTensorEncodingAttr : public ::mlir::Attribute::AttrBase<SparseTensorEncodingAttr, ::mlir::Attribute, detail::SparseTensorEncodingAttrStorage, ::mlir::VerifiableTensorEncoding::Trait> {
public:
  using Base::Base;
  /// Returns the type for position storage based on posWidth
  Type getPosType() const;

  /// Returns the type for coordinate storage based on crdWidth
  Type getCrdType() const;

  /// Constructs a new encoding with the dimOrdering and higherOrdering
  /// reset to the default/identity.
  SparseTensorEncodingAttr withoutOrdering() const;

  /// Constructs a new encoding with the pointer and index bitwidth
  /// reset to the default.
  SparseTensorEncodingAttr withoutBitWidths() const;

  /// Returns true if every level is dense.  Also returns true for
  /// the null encoding (since dense-tensors are always all-dense).
  bool isAllDense() const;

  /// Returns true if every level is ordered.  Also returns true for
  /// the null encoding (since dense-tensors are always all-ordered).
  bool isAllOrdered() const;

  /// Returns true if the encoding has an identity dimension ordering.
  /// Also returns true for the null encoding (since dense-tensors
  /// always have the identity ordering).
  bool hasIdDimOrdering() const;

  /// Returns the number of storage levels.  Asserts that the encoding
  /// is non-null (since there is no fixed result that's valid for
  /// every dense-tensor).
  ::mlir::sparse_tensor::Level getLvlRank() const;

  /// Safely looks up the level-type for the requested level.  (Returns
  /// `DimLevelType::Dense` for the null encoding, since dense-tensors
  /// are always all-dense.)
  ::mlir::sparse_tensor::DimLevelType getLvlType(::mlir::sparse_tensor::Level l) const;

  bool isDenseLvl(::mlir::sparse_tensor::Level l) const { return isDenseDLT(getLvlType(l)); }
  bool isCompressedLvl(::mlir::sparse_tensor::Level l) const { return isCompressedDLT(getLvlType(l)); }
  bool isCompressedWithHiLvl(::mlir::sparse_tensor::Level l) const { return isCompressedWithHiDLT(getLvlType(l)); }
  bool isSingletonLvl(::mlir::sparse_tensor::Level l) const { return isSingletonDLT(getLvlType(l)); }
  bool isOrderedLvl(::mlir::sparse_tensor::Level l) const { return isOrderedDLT(getLvlType(l)); }
  bool isUniqueLvl(::mlir::sparse_tensor::Level l) const { return isUniqueDLT(getLvlType(l)); }

  bool isSlice() const {
    return !getDimSlices().empty();
  }

  std::optional<uint64_t> getStaticDimSliceOffset(::mlir::sparse_tensor::Dimension dim) const;
  std::optional<uint64_t> getStaticDimSliceSize(::mlir::sparse_tensor::Dimension dim) const;
  std::optional<uint64_t> getStaticDimSliceStride(::mlir::sparse_tensor::Dimension dim) const;
  std::optional<uint64_t> getStaticLvlSliceOffset(::mlir::sparse_tensor::Level lvl) const;
  std::optional<uint64_t> getStaticLvlSliceSize(::mlir::sparse_tensor::Level lvl) const;
  std::optional<uint64_t> getStaticLvlSliceStride(::mlir::sparse_tensor::Level lvl) const;
  using Base::getChecked;
  static SparseTensorEncodingAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::sparse_tensor::DimLevelType> dimLevelType, AffineMap dimOrdering, AffineMap higherOrdering, unsigned posWidth, unsigned crdWidth, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices);
  static SparseTensorEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::sparse_tensor::DimLevelType> dimLevelType, AffineMap dimOrdering, AffineMap higherOrdering, unsigned posWidth, unsigned crdWidth, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices);
  static SparseTensorEncodingAttr get(::mlir::MLIRContext *context, ArrayRef<::mlir::sparse_tensor::DimLevelType> dimLevelType, AffineMap dimOrdering, AffineMap higherOrdering, unsigned posWidth, unsigned crdWidth);
  static SparseTensorEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ArrayRef<::mlir::sparse_tensor::DimLevelType> dimLevelType, AffineMap dimOrdering, AffineMap higherOrdering, unsigned posWidth, unsigned crdWidth);
  static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<::mlir::sparse_tensor::DimLevelType> dimLevelType, AffineMap dimOrdering, AffineMap higherOrdering, unsigned posWidth, unsigned crdWidth, ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> dimSlices);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"encoding"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<::mlir::sparse_tensor::DimLevelType> getDimLevelType() const;
  AffineMap getDimOrdering() const;
  AffineMap getHigherOrdering() const;
  unsigned getPosWidth() const;
  unsigned getCrdWidth() const;
  ::llvm::ArrayRef<::mlir::sparse_tensor::SparseTensorDimSliceAttr> getDimSlices() const;
  ::mlir::LogicalResult verifyEncoding(ArrayRef<int64_t> shape, Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const;
};
namespace detail {
struct SparseTensorSortKindAttrStorage;
} // namespace detail
class SparseTensorSortKindAttr : public ::mlir::Attribute::AttrBase<SparseTensorSortKindAttr, ::mlir::Attribute, detail::SparseTensorSortKindAttrStorage> {
public:
  using Base::Base;
  static SparseTensorSortKindAttr get(::mlir::MLIRContext *context, ::mlir::sparse_tensor::SparseTensorSortKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"SparseTensorSortAlgorithm"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::sparse_tensor::SparseTensorSortKind getValue() const;
};
namespace detail {
struct StorageSpecifierKindAttrStorage;
} // namespace detail
class StorageSpecifierKindAttr : public ::mlir::Attribute::AttrBase<StorageSpecifierKindAttr, ::mlir::Attribute, detail::StorageSpecifierKindAttrStorage> {
public:
  using Base::Base;
  static StorageSpecifierKindAttr get(::mlir::MLIRContext *context, ::mlir::sparse_tensor::StorageSpecifierKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"kind"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::sparse_tensor::StorageSpecifierKind getValue() const;
};
} // namespace sparse_tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorDimSliceAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorSortKindAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::StorageSpecifierKindAttr)

#endif  // GET_ATTRDEF_CLASSES

