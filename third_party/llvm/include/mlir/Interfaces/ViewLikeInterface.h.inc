/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class OffsetSizeAndStrideOpInterface;
namespace detail {
struct OffsetSizeAndStrideOpInterfaceInterfaceTraits {
  struct Concept {
    unsigned (*getOffsetSizeAndStrideStartOperandIndex)();
    std::array<unsigned, 3> (*getArrayAttrMaxRanks)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OperandRange (*offsets)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OperandRange (*sizes)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OperandRange (*strides)(const Concept *impl, ::mlir::Operation *);
    ::mlir::ArrayAttr (*static_offsets)(const Concept *impl, ::mlir::Operation *);
    ::mlir::ArrayAttr (*static_sizes)(const Concept *impl, ::mlir::Operation *);
    ::mlir::ArrayAttr (*static_strides)(const Concept *impl, ::mlir::Operation *);
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> (*getMixedOffsets)(const Concept *impl, ::mlir::Operation *);
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> (*getMixedSizes)(const Concept *impl, ::mlir::Operation *);
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> (*getMixedStrides)(const Concept *impl, ::mlir::Operation *);
    bool (*isDynamicOffset)(const Concept *impl, ::mlir::Operation *, unsigned);
    bool (*isDynamicSize)(const Concept *impl, ::mlir::Operation *, unsigned);
    bool (*isDynamicStride)(const Concept *impl, ::mlir::Operation *, unsigned);
    int64_t (*getStaticOffset)(const Concept *impl, ::mlir::Operation *, unsigned);
    int64_t (*getStaticSize)(const Concept *impl, ::mlir::Operation *, unsigned);
    int64_t (*getStaticStride)(const Concept *impl, ::mlir::Operation *, unsigned);
    unsigned (*getIndexOfDynamicOffset)(const Concept *impl, ::mlir::Operation *, unsigned);
    unsigned (*getIndexOfDynamicSize)(const Concept *impl, ::mlir::Operation *, unsigned);
    unsigned (*getIndexOfDynamicStride)(const Concept *impl, ::mlir::Operation *, unsigned);
    unsigned (*getNumDynamicEntriesUpToIdx)(const Concept *impl, ::mlir::Operation *, ::mlir::ArrayAttr, ::llvm::function_ref<bool(int64_t)>, unsigned);
    ::mlir::Value (*getDynamicOffset)(const Concept *impl, ::mlir::Operation *, unsigned);
    ::mlir::Value (*getDynamicSize)(const Concept *impl, ::mlir::Operation *, unsigned);
    ::mlir::Value (*getDynamicStride)(const Concept *impl, ::mlir::Operation *, unsigned);
    bool (*isSameAs)(const Concept *impl, ::mlir::Operation *, ::mlir::OffsetSizeAndStrideOpInterface, ::llvm::function_ref<bool(::mlir::OpFoldResult, ::mlir::OpFoldResult)>);
    bool (*hasUnitStride)(const Concept *impl, ::mlir::Operation *);
    bool (*hasZeroOffset)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::OffsetSizeAndStrideOpInterface;
    Model() : Concept{getOffsetSizeAndStrideStartOperandIndex, getArrayAttrMaxRanks, offsets, sizes, strides, static_offsets, static_sizes, static_strides, getMixedOffsets, getMixedSizes, getMixedStrides, isDynamicOffset, isDynamicSize, isDynamicStride, getStaticOffset, getStaticSize, getStaticStride, getIndexOfDynamicOffset, getIndexOfDynamicSize, getIndexOfDynamicStride, getNumDynamicEntriesUpToIdx, getDynamicOffset, getDynamicSize, getDynamicStride, isSameAs, hasUnitStride, hasZeroOffset} {}

    static inline unsigned getOffsetSizeAndStrideStartOperandIndex();
    static inline std::array<unsigned, 3> getArrayAttrMaxRanks(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OperandRange offsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OperandRange sizes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OperandRange strides(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr static_offsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr static_sizes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr static_strides(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedOffsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedSizes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedStrides(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isDynamicOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline bool isDynamicSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline bool isDynamicStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline int64_t getStaticOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline int64_t getStaticSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline int64_t getStaticStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline unsigned getIndexOfDynamicOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline unsigned getIndexOfDynamicSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline unsigned getIndexOfDynamicStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline unsigned getNumDynamicEntriesUpToIdx(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attr, ::llvm::function_ref<bool(int64_t)> isDynamic, unsigned idx);
    static inline ::mlir::Value getDynamicOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline ::mlir::Value getDynamicSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline ::mlir::Value getDynamicStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline bool isSameAs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OffsetSizeAndStrideOpInterface other, ::llvm::function_ref<bool(::mlir::OpFoldResult, ::mlir::OpFoldResult)> cmp);
    static inline bool hasUnitStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasZeroOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::OffsetSizeAndStrideOpInterface;
    FallbackModel() : Concept{getOffsetSizeAndStrideStartOperandIndex, getArrayAttrMaxRanks, offsets, sizes, strides, static_offsets, static_sizes, static_strides, getMixedOffsets, getMixedSizes, getMixedStrides, isDynamicOffset, isDynamicSize, isDynamicStride, getStaticOffset, getStaticSize, getStaticStride, getIndexOfDynamicOffset, getIndexOfDynamicSize, getIndexOfDynamicStride, getNumDynamicEntriesUpToIdx, getDynamicOffset, getDynamicSize, getDynamicStride, isSameAs, hasUnitStride, hasZeroOffset} {}

    static inline unsigned getOffsetSizeAndStrideStartOperandIndex();
    static inline std::array<unsigned, 3> getArrayAttrMaxRanks(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OperandRange offsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OperandRange sizes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OperandRange strides(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr static_offsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr static_sizes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ArrayAttr static_strides(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedOffsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedSizes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedStrides(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isDynamicOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline bool isDynamicSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline bool isDynamicStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline int64_t getStaticOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline int64_t getStaticSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline int64_t getStaticStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline unsigned getIndexOfDynamicOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline unsigned getIndexOfDynamicSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline unsigned getIndexOfDynamicStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline unsigned getNumDynamicEntriesUpToIdx(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attr, ::llvm::function_ref<bool(int64_t)> isDynamic, unsigned idx);
    static inline ::mlir::Value getDynamicOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline ::mlir::Value getDynamicSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline ::mlir::Value getDynamicStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline bool isSameAs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OffsetSizeAndStrideOpInterface other, ::llvm::function_ref<bool(::mlir::OpFoldResult, ::mlir::OpFoldResult)> cmp);
    static inline bool hasUnitStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasZeroOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::OperandRange offsets(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::OperandRange sizes(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::OperandRange strides(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::ArrayAttr static_offsets(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::ArrayAttr static_sizes(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::ArrayAttr static_strides(::mlir::Operation *tablegen_opaque_val) const;
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedOffsets(::mlir::Operation *tablegen_opaque_val) const;
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedSizes(::mlir::Operation *tablegen_opaque_val) const;
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedStrides(::mlir::Operation *tablegen_opaque_val) const;
    bool isDynamicOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx) const;
    bool isDynamicSize(::mlir::Operation *tablegen_opaque_val, unsigned idx) const;
    bool isDynamicStride(::mlir::Operation *tablegen_opaque_val, unsigned idx) const;
    int64_t getStaticOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx) const;
    int64_t getStaticSize(::mlir::Operation *tablegen_opaque_val, unsigned idx) const;
    int64_t getStaticStride(::mlir::Operation *tablegen_opaque_val, unsigned idx) const;
    unsigned getIndexOfDynamicOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx) const;
    unsigned getIndexOfDynamicSize(::mlir::Operation *tablegen_opaque_val, unsigned idx) const;
    unsigned getIndexOfDynamicStride(::mlir::Operation *tablegen_opaque_val, unsigned idx) const;
    unsigned getNumDynamicEntriesUpToIdx(::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attr, ::llvm::function_ref<bool(int64_t)> isDynamic, unsigned idx) const;
    ::mlir::Value getDynamicOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx) const;
    ::mlir::Value getDynamicSize(::mlir::Operation *tablegen_opaque_val, unsigned idx) const;
    ::mlir::Value getDynamicStride(::mlir::Operation *tablegen_opaque_val, unsigned idx) const;
    bool isSameAs(::mlir::Operation *tablegen_opaque_val, ::mlir::OffsetSizeAndStrideOpInterface other, ::llvm::function_ref<bool(::mlir::OpFoldResult, ::mlir::OpFoldResult)> cmp) const;
    bool hasUnitStride(::mlir::Operation *tablegen_opaque_val) const;
    bool hasZeroOffset(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct OffsetSizeAndStrideOpInterfaceTrait;

} // namespace detail
class OffsetSizeAndStrideOpInterface : public ::mlir::OpInterface<OffsetSizeAndStrideOpInterface, detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<OffsetSizeAndStrideOpInterface, detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::OffsetSizeAndStrideOpInterfaceTrait<ConcreteOp> {};
  /// Return the number of leading operands before the `offsets`, `sizes` and
  /// and `strides` operands.
  unsigned getOffsetSizeAndStrideStartOperandIndex();
  /// Return the expected rank of each of the`static_offsets`, `static_sizes`
  /// and `static_strides` attributes.
  std::array<unsigned, 3> getArrayAttrMaxRanks();
  /// Return the dynamic offset operands.
  ::mlir::OperandRange offsets();
  /// Return the dynamic size operands.
  ::mlir::OperandRange sizes();
  /// Return the dynamic stride operands.
  ::mlir::OperandRange strides();
  /// Return the static offset attributes.
  ::mlir::ArrayAttr static_offsets();
  /// Return the static size attributes.
  ::mlir::ArrayAttr static_sizes();
  /// Return the dynamic stride attributes.
  ::mlir::ArrayAttr static_strides();
  /// Return a vector of all the static or dynamic sizes of the op.
  ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedOffsets();
  /// Return a vector of all the static or dynamic sizes of the op.
  ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedSizes();
  /// Return a vector of all the static or dynamic strides of the op.
  ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedStrides();
  /// Return true if the offset `idx` is dynamic.
  bool isDynamicOffset(unsigned idx);
  /// Return true if the size `idx` is dynamic.
  bool isDynamicSize(unsigned idx);
  /// Return true if the stride `idx` is dynamic.
  bool isDynamicStride(unsigned idx);
  /// Assert the offset `idx` is a static constant and return its value.
  int64_t getStaticOffset(unsigned idx);
  /// Assert the size `idx` is a static constant and return its value.
  int64_t getStaticSize(unsigned idx);
  /// Assert the stride `idx` is a static constant and return its value.
  int64_t getStaticStride(unsigned idx);
  /// Assert the offset `idx` is dynamic and return the position of the
  /// corresponding operand.
  unsigned getIndexOfDynamicOffset(unsigned idx);
  /// Assert the size `idx` is dynamic and return the position of the
  /// corresponding operand.
  unsigned getIndexOfDynamicSize(unsigned idx);
  /// Assert the stride `idx` is dynamic and return the position of the
  /// corresponding operand.
  unsigned getIndexOfDynamicStride(unsigned idx);
  /// Helper method to compute the number of dynamic entries of `attr`, up to
  /// `idx` using `isDynamic` to determine whether an entry is dynamic.
  unsigned getNumDynamicEntriesUpToIdx(::mlir::ArrayAttr attr, ::llvm::function_ref<bool(int64_t)> isDynamic, unsigned idx);
  /// Assert the offset `idx` is dynamic and return its value.
  ::mlir::Value getDynamicOffset(unsigned idx);
  /// Assert the size `idx` is dynamic and return its value.
  ::mlir::Value getDynamicSize(unsigned idx);
  /// Assert the stride `idx` is dynamic and return its value.
  ::mlir::Value getDynamicStride(unsigned idx);
  /// Return true if all `other`'s offsets, sizes and strides are the same.
  /// Takes a custom `cmp` comparison function on OpFoldResult to avoid taking
  /// a dialect dependence.
  bool isSameAs(::mlir::OffsetSizeAndStrideOpInterface other, ::llvm::function_ref<bool(::mlir::OpFoldResult, ::mlir::OpFoldResult)> cmp);
  /// Return true if all strides are guaranteed to be 1.
  bool hasUnitStride();
  /// Return true if all offsets are guaranteed to be 0.
  bool hasZeroOffset();

    static unsigned getOffsetOperandGroupPosition() { return 0; }
    static unsigned getSizeOperandGroupPosition() { return 1; }
    static unsigned getStrideOperandGroupPosition() { return 2; }
    static ::llvm::StringRef getStaticOffsetsAttrName() {
      return "static_offsets";
    }
    static ::llvm::StringRef getStaticSizesAttrName() {
      return "static_sizes";
    }
    static ::llvm::StringRef getStaticStridesAttrName() {
      return "static_strides";
    }
    static ::llvm::ArrayRef<::llvm::StringRef> getSpecialAttrNames() {
      static ::llvm::SmallVector<::llvm::StringRef, 4> names{
        ::mlir::OffsetSizeAndStrideOpInterface::getStaticOffsetsAttrName(),
        ::mlir::OffsetSizeAndStrideOpInterface::getStaticSizesAttrName(),
        ::mlir::OffsetSizeAndStrideOpInterface::getStaticStridesAttrName(),
        ::mlir::OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr()};
      return names;
    }
  
};
namespace detail {
  template <typename ConcreteOp>
  struct OffsetSizeAndStrideOpInterfaceTrait : public ::mlir::OpInterface<OffsetSizeAndStrideOpInterface, detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Return the dynamic offset operands.
    ::mlir::OperandRange offsets() {
      return (*static_cast<ConcreteOp *>(this)).getOffsets();
    }
    /// Return the dynamic size operands.
    ::mlir::OperandRange sizes() {
      return (*static_cast<ConcreteOp *>(this)).getSizes();
    }
    /// Return the dynamic stride operands.
    ::mlir::OperandRange strides() {
      return (*static_cast<ConcreteOp *>(this)).getStrides();
    }
    /// Return the static offset attributes.
    ::mlir::ArrayAttr static_offsets() {
      return (*static_cast<ConcreteOp *>(this)).getStaticOffsets();
    }
    /// Return the static size attributes.
    ::mlir::ArrayAttr static_sizes() {
      return (*static_cast<ConcreteOp *>(this)).getStaticSizes();
    }
    /// Return the dynamic stride attributes.
    ::mlir::ArrayAttr static_strides() {
      return (*static_cast<ConcreteOp *>(this)).getStaticStrides();
    }
    /// Return a vector of all the static or dynamic sizes of the op.
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedOffsets() {
      return ::mlir::getMixedStridesOrOffsets((*static_cast<ConcreteOp *>(this)).getStaticOffsets(),
                                                (*static_cast<ConcreteOp *>(this)).getOffsets());
    }
    /// Return a vector of all the static or dynamic sizes of the op.
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedSizes() {
      return ::mlir::getMixedSizes((*static_cast<ConcreteOp *>(this)).getStaticSizes(), (*static_cast<ConcreteOp *>(this)).sizes());
    }
    /// Return a vector of all the static or dynamic strides of the op.
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedStrides() {
      return ::mlir::getMixedStridesOrOffsets((*static_cast<ConcreteOp *>(this)).getStaticStrides(),
                                                (*static_cast<ConcreteOp *>(this)).getStrides());
    }
    /// Return true if the offset `idx` is dynamic.
    bool isDynamicOffset(unsigned idx) {
      ::llvm::APInt v = *(static_offsets()
          .template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return ::mlir::ShapedType::isDynamicStrideOrOffset(v.getSExtValue());
    }
    /// Return true if the size `idx` is dynamic.
    bool isDynamicSize(unsigned idx) {
      ::llvm::APInt v = *(static_sizes()
          .template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return ::mlir::ShapedType::isDynamic(v.getSExtValue());
    }
    /// Return true if the stride `idx` is dynamic.
    bool isDynamicStride(unsigned idx) {
      ::llvm::APInt v = *(static_strides()
          .template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return ::mlir::ShapedType::isDynamicStrideOrOffset(v.getSExtValue());
    }
    /// Assert the offset `idx` is a static constant and return its value.
    int64_t getStaticOffset(unsigned idx) {
      assert(!(*static_cast<ConcreteOp *>(this)).isDynamicOffset(idx) && "expected static offset");
        ::llvm::APInt v = *(static_offsets().
          template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return v.getSExtValue();
    }
    /// Assert the size `idx` is a static constant and return its value.
    int64_t getStaticSize(unsigned idx) {
      assert(!(*static_cast<ConcreteOp *>(this)).isDynamicSize(idx) && "expected static size");
        ::llvm::APInt v = *(static_sizes().
          template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return v.getSExtValue();
    }
    /// Assert the stride `idx` is a static constant and return its value.
    int64_t getStaticStride(unsigned idx) {
      assert(!(*static_cast<ConcreteOp *>(this)).isDynamicStride(idx) && "expected static stride");
        ::llvm::APInt v = *(static_strides().
          template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return v.getSExtValue();
    }
    /// Assert the offset `idx` is dynamic and return the position of the
    /// corresponding operand.
    unsigned getIndexOfDynamicOffset(unsigned idx) {
      assert((*static_cast<ConcreteOp *>(this)).isDynamicOffset(idx) && "expected dynamic offset");
        auto numDynamic = getNumDynamicEntriesUpToIdx(
          static_offsets().template cast<::mlir::ArrayAttr>(),
          ::mlir::ShapedType::isDynamicStrideOrOffset,
          idx);
        return (*static_cast<ConcreteOp *>(this)).getOffsetSizeAndStrideStartOperandIndex() + numDynamic;
    }
    /// Assert the size `idx` is dynamic and return the position of the
    /// corresponding operand.
    unsigned getIndexOfDynamicSize(unsigned idx) {
      assert((*static_cast<ConcreteOp *>(this)).isDynamicSize(idx) && "expected dynamic size");
        auto numDynamic = getNumDynamicEntriesUpToIdx(
          static_sizes().template cast<::mlir::ArrayAttr>(), ::mlir::ShapedType::isDynamic, idx);
        return (*static_cast<ConcreteOp *>(this)).getOffsetSizeAndStrideStartOperandIndex() +
          offsets().size() + numDynamic;
    }
    /// Assert the stride `idx` is dynamic and return the position of the
    /// corresponding operand.
    unsigned getIndexOfDynamicStride(unsigned idx) {
      assert((*static_cast<ConcreteOp *>(this)).isDynamicStride(idx) && "expected dynamic stride");
        auto numDynamic = getNumDynamicEntriesUpToIdx(
          static_strides().template cast<::mlir::ArrayAttr>(),
          ::mlir::ShapedType::isDynamicStrideOrOffset,
          idx);
        return (*static_cast<ConcreteOp *>(this)).getOffsetSizeAndStrideStartOperandIndex() +
          offsets().size() + sizes().size() + numDynamic;
    }
    /// Helper method to compute the number of dynamic entries of `attr`, up to
    /// `idx` using `isDynamic` to determine whether an entry is dynamic.
    unsigned getNumDynamicEntriesUpToIdx(::mlir::ArrayAttr attr, ::llvm::function_ref<bool(int64_t)> isDynamic, unsigned idx) {
      return std::count_if(
          attr.getValue().begin(), attr.getValue().begin() + idx,
          [&](::mlir::Attribute attr) {
            return isDynamic(attr.cast<::mlir::IntegerAttr>().getInt());
          });
    }
    /// Assert the offset `idx` is dynamic and return its value.
    ::mlir::Value getDynamicOffset(unsigned idx) {
      return (*static_cast<ConcreteOp *>(this)).getOperand(getIndexOfDynamicOffset(idx));
    }
    /// Assert the size `idx` is dynamic and return its value.
    ::mlir::Value getDynamicSize(unsigned idx) {
      return (*static_cast<ConcreteOp *>(this)).getOperand(getIndexOfDynamicSize(idx));
    }
    /// Assert the stride `idx` is dynamic and return its value.
    ::mlir::Value getDynamicStride(unsigned idx) {
      return (*static_cast<ConcreteOp *>(this)).getOperand(getIndexOfDynamicStride(idx));
    }
    /// Return true if all `other`'s offsets, sizes and strides are the same.
    /// Takes a custom `cmp` comparison function on OpFoldResult to avoid taking
    /// a dialect dependence.
    bool isSameAs(::mlir::OffsetSizeAndStrideOpInterface other, ::llvm::function_ref<bool(::mlir::OpFoldResult, ::mlir::OpFoldResult)> cmp) {
      return ::mlir::detail::sameOffsetsSizesAndStrides(
          ::mlir::cast<::mlir::OffsetSizeAndStrideOpInterface>(
            (*static_cast<ConcreteOp *>(this)).getOperation()), other, cmp);
    }
    /// Return true if all strides are guaranteed to be 1.
    bool hasUnitStride() {
      return ::llvm::all_of(getMixedStrides(), [](::mlir::OpFoldResult ofr) {
          return ::mlir::getConstantIntValue(ofr) == static_cast<int64_t>(1);
        });
    }
    /// Return true if all offsets are guaranteed to be 0.
    bool hasZeroOffset() {
      return ::llvm::all_of(getMixedOffsets(), [](::mlir::OpFoldResult ofr) {
          return ::mlir::getConstantIntValue(ofr) == static_cast<int64_t>(0);
        });
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return ::mlir::detail::verifyOffsetSizeAndStrideOp(
        ::mlir::cast<::mlir::OffsetSizeAndStrideOpInterface>(op));
    }
  };
}// namespace detail
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getOffsetSizeAndStrideStartOperandIndex() {
  return ConcreteOp::getOffsetSizeAndStrideStartOperandIndex();
}
template<typename ConcreteOp>
std::array<unsigned, 3> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getArrayAttrMaxRanks(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getArrayAttrMaxRanks();
}
template<typename ConcreteOp>
::mlir::OperandRange detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::offsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).offsets();
}
template<typename ConcreteOp>
::mlir::OperandRange detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::sizes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).sizes();
}
template<typename ConcreteOp>
::mlir::OperandRange detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::strides(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).strides();
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::static_offsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).static_offsets();
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::static_sizes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).static_sizes();
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::static_strides(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).static_strides();
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::OpFoldResult, 4> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMixedOffsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMixedOffsets();
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::OpFoldResult, 4> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMixedSizes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMixedSizes();
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::OpFoldResult, 4> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMixedStrides(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMixedStrides();
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDynamicOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDynamicOffset(idx);
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDynamicSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDynamicSize(idx);
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDynamicStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDynamicStride(idx);
}
template<typename ConcreteOp>
int64_t detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getStaticOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticOffset(idx);
}
template<typename ConcreteOp>
int64_t detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getStaticSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticSize(idx);
}
template<typename ConcreteOp>
int64_t detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getStaticStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticStride(idx);
}
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getIndexOfDynamicOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexOfDynamicOffset(idx);
}
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getIndexOfDynamicSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexOfDynamicSize(idx);
}
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getIndexOfDynamicStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexOfDynamicStride(idx);
}
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getNumDynamicEntriesUpToIdx(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attr, ::llvm::function_ref<bool(int64_t)> isDynamic, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumDynamicEntriesUpToIdx(attr, isDynamic, idx);
}
template<typename ConcreteOp>
::mlir::Value detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDynamicOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDynamicOffset(idx);
}
template<typename ConcreteOp>
::mlir::Value detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDynamicSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDynamicSize(idx);
}
template<typename ConcreteOp>
::mlir::Value detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDynamicStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDynamicStride(idx);
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::isSameAs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OffsetSizeAndStrideOpInterface other, ::llvm::function_ref<bool(::mlir::OpFoldResult, ::mlir::OpFoldResult)> cmp) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isSameAs(other, cmp);
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::hasUnitStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasUnitStride();
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::Model<ConcreteOp>::hasZeroOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasZeroOffset();
}
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getOffsetSizeAndStrideStartOperandIndex() {
  return ConcreteOp::getOffsetSizeAndStrideStartOperandIndex();
}
template<typename ConcreteOp>
std::array<unsigned, 3> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getArrayAttrMaxRanks(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getArrayAttrMaxRanks(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OperandRange detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::offsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->offsets(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OperandRange detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::sizes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->sizes(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OperandRange detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::strides(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->strides(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::static_offsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->static_offsets(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::static_sizes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->static_sizes(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::ArrayAttr detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::static_strides(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->static_strides(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::OpFoldResult, 4> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMixedOffsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMixedOffsets(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::OpFoldResult, 4> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMixedSizes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMixedSizes(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::OpFoldResult, 4> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMixedStrides(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMixedStrides(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isDynamicOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return static_cast<const ConcreteOp *>(impl)->isDynamicOffset(tablegen_opaque_val, idx);
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isDynamicSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return static_cast<const ConcreteOp *>(impl)->isDynamicSize(tablegen_opaque_val, idx);
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isDynamicStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return static_cast<const ConcreteOp *>(impl)->isDynamicStride(tablegen_opaque_val, idx);
}
template<typename ConcreteOp>
int64_t detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getStaticOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return static_cast<const ConcreteOp *>(impl)->getStaticOffset(tablegen_opaque_val, idx);
}
template<typename ConcreteOp>
int64_t detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getStaticSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return static_cast<const ConcreteOp *>(impl)->getStaticSize(tablegen_opaque_val, idx);
}
template<typename ConcreteOp>
int64_t detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getStaticStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return static_cast<const ConcreteOp *>(impl)->getStaticStride(tablegen_opaque_val, idx);
}
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getIndexOfDynamicOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return static_cast<const ConcreteOp *>(impl)->getIndexOfDynamicOffset(tablegen_opaque_val, idx);
}
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getIndexOfDynamicSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return static_cast<const ConcreteOp *>(impl)->getIndexOfDynamicSize(tablegen_opaque_val, idx);
}
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getIndexOfDynamicStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return static_cast<const ConcreteOp *>(impl)->getIndexOfDynamicStride(tablegen_opaque_val, idx);
}
template<typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getNumDynamicEntriesUpToIdx(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attr, ::llvm::function_ref<bool(int64_t)> isDynamic, unsigned idx) {
  return static_cast<const ConcreteOp *>(impl)->getNumDynamicEntriesUpToIdx(tablegen_opaque_val, attr, isDynamic, idx);
}
template<typename ConcreteOp>
::mlir::Value detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDynamicOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return static_cast<const ConcreteOp *>(impl)->getDynamicOffset(tablegen_opaque_val, idx);
}
template<typename ConcreteOp>
::mlir::Value detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDynamicSize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return static_cast<const ConcreteOp *>(impl)->getDynamicSize(tablegen_opaque_val, idx);
}
template<typename ConcreteOp>
::mlir::Value detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDynamicStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return static_cast<const ConcreteOp *>(impl)->getDynamicStride(tablegen_opaque_val, idx);
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isSameAs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OffsetSizeAndStrideOpInterface other, ::llvm::function_ref<bool(::mlir::OpFoldResult, ::mlir::OpFoldResult)> cmp) {
  return static_cast<const ConcreteOp *>(impl)->isSameAs(tablegen_opaque_val, other, cmp);
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::hasUnitStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasUnitStride(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::hasZeroOffset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasZeroOffset(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OperandRange detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::offsets(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOffsets();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OperandRange detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::sizes(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSizes();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OperandRange detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::strides(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStrides();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::ArrayAttr detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::static_offsets(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticOffsets();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::ArrayAttr detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::static_sizes(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticSizes();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::ArrayAttr detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::static_strides(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticStrides();
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::SmallVector<::mlir::OpFoldResult, 4> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMixedOffsets(::mlir::Operation *tablegen_opaque_val) const {
return ::mlir::getMixedStridesOrOffsets((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticOffsets(),
                                                (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOffsets());
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::SmallVector<::mlir::OpFoldResult, 4> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMixedSizes(::mlir::Operation *tablegen_opaque_val) const {
return ::mlir::getMixedSizes((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticSizes(), (llvm::cast<ConcreteOp>(tablegen_opaque_val)).sizes());
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::SmallVector<::mlir::OpFoldResult, 4> detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMixedStrides(::mlir::Operation *tablegen_opaque_val) const {
return ::mlir::getMixedStridesOrOffsets((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticStrides(),
                                                (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStrides());
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isDynamicOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx) const {
::llvm::APInt v = *(static_offsets()
          .template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return ::mlir::ShapedType::isDynamicStrideOrOffset(v.getSExtValue());
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isDynamicSize(::mlir::Operation *tablegen_opaque_val, unsigned idx) const {
::llvm::APInt v = *(static_sizes()
          .template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return ::mlir::ShapedType::isDynamic(v.getSExtValue());
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isDynamicStride(::mlir::Operation *tablegen_opaque_val, unsigned idx) const {
::llvm::APInt v = *(static_strides()
          .template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return ::mlir::ShapedType::isDynamicStrideOrOffset(v.getSExtValue());
}
template<typename ConcreteModel, typename ConcreteOp>
int64_t detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getStaticOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx) const {
assert(!(llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDynamicOffset(idx) && "expected static offset");
        ::llvm::APInt v = *(static_offsets().
          template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return v.getSExtValue();
}
template<typename ConcreteModel, typename ConcreteOp>
int64_t detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getStaticSize(::mlir::Operation *tablegen_opaque_val, unsigned idx) const {
assert(!(llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDynamicSize(idx) && "expected static size");
        ::llvm::APInt v = *(static_sizes().
          template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return v.getSExtValue();
}
template<typename ConcreteModel, typename ConcreteOp>
int64_t detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getStaticStride(::mlir::Operation *tablegen_opaque_val, unsigned idx) const {
assert(!(llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDynamicStride(idx) && "expected static stride");
        ::llvm::APInt v = *(static_strides().
          template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return v.getSExtValue();
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getIndexOfDynamicOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx) const {
assert((llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDynamicOffset(idx) && "expected dynamic offset");
        auto numDynamic = getNumDynamicEntriesUpToIdx(
          static_offsets().template cast<::mlir::ArrayAttr>(),
          ::mlir::ShapedType::isDynamicStrideOrOffset,
          idx);
        return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOffsetSizeAndStrideStartOperandIndex() + numDynamic;
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getIndexOfDynamicSize(::mlir::Operation *tablegen_opaque_val, unsigned idx) const {
assert((llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDynamicSize(idx) && "expected dynamic size");
        auto numDynamic = getNumDynamicEntriesUpToIdx(
          static_sizes().template cast<::mlir::ArrayAttr>(), ::mlir::ShapedType::isDynamic, idx);
        return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOffsetSizeAndStrideStartOperandIndex() +
          offsets().size() + numDynamic;
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getIndexOfDynamicStride(::mlir::Operation *tablegen_opaque_val, unsigned idx) const {
assert((llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDynamicStride(idx) && "expected dynamic stride");
        auto numDynamic = getNumDynamicEntriesUpToIdx(
          static_strides().template cast<::mlir::ArrayAttr>(),
          ::mlir::ShapedType::isDynamicStrideOrOffset,
          idx);
        return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOffsetSizeAndStrideStartOperandIndex() +
          offsets().size() + sizes().size() + numDynamic;
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumDynamicEntriesUpToIdx(::mlir::Operation *tablegen_opaque_val, ::mlir::ArrayAttr attr, ::llvm::function_ref<bool(int64_t)> isDynamic, unsigned idx) const {
return std::count_if(
          attr.getValue().begin(), attr.getValue().begin() + idx,
          [&](::mlir::Attribute attr) {
            return isDynamic(attr.cast<::mlir::IntegerAttr>().getInt());
          });
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Value detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDynamicOffset(::mlir::Operation *tablegen_opaque_val, unsigned idx) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperand(getIndexOfDynamicOffset(idx));
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Value detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDynamicSize(::mlir::Operation *tablegen_opaque_val, unsigned idx) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperand(getIndexOfDynamicSize(idx));
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Value detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDynamicStride(::mlir::Operation *tablegen_opaque_val, unsigned idx) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperand(getIndexOfDynamicStride(idx));
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isSameAs(::mlir::Operation *tablegen_opaque_val, ::mlir::OffsetSizeAndStrideOpInterface other, ::llvm::function_ref<bool(::mlir::OpFoldResult, ::mlir::OpFoldResult)> cmp) const {
return ::mlir::detail::sameOffsetsSizesAndStrides(
          ::mlir::cast<::mlir::OffsetSizeAndStrideOpInterface>(
            (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()), other, cmp);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasUnitStride(::mlir::Operation *tablegen_opaque_val) const {
return ::llvm::all_of(getMixedStrides(), [](::mlir::OpFoldResult ofr) {
          return ::mlir::getConstantIntValue(ofr) == static_cast<int64_t>(1);
        });
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::OffsetSizeAndStrideOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasZeroOffset(::mlir::Operation *tablegen_opaque_val) const {
return ::llvm::all_of(getMixedOffsets(), [](::mlir::OpFoldResult ofr) {
          return ::mlir::getConstantIntValue(ofr) == static_cast<int64_t>(0);
        });
}
} // namespace mlir
namespace mlir {
class ViewLikeOpInterface;
namespace detail {
struct ViewLikeOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::Value (*getViewSource)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ViewLikeOpInterface;
    Model() : Concept{getViewSource} {}

    static inline ::mlir::Value getViewSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ViewLikeOpInterface;
    FallbackModel() : Concept{getViewSource} {}

    static inline ::mlir::Value getViewSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};template <typename ConcreteOp>
struct ViewLikeOpInterfaceTrait;

} // namespace detail
class ViewLikeOpInterface : public ::mlir::OpInterface<ViewLikeOpInterface, detail::ViewLikeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ViewLikeOpInterface, detail::ViewLikeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ViewLikeOpInterfaceTrait<ConcreteOp> {};
  /// Returns the source buffer from which the view is created.
  ::mlir::Value getViewSource();
};
namespace detail {
  template <typename ConcreteOp>
  struct ViewLikeOpInterfaceTrait : public ::mlir::OpInterface<ViewLikeOpInterface, detail::ViewLikeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::Value detail::ViewLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getViewSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getViewSource();
}
template<typename ConcreteOp>
::mlir::Value detail::ViewLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getViewSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getViewSource(tablegen_opaque_val);
}
} // namespace mlir
