/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class MemRefElementTypeInterface;
namespace detail {
struct MemRefElementTypeInterfaceInterfaceTraits {
  struct Concept {
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::MemRefElementTypeInterface;
    Model() : Concept{} {}

  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::MemRefElementTypeInterface;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
  };
};template <typename ConcreteType>
struct MemRefElementTypeInterfaceTrait;

} // namespace detail
class MemRefElementTypeInterface : public ::mlir::TypeInterface<MemRefElementTypeInterface, detail::MemRefElementTypeInterfaceInterfaceTraits> {
public:
  using ::mlir::TypeInterface<MemRefElementTypeInterface, detail::MemRefElementTypeInterfaceInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::MemRefElementTypeInterfaceTrait<ConcreteType> {};
};
namespace detail {
  template <typename ConcreteType>
  struct MemRefElementTypeInterfaceTrait : public ::mlir::TypeInterface<MemRefElementTypeInterface, detail::MemRefElementTypeInterfaceInterfaceTraits>::Trait<ConcreteType> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class ShapedType;
namespace detail {
struct ShapedTypeInterfaceTraits {
  struct Concept {
    ::mlir::ShapedType (*cloneWith)(const Concept *impl, ::mlir::Type , ::llvm::Optional<::llvm::ArrayRef<int64_t>>, ::mlir::Type);
    ::mlir::Type (*getElementType)(const Concept *impl, ::mlir::Type );
    bool (*hasRank)(const Concept *impl, ::mlir::Type );
    ::llvm::ArrayRef<int64_t> (*getShape)(const Concept *impl, ::mlir::Type );
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ShapedType;
    Model() : Concept{cloneWith, getElementType, hasRank, getShape} {}

    static inline ::mlir::ShapedType cloneWith(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::llvm::Optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType);
    static inline ::mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline bool hasRank(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline ::llvm::ArrayRef<int64_t> getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ShapedType;
    FallbackModel() : Concept{cloneWith, getElementType, hasRank, getShape} {}

    static inline ::mlir::ShapedType cloneWith(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::llvm::Optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType);
    static inline ::mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline bool hasRank(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline ::llvm::ArrayRef<int64_t> getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
  };
};template <typename ConcreteType>
struct ShapedTypeTrait;

} // namespace detail
class ShapedType : public ::mlir::TypeInterface<ShapedType, detail::ShapedTypeInterfaceTraits> {
public:
  using ::mlir::TypeInterface<ShapedType, detail::ShapedTypeInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::ShapedTypeTrait<ConcreteType> {};
  /// Returns a clone of this type with the given shape and element
  /// type. If a shape is not provided, the current shape of the type is used.
  ::mlir::ShapedType cloneWith(::llvm::Optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) const;
  /// Returns the element type of this shaped type.
  ::mlir::Type getElementType() const;
  /// Returns if this type is ranked, i.e. it has a known number of dimensions.
  bool hasRank() const;
  /// Returns the shape of this type if it is ranked, otherwise asserts.
  ::llvm::ArrayRef<int64_t> getShape() const;

    // TODO: merge these two special values in a single one used everywhere.
    // Unfortunately, uses of `-1` have crept deep into the codebase now and are
    // hard to track.
    static constexpr int64_t kDynamicSize = -1;
    static constexpr int64_t kDynamicStrideOrOffset =
        std::numeric_limits<int64_t>::min();

    /// Whether the given dimension size indicates a dynamic dimension.
    static constexpr bool isDynamic(int64_t dSize) {
      return dSize == kDynamicSize;
    }
    static constexpr bool isDynamicStrideOrOffset(int64_t dStrideOrOffset) {
      return dStrideOrOffset == kDynamicStrideOrOffset;
    }

    /// Return the number of elements present in the given shape.
    static int64_t getNumElements(ArrayRef<int64_t> shape);

    /// Returns the total amount of bits occupied by a value of this type. This
    /// does not take into account any memory layout or widening constraints,
    /// e.g. a vector<3xi57> may report to occupy 3x57=171 bit, even though in
    /// practice it will likely be stored as in a 4xi64 vector register. Fails
    /// with an assertion if the size cannot be computed statically, e.g. if the
    /// type has a dynamic shape or if its elemental type does not have a known
    /// bit width.
    int64_t getSizeInBits() const;
  

    /// Return a clone of this type with the given new shape and element type.
    auto clone(::llvm::ArrayRef<int64_t> shape, Type elementType) {
      return (*this).cloneWith(shape, elementType);
    }
    /// Return a clone of this type with the given new shape.
    auto clone(::llvm::ArrayRef<int64_t> shape) {
      return (*this).cloneWith(shape, (*this).getElementType());
    }
    /// Return a clone of this type with the given new element type.
    auto clone(::mlir::Type elementType) {
      return (*this).cloneWith(/*shape=*/llvm::None, elementType);
    }

    /// If an element type is an integer or a float, return its width. Otherwise,
    /// abort.
    unsigned getElementTypeBitWidth() const {
      return (*this).getElementType().getIntOrFloatBitWidth();
    }

    /// If this is a ranked type, return the rank. Otherwise, abort.
    int64_t getRank() const {
      assert((*this).hasRank() && "cannot query rank of unranked shaped type");
      return (*this).getShape().size();
    }

    /// If it has static shape, return the number of elements. Otherwise, abort.
    int64_t getNumElements() const {
      assert(hasStaticShape() && "cannot get element count of dynamic shaped type");
      return ::mlir::ShapedType::getNumElements((*this).getShape());
    }

    /// Returns true if this dimension has a dynamic size (for ranked types);
    /// aborts for unranked types.
    bool isDynamicDim(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return ::mlir::ShapedType::isDynamic((*this).getShape()[idx]);
    }

    /// Returns if this type has a static shape, i.e. if the type is ranked and
    /// all dimensions have known size (>= 0).
    bool hasStaticShape() const {
      return (*this).hasRank() &&
             llvm::none_of((*this).getShape(), ::mlir::ShapedType::isDynamic);
    }

    /// Returns if this type has a static shape and the shape is equal to
    /// `shape` return true.
    bool hasStaticShape(::llvm::ArrayRef<int64_t> shape) const {
      return hasStaticShape() && (*this).getShape() == shape;
    }

    /// If this is a ranked type, return the number of dimensions with dynamic
    /// size. Otherwise, abort.
    int64_t getNumDynamicDims() const {
      return llvm::count_if((*this).getShape(), ::mlir::ShapedType::isDynamic);
    }

    /// If this is ranked type, return the size of the specified dimension.
    /// Otherwise, abort.
    int64_t getDimSize(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return (*this).getShape()[idx];
    }

    /// Returns the position of the dynamic dimension relative to just the dynamic
    /// dimensions, given its `index` within the shape.
    unsigned getDynamicDimIndex(unsigned index) const {
      assert(index < getRank() && "invalid index");
      assert(::mlir::ShapedType::isDynamic(getDimSize(index)) && "invalid index");
      return llvm::count_if((*this).getShape().take_front(index),
                            ::mlir::ShapedType::isDynamic);
    }
  };
namespace detail {
  template <typename ConcreteType>
  struct ShapedTypeTrait : public ::mlir::TypeInterface<ShapedType, detail::ShapedTypeInterfaceTraits>::Trait<ConcreteType> {

    /// Return a clone of this type with the given new shape and element type.
    auto clone(::llvm::ArrayRef<int64_t> shape, Type elementType) {
      return (*static_cast<const ConcreteType *>(this)).cloneWith(shape, elementType);
    }
    /// Return a clone of this type with the given new shape.
    auto clone(::llvm::ArrayRef<int64_t> shape) {
      return (*static_cast<const ConcreteType *>(this)).cloneWith(shape, (*static_cast<const ConcreteType *>(this)).getElementType());
    }
    /// Return a clone of this type with the given new element type.
    auto clone(::mlir::Type elementType) {
      return (*static_cast<const ConcreteType *>(this)).cloneWith(/*shape=*/llvm::None, elementType);
    }

    /// If an element type is an integer or a float, return its width. Otherwise,
    /// abort.
    unsigned getElementTypeBitWidth() const {
      return (*static_cast<const ConcreteType *>(this)).getElementType().getIntOrFloatBitWidth();
    }

    /// If this is a ranked type, return the rank. Otherwise, abort.
    int64_t getRank() const {
      assert((*static_cast<const ConcreteType *>(this)).hasRank() && "cannot query rank of unranked shaped type");
      return (*static_cast<const ConcreteType *>(this)).getShape().size();
    }

    /// If it has static shape, return the number of elements. Otherwise, abort.
    int64_t getNumElements() const {
      assert(hasStaticShape() && "cannot get element count of dynamic shaped type");
      return ::mlir::ShapedType::getNumElements((*static_cast<const ConcreteType *>(this)).getShape());
    }

    /// Returns true if this dimension has a dynamic size (for ranked types);
    /// aborts for unranked types.
    bool isDynamicDim(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return ::mlir::ShapedType::isDynamic((*static_cast<const ConcreteType *>(this)).getShape()[idx]);
    }

    /// Returns if this type has a static shape, i.e. if the type is ranked and
    /// all dimensions have known size (>= 0).
    bool hasStaticShape() const {
      return (*static_cast<const ConcreteType *>(this)).hasRank() &&
             llvm::none_of((*static_cast<const ConcreteType *>(this)).getShape(), ::mlir::ShapedType::isDynamic);
    }

    /// Returns if this type has a static shape and the shape is equal to
    /// `shape` return true.
    bool hasStaticShape(::llvm::ArrayRef<int64_t> shape) const {
      return hasStaticShape() && (*static_cast<const ConcreteType *>(this)).getShape() == shape;
    }

    /// If this is a ranked type, return the number of dimensions with dynamic
    /// size. Otherwise, abort.
    int64_t getNumDynamicDims() const {
      return llvm::count_if((*static_cast<const ConcreteType *>(this)).getShape(), ::mlir::ShapedType::isDynamic);
    }

    /// If this is ranked type, return the size of the specified dimension.
    /// Otherwise, abort.
    int64_t getDimSize(unsigned idx) const {
      assert(idx < getRank() && "invalid index for shaped type");
      return (*static_cast<const ConcreteType *>(this)).getShape()[idx];
    }

    /// Returns the position of the dynamic dimension relative to just the dynamic
    /// dimensions, given its `index` within the shape.
    unsigned getDynamicDimIndex(unsigned index) const {
      assert(index < getRank() && "invalid index");
      assert(::mlir::ShapedType::isDynamic(getDimSize(index)) && "invalid index");
      return llvm::count_if((*static_cast<const ConcreteType *>(this)).getShape().take_front(index),
                            ::mlir::ShapedType::isDynamic);
    }
  
  };
}// namespace detail
template<typename ConcreteType>
::mlir::ShapedType detail::ShapedTypeInterfaceTraits::Model<ConcreteType>::cloneWith(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::llvm::Optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) {
  return (tablegen_opaque_val.cast<ConcreteType>()).cloneWith(shape, elementType);
}
template<typename ConcreteType>
::mlir::Type detail::ShapedTypeInterfaceTraits::Model<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (tablegen_opaque_val.cast<ConcreteType>()).getElementType();
}
template<typename ConcreteType>
bool detail::ShapedTypeInterfaceTraits::Model<ConcreteType>::hasRank(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (tablegen_opaque_val.cast<ConcreteType>()).hasRank();
}
template<typename ConcreteType>
::llvm::ArrayRef<int64_t> detail::ShapedTypeInterfaceTraits::Model<ConcreteType>::getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (tablegen_opaque_val.cast<ConcreteType>()).getShape();
}
template<typename ConcreteType>
::mlir::ShapedType detail::ShapedTypeInterfaceTraits::FallbackModel<ConcreteType>::cloneWith(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::llvm::Optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) {
  return static_cast<const ConcreteType *>(impl)->cloneWith(tablegen_opaque_val, shape, elementType);
}
template<typename ConcreteType>
::mlir::Type detail::ShapedTypeInterfaceTraits::FallbackModel<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getElementType(tablegen_opaque_val);
}
template<typename ConcreteType>
bool detail::ShapedTypeInterfaceTraits::FallbackModel<ConcreteType>::hasRank(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->hasRank(tablegen_opaque_val);
}
template<typename ConcreteType>
::llvm::ArrayRef<int64_t> detail::ShapedTypeInterfaceTraits::FallbackModel<ConcreteType>::getShape(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getShape(tablegen_opaque_val);
}
} // namespace mlir
