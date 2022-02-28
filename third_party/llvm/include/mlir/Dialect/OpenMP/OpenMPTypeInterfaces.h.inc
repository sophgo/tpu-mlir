/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace omp {
class PointerLikeType;
namespace detail {
struct PointerLikeTypeInterfaceTraits {
  struct Concept {
    ::mlir::Type (*getElementType)(const Concept *impl, ::mlir::Type );
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::omp::PointerLikeType;
    Model() : Concept{getElementType} {}

    static inline ::mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::omp::PointerLikeType;
    FallbackModel() : Concept{getElementType} {}

    static inline ::mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
  };
};template <typename ConcreteType>
struct PointerLikeTypeTrait;

} // namespace detail
class PointerLikeType : public ::mlir::TypeInterface<PointerLikeType, detail::PointerLikeTypeInterfaceTraits> {
public:
  using ::mlir::TypeInterface<PointerLikeType, detail::PointerLikeTypeInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::PointerLikeTypeTrait<ConcreteType> {};
  ::mlir::Type getElementType() const;
};
namespace detail {
  template <typename ConcreteType>
  struct PointerLikeTypeTrait : public ::mlir::TypeInterface<PointerLikeType, detail::PointerLikeTypeInterfaceTraits>::Trait<ConcreteType> {
  };
}// namespace detail
template<typename ConcreteType>
::mlir::Type detail::PointerLikeTypeInterfaceTraits::Model<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (tablegen_opaque_val.cast<ConcreteType>()).getElementType();
}
template<typename ConcreteType>
::mlir::Type detail::PointerLikeTypeInterfaceTraits::FallbackModel<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getElementType(tablegen_opaque_val);
}
} // namespace omp
} // namespace mlir
