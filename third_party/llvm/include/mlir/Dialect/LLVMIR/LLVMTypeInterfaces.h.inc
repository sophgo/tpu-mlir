/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace LLVM {
class PointerElementTypeInterface;
namespace detail {
struct PointerElementTypeInterfaceInterfaceTraits {
  struct Concept {
    unsigned (*getSizeInBytes)(const Concept *impl, ::mlir::Type , const DataLayout &);
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::LLVM::PointerElementTypeInterface;
    Model() : Concept{getSizeInBytes} {}

    static inline unsigned getSizeInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, const DataLayout & dataLayout);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::LLVM::PointerElementTypeInterface;
    FallbackModel() : Concept{getSizeInBytes} {}

    static inline unsigned getSizeInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, const DataLayout & dataLayout);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    unsigned getSizeInBytes(::mlir::Type tablegen_opaque_val, const DataLayout &dataLayout) const;
  };
};template <typename ConcreteType>
struct PointerElementTypeInterfaceTrait;

} // namespace detail
class PointerElementTypeInterface : public ::mlir::TypeInterface<PointerElementTypeInterface, detail::PointerElementTypeInterfaceInterfaceTraits> {
public:
  using ::mlir::TypeInterface<PointerElementTypeInterface, detail::PointerElementTypeInterfaceInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::PointerElementTypeInterfaceTrait<ConcreteType> {};
  unsigned getSizeInBytes(const DataLayout & dataLayout) const;
};
namespace detail {
  template <typename ConcreteType>
  struct PointerElementTypeInterfaceTrait : public ::mlir::TypeInterface<PointerElementTypeInterface, detail::PointerElementTypeInterfaceInterfaceTraits>::Trait<ConcreteType> {
    unsigned getSizeInBytes(const DataLayout & dataLayout) const {
      return dataLayout.getTypeSize((*static_cast<const ConcreteType *>(this)));
    }
  };
}// namespace detail
template<typename ConcreteType>
unsigned detail::PointerElementTypeInterfaceInterfaceTraits::Model<ConcreteType>::getSizeInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, const DataLayout & dataLayout) {
  return (tablegen_opaque_val.cast<ConcreteType>()).getSizeInBytes(dataLayout);
}
template<typename ConcreteType>
unsigned detail::PointerElementTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getSizeInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, const DataLayout & dataLayout) {
  return static_cast<const ConcreteType *>(impl)->getSizeInBytes(tablegen_opaque_val, dataLayout);
}
template<typename ConcreteModel, typename ConcreteType>
unsigned detail::PointerElementTypeInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getSizeInBytes(::mlir::Type tablegen_opaque_val, const DataLayout &dataLayout) const {
return dataLayout.getTypeSize((tablegen_opaque_val.cast<ConcreteType>()));
}
} // namespace LLVM
} // namespace mlir
