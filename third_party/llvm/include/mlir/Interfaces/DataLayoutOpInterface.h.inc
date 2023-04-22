/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class DataLayoutOpInterface;
namespace detail {
struct DataLayoutOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::DataLayoutSpecInterface (*getDataLayoutSpec)(const Concept *impl, ::mlir::Operation *);
    unsigned (*getTypeSize)(::mlir::Type, const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    unsigned (*getTypeSizeInBits)(::mlir::Type, const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    unsigned (*getTypeABIAlignment)(::mlir::Type, const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    unsigned (*getTypePreferredAlignment)(::mlir::Type, const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    ::mlir::Attribute (*getAllocaMemorySpace)(::mlir::DataLayoutEntryInterface);
    unsigned (*getStackAlignment)(::mlir::DataLayoutEntryInterface);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DataLayoutOpInterface;
    Model() : Concept{getDataLayoutSpec, getTypeSize, getTypeSizeInBits, getTypeABIAlignment, getTypePreferredAlignment, getAllocaMemorySpace, getStackAlignment} {}

    static inline ::mlir::DataLayoutSpecInterface getDataLayoutSpec(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline ::mlir::Attribute getAllocaMemorySpace(::mlir::DataLayoutEntryInterface entry);
    static inline unsigned getStackAlignment(::mlir::DataLayoutEntryInterface entry);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DataLayoutOpInterface;
    FallbackModel() : Concept{getDataLayoutSpec, getTypeSize, getTypeSizeInBits, getTypeABIAlignment, getTypePreferredAlignment, getAllocaMemorySpace, getStackAlignment} {}

    static inline ::mlir::DataLayoutSpecInterface getDataLayoutSpec(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline ::mlir::Attribute getAllocaMemorySpace(::mlir::DataLayoutEntryInterface entry);
    static inline unsigned getStackAlignment(::mlir::DataLayoutEntryInterface entry);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    static unsigned getTypeSize(::mlir::Type type, const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params);
    static unsigned getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params);
    static unsigned getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params);
    static unsigned getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params);
    static ::mlir::Attribute getAllocaMemorySpace(::mlir::DataLayoutEntryInterface entry);
    static unsigned getStackAlignment(::mlir::DataLayoutEntryInterface entry);
  };
};template <typename ConcreteOp>
struct DataLayoutOpInterfaceTrait;

} // namespace detail
class DataLayoutOpInterface : public ::mlir::OpInterface<DataLayoutOpInterface, detail::DataLayoutOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DataLayoutOpInterface, detail::DataLayoutOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DataLayoutOpInterfaceTrait<ConcreteOp> {};
  /// Returns the data layout specification for this op, or null if it does not exist.
  ::mlir::DataLayoutSpecInterface getDataLayoutSpec();
  /// Returns the size of the given type computed using the relevant entries. The data layout object can be used for recursive queries.
  unsigned getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
  /// Returns the size of the given type in bits computed using the relevant entries. The data layout object can be used for recursive queries.
  unsigned getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
  /// Returns the alignment required by the ABI for the given type computed using the relevant entries. The data layout object can be used for recursive queries.
  unsigned getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
  /// Returns the alignment preferred by the given type computed using the relevant entries. The data layoutobject can be used for recursive queries.
  unsigned getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
  /// Returns the memory space used by the ABI computed using the relevant entries. The data layout object can be used for recursive queries.
  ::mlir::Attribute getAllocaMemorySpace(::mlir::DataLayoutEntryInterface entry);
  /// Returns the natural stack alignment in bits computed using the relevant entries. The data layout object can be used for recursive queries.
  unsigned getStackAlignment(::mlir::DataLayoutEntryInterface entry);
};
namespace detail {
  template <typename ConcreteOp>
  struct DataLayoutOpInterfaceTrait : public ::mlir::OpInterface<DataLayoutOpInterface, detail::DataLayoutOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns the size of the given type computed using the relevant entries. The data layout object can be used for recursive queries.
    static unsigned getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      unsigned bits = ConcreteOp::getTypeSizeInBits(type, dataLayout, params);
        return ::llvm::divideCeil(bits, 8);
    }
    /// Returns the size of the given type in bits computed using the relevant entries. The data layout object can be used for recursive queries.
    static unsigned getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return ::mlir::detail::getDefaultTypeSizeInBits(type, dataLayout,
                                                        params);
    }
    /// Returns the alignment required by the ABI for the given type computed using the relevant entries. The data layout object can be used for recursive queries.
    static unsigned getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return ::mlir::detail::getDefaultABIAlignment(type, dataLayout, params);
    }
    /// Returns the alignment preferred by the given type computed using the relevant entries. The data layoutobject can be used for recursive queries.
    static unsigned getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return ::mlir::detail::getDefaultPreferredAlignment(type, dataLayout,
                                                            params);
    }
    /// Returns the memory space used by the ABI computed using the relevant entries. The data layout object can be used for recursive queries.
    static ::mlir::Attribute getAllocaMemorySpace(::mlir::DataLayoutEntryInterface entry) {
      return ::mlir::detail::getDefaultAllocaMemorySpace(entry);
    }
    /// Returns the natural stack alignment in bits computed using the relevant entries. The data layout object can be used for recursive queries.
    static unsigned getStackAlignment(::mlir::DataLayoutEntryInterface entry) {
      return ::mlir::detail::getDefaultStackAlignment(entry);
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return ::mlir::detail::verifyDataLayoutOp(op);
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::DataLayoutSpecInterface detail::DataLayoutOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDataLayoutSpec(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDataLayoutSpec();
}
template<typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return ConcreteOp::getTypeSize(type, dataLayout, params);
}
template<typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return ConcreteOp::getTypeSizeInBits(type, dataLayout, params);
}
template<typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return ConcreteOp::getTypeABIAlignment(type, dataLayout, params);
}
template<typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return ConcreteOp::getTypePreferredAlignment(type, dataLayout, params);
}
template<typename ConcreteOp>
::mlir::Attribute detail::DataLayoutOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAllocaMemorySpace(::mlir::DataLayoutEntryInterface entry) {
  return ConcreteOp::getAllocaMemorySpace(entry);
}
template<typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::Model<ConcreteOp>::getStackAlignment(::mlir::DataLayoutEntryInterface entry) {
  return ConcreteOp::getStackAlignment(entry);
}
template<typename ConcreteOp>
::mlir::DataLayoutSpecInterface detail::DataLayoutOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDataLayoutSpec(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDataLayoutSpec(tablegen_opaque_val);
}
template<typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return ConcreteOp::getTypeSize(type, dataLayout, params);
}
template<typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return ConcreteOp::getTypeSizeInBits(type, dataLayout, params);
}
template<typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return ConcreteOp::getTypeABIAlignment(type, dataLayout, params);
}
template<typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
  return ConcreteOp::getTypePreferredAlignment(type, dataLayout, params);
}
template<typename ConcreteOp>
::mlir::Attribute detail::DataLayoutOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAllocaMemorySpace(::mlir::DataLayoutEntryInterface entry) {
  return ConcreteOp::getAllocaMemorySpace(entry);
}
template<typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getStackAlignment(::mlir::DataLayoutEntryInterface entry) {
  return ConcreteOp::getStackAlignment(entry);
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTypeSize(::mlir::Type type, const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params) {
unsigned bits = ConcreteOp::getTypeSizeInBits(type, dataLayout, params);
        return ::llvm::divideCeil(bits, 8);
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params) {
return ::mlir::detail::getDefaultTypeSizeInBits(type, dataLayout,
                                                        params);
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params) {
return ::mlir::detail::getDefaultABIAlignment(type, dataLayout, params);
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params) {
return ::mlir::detail::getDefaultPreferredAlignment(type, dataLayout,
                                                            params);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Attribute detail::DataLayoutOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAllocaMemorySpace(::mlir::DataLayoutEntryInterface entry) {
return ::mlir::detail::getDefaultAllocaMemorySpace(entry);
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::DataLayoutOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getStackAlignment(::mlir::DataLayoutEntryInterface entry) {
return ::mlir::detail::getDefaultStackAlignment(entry);
}
} // namespace mlir
