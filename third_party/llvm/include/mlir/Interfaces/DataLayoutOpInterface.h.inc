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
    ::mlir::DataLayoutSpecInterface (*getDataLayoutSpec)(const Concept *impl, ::mlir::Operation *);
    unsigned (*getTypeSize)(::mlir::Type, const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    unsigned (*getTypeSizeInBits)(::mlir::Type, const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    unsigned (*getTypeABIAlignment)(::mlir::Type, const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
    unsigned (*getTypePreferredAlignment)(::mlir::Type, const ::mlir::DataLayout &, ::mlir::DataLayoutEntryListRef);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DataLayoutOpInterface;
    Model() : Concept{getDataLayoutSpec, getTypeSize, getTypeSizeInBits, getTypeABIAlignment, getTypePreferredAlignment} {}

    static inline ::mlir::DataLayoutSpecInterface getDataLayoutSpec(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DataLayoutOpInterface;
    FallbackModel() : Concept{getDataLayoutSpec, getTypeSize, getTypeSizeInBits, getTypeABIAlignment, getTypePreferredAlignment} {}

    static inline ::mlir::DataLayoutSpecInterface getDataLayoutSpec(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline unsigned getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
    static inline unsigned getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    static unsigned getTypeSize(::mlir::Type type, const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params);
    static unsigned getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params);
    static unsigned getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params);
    static unsigned getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout &dataLayout, ::mlir::DataLayoutEntryListRef params);
  };
};template <typename ConcreteOp>
struct DataLayoutOpInterfaceTrait;

} // namespace detail
class DataLayoutOpInterface : public ::mlir::OpInterface<DataLayoutOpInterface, detail::DataLayoutOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DataLayoutOpInterface, detail::DataLayoutOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DataLayoutOpInterfaceTrait<ConcreteOp> {};
  ::mlir::DataLayoutSpecInterface getDataLayoutSpec();
  unsigned getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
  unsigned getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
  unsigned getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
  unsigned getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params);
};
namespace detail {
  template <typename ConcreteOp>
  struct DataLayoutOpInterfaceTrait : public ::mlir::OpInterface<DataLayoutOpInterface, detail::DataLayoutOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static unsigned getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      unsigned bits = ConcreteOp::getTypeSizeInBits(type, dataLayout, params);
        return ::llvm::divideCeil(bits, 8);
    }
    static unsigned getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return ::mlir::detail::getDefaultTypeSizeInBits(type, dataLayout,
                                                        params);
    }
    static unsigned getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return ::mlir::detail::getDefaultABIAlignment(type, dataLayout, params);
    }
    static unsigned getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return ::mlir::detail::getDefaultPreferredAlignment(type, dataLayout,
                                                            params);
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return ::mlir::detail::verifyDataLayoutOp(op);
    }
  };
}// namespace detail
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
} // namespace mlir
