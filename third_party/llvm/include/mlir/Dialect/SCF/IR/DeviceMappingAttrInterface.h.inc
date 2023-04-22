/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class DeviceMappingAttrInterface;
namespace detail {
struct DeviceMappingAttrInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    int64_t (*getMappingId)(const Concept *impl, ::mlir::Attribute );
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DeviceMappingAttrInterface;
    Model() : Concept{getMappingId} {}

    static inline int64_t getMappingId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DeviceMappingAttrInterface;
    FallbackModel() : Concept{getMappingId} {}

    static inline int64_t getMappingId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};template <typename ConcreteAttr>
struct DeviceMappingAttrInterfaceTrait;

} // namespace detail
class DeviceMappingAttrInterface : public ::mlir::AttributeInterface<DeviceMappingAttrInterface, detail::DeviceMappingAttrInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<DeviceMappingAttrInterface, detail::DeviceMappingAttrInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::DeviceMappingAttrInterfaceTrait<ConcreteAttr> {};
  /// Returns mapping as an integer from the attribute.
  int64_t getMappingId() const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct DeviceMappingAttrInterfaceTrait : public ::mlir::AttributeInterface<DeviceMappingAttrInterface, detail::DeviceMappingAttrInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteAttr>
int64_t detail::DeviceMappingAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::getMappingId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).getMappingId();
}
template<typename ConcreteAttr>
int64_t detail::DeviceMappingAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getMappingId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getMappingId(tablegen_opaque_val);
}
} // namespace mlir
