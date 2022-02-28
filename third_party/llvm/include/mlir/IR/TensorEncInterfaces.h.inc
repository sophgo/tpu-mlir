/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class VerifiableTensorEncoding;
namespace detail {
struct VerifiableTensorEncodingInterfaceTraits {
  struct Concept {
    ::mlir::LogicalResult (*verifyEncoding)(const Concept *impl, ::mlir::Attribute , ArrayRef<int64_t>, Type, ::llvm::function_ref<::mlir::InFlightDiagnostic()>);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::VerifiableTensorEncoding;
    Model() : Concept{verifyEncoding} {}

    static inline ::mlir::LogicalResult verifyEncoding(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape, Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::VerifiableTensorEncoding;
    FallbackModel() : Concept{verifyEncoding} {}

    static inline ::mlir::LogicalResult verifyEncoding(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape, Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
  };
};template <typename ConcreteAttr>
struct VerifiableTensorEncodingTrait;

} // namespace detail
class VerifiableTensorEncoding : public ::mlir::AttributeInterface<VerifiableTensorEncoding, detail::VerifiableTensorEncodingInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<VerifiableTensorEncoding, detail::VerifiableTensorEncodingInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::VerifiableTensorEncodingTrait<ConcreteAttr> {};
  ::mlir::LogicalResult verifyEncoding(ArrayRef<int64_t> shape, Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct VerifiableTensorEncodingTrait : public ::mlir::AttributeInterface<VerifiableTensorEncoding, detail::VerifiableTensorEncodingInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
template<typename ConcreteAttr>
::mlir::LogicalResult detail::VerifiableTensorEncodingInterfaceTraits::Model<ConcreteAttr>::verifyEncoding(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape, Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  return (tablegen_opaque_val.cast<ConcreteAttr>()).verifyEncoding(shape, elementType, emitError);
}
template<typename ConcreteAttr>
::mlir::LogicalResult detail::VerifiableTensorEncodingInterfaceTraits::FallbackModel<ConcreteAttr>::verifyEncoding(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape, Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  return static_cast<const ConcreteAttr *>(impl)->verifyEncoding(tablegen_opaque_val, shape, elementType, emitError);
}
} // namespace mlir
