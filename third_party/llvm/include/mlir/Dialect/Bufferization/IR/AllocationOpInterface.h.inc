/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace bufferization {
class AllocationOpInterface;
namespace detail {
struct AllocationOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::Optional<::mlir::Operation*> (*buildDealloc)(::mlir::OpBuilder&, ::mlir::Value);
    ::mlir::Optional<::mlir::Value> (*buildClone)(::mlir::OpBuilder&, ::mlir::Value);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::bufferization::AllocationOpInterface;
    Model() : Concept{buildDealloc, buildClone} {}

    static inline ::mlir::Optional<::mlir::Operation*> buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc);
    static inline ::mlir::Optional<::mlir::Value> buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::bufferization::AllocationOpInterface;
    FallbackModel() : Concept{buildDealloc, buildClone} {}

    static inline ::mlir::Optional<::mlir::Operation*> buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc);
    static inline ::mlir::Optional<::mlir::Value> buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    static ::mlir::Optional<::mlir::Operation*> buildDealloc(::mlir::OpBuilder&builder, ::mlir::Value alloc);
    static ::mlir::Optional<::mlir::Value> buildClone(::mlir::OpBuilder&builder, ::mlir::Value alloc);
  };
};template <typename ConcreteOp>
struct AllocationOpInterfaceTrait;

} // namespace detail
class AllocationOpInterface : public ::mlir::OpInterface<AllocationOpInterface, detail::AllocationOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AllocationOpInterface, detail::AllocationOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AllocationOpInterfaceTrait<ConcreteOp> {};
  ::mlir::Optional<::mlir::Operation*> buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc);
  ::mlir::Optional<::mlir::Value> buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc);
};
namespace detail {
  template <typename ConcreteOp>
  struct AllocationOpInterfaceTrait : public ::mlir::OpInterface<AllocationOpInterface, detail::AllocationOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::mlir::Optional<::mlir::Operation*> buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
      return llvm::None;
    }
    static ::mlir::Optional<::mlir::Value> buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
      return llvm::None;
    }
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::Optional<::mlir::Operation*> detail::AllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
  return ConcreteOp::buildDealloc(builder, alloc);
}
template<typename ConcreteOp>
::mlir::Optional<::mlir::Value> detail::AllocationOpInterfaceInterfaceTraits::Model<ConcreteOp>::buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
  return ConcreteOp::buildClone(builder, alloc);
}
template<typename ConcreteOp>
::mlir::Optional<::mlir::Operation*> detail::AllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
  return ConcreteOp::buildDealloc(builder, alloc);
}
template<typename ConcreteOp>
::mlir::Optional<::mlir::Value> detail::AllocationOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
  return ConcreteOp::buildClone(builder, alloc);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Optional<::mlir::Operation*> detail::AllocationOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::buildDealloc(::mlir::OpBuilder&builder, ::mlir::Value alloc) {
return llvm::None;
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Optional<::mlir::Value> detail::AllocationOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::buildClone(::mlir::OpBuilder&builder, ::mlir::Value alloc) {
return llvm::None;
}
} // namespace bufferization
} // namespace mlir
