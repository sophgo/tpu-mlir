/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace transform {
class TransformOpInterface;
namespace detail {
struct TransformOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::LogicalResult (*apply)(const Concept *impl, ::mlir::Operation *, ::mlir::transform::TransformResults &, ::mlir::transform::TransformState &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::transform::TransformOpInterface;
    Model() : Concept{apply} {}

    static inline ::mlir::LogicalResult apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::transform::TransformOpInterface;
    FallbackModel() : Concept{apply} {}

    static inline ::mlir::LogicalResult apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
  };
};template <typename ConcreteOp>
struct TransformOpInterfaceTrait;

} // namespace detail
class TransformOpInterface : public ::mlir::OpInterface<TransformOpInterface, detail::TransformOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TransformOpInterface, detail::TransformOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TransformOpInterfaceTrait<ConcreteOp> {};
  ::mlir::LogicalResult apply(::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state);
};
namespace detail {
  template <typename ConcreteOp>
  struct TransformOpInterfaceTrait : public ::mlir::OpInterface<TransformOpInterface, detail::TransformOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::LogicalResult detail::TransformOpInterfaceInterfaceTraits::Model<ConcreteOp>::apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).apply(transformResults, state);
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::TransformOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state) {
  return static_cast<const ConcreteOp *>(impl)->apply(tablegen_opaque_val, transformResults, state);
}
} // namespace transform
} // namespace mlir
