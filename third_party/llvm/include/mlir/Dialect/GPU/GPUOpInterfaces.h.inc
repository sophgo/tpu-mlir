/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {
class AsyncOpInterface;
namespace detail {
struct AsyncOpInterfaceInterfaceTraits {
  struct Concept {
    OperandRange (*getAsyncDependencies)(const Concept *impl, ::mlir::Operation *);
    void (*addAsyncDependency)(const Concept *impl, ::mlir::Operation *, Value);
    OpResult (*getAsyncToken)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::gpu::AsyncOpInterface;
    Model() : Concept{getAsyncDependencies, addAsyncDependency, getAsyncToken} {}

    static inline OperandRange getAsyncDependencies(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void addAsyncDependency(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value token);
    static inline OpResult getAsyncToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::gpu::AsyncOpInterface;
    FallbackModel() : Concept{getAsyncDependencies, addAsyncDependency, getAsyncToken} {}

    static inline OperandRange getAsyncDependencies(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void addAsyncDependency(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value token);
    static inline OpResult getAsyncToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    OperandRange getAsyncDependencies(::mlir::Operation *tablegen_opaque_val) const;
    void addAsyncDependency(::mlir::Operation *tablegen_opaque_val, Value token) const;
    OpResult getAsyncToken(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct AsyncOpInterfaceTrait;

} // namespace detail
class AsyncOpInterface : public ::mlir::OpInterface<AsyncOpInterface, detail::AsyncOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AsyncOpInterface, detail::AsyncOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AsyncOpInterfaceTrait<ConcreteOp> {};
  OperandRange getAsyncDependencies();
  void addAsyncDependency(Value token);
  OpResult getAsyncToken();
};
namespace detail {
  template <typename ConcreteOp>
  struct AsyncOpInterfaceTrait : public ::mlir::OpInterface<AsyncOpInterface, detail::AsyncOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    OperandRange getAsyncDependencies() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncDependencies();
    }
    void addAsyncDependency(Value token) {
      ::mlir::gpu::addAsyncDependency(this->getOperation(), token);
    }
    OpResult getAsyncToken() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncToken().template dyn_cast_or_null<OpResult>();
    }
  };
}// namespace detail
template<typename ConcreteOp>
OperandRange detail::AsyncOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAsyncDependencies(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsyncDependencies();
}
template<typename ConcreteOp>
void detail::AsyncOpInterfaceInterfaceTraits::Model<ConcreteOp>::addAsyncDependency(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value token) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).addAsyncDependency(token);
}
template<typename ConcreteOp>
OpResult detail::AsyncOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAsyncToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsyncToken();
}
template<typename ConcreteOp>
OperandRange detail::AsyncOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAsyncDependencies(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAsyncDependencies(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::AsyncOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::addAsyncDependency(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value token) {
  return static_cast<const ConcreteOp *>(impl)->addAsyncDependency(tablegen_opaque_val, token);
}
template<typename ConcreteOp>
OpResult detail::AsyncOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAsyncToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAsyncToken(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
OperandRange detail::AsyncOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAsyncDependencies(::mlir::Operation *tablegen_opaque_val) const {
ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncDependencies();
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::AsyncOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::addAsyncDependency(::mlir::Operation *tablegen_opaque_val, Value token) const {
::mlir::gpu::addAsyncDependency(this->getOperation(), token);
}
template<typename ConcreteModel, typename ConcreteOp>
OpResult detail::AsyncOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAsyncToken(::mlir::Operation *tablegen_opaque_val) const {
ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncToken().template dyn_cast_or_null<OpResult>();
}
} // namespace gpu
} // namespace mlir
