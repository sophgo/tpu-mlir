/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class InferIntRangeInterface;
namespace detail {
struct InferIntRangeInterfaceInterfaceTraits {
  struct Concept {
    void (*inferResultRanges)(const Concept *impl, ::mlir::Operation *, ::llvm::ArrayRef<::mlir::ConstantIntRanges>, ::mlir::SetIntRangeFn);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::InferIntRangeInterface;
    Model() : Concept{inferResultRanges} {}

    static inline void inferResultRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::InferIntRangeInterface;
    FallbackModel() : Concept{inferResultRanges} {}

    static inline void inferResultRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
  };
};template <typename ConcreteOp>
struct InferIntRangeInterfaceTrait;

} // namespace detail
class InferIntRangeInterface : public ::mlir::OpInterface<InferIntRangeInterface, detail::InferIntRangeInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<InferIntRangeInterface, detail::InferIntRangeInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::InferIntRangeInterfaceTrait<ConcreteOp> {};
  void inferResultRanges(::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges);
};
namespace detail {
  template <typename ConcreteOp>
  struct InferIntRangeInterfaceTrait : public ::mlir::OpInterface<InferIntRangeInterface, detail::InferIntRangeInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
template<typename ConcreteOp>
void detail::InferIntRangeInterfaceInterfaceTraits::Model<ConcreteOp>::inferResultRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferResultRanges(argRanges, setResultRanges);
}
template<typename ConcreteOp>
void detail::InferIntRangeInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::inferResultRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges) {
  return static_cast<const ConcreteOp *>(impl)->inferResultRanges(tablegen_opaque_val, argRanges, setResultRanges);
}
} // namespace mlir
