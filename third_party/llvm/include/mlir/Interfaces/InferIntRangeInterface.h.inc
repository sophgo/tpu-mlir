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
    using ConcreteEntity = ConcreteOp;
  };
};template <typename ConcreteOp>
struct InferIntRangeInterfaceTrait;

} // namespace detail
class InferIntRangeInterface : public ::mlir::OpInterface<InferIntRangeInterface, detail::InferIntRangeInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<InferIntRangeInterface, detail::InferIntRangeInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::InferIntRangeInterfaceTrait<ConcreteOp> {};
  /// Infer the bounds on the results of this op given the bounds on its arguments.
  /// For each result value or block argument (that isn't a branch argument,
  /// since the dataflow analysis handles those case), the method should call
  /// `setValueRange` with that `Value` as an argument. When `setValueRange`
  /// is not called for some value, it will recieve a default value of the mimimum
  /// and maximum values for its type (the unbounded range).
  /// 
  /// When called on an op that also implements the RegionBranchOpInterface
  /// or BranchOpInterface, this method should not attempt to infer the values
  /// of the branch results, as this will be handled by the analyses that use
  /// this interface.
  /// 
  /// This function will only be called when at least one result of the op is a
  /// scalar integer value or the op has a region.
  /// 
  /// `argRanges` contains one `IntRangeAttrs` for each argument to the op in ODS
  ///  order. Non-integer arguments will have the an unbounded range of width-0
  ///  APInts in their `argRanges` element.
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
