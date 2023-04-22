/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class LoopLikeOpInterface;
namespace detail {
struct LoopLikeOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*isDefinedOutsideOfLoop)(const Concept *impl, ::mlir::Operation *, ::mlir::Value );
    ::mlir::Region &(*getLoopBody)(const Concept *impl, ::mlir::Operation *);
    void (*moveOutOfLoop)(const Concept *impl, ::mlir::Operation *, ::mlir::Operation *);
    ::std::optional<::mlir::Value> (*getSingleInductionVar)(const Concept *impl, ::mlir::Operation *);
    ::std::optional<::mlir::OpFoldResult> (*getSingleLowerBound)(const Concept *impl, ::mlir::Operation *);
    ::std::optional<::mlir::OpFoldResult> (*getSingleStep)(const Concept *impl, ::mlir::Operation *);
    ::std::optional<::mlir::OpFoldResult> (*getSingleUpperBound)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::LoopLikeOpInterface;
    Model() : Concept{isDefinedOutsideOfLoop, getLoopBody, moveOutOfLoop, getSingleInductionVar, getSingleLowerBound, getSingleStep, getSingleUpperBound} {}

    static inline bool isDefinedOutsideOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value  value);
    static inline ::mlir::Region &getLoopBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void moveOutOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Operation * op);
    static inline ::std::optional<::mlir::Value> getSingleInductionVar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::std::optional<::mlir::OpFoldResult> getSingleLowerBound(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::std::optional<::mlir::OpFoldResult> getSingleStep(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::std::optional<::mlir::OpFoldResult> getSingleUpperBound(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::LoopLikeOpInterface;
    FallbackModel() : Concept{isDefinedOutsideOfLoop, getLoopBody, moveOutOfLoop, getSingleInductionVar, getSingleLowerBound, getSingleStep, getSingleUpperBound} {}

    static inline bool isDefinedOutsideOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value  value);
    static inline ::mlir::Region &getLoopBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void moveOutOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Operation * op);
    static inline ::std::optional<::mlir::Value> getSingleInductionVar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::std::optional<::mlir::OpFoldResult> getSingleLowerBound(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::std::optional<::mlir::OpFoldResult> getSingleStep(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::std::optional<::mlir::OpFoldResult> getSingleUpperBound(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    bool isDefinedOutsideOfLoop(::mlir::Operation *tablegen_opaque_val, ::mlir::Value value) const;
    void moveOutOfLoop(::mlir::Operation *tablegen_opaque_val, ::mlir::Operation *op) const;
    ::std::optional<::mlir::Value> getSingleInductionVar(::mlir::Operation *tablegen_opaque_val) const;
    ::std::optional<::mlir::OpFoldResult> getSingleLowerBound(::mlir::Operation *tablegen_opaque_val) const;
    ::std::optional<::mlir::OpFoldResult> getSingleStep(::mlir::Operation *tablegen_opaque_val) const;
    ::std::optional<::mlir::OpFoldResult> getSingleUpperBound(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct LoopLikeOpInterfaceTrait;

} // namespace detail
class LoopLikeOpInterface : public ::mlir::OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::LoopLikeOpInterfaceTrait<ConcreteOp> {};
  /// Returns true if the given value is defined outside of the loop.
  /// A sensible implementation could be to check whether the value's defining
  /// operation lies outside of the loops body region. If the loop uses
  /// explicit capture of dependencies, an implementation could check whether
  /// the value corresponds to a captured dependency.
  bool isDefinedOutsideOfLoop(::mlir::Value  value);
  /// Returns the region that makes up the body of the loop and should be
  /// inspected for loop-invariant operations.
  ::mlir::Region &getLoopBody();
  /// Moves the given loop-invariant operation out of the loop.
  void moveOutOfLoop(::mlir::Operation * op);
  /// If there is a single induction variable return it, otherwise return
  /// std::nullopt.
  ::std::optional<::mlir::Value> getSingleInductionVar();
  /// Return the single lower bound value or attribute if it exists, otherwise
  /// return std::nullopt.
  ::std::optional<::mlir::OpFoldResult> getSingleLowerBound();
  /// Return the single step value or attribute if it exists, otherwise
  /// return std::nullopt.
  ::std::optional<::mlir::OpFoldResult> getSingleStep();
  /// Return the single upper bound value or attribute if it exists, otherwise
  /// return std::nullopt.
  ::std::optional<::mlir::OpFoldResult> getSingleUpperBound();

    /// Returns if a block is inside a loop (within the current function). This
    /// can either be because the block is nested inside a LoopLikeInterface, or
    /// because the control flow graph is cyclic
    static bool blockIsInLoop(Block *block);
};
namespace detail {
  template <typename ConcreteOp>
  struct LoopLikeOpInterfaceTrait : public ::mlir::OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns true if the given value is defined outside of the loop.
    /// A sensible implementation could be to check whether the value's defining
    /// operation lies outside of the loops body region. If the loop uses
    /// explicit capture of dependencies, an implementation could check whether
    /// the value corresponds to a captured dependency.
    bool isDefinedOutsideOfLoop(::mlir::Value  value) {
      return value.getParentRegion()->isProperAncestor(&(*static_cast<ConcreteOp *>(this)).getLoopBody());
    }
    /// Moves the given loop-invariant operation out of the loop.
    void moveOutOfLoop(::mlir::Operation * op) {
      op->moveBefore((*static_cast<ConcreteOp *>(this)));
    }
    /// If there is a single induction variable return it, otherwise return
    /// std::nullopt.
    ::std::optional<::mlir::Value> getSingleInductionVar() {
      return std::nullopt;
    }
    /// Return the single lower bound value or attribute if it exists, otherwise
    /// return std::nullopt.
    ::std::optional<::mlir::OpFoldResult> getSingleLowerBound() {
      return std::nullopt;
    }
    /// Return the single step value or attribute if it exists, otherwise
    /// return std::nullopt.
    ::std::optional<::mlir::OpFoldResult> getSingleStep() {
      return std::nullopt;
    }
    /// Return the single upper bound value or attribute if it exists, otherwise
    /// return std::nullopt.
    ::std::optional<::mlir::OpFoldResult> getSingleUpperBound() {
      return std::nullopt;
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
bool detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDefinedOutsideOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value  value) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDefinedOutsideOfLoop(value);
}
template<typename ConcreteOp>
::mlir::Region &detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getLoopBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopBody();
}
template<typename ConcreteOp>
void detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::moveOutOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Operation * op) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).moveOutOfLoop(op);
}
template<typename ConcreteOp>
::std::optional<::mlir::Value> detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSingleInductionVar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSingleInductionVar();
}
template<typename ConcreteOp>
::std::optional<::mlir::OpFoldResult> detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSingleLowerBound(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSingleLowerBound();
}
template<typename ConcreteOp>
::std::optional<::mlir::OpFoldResult> detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSingleStep(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSingleStep();
}
template<typename ConcreteOp>
::std::optional<::mlir::OpFoldResult> detail::LoopLikeOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSingleUpperBound(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSingleUpperBound();
}
template<typename ConcreteOp>
bool detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isDefinedOutsideOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value  value) {
  return static_cast<const ConcreteOp *>(impl)->isDefinedOutsideOfLoop(tablegen_opaque_val, value);
}
template<typename ConcreteOp>
::mlir::Region &detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getLoopBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLoopBody(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::moveOutOfLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Operation * op) {
  return static_cast<const ConcreteOp *>(impl)->moveOutOfLoop(tablegen_opaque_val, op);
}
template<typename ConcreteOp>
::std::optional<::mlir::Value> detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSingleInductionVar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSingleInductionVar(tablegen_opaque_val);
}
template<typename ConcreteOp>
::std::optional<::mlir::OpFoldResult> detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSingleLowerBound(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSingleLowerBound(tablegen_opaque_val);
}
template<typename ConcreteOp>
::std::optional<::mlir::OpFoldResult> detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSingleStep(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSingleStep(tablegen_opaque_val);
}
template<typename ConcreteOp>
::std::optional<::mlir::OpFoldResult> detail::LoopLikeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSingleUpperBound(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSingleUpperBound(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isDefinedOutsideOfLoop(::mlir::Operation *tablegen_opaque_val, ::mlir::Value value) const {
return value.getParentRegion()->isProperAncestor(&(llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopBody());
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::moveOutOfLoop(::mlir::Operation *tablegen_opaque_val, ::mlir::Operation *op) const {
op->moveBefore((llvm::cast<ConcreteOp>(tablegen_opaque_val)));
}
template<typename ConcreteModel, typename ConcreteOp>
::std::optional<::mlir::Value> detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSingleInductionVar(::mlir::Operation *tablegen_opaque_val) const {
return std::nullopt;
}
template<typename ConcreteModel, typename ConcreteOp>
::std::optional<::mlir::OpFoldResult> detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSingleLowerBound(::mlir::Operation *tablegen_opaque_val) const {
return std::nullopt;
}
template<typename ConcreteModel, typename ConcreteOp>
::std::optional<::mlir::OpFoldResult> detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSingleStep(::mlir::Operation *tablegen_opaque_val) const {
return std::nullopt;
}
template<typename ConcreteModel, typename ConcreteOp>
::std::optional<::mlir::OpFoldResult> detail::LoopLikeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getSingleUpperBound(::mlir::Operation *tablegen_opaque_val) const {
return std::nullopt;
}
} // namespace mlir
