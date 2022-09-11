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
    ::mlir::DiagnosedSilenceableFailure (*apply)(const Concept *impl, ::mlir::Operation *, ::mlir::transform::TransformResults &, ::mlir::transform::TransformState &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::transform::TransformOpInterface;
    Model() : Concept{apply} {}

    static inline ::mlir::DiagnosedSilenceableFailure apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::transform::TransformOpInterface;
    FallbackModel() : Concept{apply} {}

    static inline ::mlir::DiagnosedSilenceableFailure apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};template <typename ConcreteOp>
struct TransformOpInterfaceTrait;

} // namespace detail
class TransformOpInterface : public ::mlir::OpInterface<TransformOpInterface, detail::TransformOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TransformOpInterface, detail::TransformOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TransformOpInterfaceTrait<ConcreteOp> {};
  /// Applies the transformation represented by the current operation. This
  /// accepts as arguments the object that must be populated with results of
  /// the current transformation and a transformation state object that can be
  /// used for queries, e.g., to obtain the list of operations on which the
  /// transformation represented by the current op is targeted. Returns a
  /// special status object indicating whether the transformation succeeded
  /// or failed, and, if it failed, whether the failure is recoverable or not.
  ::mlir::DiagnosedSilenceableFailure apply(::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state);

    /// Emits a generic transform error for the current transform operation
    /// targeting the given Payload IR operation and returns failure. Should
    /// be only used as a last resort when the transformation itself provides
    /// no further indication as to the reason of the failure.
    ::mlir::LogicalResult reportUnknownTransformError(
        ::mlir::Operation *target) {
      ::mlir::InFlightDiagnostic diag = (*this)->emitError() << "failed to apply";
      diag.attachNote(target->getLoc()) << "attempted to apply to this op";
      return diag;
    }

    /// Creates the silenceable failure object with a diagnostic located at the
    /// current operation.
    DiagnosedSilenceableFailure emitSilenceableError() {
      Diagnostic diag((*this)->getLoc(), DiagnosticSeverity::Error);
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }

    /// Creates the default silenceable failure for a transform op that failed
    /// to properly apply to a target.
    DiagnosedSilenceableFailure emitDefaultSilenceableFailure(
        Operation *target) {
      Diagnostic diag((*this)->getLoc(), DiagnosticSeverity::Error);
      diag << (*this)->getName() << " failed to apply";
      diag.attachNote(target->getLoc()) << "when applied to this op";
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }
  };
namespace detail {
  template <typename ConcreteOp>
  struct TransformOpInterfaceTrait : public ::mlir::OpInterface<TransformOpInterface, detail::TransformOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {

    /// Emits a generic transform error for the current transform operation
    /// targeting the given Payload IR operation and returns failure. Should
    /// be only used as a last resort when the transformation itself provides
    /// no further indication as to the reason of the failure.
    ::mlir::LogicalResult reportUnknownTransformError(
        ::mlir::Operation *target) {
      ::mlir::InFlightDiagnostic diag = (*static_cast<ConcreteOp *>(this))->emitError() << "failed to apply";
      diag.attachNote(target->getLoc()) << "attempted to apply to this op";
      return diag;
    }

    /// Creates the silenceable failure object with a diagnostic located at the
    /// current operation.
    DiagnosedSilenceableFailure emitSilenceableError() {
      Diagnostic diag((*static_cast<ConcreteOp *>(this))->getLoc(), DiagnosticSeverity::Error);
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }

    /// Creates the default silenceable failure for a transform op that failed
    /// to properly apply to a target.
    DiagnosedSilenceableFailure emitDefaultSilenceableFailure(
        Operation *target) {
      Diagnostic diag((*static_cast<ConcreteOp *>(this))->getLoc(), DiagnosticSeverity::Error);
      diag << (*static_cast<ConcreteOp *>(this))->getName() << " failed to apply";
      diag.attachNote(target->getLoc()) << "when applied to this op";
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }
  
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::DiagnosedSilenceableFailure detail::TransformOpInterfaceInterfaceTraits::Model<ConcreteOp>::apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).apply(transformResults, state);
}
template<typename ConcreteOp>
::mlir::DiagnosedSilenceableFailure detail::TransformOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::apply(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state) {
  return static_cast<const ConcreteOp *>(impl)->apply(tablegen_opaque_val, transformResults, state);
}
} // namespace transform
} // namespace mlir
