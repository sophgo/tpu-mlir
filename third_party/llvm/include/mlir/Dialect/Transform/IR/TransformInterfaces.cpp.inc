/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Applies the transformation represented by the current operation. This
/// accepts as arguments the object that must be populated with results of
/// the current transformation and a transformation state object that can be
/// used for queries, e.g., to obtain the list of operations on which the
/// transformation represented by the current op is targeted. Returns a
/// special status object indicating whether the transformation succeeded
/// or failed, and, if it failed, whether the failure is recoverable or not.
::mlir::DiagnosedSilenceableFailure mlir::transform::TransformOpInterface::apply(::mlir::transform::TransformResults & transformResults, ::mlir::transform::TransformState & state) {
      return getImpl()->apply(getImpl(), getOperation(), transformResults, state);
  }
