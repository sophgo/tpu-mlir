/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Verifies the encoding is valid for a tensor type with the
/// given shape and element type. Generates a diagnostic using
/// the supplied callback on failure.
::mlir::LogicalResult mlir::VerifiableTensorEncoding::verifyEncoding(ArrayRef<int64_t> shape, Type elementType, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const {
      return getImpl()->verifyEncoding(getImpl(), *this, shape, elementType, emitError);
  }
