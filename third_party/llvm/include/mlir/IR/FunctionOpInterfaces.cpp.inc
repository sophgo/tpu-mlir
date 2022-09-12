/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the function argument types based exclusively on
/// the type (to allow for this method may be called on function
/// declarations).
::llvm::ArrayRef<::mlir::Type> mlir::FunctionOpInterface::getArgumentTypes() {
      return getImpl()->getArgumentTypes(getImpl(), getOperation());
  }
/// Returns the function result types based exclusively on
/// the type (to allow for this method may be called on function
/// declarations).
::llvm::ArrayRef<::mlir::Type> mlir::FunctionOpInterface::getResultTypes() {
      return getImpl()->getResultTypes(getImpl(), getOperation());
  }
/// Returns a clone of the function type with the given argument and
/// result types.
/// 
/// Note: The default implementation assumes the function type has
///       an appropriate clone method:
///         `Type clone(ArrayRef<Type> inputs, ArrayRef<Type> results)`
::mlir::Type mlir::FunctionOpInterface::cloneTypeWith(::mlir::TypeRange inputs, ::mlir::TypeRange results) {
      return getImpl()->cloneTypeWith(getImpl(), getOperation(), inputs, results);
  }
/// Verify the contents of the body of this function.
/// 
/// Note: The default implementation merely checks that if the entry block
/// exists, it has the same number and type of arguments as the function type.
::mlir::LogicalResult mlir::FunctionOpInterface::verifyBody() {
      return getImpl()->verifyBody(getImpl(), getOperation());
  }
/// Verify the type attribute of the function for derived op-specific
/// invariants.
::mlir::LogicalResult mlir::FunctionOpInterface::verifyType() {
      return getImpl()->verifyType(getImpl(), getOperation());
  }
