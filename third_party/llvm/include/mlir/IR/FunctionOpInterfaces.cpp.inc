/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::llvm::ArrayRef<::mlir::Type> mlir::FunctionOpInterface::getArgumentTypes() {
      return getImpl()->getArgumentTypes(getImpl(), getOperation());
  }
::llvm::ArrayRef<::mlir::Type> mlir::FunctionOpInterface::getResultTypes() {
      return getImpl()->getResultTypes(getImpl(), getOperation());
  }
::mlir::Type mlir::FunctionOpInterface::cloneTypeWith(::mlir::TypeRange inputs, ::mlir::TypeRange results) {
      return getImpl()->cloneTypeWith(getImpl(), getOperation(), inputs, results);
  }
::mlir::LogicalResult mlir::FunctionOpInterface::verifyBody() {
      return getImpl()->verifyBody(getImpl(), getOperation());
  }
::mlir::LogicalResult mlir::FunctionOpInterface::verifyType() {
      return getImpl()->verifyType(getImpl(), getOperation());
  }
