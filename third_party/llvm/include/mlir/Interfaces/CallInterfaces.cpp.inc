/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::CallInterfaceCallable mlir::CallOpInterface::getCallableForCallee() {
      return getImpl()->getCallableForCallee(getImpl(), getOperation());
  }
::mlir::Operation::operand_range mlir::CallOpInterface::getArgOperands() {
      return getImpl()->getArgOperands(getImpl(), getOperation());
  }
::mlir::Region *mlir::CallableOpInterface::getCallableRegion() {
      return getImpl()->getCallableRegion(getImpl(), getOperation());
  }
::mlir::ArrayRef<::mlir::Type> mlir::CallableOpInterface::getCallableResults() {
      return getImpl()->getCallableResults(getImpl(), getOperation());
  }
