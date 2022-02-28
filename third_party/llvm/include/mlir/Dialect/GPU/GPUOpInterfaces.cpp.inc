/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

OperandRange mlir::gpu::AsyncOpInterface::getAsyncDependencies() {
      return getImpl()->getAsyncDependencies(getImpl(), getOperation());
  }
void mlir::gpu::AsyncOpInterface::addAsyncDependency(Value token) {
      return getImpl()->addAsyncDependency(getImpl(), getOperation(), token);
  }
OpResult mlir::gpu::AsyncOpInterface::getAsyncToken() {
      return getImpl()->getAsyncToken(getImpl(), getOperation());
  }
