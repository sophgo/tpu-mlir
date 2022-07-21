/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::Optional<::mlir::Operation*> mlir::bufferization::AllocationOpInterface::buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
      return getImpl()->buildDealloc(builder, alloc);
  }
::mlir::Optional<::mlir::Value> mlir::bufferization::AllocationOpInterface::buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
      return getImpl()->buildClone(builder, alloc);
  }
