/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Builds a deallocation operation using the provided builder and the
/// current allocation value (which refers to the current Op implementing
/// this interface). The allocation value is a result of the current
/// operation implementing this interface. If there is no compatible
/// deallocation operation, this method can return ::llvm::None.
::mlir::Optional<::mlir::Operation*> mlir::bufferization::AllocationOpInterface::buildDealloc(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
      return getImpl()->buildDealloc(builder, alloc);
  }
/// Builds a clone operation using the provided builder and the current
/// allocation value (which refers to the current Op implementing this
/// interface). The allocation value is a result of the current operation
/// implementing this interface. If there is no compatible clone operation,
/// this method can return ::llvm::None.
::mlir::Optional<::mlir::Value> mlir::bufferization::AllocationOpInterface::buildClone(::mlir::OpBuilder& builder, ::mlir::Value alloc) {
      return getImpl()->buildClone(builder, alloc);
  }
