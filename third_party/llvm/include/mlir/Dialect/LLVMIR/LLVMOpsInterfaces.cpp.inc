/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Get fastmath flags
::mlir::LLVM::FastmathFlags mlir::LLVM::FastmathFlagsInterface::getFastmathFlags() {
      return getImpl()->getFastmathFlags(getImpl(), getOperation());
  }
