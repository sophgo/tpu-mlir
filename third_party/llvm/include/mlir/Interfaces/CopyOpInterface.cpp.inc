/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::Value mlir::CopyOpInterface::getSource() {
      return getImpl()->getSource(getImpl(), getOperation());
  }
::mlir::Value mlir::CopyOpInterface::getTarget() {
      return getImpl()->getTarget(getImpl(), getOperation());
  }
