/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Availability Interface Definitions                                         *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Capability>, 1> mlir::spirv::QueryCapabilityInterface::getCapabilities() {
  return getImpl()->getCapabilities(getImpl(), getOperation());
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::spirv::Extension>, 1> mlir::spirv::QueryExtensionInterface::getExtensions() {
  return getImpl()->getExtensions(getImpl(), getOperation());
}
llvm::Optional<::mlir::spirv::Version> mlir::spirv::QueryMinVersionInterface::getMinVersion() {
  return getImpl()->getMinVersion(getImpl(), getOperation());
}
llvm::Optional<::mlir::spirv::Version> mlir::spirv::QueryMaxVersionInterface::getMaxVersion() {
  return getImpl()->getMaxVersion(getImpl(), getOperation());
}
