/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

void mlir::SubElementTypeInterface::walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
      return getImpl()->walkImmediateSubElements(getImpl(), *this, walkAttrsFn, walkTypesFn);
  }
::mlir::SubElementTypeInterface mlir::SubElementTypeInterface::replaceImmediateSubAttribute(::llvm::ArrayRef<std::pair<size_t, ::mlir::Attribute>> replacements) const {
      return getImpl()->replaceImmediateSubAttribute(getImpl(), *this, replacements);
  }
