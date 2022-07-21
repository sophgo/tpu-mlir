/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::DataLayoutSpecInterface mlir::DataLayoutOpInterface::getDataLayoutSpec() {
      return getImpl()->getDataLayoutSpec(getImpl(), getOperation());
  }
unsigned mlir::DataLayoutOpInterface::getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return getImpl()->getTypeSize(type, dataLayout, params);
  }
unsigned mlir::DataLayoutOpInterface::getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return getImpl()->getTypeSizeInBits(type, dataLayout, params);
  }
unsigned mlir::DataLayoutOpInterface::getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return getImpl()->getTypeABIAlignment(type, dataLayout, params);
  }
unsigned mlir::DataLayoutOpInterface::getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return getImpl()->getTypePreferredAlignment(type, dataLayout, params);
  }
