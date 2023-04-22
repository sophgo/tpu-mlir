/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the data layout specification for this op, or null if it does not exist.
::mlir::DataLayoutSpecInterface mlir::DataLayoutOpInterface::getDataLayoutSpec() {
      return getImpl()->getDataLayoutSpec(getImpl(), getOperation());
  }
/// Returns the size of the given type computed using the relevant entries. The data layout object can be used for recursive queries.
unsigned mlir::DataLayoutOpInterface::getTypeSize(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return getImpl()->getTypeSize(type, dataLayout, params);
  }
/// Returns the size of the given type in bits computed using the relevant entries. The data layout object can be used for recursive queries.
unsigned mlir::DataLayoutOpInterface::getTypeSizeInBits(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return getImpl()->getTypeSizeInBits(type, dataLayout, params);
  }
/// Returns the alignment required by the ABI for the given type computed using the relevant entries. The data layout object can be used for recursive queries.
unsigned mlir::DataLayoutOpInterface::getTypeABIAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return getImpl()->getTypeABIAlignment(type, dataLayout, params);
  }
/// Returns the alignment preferred by the given type computed using the relevant entries. The data layoutobject can be used for recursive queries.
unsigned mlir::DataLayoutOpInterface::getTypePreferredAlignment(::mlir::Type type, const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) {
      return getImpl()->getTypePreferredAlignment(type, dataLayout, params);
  }
/// Returns the memory space used by the ABI computed using the relevant entries. The data layout object can be used for recursive queries.
::mlir::Attribute mlir::DataLayoutOpInterface::getAllocaMemorySpace(::mlir::DataLayoutEntryInterface entry) {
      return getImpl()->getAllocaMemorySpace(entry);
  }
/// Returns the natural stack alignment in bits computed using the relevant entries. The data layout object can be used for recursive queries.
unsigned mlir::DataLayoutOpInterface::getStackAlignment(::mlir::DataLayoutEntryInterface entry) {
      return getImpl()->getStackAlignment(entry);
  }
