/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the size of this type in bytes.
unsigned mlir::DataLayoutTypeInterface::getTypeSize(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const {
      return getImpl()->getTypeSize(getImpl(), *this, dataLayout, params);
  }
/// Returns the size of this type in bits.
unsigned mlir::DataLayoutTypeInterface::getTypeSizeInBits(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const {
      return getImpl()->getTypeSizeInBits(getImpl(), *this, dataLayout, params);
  }
/// Returns the ABI-required alignment for this type, in bytes
unsigned mlir::DataLayoutTypeInterface::getABIAlignment(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const {
      return getImpl()->getABIAlignment(getImpl(), *this, dataLayout, params);
  }
/// Returns the preferred alignment for this type, in bytes.
unsigned mlir::DataLayoutTypeInterface::getPreferredAlignment(const ::mlir::DataLayout & dataLayout, ::mlir::DataLayoutEntryListRef params) const {
      return getImpl()->getPreferredAlignment(getImpl(), *this, dataLayout, params);
  }
/// Returns true if the two lists of entries are compatible, that is, that `newLayout` spec entries can be nested in an op with `oldLayout` spec entries.
bool mlir::DataLayoutTypeInterface::areCompatible(::mlir::DataLayoutEntryListRef oldLayout, ::mlir::DataLayoutEntryListRef newLayout) const {
      return getImpl()->areCompatible(getImpl(), *this, oldLayout, newLayout);
  }
/// Verifies that the given list of entries is valid for this type.
::mlir::LogicalResult mlir::DataLayoutTypeInterface::verifyEntries(::mlir::DataLayoutEntryListRef entries, ::mlir::Location loc) const {
      return getImpl()->verifyEntries(getImpl(), *this, entries, loc);
  }
