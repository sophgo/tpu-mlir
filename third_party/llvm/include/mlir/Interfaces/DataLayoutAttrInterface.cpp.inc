/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the key of the this layout entry.
::mlir::DataLayoutEntryKey mlir::DataLayoutEntryInterface::getKey() const {
      return getImpl()->getKey(getImpl(), *this);
  }
/// Returns the value of this layout entry.
::mlir::Attribute mlir::DataLayoutEntryInterface::getValue() const {
      return getImpl()->getValue(getImpl(), *this);
  }
/// Checks that the entry is well-formed, reports errors at the provided location.
::mlir::LogicalResult mlir::DataLayoutEntryInterface::verifyEntry(::mlir::Location loc) const {
      return getImpl()->verifyEntry(getImpl(), *this, loc);
  }
/// Combines the current layout with the given list of layouts, provided from the outermost (oldest) to the innermost (newest). Returns null on failure.
::mlir::DataLayoutSpecInterface mlir::DataLayoutSpecInterface::combineWith(::llvm::ArrayRef<::mlir::DataLayoutSpecInterface> specs) const {
      return getImpl()->combineWith(getImpl(), *this, specs);
  }
/// Returns the list of layout entries.
::mlir::DataLayoutEntryListRef mlir::DataLayoutSpecInterface::getEntries() const {
      return getImpl()->getEntries(getImpl(), *this);
  }
/// Returns the alloca memory space identifier.
::mlir::StringAttr mlir::DataLayoutSpecInterface::getAllocaMemorySpaceIdentifier(::mlir::MLIRContext * context) const {
      return getImpl()->getAllocaMemorySpaceIdentifier(getImpl(), *this, context);
  }
/// Returns the stack alignment identifier.
::mlir::StringAttr mlir::DataLayoutSpecInterface::getStackAlignmentIdentifier(::mlir::MLIRContext * context) const {
      return getImpl()->getStackAlignmentIdentifier(getImpl(), *this, context);
  }
/// Returns a copy of the entries related to a specific type class regardles of type parameters.
::mlir::DataLayoutEntryList mlir::DataLayoutSpecInterface::getSpecForType(::mlir::TypeID type) const {
      return getImpl()->getSpecForType(getImpl(), *this, type);
  }
/// Returns the entry related to the given identifier, if present.
::mlir::DataLayoutEntryInterface mlir::DataLayoutSpecInterface::getSpecForIdentifier(::mlir::StringAttr identifier) const {
      return getImpl()->getSpecForIdentifier(getImpl(), *this, identifier);
  }
/// Verifies the validity of the specification and reports any errors at the given location.
::mlir::LogicalResult mlir::DataLayoutSpecInterface::verifySpec(::mlir::Location loc) const {
      return getImpl()->verifySpec(getImpl(), *this, loc);
  }
