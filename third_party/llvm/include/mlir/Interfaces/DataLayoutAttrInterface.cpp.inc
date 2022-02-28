/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::DataLayoutEntryKey mlir::DataLayoutEntryInterface::getKey() const {
      return getImpl()->getKey(getImpl(), *this);
  }
::mlir::Attribute mlir::DataLayoutEntryInterface::getValue() const {
      return getImpl()->getValue(getImpl(), *this);
  }
::mlir::LogicalResult mlir::DataLayoutEntryInterface::verifyEntry(::mlir::Location loc) const {
      return getImpl()->verifyEntry(getImpl(), *this, loc);
  }
::mlir::DataLayoutSpecInterface mlir::DataLayoutSpecInterface::combineWith(::llvm::ArrayRef<::mlir::DataLayoutSpecInterface> specs) const {
      return getImpl()->combineWith(getImpl(), *this, specs);
  }
::mlir::DataLayoutEntryListRef mlir::DataLayoutSpecInterface::getEntries() const {
      return getImpl()->getEntries(getImpl(), *this);
  }
::mlir::DataLayoutEntryList mlir::DataLayoutSpecInterface::getSpecForType(::mlir::TypeID type) const {
      return getImpl()->getSpecForType(getImpl(), *this, type);
  }
::mlir::DataLayoutEntryInterface mlir::DataLayoutSpecInterface::getSpecForIdentifier(::mlir::StringAttr identifier) const {
      return getImpl()->getSpecForIdentifier(getImpl(), *this, identifier);
  }
::mlir::LogicalResult mlir::DataLayoutSpecInterface::verifySpec(::mlir::Location loc) const {
      return getImpl()->verifySpec(getImpl(), *this, loc);
  }
