/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

StringAttr mlir::SymbolOpInterface::getNameAttr() {
      return getImpl()->getNameAttr(getImpl(), getOperation());
  }
void mlir::SymbolOpInterface::setName(StringAttr name) {
      return getImpl()->setName(getImpl(), getOperation(), name);
  }
mlir::SymbolTable::Visibility mlir::SymbolOpInterface::getVisibility() {
      return getImpl()->getVisibility(getImpl(), getOperation());
  }
bool mlir::SymbolOpInterface::isNested() {
      return getImpl()->isNested(getImpl(), getOperation());
  }
bool mlir::SymbolOpInterface::isPrivate() {
      return getImpl()->isPrivate(getImpl(), getOperation());
  }
bool mlir::SymbolOpInterface::isPublic() {
      return getImpl()->isPublic(getImpl(), getOperation());
  }
void mlir::SymbolOpInterface::setVisibility(mlir::SymbolTable::Visibility vis) {
      return getImpl()->setVisibility(getImpl(), getOperation(), vis);
  }
void mlir::SymbolOpInterface::setNested() {
      return getImpl()->setNested(getImpl(), getOperation());
  }
void mlir::SymbolOpInterface::setPrivate() {
      return getImpl()->setPrivate(getImpl(), getOperation());
  }
void mlir::SymbolOpInterface::setPublic() {
      return getImpl()->setPublic(getImpl(), getOperation());
  }
Optional<::mlir::SymbolTable::UseRange> mlir::SymbolOpInterface::getSymbolUses(Operation * from) {
      return getImpl()->getSymbolUses(getImpl(), getOperation(), from);
  }
bool mlir::SymbolOpInterface::symbolKnownUseEmpty(Operation * from) {
      return getImpl()->symbolKnownUseEmpty(getImpl(), getOperation(), from);
  }
LogicalResult mlir::SymbolOpInterface::replaceAllSymbolUses(StringAttr newSymbol, Operation * from) {
      return getImpl()->replaceAllSymbolUses(getImpl(), getOperation(), newSymbol, from);
  }
bool mlir::SymbolOpInterface::isOptionalSymbol() {
      return getImpl()->isOptionalSymbol(getImpl(), getOperation());
  }
bool mlir::SymbolOpInterface::canDiscardOnUseEmpty() {
      return getImpl()->canDiscardOnUseEmpty(getImpl(), getOperation());
  }
bool mlir::SymbolOpInterface::isDeclaration() {
      return getImpl()->isDeclaration(getImpl(), getOperation());
  }
::mlir::LogicalResult mlir::SymbolUserOpInterface::verifySymbolUses(::mlir::SymbolTableCollection & symbolTable) {
      return getImpl()->verifySymbolUses(getImpl(), getOperation(), symbolTable);
  }
