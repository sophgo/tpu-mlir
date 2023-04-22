/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the type of the function.
::mlir::Type mlir::FunctionOpInterface::getFunctionType() {
      return getImpl()->getFunctionType(getImpl(), getOperation());
  }
/// Set the type of the function. This method should perform an unsafe
/// modification to the function type; it should not update argument or
/// result attributes.
void mlir::FunctionOpInterface::setFunctionTypeAttr(::mlir::TypeAttr type) {
      return getImpl()->setFunctionTypeAttr(getImpl(), getOperation(), type);
  }
/// Get the array of argument attribute dictionaries. The method should return
/// an array attribute containing only dictionary attributes equal in number
/// to the number of function arguments. Alternatively, the method can return
/// null to indicate that the function has no argument attributes.
::mlir::ArrayAttr mlir::FunctionOpInterface::getArgAttrsAttr() {
      return getImpl()->getArgAttrsAttr(getImpl(), getOperation());
  }
/// Get the array of result attribute dictionaries. The method should return
/// an array attribute containing only dictionary attributes equal in number
/// to the number of function results. Alternatively, the method can return
/// null to indicate that the function has no result attributes.
::mlir::ArrayAttr mlir::FunctionOpInterface::getResAttrsAttr() {
      return getImpl()->getResAttrsAttr(getImpl(), getOperation());
  }
/// Set the array of argument attribute dictionaries.
void mlir::FunctionOpInterface::setArgAttrsAttr(::mlir::ArrayAttr attrs) {
      return getImpl()->setArgAttrsAttr(getImpl(), getOperation(), attrs);
  }
/// Set the array of result attribute dictionaries.
void mlir::FunctionOpInterface::setResAttrsAttr(::mlir::ArrayAttr attrs) {
      return getImpl()->setResAttrsAttr(getImpl(), getOperation(), attrs);
  }
/// Remove the array of argument attribute dictionaries. This is the same as
/// setting all argument attributes to an empty dictionary. The method should
/// return the removed attribute.
::mlir::Attribute mlir::FunctionOpInterface::removeArgAttrsAttr() {
      return getImpl()->removeArgAttrsAttr(getImpl(), getOperation());
  }
/// Remove the array of result attribute dictionaries. This is the same as
/// setting all result attributes to an empty dictionary. The method should
/// return the removed attribute.
::mlir::Attribute mlir::FunctionOpInterface::removeResAttrsAttr() {
      return getImpl()->removeResAttrsAttr(getImpl(), getOperation());
  }
/// Returns the function argument types based exclusively on
/// the type (to allow for this method may be called on function
/// declarations).
::llvm::ArrayRef<::mlir::Type> mlir::FunctionOpInterface::getArgumentTypes() {
      return getImpl()->getArgumentTypes(getImpl(), getOperation());
  }
/// Returns the function result types based exclusively on
/// the type (to allow for this method may be called on function
/// declarations).
::llvm::ArrayRef<::mlir::Type> mlir::FunctionOpInterface::getResultTypes() {
      return getImpl()->getResultTypes(getImpl(), getOperation());
  }
/// Returns a clone of the function type with the given argument and
/// result types.
/// 
/// Note: The default implementation assumes the function type has
///       an appropriate clone method:
///         `Type clone(ArrayRef<Type> inputs, ArrayRef<Type> results)`
::mlir::Type mlir::FunctionOpInterface::cloneTypeWith(::mlir::TypeRange inputs, ::mlir::TypeRange results) {
      return getImpl()->cloneTypeWith(getImpl(), getOperation(), inputs, results);
  }
/// Verify the contents of the body of this function.
/// 
/// Note: The default implementation merely checks that if the entry block
/// exists, it has the same number and type of arguments as the function type.
::mlir::LogicalResult mlir::FunctionOpInterface::verifyBody() {
      return getImpl()->verifyBody(getImpl(), getOperation());
  }
/// Verify the type attribute of the function for derived op-specific
/// invariants.
::mlir::LogicalResult mlir::FunctionOpInterface::verifyType() {
      return getImpl()->verifyType(getImpl(), getOperation());
  }
/// Returns the name of this symbol.
StringAttr mlir::FunctionOpInterface::getNameAttr() {
      return getImpl()->implSymbolOpInterface->getNameAttr(getImpl()->implSymbolOpInterface, getOperation());
  }
/// Sets the name of this symbol.
void mlir::FunctionOpInterface::setName(::mlir::StringAttr name) {
      return getImpl()->implSymbolOpInterface->setName(getImpl()->implSymbolOpInterface, getOperation(), name);
  }
/// Gets the visibility of this symbol.
mlir::SymbolTable::Visibility mlir::FunctionOpInterface::getVisibility() {
      return getImpl()->implSymbolOpInterface->getVisibility(getImpl()->implSymbolOpInterface, getOperation());
  }
/// Returns true if this symbol has nested visibility.
bool mlir::FunctionOpInterface::isNested() {
      return getImpl()->implSymbolOpInterface->isNested(getImpl()->implSymbolOpInterface, getOperation());
  }
/// Returns true if this symbol has private visibility.
bool mlir::FunctionOpInterface::isPrivate() {
      return getImpl()->implSymbolOpInterface->isPrivate(getImpl()->implSymbolOpInterface, getOperation());
  }
/// Returns true if this symbol has public visibility.
bool mlir::FunctionOpInterface::isPublic() {
      return getImpl()->implSymbolOpInterface->isPublic(getImpl()->implSymbolOpInterface, getOperation());
  }
/// Sets the visibility of this symbol.
void mlir::FunctionOpInterface::setVisibility(mlir::SymbolTable::Visibility vis) {
      return getImpl()->implSymbolOpInterface->setVisibility(getImpl()->implSymbolOpInterface, getOperation(), vis);
  }
/// Sets the visibility of this symbol to be nested.
void mlir::FunctionOpInterface::setNested() {
      return getImpl()->implSymbolOpInterface->setNested(getImpl()->implSymbolOpInterface, getOperation());
  }
/// Sets the visibility of this symbol to be private.
void mlir::FunctionOpInterface::setPrivate() {
      return getImpl()->implSymbolOpInterface->setPrivate(getImpl()->implSymbolOpInterface, getOperation());
  }
/// Sets the visibility of this symbol to be public.
void mlir::FunctionOpInterface::setPublic() {
      return getImpl()->implSymbolOpInterface->setPublic(getImpl()->implSymbolOpInterface, getOperation());
  }
/// Get all of the uses of the current symbol that are nested within the
/// given operation 'from'.
/// Note: See mlir::SymbolTable::getSymbolUses for more details.
::std::optional<::mlir::SymbolTable::UseRange> mlir::FunctionOpInterface::getSymbolUses(::mlir::Operation * from) {
      return getImpl()->implSymbolOpInterface->getSymbolUses(getImpl()->implSymbolOpInterface, getOperation(), from);
  }
/// Return if the current symbol is known to have no uses that are nested
/// within the given operation 'from'.
/// Note: See mlir::SymbolTable::symbolKnownUseEmpty for more details.
bool mlir::FunctionOpInterface::symbolKnownUseEmpty(::mlir::Operation * from) {
      return getImpl()->implSymbolOpInterface->symbolKnownUseEmpty(getImpl()->implSymbolOpInterface, getOperation(), from);
  }
/// Attempt to replace all uses of the current symbol with the provided
/// symbol 'newSymbol' that are nested within the given operation 'from'.
/// Note: See mlir::SymbolTable::replaceAllSymbolUses for more details.
::mlir::LogicalResult mlir::FunctionOpInterface::replaceAllSymbolUses(::mlir::StringAttr newSymbol, ::mlir::Operation * from) {
      return getImpl()->implSymbolOpInterface->replaceAllSymbolUses(getImpl()->implSymbolOpInterface, getOperation(), newSymbol, from);
  }
/// Returns true if this operation optionally defines a symbol based on the
/// presence of the symbol name.
bool mlir::FunctionOpInterface::isOptionalSymbol() {
      return getImpl()->implSymbolOpInterface->isOptionalSymbol(getImpl()->implSymbolOpInterface, getOperation());
  }
/// Returns true if this operation can be discarded if it has no remaining
/// symbol uses.
bool mlir::FunctionOpInterface::canDiscardOnUseEmpty() {
      return getImpl()->implSymbolOpInterface->canDiscardOnUseEmpty(getImpl()->implSymbolOpInterface, getOperation());
  }
/// Returns true if this operation is a declaration of a symbol (as opposed
/// to a definition).
bool mlir::FunctionOpInterface::isDeclaration() {
      return getImpl()->implSymbolOpInterface->isDeclaration(getImpl()->implSymbolOpInterface, getOperation());
  }
