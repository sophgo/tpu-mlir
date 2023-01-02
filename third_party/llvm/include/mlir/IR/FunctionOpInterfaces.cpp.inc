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
