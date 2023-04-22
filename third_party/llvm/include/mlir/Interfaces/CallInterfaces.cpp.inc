/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the callee of this call-like operation. A `callee` is either a
/// reference to a symbol, via SymbolRefAttr, or a reference to a defined
/// SSA value. If the reference is an SSA value, the SSA value corresponds
/// to a region of a lambda-like operation.
::mlir::CallInterfaceCallable mlir::CallOpInterface::getCallableForCallee() {
      return getImpl()->getCallableForCallee(getImpl(), getOperation());
  }
/// Returns the operands within this call that are used as arguments to the
/// callee.
::mlir::Operation::operand_range mlir::CallOpInterface::getArgOperands() {
      return getImpl()->getArgOperands(getImpl(), getOperation());
  }
/// Returns the region on the current operation that is callable. This may
/// return null in the case of an external callable object, e.g. an external
/// function.
::mlir::Region *mlir::CallableOpInterface::getCallableRegion() {
      return getImpl()->getCallableRegion(getImpl(), getOperation());
  }
/// Returns the results types that the callable region produces when
/// executed.
::llvm::ArrayRef<::mlir::Type> mlir::CallableOpInterface::getCallableResults() {
      return getImpl()->getCallableResults(getImpl(), getOperation());
  }
/// Returns the argument attributes for all callable region arguments or
/// null if there are none.
::mlir::ArrayAttr mlir::CallableOpInterface::getCallableArgAttrs() {
      return getImpl()->getCallableArgAttrs(getImpl(), getOperation());
  }
/// Returns the result attributes for all callable region results or null
/// if there are none.
::mlir::ArrayAttr mlir::CallableOpInterface::getCallableResAttrs() {
      return getImpl()->getCallableResAttrs(getImpl(), getOperation());
  }
