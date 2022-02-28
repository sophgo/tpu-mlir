/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::FailureOr<::mlir::detail::ElementsAttrIndexer> mlir::ElementsAttr::getValuesImpl(::mlir::TypeID elementID) const {
      return getImpl()->getValuesImpl(getImpl(), *this, elementID);
  }
bool mlir::ElementsAttr::isSplat() const {
      return getImpl()->isSplat(getImpl(), *this);
  }
::mlir::AffineMap mlir::MemRefLayoutAttrInterface::getAffineMap() const {
      return getImpl()->getAffineMap(getImpl(), *this);
  }
bool mlir::MemRefLayoutAttrInterface::isIdentity() const {
      return getImpl()->isIdentity(getImpl(), *this);
  }
::mlir::LogicalResult mlir::MemRefLayoutAttrInterface::verifyLayout(::llvm::ArrayRef<int64_t> shape, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const {
      return getImpl()->verifyLayout(getImpl(), *this, shape, emitError);
  }
