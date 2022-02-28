/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::ShapedType mlir::ShapedType::cloneWith(::llvm::Optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) const {
      return getImpl()->cloneWith(getImpl(), *this, shape, elementType);
  }
::mlir::Type mlir::ShapedType::getElementType() const {
      return getImpl()->getElementType(getImpl(), *this);
  }
bool mlir::ShapedType::hasRank() const {
      return getImpl()->hasRank(getImpl(), *this);
  }
::llvm::ArrayRef<int64_t> mlir::ShapedType::getShape() const {
      return getImpl()->getShape(getImpl(), *this);
  }
