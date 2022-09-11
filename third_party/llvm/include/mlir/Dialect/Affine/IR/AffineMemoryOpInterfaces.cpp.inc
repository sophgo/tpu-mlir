/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the AffineMapAttr associated with 'memref'.
NamedAttribute AffineMapAccessInterface::getAffineMapAttrForMemRef(Value memref) {
      return getImpl()->getAffineMapAttrForMemRef(getImpl(), getOperation(), memref);
  }
/// Returns the memref operand to read from.
Value AffineReadOpInterface::getMemRef() {
      return getImpl()->getMemRef(getImpl(), getOperation());
  }
/// Returns the type of the memref operand.
MemRefType AffineReadOpInterface::getMemRefType() {
      return getImpl()->getMemRefType(getImpl(), getOperation());
  }
/// Returns affine map operands.
Operation::operand_range AffineReadOpInterface::getMapOperands() {
      return getImpl()->getMapOperands(getImpl(), getOperation());
  }
/// Returns the affine map used to index the memref for this operation.
AffineMap AffineReadOpInterface::getAffineMap() {
      return getImpl()->getAffineMap(getImpl(), getOperation());
  }
/// Returns the value read by this operation.
Value AffineReadOpInterface::getValue() {
      return getImpl()->getValue(getImpl(), getOperation());
  }
/// Returns the memref operand to write to.
Value AffineWriteOpInterface::getMemRef() {
      return getImpl()->getMemRef(getImpl(), getOperation());
  }
/// Returns the type of the memref operand.
MemRefType AffineWriteOpInterface::getMemRefType() {
      return getImpl()->getMemRefType(getImpl(), getOperation());
  }
/// Returns affine map operands.
Operation::operand_range AffineWriteOpInterface::getMapOperands() {
      return getImpl()->getMapOperands(getImpl(), getOperation());
  }
/// Returns the affine map used to index the memref for this operation.
AffineMap AffineWriteOpInterface::getAffineMap() {
      return getImpl()->getAffineMap(getImpl(), getOperation());
  }
/// Returns the value to store.
Value AffineWriteOpInterface::getValueToStore() {
      return getImpl()->getValueToStore(getImpl(), getOperation());
  }
