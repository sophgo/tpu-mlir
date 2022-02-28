/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

NamedAttribute AffineMapAccessInterface::getAffineMapAttrForMemRef(Value memref) {
      return getImpl()->getAffineMapAttrForMemRef(getImpl(), getOperation(), memref);
  }
Value AffineReadOpInterface::getMemRef() {
      return getImpl()->getMemRef(getImpl(), getOperation());
  }
MemRefType AffineReadOpInterface::getMemRefType() {
      return getImpl()->getMemRefType(getImpl(), getOperation());
  }
Operation::operand_range AffineReadOpInterface::getMapOperands() {
      return getImpl()->getMapOperands(getImpl(), getOperation());
  }
AffineMap AffineReadOpInterface::getAffineMap() {
      return getImpl()->getAffineMap(getImpl(), getOperation());
  }
Value AffineReadOpInterface::getValue() {
      return getImpl()->getValue(getImpl(), getOperation());
  }
Value AffineWriteOpInterface::getMemRef() {
      return getImpl()->getMemRef(getImpl(), getOperation());
  }
MemRefType AffineWriteOpInterface::getMemRefType() {
      return getImpl()->getMemRefType(getImpl(), getOperation());
  }
Operation::operand_range AffineWriteOpInterface::getMapOperands() {
      return getImpl()->getMapOperands(getImpl(), getOperation());
  }
AffineMap AffineWriteOpInterface::getAffineMap() {
      return getImpl()->getAffineMap(getImpl(), getOperation());
  }
Value AffineWriteOpInterface::getValueToStore() {
      return getImpl()->getValueToStore(getImpl(), getOperation());
  }
