/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

bool mlir::DerivedAttributeOpInterface::isDerivedAttribute(::mlir::StringRef name) {
      return getImpl()->isDerivedAttribute(name);
  }
::mlir::DictionaryAttr mlir::DerivedAttributeOpInterface::materializeDerivedAttributes() {
      return getImpl()->materializeDerivedAttributes(getImpl(), getOperation());
  }
