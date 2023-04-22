/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Return start and end indices of the init operands range.
std::pair<int64_t, int64_t> mlir::DestinationStyleOpInterface::getDpsInitsPositionRange() {
      return getImpl()->getDpsInitsPositionRange(getImpl(), getOperation());
  }
/// Return the number of inits.
int64_t mlir::DestinationStyleOpInterface::getNumDpsInits() {
      return getImpl()->getNumDpsInits(getImpl(), getOperation());
  }
/// Return the init operands.
::mlir::OpOperandVector mlir::DestinationStyleOpInterface::getDpsInitOperands() {
      return getImpl()->getDpsInitOperands(getImpl(), getOperation());
  }
/// Return the `i`-th init operand.
::mlir::OpOperand *mlir::DestinationStyleOpInterface::getDpsInitOperand(int64_t i) {
      return getImpl()->getDpsInitOperand(getImpl(), getOperation(), i);
  }
/// Set the `i`-th init operand.
void mlir::DestinationStyleOpInterface::setDpsInitOperand(int64_t i, ::mlir::Value value) {
      return getImpl()->setDpsInitOperand(getImpl(), getOperation(), i, value);
  }
/// Return the number of inputs.
int64_t mlir::DestinationStyleOpInterface::getNumDpsInputs() {
      return getImpl()->getNumDpsInputs(getImpl(), getOperation());
  }
/// Return the input operands.
::mlir::OpOperandVector mlir::DestinationStyleOpInterface::getDpsInputOperands() {
      return getImpl()->getDpsInputOperands(getImpl(), getOperation());
  }
/// Return the `i`-th input operand.
::mlir::OpOperand *mlir::DestinationStyleOpInterface::getDpsInputOperand(int64_t i) {
      return getImpl()->getDpsInputOperand(getImpl(), getOperation(), i);
  }
/// Return true if `opOperand` is an input.
bool mlir::DestinationStyleOpInterface::isDpsInput(::mlir::OpOperand * opOperand) {
      return getImpl()->isDpsInput(getImpl(), getOperation(), opOperand);
  }
/// Return true if `opOperand` is an init.
bool mlir::DestinationStyleOpInterface::isDpsInit(::mlir::OpOperand * opOperand) {
      return getImpl()->isDpsInit(getImpl(), getOperation(), opOperand);
  }
/// Return true if the `opOperand` is a scalar value.
bool mlir::DestinationStyleOpInterface::isScalar(::mlir::OpOperand * opOperand) {
      return getImpl()->isScalar(getImpl(), getOperation(), opOperand);
  }
/// Return the OpResult that is tied to the given OpOperand.
::mlir::OpResult mlir::DestinationStyleOpInterface::getTiedOpResult(::mlir::OpOperand * opOperand) {
      return getImpl()->getTiedOpResult(getImpl(), getOperation(), opOperand);
  }
/// Return the OpOperand that is tied to the given OpResult.
::mlir::OpOperand *mlir::DestinationStyleOpInterface::getTiedOpOperand(::mlir::OpResult opResult) {
      return getImpl()->getTiedOpOperand(getImpl(), getOperation(), opResult);
  }
/// Return whether the op has only ranked MemRef input/inits.
bool mlir::DestinationStyleOpInterface::hasBufferSemantics() {
      return getImpl()->hasBufferSemantics(getImpl(), getOperation());
  }
/// Return whether the op has only ranked tensor inputs/inits.
bool mlir::DestinationStyleOpInterface::hasTensorSemantics() {
      return getImpl()->hasTensorSemantics(getImpl(), getOperation());
  }
