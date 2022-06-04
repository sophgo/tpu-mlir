/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::SuccessorOperands mlir::BranchOpInterface::getSuccessorOperands(unsigned index) {
      return getImpl()->getSuccessorOperands(getImpl(), getOperation(), index);
  }
::mlir::Optional<::mlir::BlockArgument> mlir::BranchOpInterface::getSuccessorBlockArgument(unsigned operandIndex) {
      return getImpl()->getSuccessorBlockArgument(getImpl(), getOperation(), operandIndex);
  }
::mlir::Block *mlir::BranchOpInterface::getSuccessorForOperands(::mlir::ArrayRef<::mlir::Attribute> operands) {
      return getImpl()->getSuccessorForOperands(getImpl(), getOperation(), operands);
  }
bool mlir::BranchOpInterface::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
      return getImpl()->areTypesCompatible(getImpl(), getOperation(), lhs, rhs);
  }
::mlir::OperandRange mlir::RegionBranchOpInterface::getSuccessorEntryOperands(unsigned index) {
      return getImpl()->getSuccessorEntryOperands(getImpl(), getOperation(), index);
  }
void mlir::RegionBranchOpInterface::getSuccessorRegions(::mlir::Optional<unsigned> index, ::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
      return getImpl()->getSuccessorRegions(getImpl(), getOperation(), index, operands, regions);
  }
void mlir::RegionBranchOpInterface::getRegionInvocationBounds(::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds) {
      return getImpl()->getRegionInvocationBounds(getImpl(), getOperation(), operands, invocationBounds);
  }
bool mlir::RegionBranchOpInterface::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
      return getImpl()->areTypesCompatible(getImpl(), getOperation(), lhs, rhs);
  }
::mlir::MutableOperandRange mlir::RegionBranchTerminatorOpInterface::getMutableSuccessorOperands(::mlir::Optional<unsigned> index) {
      return getImpl()->getMutableSuccessorOperands(getImpl(), getOperation(), index);
  }
::mlir::OperandRange mlir::RegionBranchTerminatorOpInterface::getSuccessorOperands(::mlir::Optional<unsigned> index) {
      return getImpl()->getSuccessorOperands(getImpl(), getOperation(), index);
  }
