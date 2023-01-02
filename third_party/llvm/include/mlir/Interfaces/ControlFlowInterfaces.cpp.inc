/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns the operands that correspond to the arguments of the successor
/// at the given index. It consists of a number of operands that are
/// internally produced by the operation, followed by a range of operands
/// that are forwarded. An example operation making use of produced
/// operands would be:
/// 
/// ```mlir
/// invoke %function(%0)
///     label ^success ^error(%1 : i32)
/// 
/// ^error(%e: !error, %arg0: i32):
///     ...
/// ```
/// 
/// The operand that would map to the `^error`s `%e` operand is produced
/// by the `invoke` operation, while `%1` is a forwarded operand that maps
/// to `%arg0` in the successor.
/// 
/// Produced operands always map to the first few block arguments of the
/// successor, followed by the forwarded operands. Mapping them in any
/// other order is not supported by the interface.
/// 
/// By having the forwarded operands last allows users of the interface
/// to append more forwarded operands to the branch operation without
/// interfering with other successor operands.
::mlir::SuccessorOperands mlir::BranchOpInterface::getSuccessorOperands(unsigned index) {
      return getImpl()->getSuccessorOperands(getImpl(), getOperation(), index);
  }
/// Returns the `BlockArgument` corresponding to operand `operandIndex` in
/// some successor, or None if `operandIndex` isn't a successor operand
/// index.
::std::optional<::mlir::BlockArgument> mlir::BranchOpInterface::getSuccessorBlockArgument(unsigned operandIndex) {
      return getImpl()->getSuccessorBlockArgument(getImpl(), getOperation(), operandIndex);
  }
/// Returns the successor that would be chosen with the given constant
/// operands. Returns nullptr if a single successor could not be chosen.
::mlir::Block *mlir::BranchOpInterface::getSuccessorForOperands(::llvm::ArrayRef<::mlir::Attribute> operands) {
      return getImpl()->getSuccessorForOperands(getImpl(), getOperation(), operands);
  }
/// This method is called to compare types along control-flow edges. By
/// default, the types are checked as equal.
bool mlir::BranchOpInterface::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
      return getImpl()->areTypesCompatible(getImpl(), getOperation(), lhs, rhs);
  }
/// Returns the operands of this operation used as the entry arguments when
/// entering the region at `index`, which was specified as a successor of
/// this operation by `getSuccessorRegions`, or the operands forwarded to
/// the operation's results when it branches back to itself. These operands
/// should correspond 1-1 with the successor inputs specified in
/// `getSuccessorRegions`.
::mlir::OperandRange mlir::RegionBranchOpInterface::getSuccessorEntryOperands(::std::optional<unsigned> index) {
      return getImpl()->getSuccessorEntryOperands(getImpl(), getOperation(), index);
  }
/// Returns the viable successors of a region at `index`, or the possible
/// successors when branching from the parent op if `index` is None. These
/// are the regions that may be selected during the flow of control. If
/// `index` is None, `operands` is a set of optional attributes that
/// either correspond to a constant value for each operand of this
/// operation, or null if that operand is not a constant. If `index` is
/// valid, `operands` corresponds to the entry values of the region at
/// `index`. Only a region, i.e. a valid `index`, may use the parent
/// operation as a successor. This method allows for describing which
/// regions may be executed when entering an operation, and which regions
/// are executed after having executed another region of the parent op. The
/// successor region must be non-empty.
void mlir::RegionBranchOpInterface::getSuccessorRegions(::std::optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> & regions) {
      return getImpl()->getSuccessorRegions(getImpl(), getOperation(), index, operands, regions);
  }
/// Populates `invocationBounds` with the minimum and maximum number of
/// times this operation will invoke the attached regions (assuming the
/// regions yield normally, i.e. do not abort or invoke an infinite loop).
/// The minimum number of invocations is at least 0. If the maximum number
/// of invocations cannot be statically determined, then it will not have a
/// value (i.e., it is set to `std::nullopt`).
/// 
/// `operands` is a set of optional attributes that either correspond to
/// constant values for each operand of this operation or null if that
/// operand is not a constant.
/// 
/// This method may be called speculatively on operations where the provided
/// operands are not necessarily the same as the operation's current
/// operands. This may occur in analyses that wish to determine "what would
/// be the region invocations if these were the operands?"
void mlir::RegionBranchOpInterface::getRegionInvocationBounds(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> & invocationBounds) {
      return getImpl()->getRegionInvocationBounds(getImpl(), getOperation(), operands, invocationBounds);
  }
/// This method is called to compare types along control-flow edges. By
/// default, the types are checked as equal.
bool mlir::RegionBranchOpInterface::areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs) {
      return getImpl()->areTypesCompatible(getImpl(), getOperation(), lhs, rhs);
  }
/// Returns a mutable range of operands that are semantically "returned" by
/// passing them to the region successor given by `index`.  If `index` is
/// None, this function returns the operands that are passed as a result to
/// the parent operation.
::mlir::MutableOperandRange mlir::RegionBranchTerminatorOpInterface::getMutableSuccessorOperands(::std::optional<unsigned> index) {
      return getImpl()->getMutableSuccessorOperands(getImpl(), getOperation(), index);
  }
/// Returns a range of operands that are semantically "returned" by passing
/// them to the region successor given by `index`.  If `index` is None, this
/// function returns the operands that are passed as a result to the parent
/// operation.
::mlir::OperandRange mlir::RegionBranchTerminatorOpInterface::getSuccessorOperands(::std::optional<unsigned> index) {
      return getImpl()->getSuccessorOperands(getImpl(), getOperation(), index);
  }
