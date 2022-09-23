/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Infer the bounds on the results of this op given the bounds on its arguments.
/// For each result value or block argument (that isn't a branch argument,
/// since the dataflow analysis handles those case), the method should call
/// `setValueRange` with that `Value` as an argument. When `setValueRange`
/// is not called for some value, it will recieve a default value of the mimimum
/// and maximum values for its type (the unbounded range).
/// 
/// When called on an op that also implements the RegionBranchOpInterface
/// or BranchOpInterface, this method should not attempt to infer the values
/// of the branch results, as this will be handled by the analyses that use
/// this interface.
/// 
/// This function will only be called when at least one result of the op is a
/// scalar integer value or the op has a region.
/// 
/// `argRanges` contains one `IntRangeAttrs` for each argument to the op in ODS
///  order. Non-integer arguments will have the an unbounded range of width-0
///  APInts in their `argRanges` element.
void mlir::InferIntRangeInterface::inferResultRanges(::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges, ::mlir::SetIntRangeFn setResultRanges) {
      return getImpl()->inferResultRanges(getImpl(), getOperation(), argRanges, setResultRanges);
  }
