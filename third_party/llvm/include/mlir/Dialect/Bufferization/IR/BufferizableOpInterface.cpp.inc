/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Return `true` if the given OpResult may bufferize to a new buffer
/// allocation. If it is statically unknown if the given OpResult
/// bufferizes to a buffer allocation, `true` should be returned.
bool mlir::bufferization::BufferizableOpInterface::bufferizesToAllocation(OpResult opResult) {
      return getImpl()->bufferizesToAllocation(getImpl(), getOperation(), opResult);
  }
/// Return `true` if the given OpOperand bufferizes to a memory read. This
/// method will never be called on OpOperands that do not have a tensor
/// type.
/// 
/// Note: It is always safe to consider an OpOperand as a memory read,
/// even if it does actually not read; however, this can introduce
/// unnecessary out-of-place bufferization decisions. One-Shot Analysis
/// considers OpOperands of unknown ops (that do not implement this
/// interface) as reading OpOperands.
bool mlir::bufferization::BufferizableOpInterface::bufferizesToMemoryRead(OpOperand & opOperand, const AnalysisState & state) {
      return getImpl()->bufferizesToMemoryRead(getImpl(), getOperation(), opOperand, state);
  }
/// Return `true` if the given OpOperand bufferizes to a memory write.
/// 
/// This method will never be called on OpOperands that do not have a
/// tensor type.
/// 
/// This method will never be called on OpOperands that do not have an
/// aliasing OpResult. Intuitively, it does not make sense for an
/// OpOperand to bufferize to a memory write without returning an aliasing
/// tensor, because the write would have no visible effect outside of the
/// op.
/// 
/// Note: It is always safe to consider an OpOperand as a memory write,
/// even if it does actually not write; however, this can introduce
/// unnecessary out-of-place bufferization decisions. One-Shot Analysis
/// considers OpOperands of unknown ops (that do not implement this
/// interface) as writing OpOperands.
bool mlir::bufferization::BufferizableOpInterface::bufferizesToMemoryWrite(OpOperand & opOperand, const AnalysisState & state) {
      return getImpl()->bufferizesToMemoryWrite(getImpl(), getOperation(), opOperand, state);
  }
/// Return `true` if the given OpResult is a memory write. This is the
/// case if in the following cases:
/// 
/// * The corresponding aliasing OpOperand bufferizes to a memory write.
/// * Or: There is no corresponding aliasing OpOperand.
/// 
/// If the OpResult has multiple aliasing OpOperands, this method
/// returns `true` if at least one of them bufferizes to a memory write.
bool mlir::bufferization::BufferizableOpInterface::isMemoryWrite(OpResult opResult, const AnalysisState & state) {
      return getImpl()->isMemoryWrite(getImpl(), getOperation(), opResult, state);
  }
/// Return `true` if the given OpOperand must bufferize in-place. Alias
/// sets and inplace attributes will be set up accordingly before making
/// any other bufferization decisions. This method will never be called on
/// OpOperands that do not have a tensor type.
bool mlir::bufferization::BufferizableOpInterface::mustBufferizeInPlace(OpOperand & opOperand, const AnalysisState & state) {
      return getImpl()->mustBufferizeInPlace(getImpl(), getOperation(), opOperand, state);
  }
/// Return the OpResult that aliases with a given OpOperand when
/// bufferized in-place. This method will never be called on OpOperands
/// that do not have a tensor type.
SmallVector<OpResult> mlir::bufferization::BufferizableOpInterface::getAliasingOpResult(OpOperand & opOperand, const AnalysisState & state) {
      return getImpl()->getAliasingOpResult(getImpl(), getOperation(), opOperand, state);
  }
/// Return the OpOperands that alias with a given OpResult when
/// bufferized in-place. This method will never be called on OpResults
/// that do not have a tensor type.
/// 
/// By default, this method is the inverse of `getAliasingOpResult`. Ops
/// with a region that yield values may want to override this method to
/// return the OpOperands that are yielded by the terminator.
/// 
/// Note: This method can return multiple OpOperands, indicating that the
/// given OpResult may at runtime alias with any of the OpOperands. This
/// is useful for branches and for ops such as `arith.select`.
SmallVector<OpOperand *> mlir::bufferization::BufferizableOpInterface::getAliasingOpOperand(OpResult opResult, const AnalysisState & state) {
      return getImpl()->getAliasingOpOperand(getImpl(), getOperation(), opResult, state);
  }
/// Return the buffer relation between the given OpResult and its aliasing
/// OpOperands when bufferized in-place. Most OpOperands have an
/// "equivalence" relation. This method will never be called on OpResults
/// that do not have a tensor type. It will also never be called on
/// OpResults that do not have at least one aliasing OpOperand.
/// 
/// TODO: Support other relations such as "OpOperand is included in
/// OpResult".
BufferRelation mlir::bufferization::BufferizableOpInterface::bufferRelation(OpResult opResult, const AnalysisState & state) {
      return getImpl()->bufferRelation(getImpl(), getOperation(), opResult, state);
  }
/// Resolve all inplacability conflicts by inserting explicit
/// `bufferization.alloc_tensor` ops. Examples of inplacability conflicts
/// are read-after-write conflicts or writes into non-writable buffers.
/// 
/// This method should rewrite the IR in such a way that for each tensor
/// OpOperand t, buffer(t) can be directly used when during bufferization.
/// The bufferization does no longer have to care about inplacability
/// conflicts.
/// 
/// This method can query analysis information from the given analysis
/// state.
LogicalResult mlir::bufferization::BufferizableOpInterface::resolveConflicts(RewriterBase & rewriter, const AnalysisState & state) {
      return getImpl()->resolveConflicts(getImpl(), getOperation(), rewriter, state);
  }
/// Bufferize this op, i.e., rewrite it into a memref-based equivalent.
/// Buffers of tensor SSA values can be retrieved via `getBuffer`.
/// Uses of tensor results of the existing tensor op can be replaced with
/// `replaceOpWithBufferizedValues` or `replaceOpWithNewBufferizedOp`.
/// These two functions automatically handle the tensor-to-memref type
/// conversion.
/// 
/// The implementation of this method must be consistent with the
/// remaining methods, in particular `getAliasingOpOperand`. I.e., a
/// tensor result `r` may only be replaced with:
/// a) A buffer that aliases one of buffers in getAliasingOpOperand(r).
/// b) Or: A newly allocated buffer.
/// 
/// This method will never be called on ops that do not have at least one
/// tensor operand/result.
/// 
/// The return value of this method indicates whether there was an error
/// while bufferizing this op (such as failing to create a new buffer
/// allocation op). The bufferization driver immediately stops bufferizing
/// the input IR and returns `failure` in that case. If this op is
/// expected to survive bufferization, `success` should be returned
/// (together with `allow-unknown-ops` enabled).
LogicalResult mlir::bufferization::BufferizableOpInterface::bufferize(RewriterBase & rewriter, const BufferizationOptions & options) {
      return getImpl()->bufferize(getImpl(), getOperation(), rewriter, options);
  }
/// Return `true` if the given Value can be written to in-place. Value is
/// either an OpResult of this operation or a BlockArgument of a block of
/// this operation.
/// 
/// Most OpResult buffers can be written to, but some ops such as
/// ConstantOp may bufferize to non-writable (read-only) memory locations.
/// Therefore, by default, this method returns `true` for OpResults. This
/// method will never be called on OpResults that do not have a tensor
/// type.
/// 
/// Whether a BlockArgument can be written to or not depends on the
/// operation. This method conservatively returns `false`. This method
/// will never be called on BlockArguments that do not have a tensor type.
bool mlir::bufferization::BufferizableOpInterface::isWritable(Value value, const AnalysisState & state) {
      return getImpl()->isWritable(getImpl(), getOperation(), value, state);
  }
/// Return `true` if the `uRead` and `uWrite` do not constitute a RaW
/// conflict. If they are conflicting or if it is unknown whether they are
/// conflicting, return `false`. This method will never be called with
/// OpOperands that do not have a tensor type. At least one of the two
/// given OpOperands belongs to this operation.
/// 
/// This method can be implemented to specify custom RaW analysis rules.
/// If this method returns `true` the given OpOperands are not considered
/// to be conflicting and do not force out-of-place bufferization. (There
/// may still be other conflicts that do.)
bool mlir::bufferization::BufferizableOpInterface::isNotConflicting(OpOperand * uRead, OpOperand * uWrite, const AnalysisState & state) {
      return getImpl()->isNotConflicting(getImpl(), getOperation(), uRead, uWrite, state);
  }
/// Return `failure` if this op does not pass the analysis. This method
/// is run during One-Shot Bufferize (after all post-analysis steps). If
/// the op does not pass the analysis, bufferization is aborted.
/// 
/// This method can be used to check expected invariants and limitations
/// of the current bufferization implementation.
LogicalResult mlir::bufferization::BufferizableOpInterface::verifyAnalysis(const AnalysisState & state) {
      return getImpl()->verifyAnalysis(getImpl(), getOperation(), state);
  }
/// Return the bufferized type of the given tensor value (without
/// bufferizing the IR). The value is either a BlockArgument of a block
/// that belongs to this op or an OpResult of the given op.
/// 
/// This method is useful when the bufferized type of value must be
/// predicted before modifying any IR.
FailureOr<BaseMemRefType> mlir::bufferization::BufferizableOpInterface::getBufferType(Value value, const BufferizationOptions & options, const DenseMap<Value, BaseMemRefType> fixedTypes) {
      return getImpl()->getBufferType(getImpl(), getOperation(), value, options, fixedTypes);
  }
/// Return `true` if the given region of this op is repetitive. By default
/// this information is queried from the `RegionBranchOpInterface`. Ops
/// that do not implement this inferface can override this method to
/// declare regions as repetitive.
/// 
/// The RaW conflict detection of One-Shot Analysis is more strict inside
/// repetitive regions: Op dominance cannot always be used to rule out
/// certain potential conflicts (e.g., a conflicting write happening after
/// a read), because there may not be a meaningful ordering of certain ops
/// that are executed multiple times. This is described in more detail in
/// documentation of One-Shot Analysis.
bool mlir::bufferization::BufferizableOpInterface::isRepetitiveRegion(unsigned index) {
      return getImpl()->isRepetitiveRegion(getImpl(), getOperation(), index);
  }
