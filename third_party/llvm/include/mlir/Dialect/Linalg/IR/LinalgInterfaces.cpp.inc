/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Return the input shape operands.
ValueRange mlir::linalg::DestinationStyleOpInterface::inputs() {
      return getImpl()->inputs(getImpl(), getOperation());
  }
/// Return the number of inputs.
int64_t mlir::linalg::DestinationStyleOpInterface::getNumInputs() {
      return getImpl()->getNumInputs(getImpl(), getOperation());
  }
/// Return the output shape operands.
ValueRange mlir::linalg::DestinationStyleOpInterface::outputs() {
      return getImpl()->outputs(getImpl(), getOperation());
  }
/// Return the number of outputs.
int64_t mlir::linalg::DestinationStyleOpInterface::getNumOutputs() {
      return getImpl()->getNumOutputs(getImpl(), getOperation());
  }
/// Return the number of inputs and outputs.
int64_t mlir::linalg::DestinationStyleOpInterface::getNumInputsAndOutputs() {
      return getImpl()->getNumInputsAndOutputs(getImpl(), getOperation());
  }
/// Return the input operands.
OpOperandVector mlir::linalg::DestinationStyleOpInterface::getInputOperands() {
      return getImpl()->getInputOperands(getImpl(), getOperation());
  }
/// Return the `i`-th input operand.
OpOperand*mlir::linalg::DestinationStyleOpInterface::getInputOperand(int64_t i) {
      return getImpl()->getInputOperand(getImpl(), getOperation(), i);
  }
/// Return the subset of input operands that are of buffer type.
OpOperandVector mlir::linalg::DestinationStyleOpInterface::getInputBufferOperands() {
      return getImpl()->getInputBufferOperands(getImpl(), getOperation());
  }
/// Return the subset of input operands that are of tensor type.
OpOperandVector mlir::linalg::DestinationStyleOpInterface::getInputTensorOperands() {
      return getImpl()->getInputTensorOperands(getImpl(), getOperation());
  }
/// Return the output operands.
OpOperandVector mlir::linalg::DestinationStyleOpInterface::getOutputOperands() {
      return getImpl()->getOutputOperands(getImpl(), getOperation());
  }
/// Return the `i`-th output operand.
OpOperand*mlir::linalg::DestinationStyleOpInterface::getOutputOperand(int64_t i) {
      return getImpl()->getOutputOperand(getImpl(), getOperation(), i);
  }
/// Set the `i`-th output operand.
void mlir::linalg::DestinationStyleOpInterface::setOutputOperand(int64_t i, Value value) {
      return getImpl()->setOutputOperand(getImpl(), getOperation(), i, value);
  }
/// Return the subset of output operands that are of buffer type.
OpOperandVector mlir::linalg::DestinationStyleOpInterface::getOutputBufferOperands() {
      return getImpl()->getOutputBufferOperands(getImpl(), getOperation());
  }
/// Return the subset of output operands that are of tensor type.
OpOperandVector mlir::linalg::DestinationStyleOpInterface::getOutputTensorOperands() {
      return getImpl()->getOutputTensorOperands(getImpl(), getOperation());
  }
/// Return the types of the subset of output operands that are of buffer type.
SmallVector<MemRefType> mlir::linalg::DestinationStyleOpInterface::getOutputBufferTypes() {
      return getImpl()->getOutputBufferTypes(getImpl(), getOperation());
  }
/// Return the types of the subset of output operands that are of tensor type.
SmallVector<RankedTensorType> mlir::linalg::DestinationStyleOpInterface::getOutputTensorTypes() {
      return getImpl()->getOutputTensorTypes(getImpl(), getOperation());
  }
/// Return the range over input and output operands.
OpOperandVector mlir::linalg::DestinationStyleOpInterface::getInputAndOutputOperands() {
      return getImpl()->getInputAndOutputOperands(getImpl(), getOperation());
  }
/// Return true if `opOperand` is an input tensor.
bool mlir::linalg::DestinationStyleOpInterface::isInputTensor(OpOperand * opOperand) {
      return getImpl()->isInputTensor(getImpl(), getOperation(), opOperand);
  }
/// Return true if `opOperand` is an output tensor.
bool mlir::linalg::DestinationStyleOpInterface::isOutputTensor(OpOperand * opOperand) {
      return getImpl()->isOutputTensor(getImpl(), getOperation(), opOperand);
  }
/// Return true if the `opOperand` is a scalar value.
bool mlir::linalg::DestinationStyleOpInterface::isScalar(OpOperand* opOperand) {
      return getImpl()->isScalar(getImpl(), getOperation(), opOperand);
  }
/// Return the result tied to `opOperand`.
OpResult mlir::linalg::DestinationStyleOpInterface::getTiedOpResult(OpOperand* opOperand) {
      return getImpl()->getTiedOpResult(getImpl(), getOperation(), opOperand);
  }
/// Return whether the op has only MemRef input and outputs.
bool mlir::linalg::DestinationStyleOpInterface::hasBufferSemantics() {
      return getImpl()->hasBufferSemantics(getImpl(), getOperation());
  }
/// Return whether the op has only RankedTensor input and outputs.
bool mlir::linalg::DestinationStyleOpInterface::hasTensorSemantics() {
      return getImpl()->hasTensorSemantics(getImpl(), getOperation());
  }
/// Clone the current operation with the given location and operands. This
/// is used to abstract away the optional underlying region creation. This
/// does not change the balance between input, output_buffer and
/// init_tensors operands.
Operation *mlir::linalg::DestinationStyleOpInterface::clone(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
      return getImpl()->clone(getImpl(), getOperation(), b, loc, resultTypes, operands);
  }
/// Clone the current operation with the given location, operands
/// and BlockAndValueMapping but leave the regions empty. This is
/// used to abstract away the optional underlying region creation.
/// This does not change the balance between input, output_buffer
/// and init_tensors operands.
Operation *mlir::linalg::DestinationStyleOpInterface::cloneWithoutRegions(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
      return getImpl()->cloneWithoutRegions(getImpl(), getOperation(), b, loc, resultTypes, operands);
  }
/// Returns the left-hand side operand.
Value mlir::linalg::ContractionOpInterface::lhs() {
      return getImpl()->lhs(getImpl(), getOperation());
  }
/// Returns the right-hand side operand.
Value mlir::linalg::ContractionOpInterface::rhs() {
      return getImpl()->rhs(getImpl(), getOperation());
  }
/// Returns whether the given op has indexing maps that correspond to a
/// row-major matmul operation.
bool mlir::linalg::ContractionOpInterface::isRowMajorMatmul() {
      return getImpl()->isRowMajorMatmul(getImpl(), getOperation());
  }
/// Returns whether the given op has indexing maps that correspond to a
/// column-major matmul operation.
bool mlir::linalg::ContractionOpInterface::isColumnMajorMatmul() {
      return getImpl()->isColumnMajorMatmul(getImpl(), getOperation());
  }
/// Returns whether the given op has indexing maps that correspond to a
/// row-major batch matmul operation.
bool mlir::linalg::ContractionOpInterface::isRowMajorBatchMatmul() {
      return getImpl()->isRowMajorBatchMatmul(getImpl(), getOperation());
  }
/// Return the image operand.
Value mlir::linalg::ConvolutionOpInterface::image() {
      return getImpl()->image(getImpl(), getOperation());
  }
/// Return the filter operand.
Value mlir::linalg::ConvolutionOpInterface::filter() {
      return getImpl()->filter(getImpl(), getOperation());
  }
/// Return the fill value.
Value mlir::linalg::FillOpInterface::value() {
      return getImpl()->value(getImpl(), getOperation());
  }
/// Return the output operand.
Value mlir::linalg::FillOpInterface::output() {
      return getImpl()->output(getImpl(), getOperation());
  }
/// Return the result.
Value mlir::linalg::FillOpInterface::result() {
      return getImpl()->result(getImpl(), getOperation());
  }
/// Return the number of parallel loops.
unsigned mlir::linalg::LinalgOp::getNumParallelLoops() {
      return getImpl()->getNumParallelLoops(getImpl(), getOperation());
  }
/// Return the dims that are parallel loops.
void mlir::linalg::LinalgOp::getParallelDims(SmallVectorImpl<unsigned> & res) {
      return getImpl()->getParallelDims(getImpl(), getOperation(), res);
  }
/// Return the number of reduction loops.
unsigned mlir::linalg::LinalgOp::getNumReductionLoops() {
      return getImpl()->getNumReductionLoops(getImpl(), getOperation());
  }
/// Return the dims that are reduction loops.
void mlir::linalg::LinalgOp::getReductionDims(SmallVectorImpl<unsigned> & res) {
      return getImpl()->getReductionDims(getImpl(), getOperation(), res);
  }
/// Return the number of window loops.
unsigned mlir::linalg::LinalgOp::getNumWindowLoops() {
      return getImpl()->getNumWindowLoops(getImpl(), getOperation());
  }
/// Return the dims that are window loops.
void mlir::linalg::LinalgOp::getWindowDims(SmallVectorImpl<unsigned> & res) {
      return getImpl()->getWindowDims(getImpl(), getOperation(), res);
  }
/// Return the total number of loops within the current operation.
unsigned mlir::linalg::LinalgOp::getNumLoops() {
      return getImpl()->getNumLoops(getImpl(), getOperation());
  }
/// Returns true if the current operation has only one loop and it's a
/// reduction loop.
bool mlir::linalg::LinalgOp::hasSingleReductionLoop() {
      return getImpl()->hasSingleReductionLoop(getImpl(), getOperation());
  }
/// Return true if the payload uses the value loaded from `opOperand`. This
/// is useful to avoid loading from "write-only" memory that may be
/// uninitialized, as well as properly cloning "read-write" operands.
bool mlir::linalg::LinalgOp::payloadUsesValueFromOperand(OpOperand * opOperand) {
      return getImpl()->payloadUsesValueFromOperand(getImpl(), getOperation(), opOperand);
  }
/// Return true if `opOperand` is an init tensor. This is true when it is
/// an output tensor operand whose value is used in the payload region.
bool mlir::linalg::LinalgOp::isInitTensor(OpOperand * opOperand) {
      return getImpl()->isInitTensor(getImpl(), getOperation(), opOperand);
  }
/// Return the `opOperand` rank or zero for scalars.
int64_t mlir::linalg::LinalgOp::getRank(OpOperand* opOperand) {
      return getImpl()->getRank(getImpl(), getOperation(), opOperand);
  }
/// Return the output block arguments of the region.
Block::BlockArgListType mlir::linalg::LinalgOp::getRegionOutputArgs() {
      return getImpl()->getRegionOutputArgs(getImpl(), getOperation());
  }
/// Return the `opOperand` shape or an empty vector for scalars.
ArrayRef<int64_t> mlir::linalg::LinalgOp::getShape(OpOperand* opOperand) {
      return getImpl()->getShape(getImpl(), getOperation(), opOperand);
  }
/// Return the block argument for an `opOperand`.
BlockArgument mlir::linalg::LinalgOp::getTiedBlockArgument(OpOperand * opOperand) {
      return getImpl()->getTiedBlockArgument(getImpl(), getOperation(), opOperand);
  }
/// Return the operand for a `blockArgument`.
OpOperand *mlir::linalg::LinalgOp::getTiedOpOperand(BlockArgument blockArgument) {
      return getImpl()->getTiedOpOperand(getImpl(), getOperation(), blockArgument);
  }
/// Return the input or output indexing map for `opOperand`.
AffineMap mlir::linalg::LinalgOp::getTiedIndexingMap(OpOperand* opOperand) {
      return getImpl()->getTiedIndexingMap(getImpl(), getOperation(), opOperand);
  }
/// Return the indexing map for a `result`.
AffineMap mlir::linalg::LinalgOp::getTiedIndexingMapForResult(OpResult result) {
      return getImpl()->getTiedIndexingMapForResult(getImpl(), getOperation(), result);
  }
/// Return the value yielded by the region corresponding to an output
/// `opOperand`.
OpOperand *mlir::linalg::LinalgOp::getTiedYieldValue(OpOperand* opOperand) {
      return getImpl()->getTiedYieldValue(getImpl(), getOperation(), opOperand);
  }
/// Return the single block constituting the body of the operation by
/// calling the getBody method on the concrete operation.
Block*mlir::linalg::LinalgOp::getBlock() {
      return getImpl()->getBlock(getImpl(), getOperation());
  }
/// Return the iterator types attribute within the current operation.
ArrayAttr mlir::linalg::LinalgOp::iterator_types() {
      return getImpl()->iterator_types(getImpl(), getOperation());
  }
/// Return true if the indexing map is depending on the current op instance.
/// This means that the indexing map is dynamically synthesized by using the
/// op instance's concrete attributes, instead of being static for all
/// instances of the same op kind.
bool mlir::linalg::LinalgOp::hasDynamicIndexingMaps() {
      return getImpl()->hasDynamicIndexingMaps(getImpl(), getOperation());
  }
/// Verify all attributes used by indexing maps are valid.
LogicalResult mlir::linalg::LinalgOp::verifyIndexingMapRequiredAttributes() {
      return getImpl()->verifyIndexingMapRequiredAttributes(getImpl(), getOperation());
  }
/// Return the indexing maps attribute within the current operation.
ArrayAttr mlir::linalg::LinalgOp::getIndexingMaps() {
      return getImpl()->getIndexingMaps(getImpl(), getOperation());
  }
/// Return the indexing maps within the current operation.
SmallVector<AffineMap> mlir::linalg::LinalgOp::getIndexingMapsArray() {
      return getImpl()->getIndexingMapsArray(getImpl(), getOperation());
  }
/// Return true if any of the operands has a dynamic shape.
bool mlir::linalg::LinalgOp::hasDynamicShape() {
      return getImpl()->hasDynamicShape(getImpl(), getOperation());
  }
/// Return the name registered for this op when lowering to an external
/// library call.
std::string mlir::linalg::LinalgOp::getLibraryCallName() {
      return getImpl()->getLibraryCallName(getImpl(), getOperation());
  }
/// Return whether the op accesses the iteration indices.
bool mlir::linalg::LinalgOp::hasIndexSemantics() {
      return getImpl()->hasIndexSemantics(getImpl(), getOperation());
  }
/// Hook to provide a custom AffineMap used to compute all the operand
/// subshapes given loop bounds. This is used to answer the question: "given
/// an iteration space over the codomain, what are the subshapes of the
/// operands involved in the computation".
/// The default behavior is to just concatenate all the indexing maps.
/// A custom AffineMap allows providing a map that can be used to
/// compute subshapes even in cases where the concatenation of indexing maps
/// (i.e. the data traversal order) is not a simple permutation of the loop
/// traversal order. It is then possible to define ops with skewed data
/// traversal order for which we can still easily compute hyperrectangular
/// loop bounds and subviews.
AffineMap mlir::linalg::LinalgOp::getLoopsToShapesMap() {
      return getImpl()->getLoopsToShapesMap(getImpl(), getOperation());
  }
/// Hook to provide a custom AffineMap used to construct the
/// hyperrectangular loop iteration space given all the operand subshapes.
/// This is used to answer the question:
/// "Given a list of operand ranges, what is the subportion of the iteration
/// space involved in the computation".
/// This is the inverse problem of `getLoopsToShapesMap`.
/// Return the empty AffineMap when such an AffineMap cannot be constructed.
/// The default behavior is based on a very simple inference procedure that
/// only works with permutation affine maps.
/// A more advanced Tensor-Comprehension like inference is possible but has
/// proven to be ambiguous in unfavorable case.
/// A safer and more robust alternative is to allow each op to define
/// its own AffineMap.
AffineMap mlir::linalg::LinalgOp::getShapesToLoopsMap() {
      return getImpl()->getShapesToLoopsMap(getImpl(), getOperation());
  }
/// Checks if the given operands can be dropped, and the remaining
/// operands can still compute the bounds of the op.
bool mlir::linalg::LinalgOp::canOpOperandsBeDropped(ArrayRef<OpOperand *> droppedOperands) {
      return getImpl()->canOpOperandsBeDropped(getImpl(), getOperation(), droppedOperands);
  }
/// Like `getShape`, but only returns statically-known information, without
/// generating any new IR. For each shape dimension, returns >=0 if that
/// dimension is statically known, or ShapeType::kDynamicSize otherwise.
SmallVector<int64_t> mlir::linalg::LinalgOp::getStaticShape() {
      return getImpl()->getStaticShape(getImpl(), getOperation());
  }
/// Returns the statically-known loop ranges. Composes
/// `getShapesToLoopsMap()` with the result of `getStaticShape`.
/// Returns ShapeType::kDynamicSize for non-statically-known loop ranges.
/// This is expected to be called by a valid Linalg op
SmallVector<int64_t, 4> mlir::linalg::LinalgOp::getStaticLoopRanges() {
      return getImpl()->getStaticLoopRanges(getImpl(), getOperation());
  }
/// Returns the region builder for constructing the body for linalg.generic.
/// Returns a null function if this named op does not define a region
/// builder.
std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> mlir::linalg::LinalgOp::getRegionBuilder() {
      return getImpl()->getRegionBuilder();
  }
/// Return true if all the indexing maps are projected permutations.
/// Otherwise return false.
bool mlir::linalg::LinalgOp::hasOnlyProjectedPermutations() {
      return getImpl()->hasOnlyProjectedPermutations(getImpl(), getOperation());
  }
