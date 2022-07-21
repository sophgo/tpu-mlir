/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

Value mlir::linalg::ContractionOpInterface::lhs() {
      return getImpl()->lhs(getImpl(), getOperation());
  }
Value mlir::linalg::ContractionOpInterface::rhs() {
      return getImpl()->rhs(getImpl(), getOperation());
  }
bool mlir::linalg::ContractionOpInterface::isRowMajorMatmul() {
      return getImpl()->isRowMajorMatmul(getImpl(), getOperation());
  }
bool mlir::linalg::ContractionOpInterface::isColumnMajorMatmul() {
      return getImpl()->isColumnMajorMatmul(getImpl(), getOperation());
  }
bool mlir::linalg::ContractionOpInterface::isRowMajorBatchMatmul() {
      return getImpl()->isRowMajorBatchMatmul(getImpl(), getOperation());
  }
Value mlir::linalg::ConvolutionOpInterface::image() {
      return getImpl()->image(getImpl(), getOperation());
  }
Value mlir::linalg::ConvolutionOpInterface::filter() {
      return getImpl()->filter(getImpl(), getOperation());
  }
Value mlir::linalg::FillOpInterface::value() {
      return getImpl()->value(getImpl(), getOperation());
  }
Value mlir::linalg::FillOpInterface::output() {
      return getImpl()->output(getImpl(), getOperation());
  }
Value mlir::linalg::FillOpInterface::result() {
      return getImpl()->result(getImpl(), getOperation());
  }
unsigned mlir::linalg::LinalgOp::getNumParallelLoops() {
      return getImpl()->getNumParallelLoops(getImpl(), getOperation());
  }
void mlir::linalg::LinalgOp::getParallelDims(SmallVectorImpl<unsigned> & res) {
      return getImpl()->getParallelDims(getImpl(), getOperation(), res);
  }
unsigned mlir::linalg::LinalgOp::getNumReductionLoops() {
      return getImpl()->getNumReductionLoops(getImpl(), getOperation());
  }
void mlir::linalg::LinalgOp::getReductionDims(SmallVectorImpl<unsigned> & res) {
      return getImpl()->getReductionDims(getImpl(), getOperation(), res);
  }
unsigned mlir::linalg::LinalgOp::getNumWindowLoops() {
      return getImpl()->getNumWindowLoops(getImpl(), getOperation());
  }
void mlir::linalg::LinalgOp::getWindowDims(SmallVectorImpl<unsigned> & res) {
      return getImpl()->getWindowDims(getImpl(), getOperation(), res);
  }
unsigned mlir::linalg::LinalgOp::getNumLoops() {
      return getImpl()->getNumLoops(getImpl(), getOperation());
  }
bool mlir::linalg::LinalgOp::hasSingleReductionLoop() {
      return getImpl()->hasSingleReductionLoop(getImpl(), getOperation());
  }
ValueRange mlir::linalg::LinalgOp::inputs() {
      return getImpl()->inputs(getImpl(), getOperation());
  }
int64_t mlir::linalg::LinalgOp::getNumInputs() {
      return getImpl()->getNumInputs(getImpl(), getOperation());
  }
ValueRange mlir::linalg::LinalgOp::outputs() {
      return getImpl()->outputs(getImpl(), getOperation());
  }
int64_t mlir::linalg::LinalgOp::getNumOutputs() {
      return getImpl()->getNumOutputs(getImpl(), getOperation());
  }
int64_t mlir::linalg::LinalgOp::getNumInputsAndOutputs() {
      return getImpl()->getNumInputsAndOutputs(getImpl(), getOperation());
  }
OpOperandVector mlir::linalg::LinalgOp::getInputOperands() {
      return getImpl()->getInputOperands(getImpl(), getOperation());
  }
OpOperand*mlir::linalg::LinalgOp::getInputOperand(int64_t i) {
      return getImpl()->getInputOperand(getImpl(), getOperation(), i);
  }
OpOperandVector mlir::linalg::LinalgOp::getInputBufferOperands() {
      return getImpl()->getInputBufferOperands(getImpl(), getOperation());
  }
OpOperandVector mlir::linalg::LinalgOp::getInputTensorOperands() {
      return getImpl()->getInputTensorOperands(getImpl(), getOperation());
  }
OpOperandVector mlir::linalg::LinalgOp::getOutputOperands() {
      return getImpl()->getOutputOperands(getImpl(), getOperation());
  }
OpOperand*mlir::linalg::LinalgOp::getOutputOperand(int64_t i) {
      return getImpl()->getOutputOperand(getImpl(), getOperation(), i);
  }
void mlir::linalg::LinalgOp::setOutputOperand(int64_t i, Value value) {
      return getImpl()->setOutputOperand(getImpl(), getOperation(), i, value);
  }
OpOperandVector mlir::linalg::LinalgOp::getOutputBufferOperands() {
      return getImpl()->getOutputBufferOperands(getImpl(), getOperation());
  }
OpOperandVector mlir::linalg::LinalgOp::getOutputTensorOperands() {
      return getImpl()->getOutputTensorOperands(getImpl(), getOperation());
  }
SmallVector<MemRefType> mlir::linalg::LinalgOp::getOutputBufferTypes() {
      return getImpl()->getOutputBufferTypes(getImpl(), getOperation());
  }
SmallVector<RankedTensorType> mlir::linalg::LinalgOp::getOutputTensorTypes() {
      return getImpl()->getOutputTensorTypes(getImpl(), getOperation());
  }
OpOperandVector mlir::linalg::LinalgOp::getInputAndOutputOperands() {
      return getImpl()->getInputAndOutputOperands(getImpl(), getOperation());
  }
bool mlir::linalg::LinalgOp::payloadUsesValueFromOperand(OpOperand * opOperand) {
      return getImpl()->payloadUsesValueFromOperand(getImpl(), getOperation(), opOperand);
  }
bool mlir::linalg::LinalgOp::isInputTensor(OpOperand * opOperand) {
      return getImpl()->isInputTensor(getImpl(), getOperation(), opOperand);
  }
bool mlir::linalg::LinalgOp::isOutputTensor(OpOperand * opOperand) {
      return getImpl()->isOutputTensor(getImpl(), getOperation(), opOperand);
  }
bool mlir::linalg::LinalgOp::isInitTensor(OpOperand * opOperand) {
      return getImpl()->isInitTensor(getImpl(), getOperation(), opOperand);
  }
int64_t mlir::linalg::LinalgOp::getRank(OpOperand* opOperand) {
      return getImpl()->getRank(getImpl(), getOperation(), opOperand);
  }
Block::BlockArgListType mlir::linalg::LinalgOp::getRegionOutputArgs() {
      return getImpl()->getRegionOutputArgs(getImpl(), getOperation());
  }
ArrayRef<int64_t> mlir::linalg::LinalgOp::getShape(OpOperand* opOperand) {
      return getImpl()->getShape(getImpl(), getOperation(), opOperand);
  }
bool mlir::linalg::LinalgOp::isScalar(OpOperand* opOperand) {
      return getImpl()->isScalar(getImpl(), getOperation(), opOperand);
  }
BlockArgument mlir::linalg::LinalgOp::getTiedBlockArgument(OpOperand * opOperand) {
      return getImpl()->getTiedBlockArgument(getImpl(), getOperation(), opOperand);
  }
AffineMap mlir::linalg::LinalgOp::getTiedIndexingMap(OpOperand* opOperand) {
      return getImpl()->getTiedIndexingMap(getImpl(), getOperation(), opOperand);
  }
AffineMap mlir::linalg::LinalgOp::getTiedIndexingMapForResult(OpResult result) {
      return getImpl()->getTiedIndexingMapForResult(getImpl(), getOperation(), result);
  }
OpResult mlir::linalg::LinalgOp::getTiedOpResult(OpOperand* opOperand) {
      return getImpl()->getTiedOpResult(getImpl(), getOperation(), opOperand);
  }
OpOperand *mlir::linalg::LinalgOp::getTiedYieldValue(OpOperand* opOperand) {
      return getImpl()->getTiedYieldValue(getImpl(), getOperation(), opOperand);
  }
Block*mlir::linalg::LinalgOp::getBlock() {
      return getImpl()->getBlock(getImpl(), getOperation());
  }
ArrayAttr mlir::linalg::LinalgOp::iterator_types() {
      return getImpl()->iterator_types(getImpl(), getOperation());
  }
bool mlir::linalg::LinalgOp::hasDynamicIndexingMaps() {
      return getImpl()->hasDynamicIndexingMaps(getImpl(), getOperation());
  }
LogicalResult mlir::linalg::LinalgOp::verifyIndexingMapRequiredAttributes() {
      return getImpl()->verifyIndexingMapRequiredAttributes(getImpl(), getOperation());
  }
ArrayAttr mlir::linalg::LinalgOp::indexing_maps() {
      return getImpl()->indexing_maps(getImpl(), getOperation());
  }
SmallVector<AffineMap> mlir::linalg::LinalgOp::getIndexingMaps() {
      return getImpl()->getIndexingMaps(getImpl(), getOperation());
  }
bool mlir::linalg::LinalgOp::hasDynamicShape() {
      return getImpl()->hasDynamicShape(getImpl(), getOperation());
  }
bool mlir::linalg::LinalgOp::hasBufferSemantics() {
      return getImpl()->hasBufferSemantics(getImpl(), getOperation());
  }
bool mlir::linalg::LinalgOp::hasTensorSemantics() {
      return getImpl()->hasTensorSemantics(getImpl(), getOperation());
  }
std::string mlir::linalg::LinalgOp::getLibraryCallName() {
      return getImpl()->getLibraryCallName(getImpl(), getOperation());
  }
bool mlir::linalg::LinalgOp::hasIndexSemantics() {
      return getImpl()->hasIndexSemantics(getImpl(), getOperation());
  }
AffineMap mlir::linalg::LinalgOp::getLoopsToShapesMap() {
      return getImpl()->getLoopsToShapesMap(getImpl(), getOperation());
  }
AffineMap mlir::linalg::LinalgOp::getShapesToLoopsMap() {
      return getImpl()->getShapesToLoopsMap(getImpl(), getOperation());
  }
bool mlir::linalg::LinalgOp::canOpOperandsBeDropped(ArrayRef<OpOperand *> droppedOperands) {
      return getImpl()->canOpOperandsBeDropped(getImpl(), getOperation(), droppedOperands);
  }
std::pair<int64_t, int64_t> mlir::linalg::LinalgOp::getResultsPositionInLoopsToShapeMap() {
      return getImpl()->getResultsPositionInLoopsToShapeMap(getImpl(), getOperation());
  }
SmallVector<int64_t> mlir::linalg::LinalgOp::getStaticShape() {
      return getImpl()->getStaticShape(getImpl(), getOperation());
  }
SmallVector<int64_t, 4> mlir::linalg::LinalgOp::getStaticLoopRanges() {
      return getImpl()->getStaticLoopRanges(getImpl(), getOperation());
  }
Operation *mlir::linalg::LinalgOp::clone(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
      return getImpl()->clone(getImpl(), getOperation(), b, loc, resultTypes, operands);
  }
Operation *mlir::linalg::LinalgOp::cloneWithMapper(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands, BlockAndValueMapping & bvm) {
      return getImpl()->cloneWithMapper(getImpl(), getOperation(), b, loc, resultTypes, operands, bvm);
  }
Operation *mlir::linalg::LinalgOp::cloneWithoutRegions(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
      return getImpl()->cloneWithoutRegions(getImpl(), getOperation(), b, loc, resultTypes, operands);
  }
std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> mlir::linalg::LinalgOp::getRegionBuilder() {
      return getImpl()->getRegionBuilder();
  }
bool mlir::linalg::LinalgOp::hasOnlyProjectedPermutations() {
      return getImpl()->hasOnlyProjectedPermutations(getImpl(), getOperation());
  }
