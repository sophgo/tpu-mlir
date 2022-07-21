/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

unsigned mlir::OffsetSizeAndStrideOpInterface::getOffsetSizeAndStrideStartOperandIndex() {
      return getImpl()->getOffsetSizeAndStrideStartOperandIndex();
  }
std::array<unsigned, 3> mlir::OffsetSizeAndStrideOpInterface::getArrayAttrMaxRanks() {
      return getImpl()->getArrayAttrMaxRanks(getImpl(), getOperation());
  }
::mlir::OperandRange mlir::OffsetSizeAndStrideOpInterface::offsets() {
      return getImpl()->offsets(getImpl(), getOperation());
  }
::mlir::OperandRange mlir::OffsetSizeAndStrideOpInterface::sizes() {
      return getImpl()->sizes(getImpl(), getOperation());
  }
::mlir::OperandRange mlir::OffsetSizeAndStrideOpInterface::strides() {
      return getImpl()->strides(getImpl(), getOperation());
  }
::mlir::ArrayAttr mlir::OffsetSizeAndStrideOpInterface::static_offsets() {
      return getImpl()->static_offsets(getImpl(), getOperation());
  }
::mlir::ArrayAttr mlir::OffsetSizeAndStrideOpInterface::static_sizes() {
      return getImpl()->static_sizes(getImpl(), getOperation());
  }
::mlir::ArrayAttr mlir::OffsetSizeAndStrideOpInterface::static_strides() {
      return getImpl()->static_strides(getImpl(), getOperation());
  }
::llvm::SmallVector<::mlir::OpFoldResult, 4> mlir::OffsetSizeAndStrideOpInterface::getMixedOffsets() {
      return getImpl()->getMixedOffsets(getImpl(), getOperation());
  }
::llvm::SmallVector<::mlir::OpFoldResult, 4> mlir::OffsetSizeAndStrideOpInterface::getMixedSizes() {
      return getImpl()->getMixedSizes(getImpl(), getOperation());
  }
::llvm::SmallVector<::mlir::OpFoldResult, 4> mlir::OffsetSizeAndStrideOpInterface::getMixedStrides() {
      return getImpl()->getMixedStrides(getImpl(), getOperation());
  }
bool mlir::OffsetSizeAndStrideOpInterface::isDynamicOffset(unsigned idx) {
      return getImpl()->isDynamicOffset(getImpl(), getOperation(), idx);
  }
bool mlir::OffsetSizeAndStrideOpInterface::isDynamicSize(unsigned idx) {
      return getImpl()->isDynamicSize(getImpl(), getOperation(), idx);
  }
bool mlir::OffsetSizeAndStrideOpInterface::isDynamicStride(unsigned idx) {
      return getImpl()->isDynamicStride(getImpl(), getOperation(), idx);
  }
int64_t mlir::OffsetSizeAndStrideOpInterface::getStaticOffset(unsigned idx) {
      return getImpl()->getStaticOffset(getImpl(), getOperation(), idx);
  }
int64_t mlir::OffsetSizeAndStrideOpInterface::getStaticSize(unsigned idx) {
      return getImpl()->getStaticSize(getImpl(), getOperation(), idx);
  }
int64_t mlir::OffsetSizeAndStrideOpInterface::getStaticStride(unsigned idx) {
      return getImpl()->getStaticStride(getImpl(), getOperation(), idx);
  }
unsigned mlir::OffsetSizeAndStrideOpInterface::getIndexOfDynamicOffset(unsigned idx) {
      return getImpl()->getIndexOfDynamicOffset(getImpl(), getOperation(), idx);
  }
unsigned mlir::OffsetSizeAndStrideOpInterface::getIndexOfDynamicSize(unsigned idx) {
      return getImpl()->getIndexOfDynamicSize(getImpl(), getOperation(), idx);
  }
unsigned mlir::OffsetSizeAndStrideOpInterface::getIndexOfDynamicStride(unsigned idx) {
      return getImpl()->getIndexOfDynamicStride(getImpl(), getOperation(), idx);
  }
unsigned mlir::OffsetSizeAndStrideOpInterface::getNumDynamicEntriesUpToIdx(::mlir::ArrayAttr attr, ::llvm::function_ref<bool(int64_t)> isDynamic, unsigned idx) {
      return getImpl()->getNumDynamicEntriesUpToIdx(getImpl(), getOperation(), attr, isDynamic, idx);
  }
::mlir::Value mlir::OffsetSizeAndStrideOpInterface::getDynamicOffset(unsigned idx) {
      return getImpl()->getDynamicOffset(getImpl(), getOperation(), idx);
  }
::mlir::Value mlir::OffsetSizeAndStrideOpInterface::getDynamicSize(unsigned idx) {
      return getImpl()->getDynamicSize(getImpl(), getOperation(), idx);
  }
::mlir::Value mlir::OffsetSizeAndStrideOpInterface::getDynamicStride(unsigned idx) {
      return getImpl()->getDynamicStride(getImpl(), getOperation(), idx);
  }
bool mlir::OffsetSizeAndStrideOpInterface::isSameAs(::mlir::OffsetSizeAndStrideOpInterface other, ::llvm::function_ref<bool(::mlir::OpFoldResult, ::mlir::OpFoldResult)> cmp) {
      return getImpl()->isSameAs(getImpl(), getOperation(), other, cmp);
  }
bool mlir::OffsetSizeAndStrideOpInterface::hasUnitStride() {
      return getImpl()->hasUnitStride(getImpl(), getOperation());
  }
bool mlir::OffsetSizeAndStrideOpInterface::hasZeroOffset() {
      return getImpl()->hasZeroOffset(getImpl(), getOperation());
  }
::mlir::Value mlir::ViewLikeOpInterface::getViewSource() {
      return getImpl()->getViewSource(getImpl(), getOperation());
  }
