/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

mlir::tosa::AbsOp,
mlir::tosa::AddOp,
mlir::tosa::ApplyScaleOp,
mlir::tosa::ArgMaxOp,
mlir::tosa::ArithmeticRightShiftOp,
mlir::tosa::AvgPool2dOp,
mlir::tosa::BitwiseAndOp,
mlir::tosa::BitwiseNotOp,
mlir::tosa::BitwiseOrOp,
mlir::tosa::BitwiseXorOp,
mlir::tosa::CastOp,
mlir::tosa::CeilOp,
mlir::tosa::ClampOp,
mlir::tosa::ClzOp,
mlir::tosa::ConcatOp,
mlir::tosa::ConstOp,
mlir::tosa::Conv2DOp,
mlir::tosa::Conv3DOp,
mlir::tosa::CustomOp,
mlir::tosa::DepthwiseConv2DOp,
mlir::tosa::DivOp,
mlir::tosa::EqualOp,
mlir::tosa::ExpOp,
mlir::tosa::FloorOp,
mlir::tosa::FullyConnectedOp,
mlir::tosa::GatherOp,
mlir::tosa::GreaterEqualOp,
mlir::tosa::GreaterOp,
mlir::tosa::IdentityOp,
mlir::tosa::IfOp,
mlir::tosa::LogOp,
mlir::tosa::LogicalAndOp,
mlir::tosa::LogicalLeftShiftOp,
mlir::tosa::LogicalNotOp,
mlir::tosa::LogicalOrOp,
mlir::tosa::LogicalRightShiftOp,
mlir::tosa::LogicalXorOp,
mlir::tosa::MatMulOp,
mlir::tosa::MaxPool2dOp,
mlir::tosa::MaximumOp,
mlir::tosa::MinimumOp,
mlir::tosa::MulOp,
mlir::tosa::NegateOp,
mlir::tosa::PadOp,
mlir::tosa::PowOp,
mlir::tosa::ReciprocalOp,
mlir::tosa::ReduceAllOp,
mlir::tosa::ReduceAnyOp,
mlir::tosa::ReduceMaxOp,
mlir::tosa::ReduceMinOp,
mlir::tosa::ReduceProdOp,
mlir::tosa::ReduceSumOp,
mlir::tosa::RescaleOp,
mlir::tosa::ReshapeOp,
mlir::tosa::ResizeOp,
mlir::tosa::ReverseOp,
mlir::tosa::RsqrtOp,
mlir::tosa::ScatterOp,
mlir::tosa::SelectOp,
mlir::tosa::SigmoidOp,
mlir::tosa::SliceOp,
mlir::tosa::SubOp,
mlir::tosa::TableOp,
mlir::tosa::TanhOp,
mlir::tosa::TileOp,
mlir::tosa::TransposeConv2DOp,
mlir::tosa::TransposeOp,
mlir::tosa::WhileOp,
mlir::tosa::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace tosa {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isSignlessInteger(1))) || ((type.isUnsignedInteger(8))) || (((type.isSignlessInteger(8))) || ((type.isSignlessInteger(16))) || ((type.isSignlessInteger(32))) || ((type.isSignlessInteger(48))) || ((type.isSignlessInteger(64))))) || ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64)))); }(type.cast<::mlir::ShapedType>().getElementType()))) || (((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64)))); }(type.cast<::mlir::ShapedType>().getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be signless-integer-like, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isSignlessInteger(8))) || ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(type.cast<::mlir::ShapedType>().getElementType()))) || (((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(type.cast<::mlir::ShapedType>().getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be signless-integer-8-bit-like, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::UnrankedTensorType>())) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && (((type.cast<::mlir::ShapedType>().getRank()
                         == 1)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 2)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 3)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 4))))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked.tensor of number values or 1D/2D/3D/4D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::UnrankedTensorType>())) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && (((type.cast<::mlir::ShapedType>().getRank()
                         == 0)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 1)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 2)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 3)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 4))))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked.tensor of number values or 0D/1D/2D/3D/4D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::UnrankedTensorType>())) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 4)))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked.tensor of number values or 4D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::UnrankedTensorType>())) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1)))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked.tensor of number values or 1D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::UnrankedTensorType>())) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 5)))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked.tensor of number values or 5D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 32-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 1-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps10(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::UnrankedTensorType>())) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 2)))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked.tensor of number values or 2D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps11(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::UnrankedTensorType>())) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 3)))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked.tensor of number values or 3D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps12(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 2)))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 2D tensor of 32-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps13(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().hasRank()))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps14(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 32-bit signless integer or 64-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps15(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 0)))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 0D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps16(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::UnrankedTensorType>())) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && (((type.cast<::mlir::ShapedType>().getRank()
                         == 1)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 2)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 3)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 4)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 5)) || ((type.cast<::mlir::ShapedType>().getRank()
                         == 6))))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 4))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 8))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 16))) || (((elementType.isa<mlir::quant::QuantizedType>())) && ((elementType.cast<mlir::quant::QuantizedType>().getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(type.cast<::mlir::ShapedType>().getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked.tensor of number values or 1D/2D/3D/4D/5D/6D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::BoolAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: bool attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((attr.cast<::mlir::ArrayAttr>().size() == 2)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((attr.cast<::mlir::ArrayAttr>().size() == 4)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit integer array attribute with exactly 4 elements";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<mlir::tosa::UnaryOpQuantizationAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Attribute for UnaryOp quantization information.";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::FloatAttr>())) && ((attr.cast<::mlir::FloatAttr>().getType().isF32())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit float attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::ElementsAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: constant vector/tensor attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<mlir::tosa::ConvOpQuantizationAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Attribute for Conv type op quantization information.";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((attr.cast<::mlir::ArrayAttr>().size() == 6)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit integer array attribute with exactly 6 elements";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((attr.cast<::mlir::ArrayAttr>().size() == 3)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<mlir::tosa::MatMulOpQuantizationAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Attribute for MatMulOp quantization information.";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps12(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps13(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<mlir::tosa::PadOpQuantizationAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Attribute for PadOp quantization information.";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps14(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit integer array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps15(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit integer array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps16(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.cast<StringAttr>().getValue() == "BILINEAR"  || attr.cast<StringAttr>().getValue() == "NEAREST_NEIGHBOR"))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Supported resize/upsampling strategies";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps17(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((attr.cast<::mlir::ArrayAttr>().size() <= 4)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit integer array attribute with at least 4 elements";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_TosaOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}
} // namespace tosa
} // namespace mlir
namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::AbsOp definitions
//===----------------------------------------------------------------------===//

AbsOpAdaptor::AbsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.abs", odsAttrs.getContext());
}

AbsOpAdaptor::AbsOpAdaptor(AbsOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> AbsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AbsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AbsOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange AbsOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr AbsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AbsOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AbsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AbsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AbsOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AbsOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AbsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AbsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AbsOp::getOutput() {
  return *getODSResults(0).begin();
}

void AbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void AbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AbsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AbsOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AbsOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void AbsOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::AbsOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::AddOp definitions
//===----------------------------------------------------------------------===//

AddOpAdaptor::AddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.add", odsAttrs.getContext());
}

AddOpAdaptor::AddOpAdaptor(AddOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> AddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange AddOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr AddOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AddOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AddOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> AddOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AddOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AddOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AddOp::getOutput() {
  return *getODSResults(0).begin();
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AddOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AddOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void AddOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::AddOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ApplyScaleOp definitions
//===----------------------------------------------------------------------===//

ApplyScaleOpAdaptor::ApplyScaleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.apply_scale", odsAttrs.getContext());
}

ApplyScaleOpAdaptor::ApplyScaleOpAdaptor(ApplyScaleOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ApplyScaleOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ApplyScaleOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ApplyScaleOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value ApplyScaleOpAdaptor::getMultiplier() {
  return *getODSOperands(1).begin();
}

::mlir::Value ApplyScaleOpAdaptor::getShift() {
  return *getODSOperands(2).begin();
}

::mlir::ValueRange ApplyScaleOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ApplyScaleOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr ApplyScaleOpAdaptor::getDoubleRoundAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ApplyScaleOp::getDoubleRoundAttrName(*odsOpName)).cast<::mlir::BoolAttr>();
  return attr;
}

bool ApplyScaleOpAdaptor::getDoubleRound() {
  auto attr = getDoubleRoundAttr();
  return attr.getValue();
}

::mlir::LogicalResult ApplyScaleOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_double_round;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.apply_scale' op ""requires attribute 'double_round'");
    if (namedAttrIt->getName() == ApplyScaleOp::getDoubleRoundAttrName(*odsOpName)) {
      tblgen_double_round = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_double_round && !((tblgen_double_round.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'tosa.apply_scale' op ""attribute 'double_round' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ApplyScaleOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ApplyScaleOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ApplyScaleOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value ApplyScaleOp::getMultiplier() {
  return *getODSOperands(1).begin();
}

::mlir::Value ApplyScaleOp::getShift() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ApplyScaleOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ApplyScaleOp::getMultiplierMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ApplyScaleOp::getShiftMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ApplyScaleOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ApplyScaleOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ApplyScaleOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::BoolAttr ApplyScaleOp::getDoubleRoundAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getDoubleRoundAttrName()).cast<::mlir::BoolAttr>();
}

bool ApplyScaleOp::getDoubleRound() {
  auto attr = getDoubleRoundAttr();
  return attr.getValue();
}

void ApplyScaleOp::setDoubleRoundAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getDoubleRoundAttrName(), attr);
}

void ApplyScaleOp::setDoubleRound(bool attrValue) {
  (*this)->setAttr(getDoubleRoundAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void ApplyScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value value, ::mlir::Value multiplier, ::mlir::Value shift, ::mlir::BoolAttr double_round) {
  odsState.addOperands(value);
  odsState.addOperands(multiplier);
  odsState.addOperands(shift);
  odsState.addAttribute(getDoubleRoundAttrName(odsState.name), double_round);
  odsState.addTypes(output);
}

void ApplyScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value multiplier, ::mlir::Value shift, ::mlir::BoolAttr double_round) {
  odsState.addOperands(value);
  odsState.addOperands(multiplier);
  odsState.addOperands(shift);
  odsState.addAttribute(getDoubleRoundAttrName(odsState.name), double_round);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value value, ::mlir::Value multiplier, ::mlir::Value shift, bool double_round) {
  odsState.addOperands(value);
  odsState.addOperands(multiplier);
  odsState.addOperands(shift);
  odsState.addAttribute(getDoubleRoundAttrName(odsState.name), odsBuilder.getBoolAttr(double_round));
  odsState.addTypes(output);
}

void ApplyScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value multiplier, ::mlir::Value shift, bool double_round) {
  odsState.addOperands(value);
  odsState.addOperands(multiplier);
  odsState.addOperands(shift);
  odsState.addAttribute(getDoubleRoundAttrName(odsState.name), odsBuilder.getBoolAttr(double_round));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyScaleOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ApplyScaleOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_double_round;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'double_round'");
    if (namedAttrIt->getName() == getDoubleRoundAttrName()) {
      tblgen_double_round = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps0(*this, tblgen_double_round, "double_round")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ApplyScaleOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ApplyScaleOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ApplyScaleOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ArgMaxOp definitions
//===----------------------------------------------------------------------===//

ArgMaxOpAdaptor::ArgMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.argmax", odsAttrs.getContext());
}

ArgMaxOpAdaptor::ArgMaxOpAdaptor(ArgMaxOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ArgMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ArgMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ArgMaxOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ArgMaxOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ArgMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ArgMaxOpAdaptor::getAxisAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ArgMaxOp::getAxisAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ArgMaxOpAdaptor::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult ArgMaxOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.argmax' op ""requires attribute 'axis'");
    if (namedAttrIt->getName() == ArgMaxOp::getAxisAttrName(*odsOpName)) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_axis && !(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tosa.argmax' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ArgMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ArgMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ArgMaxOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ArgMaxOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ArgMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ArgMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ArgMaxOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ArgMaxOp::getAxisAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ArgMaxOp::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

void ArgMaxOp::setAxisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAxisAttrName(), attr);
}

void ArgMaxOp::setAxis(uint64_t attrValue) {
  (*this)->setAttr(getAxisAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  odsState.addTypes(output);
}

void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ArgMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ArgMaxOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'axis'");
    if (namedAttrIt->getName() == getAxisAttrName()) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(*this, tblgen_axis, "axis")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ArgMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ArgMaxOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ArgMaxOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ArithmeticRightShiftOp definitions
//===----------------------------------------------------------------------===//

ArithmeticRightShiftOpAdaptor::ArithmeticRightShiftOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.arithmetic_right_shift", odsAttrs.getContext());
}

ArithmeticRightShiftOpAdaptor::ArithmeticRightShiftOpAdaptor(ArithmeticRightShiftOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ArithmeticRightShiftOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ArithmeticRightShiftOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ArithmeticRightShiftOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ArithmeticRightShiftOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange ArithmeticRightShiftOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ArithmeticRightShiftOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr ArithmeticRightShiftOpAdaptor::getRoundAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ArithmeticRightShiftOp::getRoundAttrName(*odsOpName)).cast<::mlir::BoolAttr>();
  return attr;
}

bool ArithmeticRightShiftOpAdaptor::getRound() {
  auto attr = getRoundAttr();
  return attr.getValue();
}

::mlir::LogicalResult ArithmeticRightShiftOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_round;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.arithmetic_right_shift' op ""requires attribute 'round'");
    if (namedAttrIt->getName() == ArithmeticRightShiftOp::getRoundAttrName(*odsOpName)) {
      tblgen_round = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_round && !((tblgen_round.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'tosa.arithmetic_right_shift' op ""attribute 'round' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ArithmeticRightShiftOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ArithmeticRightShiftOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ArithmeticRightShiftOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> ArithmeticRightShiftOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ArithmeticRightShiftOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ArithmeticRightShiftOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ArithmeticRightShiftOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ArithmeticRightShiftOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ArithmeticRightShiftOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::BoolAttr ArithmeticRightShiftOp::getRoundAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getRoundAttrName()).cast<::mlir::BoolAttr>();
}

bool ArithmeticRightShiftOp::getRound() {
  auto attr = getRoundAttr();
  return attr.getValue();
}

void ArithmeticRightShiftOp::setRoundAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getRoundAttrName(), attr);
}

void ArithmeticRightShiftOp::setRound(bool attrValue) {
  (*this)->setAttr(getRoundAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void ArithmeticRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, ::mlir::BoolAttr round) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute(getRoundAttrName(odsState.name), round);
  odsState.addTypes(output);
}

void ArithmeticRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::BoolAttr round) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute(getRoundAttrName(odsState.name), round);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ArithmeticRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, bool round) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute(getRoundAttrName(odsState.name), odsBuilder.getBoolAttr(round));
  odsState.addTypes(output);
}

void ArithmeticRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, bool round) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute(getRoundAttrName(odsState.name), odsBuilder.getBoolAttr(round));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ArithmeticRightShiftOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ArithmeticRightShiftOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_round;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'round'");
    if (namedAttrIt->getName() == getRoundAttrName()) {
      tblgen_round = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps0(*this, tblgen_round, "round")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ArithmeticRightShiftOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ArithmeticRightShiftOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ArithmeticRightShiftOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::AvgPool2dOp definitions
//===----------------------------------------------------------------------===//

AvgPool2dOpAdaptor::AvgPool2dOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.avg_pool2d", odsAttrs.getContext());
}

AvgPool2dOpAdaptor::AvgPool2dOpAdaptor(AvgPool2dOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> AvgPool2dOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AvgPool2dOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AvgPool2dOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange AvgPool2dOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr AvgPool2dOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr AvgPool2dOpAdaptor::getKernelAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, AvgPool2dOp::getKernelAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr AvgPool2dOpAdaptor::getKernel() {
  auto attr = getKernelAttr();
  return attr;
}

::mlir::ArrayAttr AvgPool2dOpAdaptor::getStrideAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, AvgPool2dOp::getStrideAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr AvgPool2dOpAdaptor::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::ArrayAttr AvgPool2dOpAdaptor::getPadAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, AvgPool2dOp::getPadAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr AvgPool2dOpAdaptor::getPad() {
  auto attr = getPadAttr();
  return attr;
}

mlir::tosa::UnaryOpQuantizationAttr AvgPool2dOpAdaptor::getQuantizationInfoAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, AvgPool2dOp::getQuantizationInfoAttrName(*odsOpName)).dyn_cast_or_null<mlir::tosa::UnaryOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::UnaryOpQuantizationAttr> AvgPool2dOpAdaptor::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::UnaryOpQuantizationAttr>(attr.cast<mlir::tosa::UnaryOpQuantizationAttr>()) : (::std::nullopt);
}

::mlir::LogicalResult AvgPool2dOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_kernel;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.avg_pool2d' op ""requires attribute 'kernel'");
    if (namedAttrIt->getName() == AvgPool2dOp::getKernelAttrName(*odsOpName)) {
      tblgen_kernel = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pad;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.avg_pool2d' op ""requires attribute 'pad'");
    if (namedAttrIt->getName() == AvgPool2dOp::getPadAttrName(*odsOpName)) {
      tblgen_pad = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_stride;
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.avg_pool2d' op ""requires attribute 'stride'");
    if (namedAttrIt->getName() == AvgPool2dOp::getStrideAttrName(*odsOpName)) {
      tblgen_stride = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == AvgPool2dOp::getQuantizationInfoAttrName(*odsOpName)) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_kernel && !((((tblgen_kernel.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_kernel.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_kernel.cast<::mlir::ArrayAttr>().size() == 2))))
    return emitError(loc, "'tosa.avg_pool2d' op ""attribute 'kernel' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");

  if (tblgen_stride && !((((tblgen_stride.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_stride.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_stride.cast<::mlir::ArrayAttr>().size() == 2))))
    return emitError(loc, "'tosa.avg_pool2d' op ""attribute 'stride' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");

  if (tblgen_pad && !((((tblgen_pad.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_pad.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_pad.cast<::mlir::ArrayAttr>().size() == 4))))
    return emitError(loc, "'tosa.avg_pool2d' op ""attribute 'pad' failed to satisfy constraint: 64-bit integer array attribute with exactly 4 elements");

  if (tblgen_quantization_info && !((tblgen_quantization_info.isa<mlir::tosa::UnaryOpQuantizationAttr>())))
    return emitError(loc, "'tosa.avg_pool2d' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for UnaryOp quantization information.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AvgPool2dOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AvgPool2dOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AvgPool2dOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AvgPool2dOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AvgPool2dOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AvgPool2dOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AvgPool2dOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr AvgPool2dOp::getKernelAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getKernelAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr AvgPool2dOp::getKernel() {
  auto attr = getKernelAttr();
  return attr;
}

::mlir::ArrayAttr AvgPool2dOp::getStrideAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getStrideAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr AvgPool2dOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::ArrayAttr AvgPool2dOp::getPadAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getPadAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr AvgPool2dOp::getPad() {
  auto attr = getPadAttr();
  return attr;
}

mlir::tosa::UnaryOpQuantizationAttr AvgPool2dOp::getQuantizationInfoAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getQuantizationInfoAttrName()).dyn_cast_or_null<mlir::tosa::UnaryOpQuantizationAttr>();
}

::std::optional<mlir::tosa::UnaryOpQuantizationAttr> AvgPool2dOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::UnaryOpQuantizationAttr>(attr.cast<mlir::tosa::UnaryOpQuantizationAttr>()) : (::std::nullopt);
}

void AvgPool2dOp::setKernelAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getKernelAttrName(), attr);
}

void AvgPool2dOp::setStrideAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getStrideAttrName(), attr);
}

void AvgPool2dOp::setPadAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPadAttrName(), attr);
}

void AvgPool2dOp::setQuantizationInfoAttr(mlir::tosa::UnaryOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute AvgPool2dOp::removeQuantizationInfoAttr() {
  return (*this)->removeAttr(getQuantizationInfoAttrName());
}

void AvgPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, ArrayAttr kernel, ArrayAttr stride, ArrayAttr pad) {
    buildAvgPool2dOpWithQuantInfo(odsBuilder, odsState, outputType,
                                  input, kernel, stride, pad);
  
}

void AvgPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::ArrayAttr kernel, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr pad, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addAttribute(getKernelAttrName(odsState.name), kernel);
  odsState.addAttribute(getStrideAttrName(odsState.name), stride);
  odsState.addAttribute(getPadAttrName(odsState.name), pad);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  odsState.addTypes(output);
}

void AvgPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr kernel, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr pad, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addAttribute(getKernelAttrName(odsState.name), kernel);
  odsState.addAttribute(getStrideAttrName(odsState.name), stride);
  odsState.addAttribute(getPadAttrName(odsState.name), pad);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AvgPool2dOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AvgPool2dOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_kernel;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'kernel'");
    if (namedAttrIt->getName() == getKernelAttrName()) {
      tblgen_kernel = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pad;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'pad'");
    if (namedAttrIt->getName() == getPadAttrName()) {
      tblgen_pad = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_stride;
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'stride'");
    if (namedAttrIt->getName() == getStrideAttrName()) {
      tblgen_stride = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getQuantizationInfoAttrName()) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_kernel, "kernel")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_stride, "stride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(*this, tblgen_pad, "pad")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps4(*this, tblgen_quantization_info, "quantization_info")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AvgPool2dOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AvgPool2dOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::AvgPool2dOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::BitwiseAndOp definitions
//===----------------------------------------------------------------------===//

BitwiseAndOpAdaptor::BitwiseAndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.bitwise_and", odsAttrs.getContext());
}

BitwiseAndOpAdaptor::BitwiseAndOpAdaptor(BitwiseAndOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> BitwiseAndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BitwiseAndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseAndOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value BitwiseAndOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange BitwiseAndOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr BitwiseAndOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BitwiseAndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BitwiseAndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BitwiseAndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BitwiseAndOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BitwiseAndOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BitwiseAndOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BitwiseAndOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BitwiseAndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BitwiseAndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BitwiseAndOp::getOutput() {
  return *getODSResults(0).begin();
}

void BitwiseAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void BitwiseAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BitwiseAndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BitwiseAndOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BitwiseAndOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void BitwiseAndOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::BitwiseAndOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::BitwiseNotOp definitions
//===----------------------------------------------------------------------===//

BitwiseNotOpAdaptor::BitwiseNotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.bitwise_not", odsAttrs.getContext());
}

BitwiseNotOpAdaptor::BitwiseNotOpAdaptor(BitwiseNotOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> BitwiseNotOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BitwiseNotOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseNotOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange BitwiseNotOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr BitwiseNotOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BitwiseNotOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BitwiseNotOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BitwiseNotOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BitwiseNotOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange BitwiseNotOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BitwiseNotOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BitwiseNotOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BitwiseNotOp::getOutput() {
  return *getODSResults(0).begin();
}

void BitwiseNotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void BitwiseNotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BitwiseNotOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BitwiseNotOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BitwiseNotOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void BitwiseNotOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::BitwiseNotOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::BitwiseOrOp definitions
//===----------------------------------------------------------------------===//

BitwiseOrOpAdaptor::BitwiseOrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.bitwise_or", odsAttrs.getContext());
}

BitwiseOrOpAdaptor::BitwiseOrOpAdaptor(BitwiseOrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> BitwiseOrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BitwiseOrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseOrOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value BitwiseOrOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange BitwiseOrOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr BitwiseOrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BitwiseOrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BitwiseOrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BitwiseOrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BitwiseOrOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BitwiseOrOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BitwiseOrOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BitwiseOrOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BitwiseOrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BitwiseOrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BitwiseOrOp::getOutput() {
  return *getODSResults(0).begin();
}

void BitwiseOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void BitwiseOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BitwiseOrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BitwiseOrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BitwiseOrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void BitwiseOrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::BitwiseOrOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::BitwiseXorOp definitions
//===----------------------------------------------------------------------===//

BitwiseXorOpAdaptor::BitwiseXorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.bitwise_xor", odsAttrs.getContext());
}

BitwiseXorOpAdaptor::BitwiseXorOpAdaptor(BitwiseXorOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> BitwiseXorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BitwiseXorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitwiseXorOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value BitwiseXorOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange BitwiseXorOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr BitwiseXorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BitwiseXorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BitwiseXorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BitwiseXorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BitwiseXorOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> BitwiseXorOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BitwiseXorOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BitwiseXorOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BitwiseXorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BitwiseXorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> BitwiseXorOp::getOutput() {
  return *getODSResults(0).begin();
}

void BitwiseXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void BitwiseXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BitwiseXorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BitwiseXorOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BitwiseXorOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void BitwiseXorOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::BitwiseXorOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::CastOp definitions
//===----------------------------------------------------------------------===//

CastOpAdaptor::CastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.cast", odsAttrs.getContext());
}

CastOpAdaptor::CastOpAdaptor(CastOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> CastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange CastOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr CastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> CastOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CastOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> CastOp::getOutput() {
  return *getODSResults(0).begin();
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(output);
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CastOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void CastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::CastOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::CeilOp definitions
//===----------------------------------------------------------------------===//

CeilOpAdaptor::CeilOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.ceil", odsAttrs.getContext());
}

CeilOpAdaptor::CeilOpAdaptor(CeilOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> CeilOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CeilOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CeilOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange CeilOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr CeilOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CeilOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CeilOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CeilOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> CeilOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CeilOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CeilOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CeilOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> CeilOp::getOutput() {
  return *getODSResults(0).begin();
}

void CeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void CeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CeilOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CeilOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CeilOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void CeilOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::CeilOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ClampOp definitions
//===----------------------------------------------------------------------===//

ClampOpAdaptor::ClampOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.clamp", odsAttrs.getContext());
}

ClampOpAdaptor::ClampOpAdaptor(ClampOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ClampOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ClampOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ClampOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ClampOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ClampOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ClampOpAdaptor::getMinIntAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, ClampOp::getMinIntAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ClampOpAdaptor::getMinInt() {
  auto attr = getMinIntAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ClampOpAdaptor::getMaxIntAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 2, ClampOp::getMaxIntAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ClampOpAdaptor::getMaxInt() {
  auto attr = getMaxIntAttr();
  return attr.getValue().getZExtValue();
}

::mlir::FloatAttr ClampOpAdaptor::getMinFpAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, ClampOp::getMinFpAttrName(*odsOpName)).cast<::mlir::FloatAttr>();
  return attr;
}

::llvm::APFloat ClampOpAdaptor::getMinFp() {
  auto attr = getMinFpAttr();
  return attr.getValue();
}

::mlir::FloatAttr ClampOpAdaptor::getMaxFpAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, ClampOp::getMaxFpAttrName(*odsOpName)).cast<::mlir::FloatAttr>();
  return attr;
}

::llvm::APFloat ClampOpAdaptor::getMaxFp() {
  auto attr = getMaxFpAttr();
  return attr.getValue();
}

::mlir::LogicalResult ClampOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_max_fp;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.clamp' op ""requires attribute 'max_fp'");
    if (namedAttrIt->getName() == ClampOp::getMaxFpAttrName(*odsOpName)) {
      tblgen_max_fp = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_max_int;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.clamp' op ""requires attribute 'max_int'");
    if (namedAttrIt->getName() == ClampOp::getMaxIntAttrName(*odsOpName)) {
      tblgen_max_int = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_min_fp;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.clamp' op ""requires attribute 'min_fp'");
    if (namedAttrIt->getName() == ClampOp::getMinFpAttrName(*odsOpName)) {
      tblgen_min_fp = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_min_int;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.clamp' op ""requires attribute 'min_int'");
    if (namedAttrIt->getName() == ClampOp::getMinIntAttrName(*odsOpName)) {
      tblgen_min_int = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_min_int && !(((tblgen_min_int.isa<::mlir::IntegerAttr>())) && ((tblgen_min_int.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tosa.clamp' op ""attribute 'min_int' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_max_int && !(((tblgen_max_int.isa<::mlir::IntegerAttr>())) && ((tblgen_max_int.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tosa.clamp' op ""attribute 'max_int' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_min_fp && !(((tblgen_min_fp.isa<::mlir::FloatAttr>())) && ((tblgen_min_fp.cast<::mlir::FloatAttr>().getType().isF32()))))
    return emitError(loc, "'tosa.clamp' op ""attribute 'min_fp' failed to satisfy constraint: 32-bit float attribute");

  if (tblgen_max_fp && !(((tblgen_max_fp.isa<::mlir::FloatAttr>())) && ((tblgen_max_fp.cast<::mlir::FloatAttr>().getType().isF32()))))
    return emitError(loc, "'tosa.clamp' op ""attribute 'max_fp' failed to satisfy constraint: 32-bit float attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ClampOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ClampOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ClampOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ClampOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ClampOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ClampOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ClampOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ClampOp::getMinIntAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getMinIntAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ClampOp::getMinInt() {
  auto attr = getMinIntAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ClampOp::getMaxIntAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 2, getMaxIntAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ClampOp::getMaxInt() {
  auto attr = getMaxIntAttr();
  return attr.getValue().getZExtValue();
}

::mlir::FloatAttr ClampOp::getMinFpAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getMinFpAttrName()).cast<::mlir::FloatAttr>();
}

::llvm::APFloat ClampOp::getMinFp() {
  auto attr = getMinFpAttr();
  return attr.getValue();
}

::mlir::FloatAttr ClampOp::getMaxFpAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getMaxFpAttrName()).cast<::mlir::FloatAttr>();
}

::llvm::APFloat ClampOp::getMaxFp() {
  auto attr = getMaxFpAttr();
  return attr.getValue();
}

void ClampOp::setMinIntAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getMinIntAttrName(), attr);
}

void ClampOp::setMinInt(uint64_t attrValue) {
  (*this)->setAttr(getMinIntAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void ClampOp::setMaxIntAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getMaxIntAttrName(), attr);
}

void ClampOp::setMaxInt(uint64_t attrValue) {
  (*this)->setAttr(getMaxIntAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void ClampOp::setMinFpAttr(::mlir::FloatAttr attr) {
  (*this)->setAttr(getMinFpAttrName(), attr);
}

void ClampOp::setMinFp(::llvm::APFloat attrValue) {
  (*this)->setAttr(getMinFpAttrName(), ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue));
}

void ClampOp::setMaxFpAttr(::mlir::FloatAttr attr) {
  (*this)->setAttr(getMaxFpAttrName(), attr);
}

void ClampOp::setMaxFp(::llvm::APFloat attrValue) {
  (*this)->setAttr(getMaxFpAttrName(), ::mlir::Builder((*this)->getContext()).getFloatAttr(::mlir::Builder((*this)->getContext()).getF32Type(), attrValue));
}

void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr min_int, ::mlir::IntegerAttr max_int, ::mlir::FloatAttr min_fp, ::mlir::FloatAttr max_fp) {
  odsState.addOperands(input);
  odsState.addAttribute(getMinIntAttrName(odsState.name), min_int);
  odsState.addAttribute(getMaxIntAttrName(odsState.name), max_int);
  odsState.addAttribute(getMinFpAttrName(odsState.name), min_fp);
  odsState.addAttribute(getMaxFpAttrName(odsState.name), max_fp);
  odsState.addTypes(output);
}

void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr min_int, ::mlir::IntegerAttr max_int, ::mlir::FloatAttr min_fp, ::mlir::FloatAttr max_fp) {
  odsState.addOperands(input);
  odsState.addAttribute(getMinIntAttrName(odsState.name), min_int);
  odsState.addAttribute(getMaxIntAttrName(odsState.name), max_int);
  odsState.addAttribute(getMinFpAttrName(odsState.name), min_fp);
  odsState.addAttribute(getMaxFpAttrName(odsState.name), max_fp);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t min_int, uint64_t max_int, ::llvm::APFloat min_fp, ::llvm::APFloat max_fp) {
  odsState.addOperands(input);
  odsState.addAttribute(getMinIntAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), min_int));
  odsState.addAttribute(getMaxIntAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), max_int));
  odsState.addAttribute(getMinFpAttrName(odsState.name), odsBuilder.getFloatAttr(odsBuilder.getF32Type(), min_fp));
  odsState.addAttribute(getMaxFpAttrName(odsState.name), odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max_fp));
  odsState.addTypes(output);
}

void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t min_int, uint64_t max_int, ::llvm::APFloat min_fp, ::llvm::APFloat max_fp) {
  odsState.addOperands(input);
  odsState.addAttribute(getMinIntAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), min_int));
  odsState.addAttribute(getMaxIntAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), max_int));
  odsState.addAttribute(getMinFpAttrName(odsState.name), odsBuilder.getFloatAttr(odsBuilder.getF32Type(), min_fp));
  odsState.addAttribute(getMaxFpAttrName(odsState.name), odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max_fp));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ClampOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ClampOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_max_fp;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'max_fp'");
    if (namedAttrIt->getName() == getMaxFpAttrName()) {
      tblgen_max_fp = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_max_int;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'max_int'");
    if (namedAttrIt->getName() == getMaxIntAttrName()) {
      tblgen_max_int = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_min_fp;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'min_fp'");
    if (namedAttrIt->getName() == getMinFpAttrName()) {
      tblgen_min_fp = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_min_int;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'min_int'");
    if (namedAttrIt->getName() == getMinIntAttrName()) {
      tblgen_min_int = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(*this, tblgen_min_int, "min_int")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(*this, tblgen_max_int, "max_int")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps5(*this, tblgen_min_fp, "min_fp")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps5(*this, tblgen_max_fp, "max_fp")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ClampOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ClampOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ClampOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ClzOp definitions
//===----------------------------------------------------------------------===//

ClzOpAdaptor::ClzOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.clz", odsAttrs.getContext());
}

ClzOpAdaptor::ClzOpAdaptor(ClzOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ClzOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ClzOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ClzOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ClzOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ClzOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ClzOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ClzOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ClzOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ClzOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ClzOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ClzOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ClzOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ClzOp::getOutput() {
  return *getODSResults(0).begin();
}

void ClzOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void ClzOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ClzOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ClzOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ClzOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ClzOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ClzOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ConcatOp definitions
//===----------------------------------------------------------------------===//

ConcatOpAdaptor::ConcatOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.concat", odsAttrs.getContext());
}

ConcatOpAdaptor::ConcatOpAdaptor(ConcatOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ConcatOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ConcatOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConcatOpAdaptor::getInput1() {
  return getODSOperands(0);
}

::mlir::ValueRange ConcatOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ConcatOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ConcatOpAdaptor::getAxisAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ConcatOp::getAxisAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ConcatOpAdaptor::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult ConcatOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.concat' op ""requires attribute 'axis'");
    if (namedAttrIt->getName() == ConcatOp::getAxisAttrName(*odsOpName)) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_axis && !(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tosa.concat' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConcatOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ConcatOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConcatOp::getInput1() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ConcatOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ConcatOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConcatOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ConcatOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ConcatOp::getAxisAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ConcatOp::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

void ConcatOp::setAxisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAxisAttrName(), attr);
}

void ConcatOp::setAxis(uint64_t attrValue) {
  (*this)->setAttr(getAxisAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void ConcatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::ValueRange input1, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input1);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  odsState.addTypes(output);
}

void ConcatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange input1, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input1);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConcatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::ValueRange input1, uint64_t axis) {
  odsState.addOperands(input1);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ConcatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange input1, uint64_t axis) {
  odsState.addOperands(input1);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConcatOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConcatOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'axis'");
    if (namedAttrIt->getName() == getAxisAttrName()) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(*this, tblgen_axis, "axis")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConcatOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ConcatOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ConcatOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ConstOp definitions
//===----------------------------------------------------------------------===//

ConstOpAdaptor::ConstOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.const", odsAttrs.getContext());
}

ConstOpAdaptor::ConstOpAdaptor(ConstOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ConstOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConstOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ConstOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ElementsAttr ConstOpAdaptor::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ConstOp::getValueAttrName(*odsOpName)).cast<::mlir::ElementsAttr>();
  return attr;
}

::mlir::ElementsAttr ConstOpAdaptor::getValue() {
  auto attr = getValueAttr();
  return attr;
}

::mlir::LogicalResult ConstOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.const' op ""requires attribute 'value'");
    if (namedAttrIt->getName() == ConstOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_value && !((tblgen_value.isa<::mlir::ElementsAttr>())))
    return emitError(loc, "'tosa.const' op ""attribute 'value' failed to satisfy constraint: constant vector/tensor attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConstOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ConstOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ConstOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::ElementsAttr ConstOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getValueAttrName()).cast<::mlir::ElementsAttr>();
}

::mlir::ElementsAttr ConstOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void ConstOp::setValueAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void ConstOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::ElementsAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(output);
}

void ConstOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ElementsAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ConstOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  auto attrName = getValueAttrName(odsState.name);
  for (auto attr : attributes) {
    if (attr.getName() != attrName) continue;
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
    odsState.addTypes({attr.getValue().cast<::mlir::TypedAttr>().getType()});
  }
}

::mlir::LogicalResult ConstOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'value'");
    if (namedAttrIt->getName() == getValueAttrName()) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps6(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConstOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ConstOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ConstOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::Conv2DOp definitions
//===----------------------------------------------------------------------===//

Conv2DOpAdaptor::Conv2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.conv2d", odsAttrs.getContext());
}

Conv2DOpAdaptor::Conv2DOpAdaptor(Conv2DOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> Conv2DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Conv2DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Conv2DOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::Value Conv2DOpAdaptor::getWeight() {
  return *getODSOperands(1).begin();
}

::mlir::Value Conv2DOpAdaptor::getBias() {
  return *getODSOperands(2).begin();
}

::mlir::ValueRange Conv2DOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr Conv2DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr Conv2DOpAdaptor::getPadAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, Conv2DOp::getPadAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr Conv2DOpAdaptor::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::ArrayAttr Conv2DOpAdaptor::getStrideAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, Conv2DOp::getStrideAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr Conv2DOpAdaptor::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::ArrayAttr Conv2DOpAdaptor::getDilationAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, Conv2DOp::getDilationAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr Conv2DOpAdaptor::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr Conv2DOpAdaptor::getQuantizationInfoAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, Conv2DOp::getQuantizationInfoAttrName(*odsOpName)).dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> Conv2DOpAdaptor::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(attr.cast<mlir::tosa::ConvOpQuantizationAttr>()) : (::std::nullopt);
}

::mlir::LogicalResult Conv2DOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dilation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.conv2d' op ""requires attribute 'dilation'");
    if (namedAttrIt->getName() == Conv2DOp::getDilationAttrName(*odsOpName)) {
      tblgen_dilation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pad;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.conv2d' op ""requires attribute 'pad'");
    if (namedAttrIt->getName() == Conv2DOp::getPadAttrName(*odsOpName)) {
      tblgen_pad = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_stride;
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.conv2d' op ""requires attribute 'stride'");
    if (namedAttrIt->getName() == Conv2DOp::getStrideAttrName(*odsOpName)) {
      tblgen_stride = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == Conv2DOp::getQuantizationInfoAttrName(*odsOpName)) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_pad && !((((tblgen_pad.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_pad.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_pad.cast<::mlir::ArrayAttr>().size() == 4))))
    return emitError(loc, "'tosa.conv2d' op ""attribute 'pad' failed to satisfy constraint: 64-bit integer array attribute with exactly 4 elements");

  if (tblgen_stride && !((((tblgen_stride.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_stride.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_stride.cast<::mlir::ArrayAttr>().size() == 2))))
    return emitError(loc, "'tosa.conv2d' op ""attribute 'stride' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");

  if (tblgen_dilation && !((((tblgen_dilation.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilation.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_dilation.cast<::mlir::ArrayAttr>().size() == 2))))
    return emitError(loc, "'tosa.conv2d' op ""attribute 'dilation' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");

  if (tblgen_quantization_info && !((tblgen_quantization_info.isa<mlir::tosa::ConvOpQuantizationAttr>())))
    return emitError(loc, "'tosa.conv2d' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for Conv type op quantization information.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv2DOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Conv2DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Conv2DOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::Value Conv2DOp::getWeight() {
  return *getODSOperands(1).begin();
}

::mlir::Value Conv2DOp::getBias() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange Conv2DOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DOp::getWeightMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange Conv2DOp::getBiasMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv2DOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Conv2DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Conv2DOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr Conv2DOp::getPadAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getPadAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr Conv2DOp::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::ArrayAttr Conv2DOp::getStrideAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getStrideAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr Conv2DOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::ArrayAttr Conv2DOp::getDilationAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getDilationAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr Conv2DOp::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr Conv2DOp::getQuantizationInfoAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getQuantizationInfoAttrName()).dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> Conv2DOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(attr.cast<mlir::tosa::ConvOpQuantizationAttr>()) : (::std::nullopt);
}

void Conv2DOp::setPadAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPadAttrName(), attr);
}

void Conv2DOp::setStrideAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getStrideAttrName(), attr);
}

void Conv2DOp::setDilationAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getDilationAttrName(), attr);
}

void Conv2DOp::setQuantizationInfoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute Conv2DOp::removeQuantizationInfoAttr() {
  return (*this)->removeAttr(getQuantizationInfoAttrName());
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, Value weight, Value bias, ArrayAttr pad, ArrayAttr stride, ArrayAttr dilation) {
    buildConvOpWithQuantInfo(odsBuilder, odsState, outputType,
                             input, weight, bias,
                             pad, stride, dilation);
  
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::ArrayAttr pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.addAttribute(getPadAttrName(odsState.name), pad);
  odsState.addAttribute(getStrideAttrName(odsState.name), stride);
  odsState.addAttribute(getDilationAttrName(odsState.name), dilation);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  odsState.addTypes(output);
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::ArrayAttr pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.addAttribute(getPadAttrName(odsState.name), pad);
  odsState.addAttribute(getStrideAttrName(odsState.name), stride);
  odsState.addAttribute(getDilationAttrName(odsState.name), dilation);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Conv2DOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Conv2DOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dilation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'dilation'");
    if (namedAttrIt->getName() == getDilationAttrName()) {
      tblgen_dilation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pad;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'pad'");
    if (namedAttrIt->getName() == getPadAttrName()) {
      tblgen_pad = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_stride;
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'stride'");
    if (namedAttrIt->getName() == getStrideAttrName()) {
      tblgen_stride = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getQuantizationInfoAttrName()) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(*this, tblgen_pad, "pad")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_stride, "stride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_dilation, "dilation")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps7(*this, tblgen_quantization_info, "quantization_info")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv2DOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void Conv2DOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::Conv2DOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::Conv3DOp definitions
//===----------------------------------------------------------------------===//

Conv3DOpAdaptor::Conv3DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.conv3d", odsAttrs.getContext());
}

Conv3DOpAdaptor::Conv3DOpAdaptor(Conv3DOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> Conv3DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Conv3DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Conv3DOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::Value Conv3DOpAdaptor::getWeight() {
  return *getODSOperands(1).begin();
}

::mlir::Value Conv3DOpAdaptor::getBias() {
  return *getODSOperands(2).begin();
}

::mlir::ValueRange Conv3DOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr Conv3DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr Conv3DOpAdaptor::getPadAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, Conv3DOp::getPadAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr Conv3DOpAdaptor::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::ArrayAttr Conv3DOpAdaptor::getStrideAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, Conv3DOp::getStrideAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr Conv3DOpAdaptor::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::ArrayAttr Conv3DOpAdaptor::getDilationAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, Conv3DOp::getDilationAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr Conv3DOpAdaptor::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr Conv3DOpAdaptor::getQuantizationInfoAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, Conv3DOp::getQuantizationInfoAttrName(*odsOpName)).dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> Conv3DOpAdaptor::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(attr.cast<mlir::tosa::ConvOpQuantizationAttr>()) : (::std::nullopt);
}

::mlir::LogicalResult Conv3DOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dilation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.conv3d' op ""requires attribute 'dilation'");
    if (namedAttrIt->getName() == Conv3DOp::getDilationAttrName(*odsOpName)) {
      tblgen_dilation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pad;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.conv3d' op ""requires attribute 'pad'");
    if (namedAttrIt->getName() == Conv3DOp::getPadAttrName(*odsOpName)) {
      tblgen_pad = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_stride;
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.conv3d' op ""requires attribute 'stride'");
    if (namedAttrIt->getName() == Conv3DOp::getStrideAttrName(*odsOpName)) {
      tblgen_stride = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == Conv3DOp::getQuantizationInfoAttrName(*odsOpName)) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_pad && !((((tblgen_pad.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_pad.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_pad.cast<::mlir::ArrayAttr>().size() == 6))))
    return emitError(loc, "'tosa.conv3d' op ""attribute 'pad' failed to satisfy constraint: 64-bit integer array attribute with exactly 6 elements");

  if (tblgen_stride && !((((tblgen_stride.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_stride.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_stride.cast<::mlir::ArrayAttr>().size() == 3))))
    return emitError(loc, "'tosa.conv3d' op ""attribute 'stride' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");

  if (tblgen_dilation && !((((tblgen_dilation.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilation.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_dilation.cast<::mlir::ArrayAttr>().size() == 3))))
    return emitError(loc, "'tosa.conv3d' op ""attribute 'dilation' failed to satisfy constraint: 64-bit integer array attribute with exactly 3 elements");

  if (tblgen_quantization_info && !((tblgen_quantization_info.isa<mlir::tosa::ConvOpQuantizationAttr>())))
    return emitError(loc, "'tosa.conv3d' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for Conv type op quantization information.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Conv3DOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Conv3DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Conv3DOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::Value Conv3DOp::getWeight() {
  return *getODSOperands(1).begin();
}

::mlir::Value Conv3DOp::getBias() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange Conv3DOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DOp::getWeightMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange Conv3DOp::getBiasMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> Conv3DOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Conv3DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Conv3DOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr Conv3DOp::getPadAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getPadAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr Conv3DOp::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::ArrayAttr Conv3DOp::getStrideAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getStrideAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr Conv3DOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::ArrayAttr Conv3DOp::getDilationAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getDilationAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr Conv3DOp::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr Conv3DOp::getQuantizationInfoAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getQuantizationInfoAttrName()).dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> Conv3DOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(attr.cast<mlir::tosa::ConvOpQuantizationAttr>()) : (::std::nullopt);
}

void Conv3DOp::setPadAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPadAttrName(), attr);
}

void Conv3DOp::setStrideAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getStrideAttrName(), attr);
}

void Conv3DOp::setDilationAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getDilationAttrName(), attr);
}

void Conv3DOp::setQuantizationInfoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute Conv3DOp::removeQuantizationInfoAttr() {
  return (*this)->removeAttr(getQuantizationInfoAttrName());
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, Value weight, Value bias, ArrayAttr pad, ArrayAttr stride, ArrayAttr dilation) {
    buildConvOpWithQuantInfo(odsBuilder, odsState, outputType,
                             input, weight, bias,
                             pad, stride, dilation);
  
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::ArrayAttr pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.addAttribute(getPadAttrName(odsState.name), pad);
  odsState.addAttribute(getStrideAttrName(odsState.name), stride);
  odsState.addAttribute(getDilationAttrName(odsState.name), dilation);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  odsState.addTypes(output);
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::ArrayAttr pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.addAttribute(getPadAttrName(odsState.name), pad);
  odsState.addAttribute(getStrideAttrName(odsState.name), stride);
  odsState.addAttribute(getDilationAttrName(odsState.name), dilation);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Conv3DOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Conv3DOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dilation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'dilation'");
    if (namedAttrIt->getName() == getDilationAttrName()) {
      tblgen_dilation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pad;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'pad'");
    if (namedAttrIt->getName() == getPadAttrName()) {
      tblgen_pad = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_stride;
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'stride'");
    if (namedAttrIt->getName() == getStrideAttrName()) {
      tblgen_stride = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getQuantizationInfoAttrName()) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps8(*this, tblgen_pad, "pad")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps9(*this, tblgen_stride, "stride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps9(*this, tblgen_dilation, "dilation")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps7(*this, tblgen_quantization_info, "quantization_info")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult Conv3DOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void Conv3DOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::Conv3DOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::CustomOp definitions
//===----------------------------------------------------------------------===//

CustomOpAdaptor::CustomOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.custom", odsAttrs.getContext());
}

CustomOpAdaptor::CustomOpAdaptor(CustomOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> CustomOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange CustomOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange CustomOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange CustomOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr CustomOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CustomOpAdaptor::getIdentifierAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, CustomOp::getIdentifierAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CustomOpAdaptor::getIdentifier() {
  auto attr = getIdentifierAttr();
  return attr.getValue();
}

::mlir::StringAttr CustomOpAdaptor::getConfigAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, CustomOp::getConfigAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CustomOpAdaptor::getConfig() {
  auto attr = getConfigAttr();
  return attr.getValue();
}

::mlir::StringAttr CustomOpAdaptor::getImplementationAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, CustomOp::getImplementationAttrsAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CustomOpAdaptor::getImplementationAttrs() {
  auto attr = getImplementationAttrsAttr();
  return attr.getValue();
}

::mlir::LogicalResult CustomOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_config;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.custom' op ""requires attribute 'config'");
    if (namedAttrIt->getName() == CustomOp::getConfigAttrName(*odsOpName)) {
      tblgen_config = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_identifier;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.custom' op ""requires attribute 'identifier'");
    if (namedAttrIt->getName() == CustomOp::getIdentifierAttrName(*odsOpName)) {
      tblgen_identifier = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_implementation_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.custom' op ""requires attribute 'implementation_attrs'");
    if (namedAttrIt->getName() == CustomOp::getImplementationAttrsAttrName(*odsOpName)) {
      tblgen_implementation_attrs = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_identifier && !((tblgen_identifier.isa<::mlir::StringAttr>())))
    return emitError(loc, "'tosa.custom' op ""attribute 'identifier' failed to satisfy constraint: string attribute");

  if (tblgen_config && !((tblgen_config.isa<::mlir::StringAttr>())))
    return emitError(loc, "'tosa.custom' op ""attribute 'config' failed to satisfy constraint: string attribute");

  if (tblgen_implementation_attrs && !((tblgen_implementation_attrs.isa<::mlir::StringAttr>())))
    return emitError(loc, "'tosa.custom' op ""attribute 'implementation_attrs' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CustomOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range CustomOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range CustomOp::getInputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange CustomOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CustomOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range CustomOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range CustomOp::getOutputs() {
  return getODSResults(0);
}

::mlir::StringAttr CustomOp::getIdentifierAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getIdentifierAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef CustomOp::getIdentifier() {
  auto attr = getIdentifierAttr();
  return attr.getValue();
}

::mlir::StringAttr CustomOp::getConfigAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getConfigAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef CustomOp::getConfig() {
  auto attr = getConfigAttr();
  return attr.getValue();
}

::mlir::StringAttr CustomOp::getImplementationAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getImplementationAttrsAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef CustomOp::getImplementationAttrs() {
  auto attr = getImplementationAttrsAttr();
  return attr.getValue();
}

void CustomOp::setIdentifierAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getIdentifierAttrName(), attr);
}

void CustomOp::setIdentifier(::llvm::StringRef attrValue) {
  (*this)->setAttr(getIdentifierAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void CustomOp::setConfigAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getConfigAttrName(), attr);
}

void CustomOp::setConfig(::llvm::StringRef attrValue) {
  (*this)->setAttr(getConfigAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void CustomOp::setImplementationAttrsAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getImplementationAttrsAttrName(), attr);
}

void CustomOp::setImplementationAttrs(::llvm::StringRef attrValue) {
  (*this)->setAttr(getImplementationAttrsAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void CustomOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::StringAttr identifier, ::mlir::StringAttr config, ::mlir::StringAttr implementation_attrs, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute(getIdentifierAttrName(odsState.name), identifier);
  odsState.addAttribute(getConfigAttrName(odsState.name), config);
  odsState.addAttribute(getImplementationAttrsAttrName(odsState.name), implementation_attrs);
  odsState.addTypes(outputs);
}

void CustomOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::llvm::StringRef identifier, ::llvm::StringRef config, ::llvm::StringRef implementation_attrs, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute(getIdentifierAttrName(odsState.name), odsBuilder.getStringAttr(identifier));
  odsState.addAttribute(getConfigAttrName(odsState.name), odsBuilder.getStringAttr(config));
  odsState.addAttribute(getImplementationAttrsAttrName(odsState.name), odsBuilder.getStringAttr(implementation_attrs));
  odsState.addTypes(outputs);
}

void CustomOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CustomOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_config;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'config'");
    if (namedAttrIt->getName() == getConfigAttrName()) {
      tblgen_config = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_identifier;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'identifier'");
    if (namedAttrIt->getName() == getIdentifierAttrName()) {
      tblgen_identifier = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_implementation_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'implementation_attrs'");
    if (namedAttrIt->getName() == getImplementationAttrsAttrName()) {
      tblgen_implementation_attrs = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps10(*this, tblgen_identifier, "identifier")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps10(*this, tblgen_config, "config")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps10(*this, tblgen_implementation_attrs, "implementation_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CustomOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::CustomOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::DepthwiseConv2DOp definitions
//===----------------------------------------------------------------------===//

DepthwiseConv2DOpAdaptor::DepthwiseConv2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.depthwise_conv2d", odsAttrs.getContext());
}

DepthwiseConv2DOpAdaptor::DepthwiseConv2DOpAdaptor(DepthwiseConv2DOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> DepthwiseConv2DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DepthwiseConv2DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DepthwiseConv2DOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::Value DepthwiseConv2DOpAdaptor::getWeight() {
  return *getODSOperands(1).begin();
}

::mlir::Value DepthwiseConv2DOpAdaptor::getBias() {
  return *getODSOperands(2).begin();
}

::mlir::ValueRange DepthwiseConv2DOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr DepthwiseConv2DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr DepthwiseConv2DOpAdaptor::getPadAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, DepthwiseConv2DOp::getPadAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr DepthwiseConv2DOpAdaptor::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::ArrayAttr DepthwiseConv2DOpAdaptor::getStrideAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, DepthwiseConv2DOp::getStrideAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr DepthwiseConv2DOpAdaptor::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::ArrayAttr DepthwiseConv2DOpAdaptor::getDilationAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, DepthwiseConv2DOp::getDilationAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr DepthwiseConv2DOpAdaptor::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr DepthwiseConv2DOpAdaptor::getQuantizationInfoAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, DepthwiseConv2DOp::getQuantizationInfoAttrName(*odsOpName)).dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> DepthwiseConv2DOpAdaptor::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(attr.cast<mlir::tosa::ConvOpQuantizationAttr>()) : (::std::nullopt);
}

::mlir::LogicalResult DepthwiseConv2DOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dilation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.depthwise_conv2d' op ""requires attribute 'dilation'");
    if (namedAttrIt->getName() == DepthwiseConv2DOp::getDilationAttrName(*odsOpName)) {
      tblgen_dilation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pad;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.depthwise_conv2d' op ""requires attribute 'pad'");
    if (namedAttrIt->getName() == DepthwiseConv2DOp::getPadAttrName(*odsOpName)) {
      tblgen_pad = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_stride;
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.depthwise_conv2d' op ""requires attribute 'stride'");
    if (namedAttrIt->getName() == DepthwiseConv2DOp::getStrideAttrName(*odsOpName)) {
      tblgen_stride = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DepthwiseConv2DOp::getQuantizationInfoAttrName(*odsOpName)) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_pad && !((((tblgen_pad.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_pad.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_pad.cast<::mlir::ArrayAttr>().size() == 4))))
    return emitError(loc, "'tosa.depthwise_conv2d' op ""attribute 'pad' failed to satisfy constraint: 64-bit integer array attribute with exactly 4 elements");

  if (tblgen_stride && !((((tblgen_stride.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_stride.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_stride.cast<::mlir::ArrayAttr>().size() == 2))))
    return emitError(loc, "'tosa.depthwise_conv2d' op ""attribute 'stride' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");

  if (tblgen_dilation && !((((tblgen_dilation.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilation.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_dilation.cast<::mlir::ArrayAttr>().size() == 2))))
    return emitError(loc, "'tosa.depthwise_conv2d' op ""attribute 'dilation' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");

  if (tblgen_quantization_info && !((tblgen_quantization_info.isa<mlir::tosa::ConvOpQuantizationAttr>())))
    return emitError(loc, "'tosa.depthwise_conv2d' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for Conv type op quantization information.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DepthwiseConv2DOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DepthwiseConv2DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DepthwiseConv2DOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::Value DepthwiseConv2DOp::getWeight() {
  return *getODSOperands(1).begin();
}

::mlir::Value DepthwiseConv2DOp::getBias() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange DepthwiseConv2DOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DOp::getWeightMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange DepthwiseConv2DOp::getBiasMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DepthwiseConv2DOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DepthwiseConv2DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DepthwiseConv2DOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr DepthwiseConv2DOp::getPadAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getPadAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr DepthwiseConv2DOp::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::ArrayAttr DepthwiseConv2DOp::getStrideAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getStrideAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr DepthwiseConv2DOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::ArrayAttr DepthwiseConv2DOp::getDilationAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getDilationAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr DepthwiseConv2DOp::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr DepthwiseConv2DOp::getQuantizationInfoAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getQuantizationInfoAttrName()).dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> DepthwiseConv2DOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(attr.cast<mlir::tosa::ConvOpQuantizationAttr>()) : (::std::nullopt);
}

void DepthwiseConv2DOp::setPadAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPadAttrName(), attr);
}

void DepthwiseConv2DOp::setStrideAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getStrideAttrName(), attr);
}

void DepthwiseConv2DOp::setDilationAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getDilationAttrName(), attr);
}

void DepthwiseConv2DOp::setQuantizationInfoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute DepthwiseConv2DOp::removeQuantizationInfoAttr() {
  return (*this)->removeAttr(getQuantizationInfoAttrName());
}

void DepthwiseConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, Value weight, Value bias, ArrayAttr pad, ArrayAttr stride, ArrayAttr dilation) {
    buildConvOpWithQuantInfo(odsBuilder, odsState, outputType,
                             input, weight, bias,
                             pad, stride, dilation);
  
}

void DepthwiseConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::ArrayAttr pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.addAttribute(getPadAttrName(odsState.name), pad);
  odsState.addAttribute(getStrideAttrName(odsState.name), stride);
  odsState.addAttribute(getDilationAttrName(odsState.name), dilation);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  odsState.addTypes(output);
}

void DepthwiseConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::ArrayAttr pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.addAttribute(getPadAttrName(odsState.name), pad);
  odsState.addAttribute(getStrideAttrName(odsState.name), stride);
  odsState.addAttribute(getDilationAttrName(odsState.name), dilation);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DepthwiseConv2DOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DepthwiseConv2DOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dilation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'dilation'");
    if (namedAttrIt->getName() == getDilationAttrName()) {
      tblgen_dilation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pad;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'pad'");
    if (namedAttrIt->getName() == getPadAttrName()) {
      tblgen_pad = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_stride;
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'stride'");
    if (namedAttrIt->getName() == getStrideAttrName()) {
      tblgen_stride = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getQuantizationInfoAttrName()) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(*this, tblgen_pad, "pad")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_stride, "stride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_dilation, "dilation")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps7(*this, tblgen_quantization_info, "quantization_info")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DepthwiseConv2DOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void DepthwiseConv2DOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::DepthwiseConv2DOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::DivOp definitions
//===----------------------------------------------------------------------===//

DivOpAdaptor::DivOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.div", odsAttrs.getContext());
}

DivOpAdaptor::DivOpAdaptor(DivOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> DivOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DivOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DivOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value DivOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange DivOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr DivOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult DivOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> DivOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> DivOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange DivOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange DivOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> DivOp::getOutput() {
  return *getODSResults(0).begin();
}

void DivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void DivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DivOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps8(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DivOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void DivOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::DivOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::EqualOp definitions
//===----------------------------------------------------------------------===//

EqualOpAdaptor::EqualOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.equal", odsAttrs.getContext());
}

EqualOpAdaptor::EqualOpAdaptor(EqualOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> EqualOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange EqualOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EqualOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value EqualOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange EqualOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr EqualOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult EqualOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> EqualOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range EqualOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> EqualOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> EqualOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange EqualOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange EqualOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> EqualOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EqualOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> EqualOp::getOutput() {
  return *getODSResults(0).begin();
}

void EqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void EqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(EqualOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void EqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EqualOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void EqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(EqualOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult EqualOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult EqualOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void EqualOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::EqualOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ExpOp definitions
//===----------------------------------------------------------------------===//

ExpOpAdaptor::ExpOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.exp", odsAttrs.getContext());
}

ExpOpAdaptor::ExpOpAdaptor(ExpOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ExpOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ExpOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExpOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ExpOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ExpOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ExpOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExpOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExpOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ExpOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ExpOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExpOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExpOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ExpOp::getOutput() {
  return *getODSResults(0).begin();
}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExpOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExpOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExpOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ExpOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ExpOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::FloorOp definitions
//===----------------------------------------------------------------------===//

FloorOpAdaptor::FloorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.floor", odsAttrs.getContext());
}

FloorOpAdaptor::FloorOpAdaptor(FloorOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> FloorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FloorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FloorOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange FloorOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr FloorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FloorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FloorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FloorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> FloorOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FloorOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FloorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FloorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> FloorOp::getOutput() {
  return *getODSResults(0).begin();
}

void FloorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void FloorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FloorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FloorOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FloorOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void FloorOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::FloorOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::FullyConnectedOp definitions
//===----------------------------------------------------------------------===//

FullyConnectedOpAdaptor::FullyConnectedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.fully_connected", odsAttrs.getContext());
}

FullyConnectedOpAdaptor::FullyConnectedOpAdaptor(FullyConnectedOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> FullyConnectedOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FullyConnectedOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FullyConnectedOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::Value FullyConnectedOpAdaptor::getWeight() {
  return *getODSOperands(1).begin();
}

::mlir::Value FullyConnectedOpAdaptor::getBias() {
  return *getODSOperands(2).begin();
}

::mlir::ValueRange FullyConnectedOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr FullyConnectedOpAdaptor::getAttributes() {
  return odsAttrs;
}

mlir::tosa::ConvOpQuantizationAttr FullyConnectedOpAdaptor::getQuantizationInfoAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FullyConnectedOp::getQuantizationInfoAttrName(*odsOpName)).dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> FullyConnectedOpAdaptor::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(attr.cast<mlir::tosa::ConvOpQuantizationAttr>()) : (::std::nullopt);
}

::mlir::LogicalResult FullyConnectedOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FullyConnectedOp::getQuantizationInfoAttrName(*odsOpName)) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_quantization_info && !((tblgen_quantization_info.isa<mlir::tosa::ConvOpQuantizationAttr>())))
    return emitError(loc, "'tosa.fully_connected' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for Conv type op quantization information.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FullyConnectedOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FullyConnectedOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FullyConnectedOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::Value FullyConnectedOp::getWeight() {
  return *getODSOperands(1).begin();
}

::mlir::Value FullyConnectedOp::getBias() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange FullyConnectedOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FullyConnectedOp::getWeightMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FullyConnectedOp::getBiasMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FullyConnectedOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FullyConnectedOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FullyConnectedOp::getOutput() {
  return *getODSResults(0).begin();
}

mlir::tosa::ConvOpQuantizationAttr FullyConnectedOp::getQuantizationInfoAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getQuantizationInfoAttrName()).dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> FullyConnectedOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(attr.cast<mlir::tosa::ConvOpQuantizationAttr>()) : (::std::nullopt);
}

void FullyConnectedOp::setQuantizationInfoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute FullyConnectedOp::removeQuantizationInfoAttr() {
  return (*this)->removeAttr(getQuantizationInfoAttrName());
}

void FullyConnectedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, Value weight, Value bias) {
    buildFCOpWithQuantInfo(odsBuilder, odsState, outputType,
                           input, weight, bias);
  
}

void FullyConnectedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  odsState.addTypes(output);
}

void FullyConnectedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FullyConnectedOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FullyConnectedOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getQuantizationInfoAttrName()) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps7(*this, tblgen_quantization_info, "quantization_info")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FullyConnectedOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void FullyConnectedOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::FullyConnectedOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::GatherOp definitions
//===----------------------------------------------------------------------===//

GatherOpAdaptor::GatherOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.gather", odsAttrs.getContext());
}

GatherOpAdaptor::GatherOpAdaptor(GatherOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> GatherOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GatherOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GatherOpAdaptor::getValues() {
  return *getODSOperands(0).begin();
}

::mlir::Value GatherOpAdaptor::getIndices() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange GatherOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr GatherOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GatherOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GatherOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GatherOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GatherOp::getValues() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> GatherOp::getIndices() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange GatherOp::getValuesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GatherOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GatherOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GatherOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GatherOp::getOutput() {
  return *getODSResults(0).begin();
}

void GatherOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value values, ::mlir::Value indices) {
  odsState.addOperands(values);
  odsState.addOperands(indices);
  odsState.addTypes(output);
}

void GatherOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value values, ::mlir::Value indices) {
  odsState.addOperands(values);
  odsState.addOperands(indices);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GatherOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GatherOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps11(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GatherOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void GatherOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::GatherOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::GreaterEqualOp definitions
//===----------------------------------------------------------------------===//

GreaterEqualOpAdaptor::GreaterEqualOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.greater_equal", odsAttrs.getContext());
}

GreaterEqualOpAdaptor::GreaterEqualOpAdaptor(GreaterEqualOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> GreaterEqualOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GreaterEqualOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GreaterEqualOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value GreaterEqualOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange GreaterEqualOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr GreaterEqualOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GreaterEqualOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GreaterEqualOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GreaterEqualOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> GreaterEqualOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> GreaterEqualOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange GreaterEqualOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GreaterEqualOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GreaterEqualOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GreaterEqualOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> GreaterEqualOp::getOutput() {
  return *getODSResults(0).begin();
}

void GreaterEqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void GreaterEqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GreaterEqualOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GreaterEqualOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GreaterEqualOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void GreaterEqualOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::GreaterEqualOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::GreaterOp definitions
//===----------------------------------------------------------------------===//

GreaterOpAdaptor::GreaterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.greater", odsAttrs.getContext());
}

GreaterOpAdaptor::GreaterOpAdaptor(GreaterOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> GreaterOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GreaterOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GreaterOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value GreaterOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange GreaterOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr GreaterOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GreaterOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GreaterOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GreaterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> GreaterOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> GreaterOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange GreaterOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GreaterOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GreaterOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GreaterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> GreaterOp::getOutput() {
  return *getODSResults(0).begin();
}

void GreaterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void GreaterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GreaterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GreaterOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GreaterOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void GreaterOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::GreaterOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::IdentityOp definitions
//===----------------------------------------------------------------------===//

IdentityOpAdaptor::IdentityOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.identity", odsAttrs.getContext());
}

IdentityOpAdaptor::IdentityOpAdaptor(IdentityOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> IdentityOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange IdentityOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IdentityOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange IdentityOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr IdentityOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult IdentityOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> IdentityOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range IdentityOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> IdentityOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange IdentityOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> IdentityOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range IdentityOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> IdentityOp::getOutput() {
  return *getODSResults(0).begin();
}

void IdentityOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void IdentityOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IdentityOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult IdentityOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult IdentityOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void IdentityOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::IdentityOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::IfOp definitions
//===----------------------------------------------------------------------===//

IfOpAdaptor::IfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.cond_if", odsAttrs.getContext());
}

IfOpAdaptor::IfOpAdaptor(IfOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> IfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange IfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IfOpAdaptor::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange IfOpAdaptor::getInputs() {
  return getODSOperands(1);
}

::mlir::ValueRange IfOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr IfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::Region &IfOpAdaptor::getThenBranch() {
  return *odsRegions[0];
}

::mlir::Region &IfOpAdaptor::getElseBranch() {
  return *odsRegions[1];
}

::mlir::RegionRange IfOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::LogicalResult IfOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> IfOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range IfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> IfOp::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range IfOp::getInputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange IfOp::getCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange IfOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> IfOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range IfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range IfOp::getOutput() {
  return getODSResults(0);
}

::mlir::Region &IfOp::getThenBranch() {
  return (*this)->getRegion(0);
}

::mlir::Region &IfOp::getElseBranch() {
  return (*this)->getRegion(1);
}

void IfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange output, ::mlir::Value cond, ::mlir::ValueRange inputs) {
  odsState.addOperands(cond);
  odsState.addOperands(inputs);
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(output);
}

void IfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult IfOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_TosaOps0(*this, region, "then_branch", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_TosaOps0(*this, region, "else_branch", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult IfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::IfOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::LogOp definitions
//===----------------------------------------------------------------------===//

LogOpAdaptor::LogOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.log", odsAttrs.getContext());
}

LogOpAdaptor::LogOpAdaptor(LogOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> LogOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange LogOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr LogOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LogOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LogOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange LogOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LogOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LogOp::getOutput() {
  return *getODSResults(0).begin();
}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LogOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LogOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void LogOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::LogOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::LogicalAndOp definitions
//===----------------------------------------------------------------------===//

LogicalAndOpAdaptor::LogicalAndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.logical_and", odsAttrs.getContext());
}

LogicalAndOpAdaptor::LogicalAndOpAdaptor(LogicalAndOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> LogicalAndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogicalAndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalAndOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalAndOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange LogicalAndOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr LogicalAndOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogicalAndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LogicalAndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogicalAndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LogicalAndOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> LogicalAndOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange LogicalAndOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LogicalAndOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LogicalAndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogicalAndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LogicalAndOp::getZ() {
  return *getODSResults(0).begin();
}

void LogicalAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(z);
}

void LogicalAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogicalAndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LogicalAndOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LogicalAndOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void LogicalAndOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::LogicalAndOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::LogicalLeftShiftOp definitions
//===----------------------------------------------------------------------===//

LogicalLeftShiftOpAdaptor::LogicalLeftShiftOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.logical_left_shift", odsAttrs.getContext());
}

LogicalLeftShiftOpAdaptor::LogicalLeftShiftOpAdaptor(LogicalLeftShiftOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> LogicalLeftShiftOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogicalLeftShiftOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalLeftShiftOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalLeftShiftOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange LogicalLeftShiftOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr LogicalLeftShiftOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogicalLeftShiftOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LogicalLeftShiftOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogicalLeftShiftOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LogicalLeftShiftOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> LogicalLeftShiftOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange LogicalLeftShiftOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LogicalLeftShiftOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LogicalLeftShiftOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogicalLeftShiftOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LogicalLeftShiftOp::getOutput() {
  return *getODSResults(0).begin();
}

void LogicalLeftShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void LogicalLeftShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogicalLeftShiftOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LogicalLeftShiftOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LogicalLeftShiftOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void LogicalLeftShiftOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::LogicalLeftShiftOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::LogicalNotOp definitions
//===----------------------------------------------------------------------===//

LogicalNotOpAdaptor::LogicalNotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.logical_not", odsAttrs.getContext());
}

LogicalNotOpAdaptor::LogicalNotOpAdaptor(LogicalNotOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> LogicalNotOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogicalNotOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalNotOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange LogicalNotOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr LogicalNotOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogicalNotOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LogicalNotOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogicalNotOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LogicalNotOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange LogicalNotOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LogicalNotOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogicalNotOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LogicalNotOp::getOutput() {
  return *getODSResults(0).begin();
}

void LogicalNotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void LogicalNotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1) {
  odsState.addOperands(input1);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(LogicalNotOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void LogicalNotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogicalNotOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LogicalNotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(LogicalNotOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult LogicalNotOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LogicalNotOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult LogicalNotOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void LogicalNotOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::LogicalNotOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::LogicalOrOp definitions
//===----------------------------------------------------------------------===//

LogicalOrOpAdaptor::LogicalOrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.logical_or", odsAttrs.getContext());
}

LogicalOrOpAdaptor::LogicalOrOpAdaptor(LogicalOrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> LogicalOrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogicalOrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalOrOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalOrOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange LogicalOrOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr LogicalOrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogicalOrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LogicalOrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogicalOrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LogicalOrOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> LogicalOrOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange LogicalOrOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LogicalOrOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LogicalOrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogicalOrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LogicalOrOp::getZ() {
  return *getODSResults(0).begin();
}

void LogicalOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(z);
}

void LogicalOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogicalOrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LogicalOrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LogicalOrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void LogicalOrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::LogicalOrOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::LogicalRightShiftOp definitions
//===----------------------------------------------------------------------===//

LogicalRightShiftOpAdaptor::LogicalRightShiftOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.logical_right_shift", odsAttrs.getContext());
}

LogicalRightShiftOpAdaptor::LogicalRightShiftOpAdaptor(LogicalRightShiftOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> LogicalRightShiftOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogicalRightShiftOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalRightShiftOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalRightShiftOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange LogicalRightShiftOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr LogicalRightShiftOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogicalRightShiftOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LogicalRightShiftOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogicalRightShiftOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LogicalRightShiftOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> LogicalRightShiftOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange LogicalRightShiftOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LogicalRightShiftOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LogicalRightShiftOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogicalRightShiftOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LogicalRightShiftOp::getOutput() {
  return *getODSResults(0).begin();
}

void LogicalRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void LogicalRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogicalRightShiftOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LogicalRightShiftOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LogicalRightShiftOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void LogicalRightShiftOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::LogicalRightShiftOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::LogicalXorOp definitions
//===----------------------------------------------------------------------===//

LogicalXorOpAdaptor::LogicalXorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.logical_xor", odsAttrs.getContext());
}

LogicalXorOpAdaptor::LogicalXorOpAdaptor(LogicalXorOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> LogicalXorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogicalXorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogicalXorOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value LogicalXorOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange LogicalXorOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr LogicalXorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogicalXorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LogicalXorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogicalXorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LogicalXorOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> LogicalXorOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange LogicalXorOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LogicalXorOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LogicalXorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogicalXorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LogicalXorOp::getZ() {
  return *getODSResults(0).begin();
}

void LogicalXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(z);
}

void LogicalXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogicalXorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LogicalXorOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LogicalXorOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void LogicalXorOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::LogicalXorOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::MatMulOp definitions
//===----------------------------------------------------------------------===//

MatMulOpAdaptor::MatMulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.matmul", odsAttrs.getContext());
}

MatMulOpAdaptor::MatMulOpAdaptor(MatMulOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> MatMulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MatMulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatMulOpAdaptor::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatMulOpAdaptor::getB() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange MatMulOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr MatMulOpAdaptor::getAttributes() {
  return odsAttrs;
}

mlir::tosa::MatMulOpQuantizationAttr MatMulOpAdaptor::getQuantizationInfoAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MatMulOp::getQuantizationInfoAttrName(*odsOpName)).dyn_cast_or_null<mlir::tosa::MatMulOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::MatMulOpQuantizationAttr> MatMulOpAdaptor::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::MatMulOpQuantizationAttr>(attr.cast<mlir::tosa::MatMulOpQuantizationAttr>()) : (::std::nullopt);
}

::mlir::LogicalResult MatMulOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == MatMulOp::getQuantizationInfoAttrName(*odsOpName)) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_quantization_info && !((tblgen_quantization_info.isa<mlir::tosa::MatMulOpQuantizationAttr>())))
    return emitError(loc, "'tosa.matmul' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for MatMulOp quantization information.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MatMulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MatMulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatMulOp::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatMulOp::getB() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MatMulOp::getAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MatMulOp::getBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MatMulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MatMulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatMulOp::getC() {
  return *getODSResults(0).begin();
}

mlir::tosa::MatMulOpQuantizationAttr MatMulOp::getQuantizationInfoAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getQuantizationInfoAttrName()).dyn_cast_or_null<mlir::tosa::MatMulOpQuantizationAttr>();
}

::std::optional<mlir::tosa::MatMulOpQuantizationAttr> MatMulOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::MatMulOpQuantizationAttr>(attr.cast<mlir::tosa::MatMulOpQuantizationAttr>()) : (::std::nullopt);
}

void MatMulOp::setQuantizationInfoAttr(mlir::tosa::MatMulOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute MatMulOp::removeQuantizationInfoAttr() {
  return (*this)->removeAttr(getQuantizationInfoAttrName());
}

void MatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value a, Value b) {
    buildMatMulOpWithQuantInfo(odsBuilder, odsState, outputType,
                               a, b);
  
}

void MatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type c, ::mlir::Value a, ::mlir::Value b, /*optional*/mlir::tosa::MatMulOpQuantizationAttr quantization_info) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  odsState.addTypes(c);
}

void MatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, /*optional*/mlir::tosa::MatMulOpQuantizationAttr quantization_info) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatMulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MatMulOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getQuantizationInfoAttrName()) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps11(*this, tblgen_quantization_info, "quantization_info")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps11(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MatMulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void MatMulOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::MatMulOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::MaxPool2dOp definitions
//===----------------------------------------------------------------------===//

MaxPool2dOpAdaptor::MaxPool2dOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.max_pool2d", odsAttrs.getContext());
}

MaxPool2dOpAdaptor::MaxPool2dOpAdaptor(MaxPool2dOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> MaxPool2dOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaxPool2dOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaxPool2dOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange MaxPool2dOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr MaxPool2dOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr MaxPool2dOpAdaptor::getKernelAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, MaxPool2dOp::getKernelAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr MaxPool2dOpAdaptor::getKernel() {
  auto attr = getKernelAttr();
  return attr;
}

::mlir::ArrayAttr MaxPool2dOpAdaptor::getStrideAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, MaxPool2dOp::getStrideAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr MaxPool2dOpAdaptor::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::ArrayAttr MaxPool2dOpAdaptor::getPadAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, MaxPool2dOp::getPadAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr MaxPool2dOpAdaptor::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::LogicalResult MaxPool2dOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_kernel;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.max_pool2d' op ""requires attribute 'kernel'");
    if (namedAttrIt->getName() == MaxPool2dOp::getKernelAttrName(*odsOpName)) {
      tblgen_kernel = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pad;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.max_pool2d' op ""requires attribute 'pad'");
    if (namedAttrIt->getName() == MaxPool2dOp::getPadAttrName(*odsOpName)) {
      tblgen_pad = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_stride;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.max_pool2d' op ""requires attribute 'stride'");
    if (namedAttrIt->getName() == MaxPool2dOp::getStrideAttrName(*odsOpName)) {
      tblgen_stride = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_kernel && !((((tblgen_kernel.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_kernel.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_kernel.cast<::mlir::ArrayAttr>().size() == 2))))
    return emitError(loc, "'tosa.max_pool2d' op ""attribute 'kernel' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");

  if (tblgen_stride && !((((tblgen_stride.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_stride.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_stride.cast<::mlir::ArrayAttr>().size() == 2))))
    return emitError(loc, "'tosa.max_pool2d' op ""attribute 'stride' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");

  if (tblgen_pad && !((((tblgen_pad.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_pad.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_pad.cast<::mlir::ArrayAttr>().size() == 4))))
    return emitError(loc, "'tosa.max_pool2d' op ""attribute 'pad' failed to satisfy constraint: 64-bit integer array attribute with exactly 4 elements");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MaxPool2dOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaxPool2dOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaxPool2dOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange MaxPool2dOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MaxPool2dOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaxPool2dOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaxPool2dOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr MaxPool2dOp::getKernelAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getKernelAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr MaxPool2dOp::getKernel() {
  auto attr = getKernelAttr();
  return attr;
}

::mlir::ArrayAttr MaxPool2dOp::getStrideAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getStrideAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr MaxPool2dOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::ArrayAttr MaxPool2dOp::getPadAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getPadAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr MaxPool2dOp::getPad() {
  auto attr = getPadAttr();
  return attr;
}

void MaxPool2dOp::setKernelAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getKernelAttrName(), attr);
}

void MaxPool2dOp::setStrideAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getStrideAttrName(), attr);
}

void MaxPool2dOp::setPadAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPadAttrName(), attr);
}

void MaxPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::ArrayAttr kernel, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr pad) {
  odsState.addOperands(input);
  odsState.addAttribute(getKernelAttrName(odsState.name), kernel);
  odsState.addAttribute(getStrideAttrName(odsState.name), stride);
  odsState.addAttribute(getPadAttrName(odsState.name), pad);
  odsState.addTypes(output);
}

void MaxPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr kernel, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr pad) {
  odsState.addOperands(input);
  odsState.addAttribute(getKernelAttrName(odsState.name), kernel);
  odsState.addAttribute(getStrideAttrName(odsState.name), stride);
  odsState.addAttribute(getPadAttrName(odsState.name), pad);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaxPool2dOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaxPool2dOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_kernel;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'kernel'");
    if (namedAttrIt->getName() == getKernelAttrName()) {
      tblgen_kernel = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pad;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'pad'");
    if (namedAttrIt->getName() == getPadAttrName()) {
      tblgen_pad = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_stride;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'stride'");
    if (namedAttrIt->getName() == getStrideAttrName()) {
      tblgen_stride = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_kernel, "kernel")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_stride, "stride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(*this, tblgen_pad, "pad")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MaxPool2dOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void MaxPool2dOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::MaxPool2dOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::MaximumOp definitions
//===----------------------------------------------------------------------===//

MaximumOpAdaptor::MaximumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.maximum", odsAttrs.getContext());
}

MaximumOpAdaptor::MaximumOpAdaptor(MaximumOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> MaximumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaximumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaximumOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaximumOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange MaximumOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr MaximumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MaximumOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MaximumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaximumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> MaximumOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> MaximumOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MaximumOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaximumOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MaximumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaximumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> MaximumOp::getOutput() {
  return *getODSResults(0).begin();
}

void MaximumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void MaximumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaximumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaximumOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MaximumOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void MaximumOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::MaximumOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::MinimumOp definitions
//===----------------------------------------------------------------------===//

MinimumOpAdaptor::MinimumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.minimum", odsAttrs.getContext());
}

MinimumOpAdaptor::MinimumOpAdaptor(MinimumOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> MinimumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MinimumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MinimumOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value MinimumOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange MinimumOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr MinimumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MinimumOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MinimumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MinimumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> MinimumOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> MinimumOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MinimumOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MinimumOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MinimumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MinimumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> MinimumOp::getOutput() {
  return *getODSResults(0).begin();
}

void MinimumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void MinimumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MinimumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MinimumOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MinimumOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void MinimumOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::MinimumOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::MulOp definitions
//===----------------------------------------------------------------------===//

MulOpAdaptor::MulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.mul", odsAttrs.getContext());
}

MulOpAdaptor::MulOpAdaptor(MulOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> MulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value MulOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange MulOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr MulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MulOpAdaptor::getShiftAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MulOp::getShiftAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t MulOpAdaptor::getShift() {
  auto attr = getShiftAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult MulOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_shift;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.mul' op ""requires attribute 'shift'");
    if (namedAttrIt->getName() == MulOp::getShiftAttrName(*odsOpName)) {
      tblgen_shift = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_shift && !(((tblgen_shift.isa<::mlir::IntegerAttr>())) && ((tblgen_shift.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'tosa.mul' op ""attribute 'shift' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> MulOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> MulOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MulOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MulOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> MulOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr MulOp::getShiftAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getShiftAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MulOp::getShift() {
  auto attr = getShiftAttr();
  return attr.getValue().getZExtValue();
}

void MulOp::setShiftAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getShiftAttrName(), attr);
}

void MulOp::setShift(uint32_t attrValue) {
  (*this)->setAttr(getShiftAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, ::mlir::IntegerAttr shift) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute(getShiftAttrName(odsState.name), shift);
  odsState.addTypes(output);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::IntegerAttr shift) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute(getShiftAttrName(odsState.name), shift);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, uint32_t shift) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute(getShiftAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), shift));
  odsState.addTypes(output);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, uint32_t shift) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addAttribute(getShiftAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), shift));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MulOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_shift;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'shift'");
    if (namedAttrIt->getName() == getShiftAttrName()) {
      tblgen_shift = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps12(*this, tblgen_shift, "shift")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void MulOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::MulOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::NegateOp definitions
//===----------------------------------------------------------------------===//

NegateOpAdaptor::NegateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.negate", odsAttrs.getContext());
}

NegateOpAdaptor::NegateOpAdaptor(NegateOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> NegateOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange NegateOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NegateOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange NegateOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr NegateOpAdaptor::getAttributes() {
  return odsAttrs;
}

mlir::tosa::UnaryOpQuantizationAttr NegateOpAdaptor::getQuantizationInfoAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, NegateOp::getQuantizationInfoAttrName(*odsOpName)).dyn_cast_or_null<mlir::tosa::UnaryOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::UnaryOpQuantizationAttr> NegateOpAdaptor::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::UnaryOpQuantizationAttr>(attr.cast<mlir::tosa::UnaryOpQuantizationAttr>()) : (::std::nullopt);
}

::mlir::LogicalResult NegateOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == NegateOp::getQuantizationInfoAttrName(*odsOpName)) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_quantization_info && !((tblgen_quantization_info.isa<mlir::tosa::UnaryOpQuantizationAttr>())))
    return emitError(loc, "'tosa.negate' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for UnaryOp quantization information.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> NegateOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range NegateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> NegateOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange NegateOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> NegateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range NegateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> NegateOp::getOutput() {
  return *getODSResults(0).begin();
}

mlir::tosa::UnaryOpQuantizationAttr NegateOp::getQuantizationInfoAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getQuantizationInfoAttrName()).dyn_cast_or_null<mlir::tosa::UnaryOpQuantizationAttr>();
}

::std::optional<mlir::tosa::UnaryOpQuantizationAttr> NegateOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::UnaryOpQuantizationAttr>(attr.cast<mlir::tosa::UnaryOpQuantizationAttr>()) : (::std::nullopt);
}

void NegateOp::setQuantizationInfoAttr(mlir::tosa::UnaryOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute NegateOp::removeQuantizationInfoAttr() {
  return (*this)->removeAttr(getQuantizationInfoAttrName());
}

void NegateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input) {
    buildUnaryOpWithQuantInfo(odsBuilder, odsState, outputType, input);
  
}

void NegateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input1);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  odsState.addTypes(output);
}

void NegateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input1);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NegateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult NegateOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getQuantizationInfoAttrName()) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps4(*this, tblgen_quantization_info, "quantization_info")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult NegateOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void NegateOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::NegateOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::PadOp definitions
//===----------------------------------------------------------------------===//

PadOpAdaptor::PadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.pad", odsAttrs.getContext());
}

PadOpAdaptor::PadOpAdaptor(PadOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> PadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange PadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PadOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value PadOpAdaptor::getPadding() {
  return *getODSOperands(1).begin();
}

::mlir::Value PadOpAdaptor::getPadConst() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange PadOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr PadOpAdaptor::getAttributes() {
  return odsAttrs;
}

mlir::tosa::PadOpQuantizationAttr PadOpAdaptor::getQuantizationInfoAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PadOp::getQuantizationInfoAttrName(*odsOpName)).dyn_cast_or_null<mlir::tosa::PadOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::PadOpQuantizationAttr> PadOpAdaptor::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::PadOpQuantizationAttr>(attr.cast<mlir::tosa::PadOpQuantizationAttr>()) : (::std::nullopt);
}

::mlir::LogicalResult PadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PadOp::getQuantizationInfoAttrName(*odsOpName)) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_quantization_info && !((tblgen_quantization_info.isa<mlir::tosa::PadOpQuantizationAttr>())))
    return emitError(loc, "'tosa.pad' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for PadOp quantization information.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PadOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range PadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> PadOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> PadOp::getPadding() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> PadOp::getPadConst() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange PadOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PadOp::getPaddingMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PadOp::getPadConstMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> PadOp::getOutput() {
  return *getODSResults(0).begin();
}

mlir::tosa::PadOpQuantizationAttr PadOp::getQuantizationInfoAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getQuantizationInfoAttrName()).dyn_cast_or_null<mlir::tosa::PadOpQuantizationAttr>();
}

::std::optional<mlir::tosa::PadOpQuantizationAttr> PadOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::PadOpQuantizationAttr>(attr.cast<mlir::tosa::PadOpQuantizationAttr>()) : (::std::nullopt);
}

void PadOp::setQuantizationInfoAttr(mlir::tosa::PadOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute PadOp::removeQuantizationInfoAttr() {
  return (*this)->removeAttr(getQuantizationInfoAttrName());
}

void PadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, Value paddings) {
    buildPadOpWithQuantInfo(odsBuilder, odsState, outputType,
                            input, paddings);
  
}

void PadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, Value paddings, Value pad_value) {
    buildExplicitValuePadOpWithQuantInfo(odsBuilder, odsState, outputType,
                                         input, paddings, pad_value);
  
}

void PadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value padding, /*optional*/::mlir::Value pad_const, /*optional*/mlir::tosa::PadOpQuantizationAttr quantization_info) {
  odsState.addOperands(input1);
  odsState.addOperands(padding);
  if (pad_const)
    odsState.addOperands(pad_const);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  odsState.addTypes(output);
}

void PadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value padding, /*optional*/::mlir::Value pad_const, /*optional*/mlir::tosa::PadOpQuantizationAttr quantization_info) {
  odsState.addOperands(input1);
  odsState.addOperands(padding);
  if (pad_const)
    odsState.addOperands(pad_const);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getQuantizationInfoAttrName()) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps13(*this, tblgen_quantization_info, "quantization_info")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps13(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps15(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps13(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PadOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void PadOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::PadOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::PowOp definitions
//===----------------------------------------------------------------------===//

PowOpAdaptor::PowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.pow", odsAttrs.getContext());
}

PowOpAdaptor::PowOpAdaptor(PowOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> PowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PowOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value PowOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange PowOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr PowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult PowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> PowOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> PowOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange PowOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PowOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> PowOp::getZ() {
  return *getODSResults(0).begin();
}

void PowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(z);
}

void PowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PowOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PowOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void PowOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::PowOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReciprocalOp definitions
//===----------------------------------------------------------------------===//

ReciprocalOpAdaptor::ReciprocalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reciprocal", odsAttrs.getContext());
}

ReciprocalOpAdaptor::ReciprocalOpAdaptor(ReciprocalOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ReciprocalOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReciprocalOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReciprocalOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ReciprocalOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ReciprocalOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ReciprocalOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReciprocalOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReciprocalOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ReciprocalOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReciprocalOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReciprocalOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReciprocalOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ReciprocalOp::getOutput() {
  return *getODSResults(0).begin();
}

void ReciprocalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void ReciprocalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReciprocalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReciprocalOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReciprocalOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ReciprocalOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ReciprocalOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReduceAllOp definitions
//===----------------------------------------------------------------------===//

ReduceAllOpAdaptor::ReduceAllOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reduce_all", odsAttrs.getContext());
}

ReduceAllOpAdaptor::ReduceAllOpAdaptor(ReduceAllOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ReduceAllOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReduceAllOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceAllOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ReduceAllOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ReduceAllOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceAllOpAdaptor::getAxisAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ReduceAllOp::getAxisAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ReduceAllOpAdaptor::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult ReduceAllOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.reduce_all' op ""requires attribute 'axis'");
    if (namedAttrIt->getName() == ReduceAllOp::getAxisAttrName(*odsOpName)) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_axis && !(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tosa.reduce_all' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReduceAllOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReduceAllOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceAllOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReduceAllOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReduceAllOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReduceAllOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceAllOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReduceAllOp::getAxisAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ReduceAllOp::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

void ReduceAllOp::setAxisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAxisAttrName(), attr);
}

void ReduceAllOp::setAxis(uint64_t attrValue) {
  (*this)->setAttr(getAxisAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void ReduceAllOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  odsState.addTypes(output);
}

void ReduceAllOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceAllOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ReduceAllOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceAllOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReduceAllOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'axis'");
    if (namedAttrIt->getName() == getAxisAttrName()) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(*this, tblgen_axis, "axis")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReduceAllOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ReduceAllOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ReduceAllOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReduceAnyOp definitions
//===----------------------------------------------------------------------===//

ReduceAnyOpAdaptor::ReduceAnyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reduce_any", odsAttrs.getContext());
}

ReduceAnyOpAdaptor::ReduceAnyOpAdaptor(ReduceAnyOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ReduceAnyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReduceAnyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceAnyOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ReduceAnyOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ReduceAnyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceAnyOpAdaptor::getAxisAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ReduceAnyOp::getAxisAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ReduceAnyOpAdaptor::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult ReduceAnyOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.reduce_any' op ""requires attribute 'axis'");
    if (namedAttrIt->getName() == ReduceAnyOp::getAxisAttrName(*odsOpName)) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_axis && !(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tosa.reduce_any' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReduceAnyOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReduceAnyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceAnyOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReduceAnyOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReduceAnyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReduceAnyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceAnyOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReduceAnyOp::getAxisAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ReduceAnyOp::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

void ReduceAnyOp::setAxisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAxisAttrName(), attr);
}

void ReduceAnyOp::setAxis(uint64_t attrValue) {
  (*this)->setAttr(getAxisAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void ReduceAnyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  odsState.addTypes(output);
}

void ReduceAnyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceAnyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ReduceAnyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceAnyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReduceAnyOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'axis'");
    if (namedAttrIt->getName() == getAxisAttrName()) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(*this, tblgen_axis, "axis")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReduceAnyOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ReduceAnyOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ReduceAnyOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReduceMaxOp definitions
//===----------------------------------------------------------------------===//

ReduceMaxOpAdaptor::ReduceMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reduce_max", odsAttrs.getContext());
}

ReduceMaxOpAdaptor::ReduceMaxOpAdaptor(ReduceMaxOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ReduceMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReduceMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceMaxOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ReduceMaxOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ReduceMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceMaxOpAdaptor::getAxisAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ReduceMaxOp::getAxisAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ReduceMaxOpAdaptor::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult ReduceMaxOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.reduce_max' op ""requires attribute 'axis'");
    if (namedAttrIt->getName() == ReduceMaxOp::getAxisAttrName(*odsOpName)) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_axis && !(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tosa.reduce_max' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReduceMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReduceMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceMaxOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReduceMaxOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReduceMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReduceMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceMaxOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReduceMaxOp::getAxisAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ReduceMaxOp::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

void ReduceMaxOp::setAxisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAxisAttrName(), attr);
}

void ReduceMaxOp::setAxis(uint64_t attrValue) {
  (*this)->setAttr(getAxisAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void ReduceMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  odsState.addTypes(output);
}

void ReduceMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ReduceMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReduceMaxOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'axis'");
    if (namedAttrIt->getName() == getAxisAttrName()) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(*this, tblgen_axis, "axis")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReduceMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ReduceMaxOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ReduceMaxOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReduceMinOp definitions
//===----------------------------------------------------------------------===//

ReduceMinOpAdaptor::ReduceMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reduce_min", odsAttrs.getContext());
}

ReduceMinOpAdaptor::ReduceMinOpAdaptor(ReduceMinOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ReduceMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReduceMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceMinOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ReduceMinOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ReduceMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceMinOpAdaptor::getAxisAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ReduceMinOp::getAxisAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ReduceMinOpAdaptor::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult ReduceMinOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.reduce_min' op ""requires attribute 'axis'");
    if (namedAttrIt->getName() == ReduceMinOp::getAxisAttrName(*odsOpName)) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_axis && !(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tosa.reduce_min' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReduceMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReduceMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceMinOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReduceMinOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReduceMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReduceMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceMinOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReduceMinOp::getAxisAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ReduceMinOp::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

void ReduceMinOp::setAxisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAxisAttrName(), attr);
}

void ReduceMinOp::setAxis(uint64_t attrValue) {
  (*this)->setAttr(getAxisAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void ReduceMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  odsState.addTypes(output);
}

void ReduceMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ReduceMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReduceMinOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'axis'");
    if (namedAttrIt->getName() == getAxisAttrName()) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(*this, tblgen_axis, "axis")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReduceMinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ReduceMinOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ReduceMinOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReduceProdOp definitions
//===----------------------------------------------------------------------===//

ReduceProdOpAdaptor::ReduceProdOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reduce_prod", odsAttrs.getContext());
}

ReduceProdOpAdaptor::ReduceProdOpAdaptor(ReduceProdOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ReduceProdOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReduceProdOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceProdOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ReduceProdOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ReduceProdOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceProdOpAdaptor::getAxisAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ReduceProdOp::getAxisAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ReduceProdOpAdaptor::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult ReduceProdOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.reduce_prod' op ""requires attribute 'axis'");
    if (namedAttrIt->getName() == ReduceProdOp::getAxisAttrName(*odsOpName)) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_axis && !(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tosa.reduce_prod' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReduceProdOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReduceProdOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceProdOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReduceProdOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReduceProdOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReduceProdOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceProdOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReduceProdOp::getAxisAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ReduceProdOp::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

void ReduceProdOp::setAxisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAxisAttrName(), attr);
}

void ReduceProdOp::setAxis(uint64_t attrValue) {
  (*this)->setAttr(getAxisAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void ReduceProdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  odsState.addTypes(output);
}

void ReduceProdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceProdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ReduceProdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceProdOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReduceProdOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'axis'");
    if (namedAttrIt->getName() == getAxisAttrName()) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(*this, tblgen_axis, "axis")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReduceProdOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ReduceProdOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ReduceProdOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReduceSumOp definitions
//===----------------------------------------------------------------------===//

ReduceSumOpAdaptor::ReduceSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reduce_sum", odsAttrs.getContext());
}

ReduceSumOpAdaptor::ReduceSumOpAdaptor(ReduceSumOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ReduceSumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReduceSumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceSumOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ReduceSumOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ReduceSumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceSumOpAdaptor::getAxisAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ReduceSumOp::getAxisAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ReduceSumOpAdaptor::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult ReduceSumOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.reduce_sum' op ""requires attribute 'axis'");
    if (namedAttrIt->getName() == ReduceSumOp::getAxisAttrName(*odsOpName)) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_axis && !(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tosa.reduce_sum' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReduceSumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReduceSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceSumOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReduceSumOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReduceSumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReduceSumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceSumOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReduceSumOp::getAxisAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ReduceSumOp::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

void ReduceSumOp::setAxisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAxisAttrName(), attr);
}

void ReduceSumOp::setAxis(uint64_t attrValue) {
  (*this)->setAttr(getAxisAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void ReduceSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  odsState.addTypes(output);
}

void ReduceSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ReduceSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceSumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReduceSumOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'axis'");
    if (namedAttrIt->getName() == getAxisAttrName()) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(*this, tblgen_axis, "axis")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReduceSumOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ReduceSumOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ReduceSumOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::RescaleOp definitions
//===----------------------------------------------------------------------===//

RescaleOpAdaptor::RescaleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.rescale", odsAttrs.getContext());
}

RescaleOpAdaptor::RescaleOpAdaptor(RescaleOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> RescaleOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange RescaleOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RescaleOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange RescaleOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr RescaleOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr RescaleOpAdaptor::getInputZpAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 5, RescaleOp::getInputZpAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t RescaleOpAdaptor::getInputZp() {
  auto attr = getInputZpAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr RescaleOpAdaptor::getOutputZpAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 3, RescaleOp::getOutputZpAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t RescaleOpAdaptor::getOutputZp() {
  auto attr = getOutputZpAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr RescaleOpAdaptor::getMultiplierAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 4, RescaleOp::getMultiplierAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr RescaleOpAdaptor::getMultiplier() {
  auto attr = getMultiplierAttr();
  return attr;
}

::mlir::ArrayAttr RescaleOpAdaptor::getShiftAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 6, odsAttrs.end() - 0, RescaleOp::getShiftAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr RescaleOpAdaptor::getShift() {
  auto attr = getShiftAttr();
  return attr;
}

::mlir::BoolAttr RescaleOpAdaptor::getScale32Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 5, odsAttrs.end() - 1, RescaleOp::getScale32AttrName(*odsOpName)).cast<::mlir::BoolAttr>();
  return attr;
}

bool RescaleOpAdaptor::getScale32() {
  auto attr = getScale32Attr();
  return attr.getValue();
}

::mlir::BoolAttr RescaleOpAdaptor::getDoubleRoundAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 6, RescaleOp::getDoubleRoundAttrName(*odsOpName)).cast<::mlir::BoolAttr>();
  return attr;
}

bool RescaleOpAdaptor::getDoubleRound() {
  auto attr = getDoubleRoundAttr();
  return attr.getValue();
}

::mlir::BoolAttr RescaleOpAdaptor::getPerChannelAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 4, odsAttrs.end() - 2, RescaleOp::getPerChannelAttrName(*odsOpName)).cast<::mlir::BoolAttr>();
  return attr;
}

bool RescaleOpAdaptor::getPerChannel() {
  auto attr = getPerChannelAttr();
  return attr.getValue();
}

::mlir::LogicalResult RescaleOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_double_round;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.rescale' op ""requires attribute 'double_round'");
    if (namedAttrIt->getName() == RescaleOp::getDoubleRoundAttrName(*odsOpName)) {
      tblgen_double_round = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_input_zp;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.rescale' op ""requires attribute 'input_zp'");
    if (namedAttrIt->getName() == RescaleOp::getInputZpAttrName(*odsOpName)) {
      tblgen_input_zp = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_multiplier;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.rescale' op ""requires attribute 'multiplier'");
    if (namedAttrIt->getName() == RescaleOp::getMultiplierAttrName(*odsOpName)) {
      tblgen_multiplier = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_output_zp;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.rescale' op ""requires attribute 'output_zp'");
    if (namedAttrIt->getName() == RescaleOp::getOutputZpAttrName(*odsOpName)) {
      tblgen_output_zp = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_per_channel;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.rescale' op ""requires attribute 'per_channel'");
    if (namedAttrIt->getName() == RescaleOp::getPerChannelAttrName(*odsOpName)) {
      tblgen_per_channel = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_scale32;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.rescale' op ""requires attribute 'scale32'");
    if (namedAttrIt->getName() == RescaleOp::getScale32AttrName(*odsOpName)) {
      tblgen_scale32 = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_shift;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.rescale' op ""requires attribute 'shift'");
    if (namedAttrIt->getName() == RescaleOp::getShiftAttrName(*odsOpName)) {
      tblgen_shift = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_input_zp && !(((tblgen_input_zp.isa<::mlir::IntegerAttr>())) && ((tblgen_input_zp.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'tosa.rescale' op ""attribute 'input_zp' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_output_zp && !(((tblgen_output_zp.isa<::mlir::IntegerAttr>())) && ((tblgen_output_zp.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'tosa.rescale' op ""attribute 'output_zp' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_multiplier && !(((tblgen_multiplier.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_multiplier.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))); }))))
    return emitError(loc, "'tosa.rescale' op ""attribute 'multiplier' failed to satisfy constraint: 32-bit integer array attribute");

  if (tblgen_shift && !(((tblgen_shift.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_shift.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))); }))))
    return emitError(loc, "'tosa.rescale' op ""attribute 'shift' failed to satisfy constraint: 32-bit integer array attribute");

  if (tblgen_scale32 && !((tblgen_scale32.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'tosa.rescale' op ""attribute 'scale32' failed to satisfy constraint: bool attribute");

  if (tblgen_double_round && !((tblgen_double_round.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'tosa.rescale' op ""attribute 'double_round' failed to satisfy constraint: bool attribute");

  if (tblgen_per_channel && !((tblgen_per_channel.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'tosa.rescale' op ""attribute 'per_channel' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RescaleOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range RescaleOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> RescaleOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange RescaleOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> RescaleOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RescaleOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> RescaleOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr RescaleOp::getInputZpAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 5, getInputZpAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t RescaleOp::getInputZp() {
  auto attr = getInputZpAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr RescaleOp::getOutputZpAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 3, getOutputZpAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t RescaleOp::getOutputZp() {
  auto attr = getOutputZpAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr RescaleOp::getMultiplierAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 4, getMultiplierAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr RescaleOp::getMultiplier() {
  auto attr = getMultiplierAttr();
  return attr;
}

::mlir::ArrayAttr RescaleOp::getShiftAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 6, (*this)->getAttrs().end() - 0, getShiftAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr RescaleOp::getShift() {
  auto attr = getShiftAttr();
  return attr;
}

::mlir::BoolAttr RescaleOp::getScale32Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 5, (*this)->getAttrs().end() - 1, getScale32AttrName()).cast<::mlir::BoolAttr>();
}

bool RescaleOp::getScale32() {
  auto attr = getScale32Attr();
  return attr.getValue();
}

::mlir::BoolAttr RescaleOp::getDoubleRoundAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 6, getDoubleRoundAttrName()).cast<::mlir::BoolAttr>();
}

bool RescaleOp::getDoubleRound() {
  auto attr = getDoubleRoundAttr();
  return attr.getValue();
}

::mlir::BoolAttr RescaleOp::getPerChannelAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 4, (*this)->getAttrs().end() - 2, getPerChannelAttrName()).cast<::mlir::BoolAttr>();
}

bool RescaleOp::getPerChannel() {
  auto attr = getPerChannelAttr();
  return attr.getValue();
}

void RescaleOp::setInputZpAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getInputZpAttrName(), attr);
}

void RescaleOp::setInputZp(uint32_t attrValue) {
  (*this)->setAttr(getInputZpAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void RescaleOp::setOutputZpAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getOutputZpAttrName(), attr);
}

void RescaleOp::setOutputZp(uint32_t attrValue) {
  (*this)->setAttr(getOutputZpAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void RescaleOp::setMultiplierAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getMultiplierAttrName(), attr);
}

void RescaleOp::setShiftAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getShiftAttrName(), attr);
}

void RescaleOp::setScale32Attr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getScale32AttrName(), attr);
}

void RescaleOp::setScale32(bool attrValue) {
  (*this)->setAttr(getScale32AttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void RescaleOp::setDoubleRoundAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getDoubleRoundAttrName(), attr);
}

void RescaleOp::setDoubleRound(bool attrValue) {
  (*this)->setAttr(getDoubleRoundAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void RescaleOp::setPerChannelAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getPerChannelAttrName(), attr);
}

void RescaleOp::setPerChannel(bool attrValue) {
  (*this)->setAttr(getPerChannelAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void RescaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr input_zp, ::mlir::IntegerAttr output_zp, ::mlir::ArrayAttr multiplier, ::mlir::ArrayAttr shift, ::mlir::BoolAttr scale32, ::mlir::BoolAttr double_round, ::mlir::BoolAttr per_channel) {
  odsState.addOperands(input);
  odsState.addAttribute(getInputZpAttrName(odsState.name), input_zp);
  odsState.addAttribute(getOutputZpAttrName(odsState.name), output_zp);
  odsState.addAttribute(getMultiplierAttrName(odsState.name), multiplier);
  odsState.addAttribute(getShiftAttrName(odsState.name), shift);
  odsState.addAttribute(getScale32AttrName(odsState.name), scale32);
  odsState.addAttribute(getDoubleRoundAttrName(odsState.name), double_round);
  odsState.addAttribute(getPerChannelAttrName(odsState.name), per_channel);
  odsState.addTypes(output);
}

void RescaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr input_zp, ::mlir::IntegerAttr output_zp, ::mlir::ArrayAttr multiplier, ::mlir::ArrayAttr shift, ::mlir::BoolAttr scale32, ::mlir::BoolAttr double_round, ::mlir::BoolAttr per_channel) {
  odsState.addOperands(input);
  odsState.addAttribute(getInputZpAttrName(odsState.name), input_zp);
  odsState.addAttribute(getOutputZpAttrName(odsState.name), output_zp);
  odsState.addAttribute(getMultiplierAttrName(odsState.name), multiplier);
  odsState.addAttribute(getShiftAttrName(odsState.name), shift);
  odsState.addAttribute(getScale32AttrName(odsState.name), scale32);
  odsState.addAttribute(getDoubleRoundAttrName(odsState.name), double_round);
  odsState.addAttribute(getPerChannelAttrName(odsState.name), per_channel);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RescaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint32_t input_zp, uint32_t output_zp, ::mlir::ArrayAttr multiplier, ::mlir::ArrayAttr shift, bool scale32, bool double_round, bool per_channel) {
  odsState.addOperands(input);
  odsState.addAttribute(getInputZpAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), input_zp));
  odsState.addAttribute(getOutputZpAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), output_zp));
  odsState.addAttribute(getMultiplierAttrName(odsState.name), multiplier);
  odsState.addAttribute(getShiftAttrName(odsState.name), shift);
  odsState.addAttribute(getScale32AttrName(odsState.name), odsBuilder.getBoolAttr(scale32));
  odsState.addAttribute(getDoubleRoundAttrName(odsState.name), odsBuilder.getBoolAttr(double_round));
  odsState.addAttribute(getPerChannelAttrName(odsState.name), odsBuilder.getBoolAttr(per_channel));
  odsState.addTypes(output);
}

void RescaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t input_zp, uint32_t output_zp, ::mlir::ArrayAttr multiplier, ::mlir::ArrayAttr shift, bool scale32, bool double_round, bool per_channel) {
  odsState.addOperands(input);
  odsState.addAttribute(getInputZpAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), input_zp));
  odsState.addAttribute(getOutputZpAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), output_zp));
  odsState.addAttribute(getMultiplierAttrName(odsState.name), multiplier);
  odsState.addAttribute(getShiftAttrName(odsState.name), shift);
  odsState.addAttribute(getScale32AttrName(odsState.name), odsBuilder.getBoolAttr(scale32));
  odsState.addAttribute(getDoubleRoundAttrName(odsState.name), odsBuilder.getBoolAttr(double_round));
  odsState.addAttribute(getPerChannelAttrName(odsState.name), odsBuilder.getBoolAttr(per_channel));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RescaleOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RescaleOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_double_round;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'double_round'");
    if (namedAttrIt->getName() == getDoubleRoundAttrName()) {
      tblgen_double_round = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_input_zp;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'input_zp'");
    if (namedAttrIt->getName() == getInputZpAttrName()) {
      tblgen_input_zp = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_multiplier;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'multiplier'");
    if (namedAttrIt->getName() == getMultiplierAttrName()) {
      tblgen_multiplier = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_output_zp;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'output_zp'");
    if (namedAttrIt->getName() == getOutputZpAttrName()) {
      tblgen_output_zp = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_per_channel;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'per_channel'");
    if (namedAttrIt->getName() == getPerChannelAttrName()) {
      tblgen_per_channel = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_scale32;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'scale32'");
    if (namedAttrIt->getName() == getScale32AttrName()) {
      tblgen_scale32 = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_shift;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'shift'");
    if (namedAttrIt->getName() == getShiftAttrName()) {
      tblgen_shift = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps12(*this, tblgen_input_zp, "input_zp")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps12(*this, tblgen_output_zp, "output_zp")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps14(*this, tblgen_multiplier, "multiplier")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps14(*this, tblgen_shift, "shift")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps0(*this, tblgen_scale32, "scale32")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps0(*this, tblgen_double_round, "double_round")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps0(*this, tblgen_per_channel, "per_channel")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult RescaleOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void RescaleOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::RescaleOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReshapeOp definitions
//===----------------------------------------------------------------------===//

ReshapeOpAdaptor::ReshapeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reshape", odsAttrs.getContext());
}

ReshapeOpAdaptor::ReshapeOpAdaptor(ReshapeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ReshapeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReshapeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReshapeOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ReshapeOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ReshapeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ReshapeOpAdaptor::getNewShapeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ReshapeOp::getNewShapeAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ReshapeOpAdaptor::getNewShape() {
  auto attr = getNewShapeAttr();
  return attr;
}

::mlir::LogicalResult ReshapeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_new_shape;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.reshape' op ""requires attribute 'new_shape'");
    if (namedAttrIt->getName() == ReshapeOp::getNewShapeAttrName(*odsOpName)) {
      tblgen_new_shape = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_new_shape && !(((tblgen_new_shape.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_new_shape.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'tosa.reshape' op ""attribute 'new_shape' failed to satisfy constraint: 64-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReshapeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReshapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ReshapeOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReshapeOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReshapeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReshapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ReshapeOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr ReshapeOp::getNewShapeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNewShapeAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ReshapeOp::getNewShape() {
  auto attr = getNewShapeAttr();
  return attr;
}

void ReshapeOp::setNewShapeAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getNewShapeAttrName(), attr);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::ArrayAttr new_shape) {
  odsState.addOperands(input1);
  odsState.addAttribute(getNewShapeAttrName(odsState.name), new_shape);
  odsState.addTypes(output);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::ArrayAttr new_shape) {
  odsState.addOperands(input1);
  odsState.addAttribute(getNewShapeAttrName(odsState.name), new_shape);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReshapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReshapeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_new_shape;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'new_shape'");
    if (namedAttrIt->getName() == getNewShapeAttrName()) {
      tblgen_new_shape = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps15(*this, tblgen_new_shape, "new_shape")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps13(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReshapeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ReshapeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ReshapeOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ResizeOp definitions
//===----------------------------------------------------------------------===//

ResizeOpAdaptor::ResizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.resize", odsAttrs.getContext());
}

ResizeOpAdaptor::ResizeOpAdaptor(ResizeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ResizeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ResizeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResizeOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ResizeOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ResizeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ResizeOpAdaptor::getScaleAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, ResizeOp::getScaleAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ResizeOpAdaptor::getScale() {
  auto attr = getScaleAttr();
  return attr;
}

::mlir::ArrayAttr ResizeOpAdaptor::getOffsetAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, ResizeOp::getOffsetAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ResizeOpAdaptor::getOffset() {
  auto attr = getOffsetAttr();
  return attr;
}

::mlir::ArrayAttr ResizeOpAdaptor::getBorderAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, ResizeOp::getBorderAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ResizeOpAdaptor::getBorder() {
  auto attr = getBorderAttr();
  return attr;
}

::mlir::StringAttr ResizeOpAdaptor::getModeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 2, ResizeOp::getModeAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef ResizeOpAdaptor::getMode() {
  auto attr = getModeAttr();
  return attr.getValue();
}

::mlir::LogicalResult ResizeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_border;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.resize' op ""requires attribute 'border'");
    if (namedAttrIt->getName() == ResizeOp::getBorderAttrName(*odsOpName)) {
      tblgen_border = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_mode;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.resize' op ""requires attribute 'mode'");
    if (namedAttrIt->getName() == ResizeOp::getModeAttrName(*odsOpName)) {
      tblgen_mode = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_offset;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.resize' op ""requires attribute 'offset'");
    if (namedAttrIt->getName() == ResizeOp::getOffsetAttrName(*odsOpName)) {
      tblgen_offset = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_scale;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.resize' op ""requires attribute 'scale'");
    if (namedAttrIt->getName() == ResizeOp::getScaleAttrName(*odsOpName)) {
      tblgen_scale = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_scale && !((((tblgen_scale.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_scale.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_scale.cast<::mlir::ArrayAttr>().size() == 4))))
    return emitError(loc, "'tosa.resize' op ""attribute 'scale' failed to satisfy constraint: 64-bit integer array attribute with exactly 4 elements");

  if (tblgen_offset && !((((tblgen_offset.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_offset.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_offset.cast<::mlir::ArrayAttr>().size() == 2))))
    return emitError(loc, "'tosa.resize' op ""attribute 'offset' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");

  if (tblgen_border && !((((tblgen_border.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_border.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_border.cast<::mlir::ArrayAttr>().size() == 2))))
    return emitError(loc, "'tosa.resize' op ""attribute 'border' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");

  if (tblgen_mode && !((tblgen_mode.cast<StringAttr>().getValue() == "BILINEAR"  || tblgen_mode.cast<StringAttr>().getValue() == "NEAREST_NEIGHBOR")))
    return emitError(loc, "'tosa.resize' op ""attribute 'mode' failed to satisfy constraint: Supported resize/upsampling strategies");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ResizeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ResizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResizeOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ResizeOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ResizeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ResizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResizeOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr ResizeOp::getScaleAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getScaleAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ResizeOp::getScale() {
  auto attr = getScaleAttr();
  return attr;
}

::mlir::ArrayAttr ResizeOp::getOffsetAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getOffsetAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ResizeOp::getOffset() {
  auto attr = getOffsetAttr();
  return attr;
}

::mlir::ArrayAttr ResizeOp::getBorderAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getBorderAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ResizeOp::getBorder() {
  auto attr = getBorderAttr();
  return attr;
}

::mlir::StringAttr ResizeOp::getModeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 2, getModeAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef ResizeOp::getMode() {
  auto attr = getModeAttr();
  return attr.getValue();
}

void ResizeOp::setScaleAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getScaleAttrName(), attr);
}

void ResizeOp::setOffsetAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOffsetAttrName(), attr);
}

void ResizeOp::setBorderAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getBorderAttrName(), attr);
}

void ResizeOp::setModeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getModeAttrName(), attr);
}

void ResizeOp::setMode(::llvm::StringRef attrValue) {
  (*this)->setAttr(getModeAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void ResizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::ArrayAttr scale, ::mlir::ArrayAttr offset, ::mlir::ArrayAttr border, ::mlir::StringAttr mode) {
  odsState.addOperands(input);
  odsState.addAttribute(getScaleAttrName(odsState.name), scale);
  odsState.addAttribute(getOffsetAttrName(odsState.name), offset);
  odsState.addAttribute(getBorderAttrName(odsState.name), border);
  odsState.addAttribute(getModeAttrName(odsState.name), mode);
  odsState.addTypes(output);
}

void ResizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr scale, ::mlir::ArrayAttr offset, ::mlir::ArrayAttr border, ::mlir::StringAttr mode) {
  odsState.addOperands(input);
  odsState.addAttribute(getScaleAttrName(odsState.name), scale);
  odsState.addAttribute(getOffsetAttrName(odsState.name), offset);
  odsState.addAttribute(getBorderAttrName(odsState.name), border);
  odsState.addAttribute(getModeAttrName(odsState.name), mode);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::ArrayAttr scale, ::mlir::ArrayAttr offset, ::mlir::ArrayAttr border, ::llvm::StringRef mode) {
  odsState.addOperands(input);
  odsState.addAttribute(getScaleAttrName(odsState.name), scale);
  odsState.addAttribute(getOffsetAttrName(odsState.name), offset);
  odsState.addAttribute(getBorderAttrName(odsState.name), border);
  odsState.addAttribute(getModeAttrName(odsState.name), odsBuilder.getStringAttr(mode));
  odsState.addTypes(output);
}

void ResizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr scale, ::mlir::ArrayAttr offset, ::mlir::ArrayAttr border, ::llvm::StringRef mode) {
  odsState.addOperands(input);
  odsState.addAttribute(getScaleAttrName(odsState.name), scale);
  odsState.addAttribute(getOffsetAttrName(odsState.name), offset);
  odsState.addAttribute(getBorderAttrName(odsState.name), border);
  odsState.addAttribute(getModeAttrName(odsState.name), odsBuilder.getStringAttr(mode));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ResizeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_border;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'border'");
    if (namedAttrIt->getName() == getBorderAttrName()) {
      tblgen_border = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_mode;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'mode'");
    if (namedAttrIt->getName() == getModeAttrName()) {
      tblgen_mode = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_offset;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'offset'");
    if (namedAttrIt->getName() == getOffsetAttrName()) {
      tblgen_offset = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_scale;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'scale'");
    if (namedAttrIt->getName() == getScaleAttrName()) {
      tblgen_scale = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(*this, tblgen_scale, "scale")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_offset, "offset")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_border, "border")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps16(*this, tblgen_mode, "mode")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ResizeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ResizeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ResizeOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReverseOp definitions
//===----------------------------------------------------------------------===//

ReverseOpAdaptor::ReverseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reverse", odsAttrs.getContext());
}

ReverseOpAdaptor::ReverseOpAdaptor(ReverseOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ReverseOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReverseOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReverseOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ReverseOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ReverseOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReverseOpAdaptor::getAxisAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ReverseOp::getAxisAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ReverseOpAdaptor::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult ReverseOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.reverse' op ""requires attribute 'axis'");
    if (namedAttrIt->getName() == ReverseOp::getAxisAttrName(*odsOpName)) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_axis && !(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tosa.reverse' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReverseOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReverseOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReverseOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReverseOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReverseOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReverseOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReverseOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ReverseOp::getAxisAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAxisAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ReverseOp::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

void ReverseOp::setAxisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAxisAttrName(), attr);
}

void ReverseOp::setAxis(uint64_t attrValue) {
  (*this)->setAttr(getAxisAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void ReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  odsState.addTypes(output);
}

void ReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addTypes(output);
}

void ReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.addAttribute(getAxisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReverseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReverseOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'axis'");
    if (namedAttrIt->getName() == getAxisAttrName()) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(*this, tblgen_axis, "axis")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReverseOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ReverseOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ReverseOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::RsqrtOp definitions
//===----------------------------------------------------------------------===//

RsqrtOpAdaptor::RsqrtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.rsqrt", odsAttrs.getContext());
}

RsqrtOpAdaptor::RsqrtOpAdaptor(RsqrtOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> RsqrtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange RsqrtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RsqrtOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange RsqrtOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr RsqrtOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult RsqrtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RsqrtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range RsqrtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> RsqrtOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange RsqrtOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> RsqrtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RsqrtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> RsqrtOp::getOutput() {
  return *getODSResults(0).begin();
}

void RsqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1) {
  odsState.addOperands(input1);
  odsState.addTypes(output);
}

void RsqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1) {
  odsState.addOperands(input1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RsqrtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RsqrtOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult RsqrtOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void RsqrtOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::RsqrtOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ScatterOp definitions
//===----------------------------------------------------------------------===//

ScatterOpAdaptor::ScatterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.scatter", odsAttrs.getContext());
}

ScatterOpAdaptor::ScatterOpAdaptor(ScatterOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ScatterOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScatterOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScatterOpAdaptor::getValuesIn() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScatterOpAdaptor::getIndices() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScatterOpAdaptor::getInput() {
  return *getODSOperands(2).begin();
}

::mlir::ValueRange ScatterOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ScatterOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScatterOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ScatterOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScatterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScatterOp::getValuesIn() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> ScatterOp::getIndices() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScatterOp::getInput() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ScatterOp::getValuesInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScatterOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScatterOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScatterOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScatterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScatterOp::getValuesOut() {
  return *getODSResults(0).begin();
}

void ScatterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type values_out, ::mlir::Value values_in, ::mlir::Value indices, ::mlir::Value input) {
  odsState.addOperands(values_in);
  odsState.addOperands(indices);
  odsState.addOperands(input);
  odsState.addTypes(values_out);
}

void ScatterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value values_in, ::mlir::Value indices, ::mlir::Value input) {
  odsState.addOperands(values_in);
  odsState.addOperands(indices);
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScatterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScatterOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps11(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ScatterOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ScatterOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ScatterOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::SelectOp definitions
//===----------------------------------------------------------------------===//

SelectOpAdaptor::SelectOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.select", odsAttrs.getContext());
}

SelectOpAdaptor::SelectOpAdaptor(SelectOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> SelectOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SelectOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SelectOpAdaptor::getPred() {
  return *getODSOperands(0).begin();
}

::mlir::Value SelectOpAdaptor::getOnTrue() {
  return *getODSOperands(1).begin();
}

::mlir::Value SelectOpAdaptor::getOnFalse() {
  return *getODSOperands(2).begin();
}

::mlir::ValueRange SelectOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr SelectOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SelectOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SelectOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SelectOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> SelectOp::getPred() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> SelectOp::getOnTrue() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> SelectOp::getOnFalse() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange SelectOp::getPredMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SelectOp::getOnTrueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SelectOp::getOnFalseMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SelectOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SelectOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> SelectOp::getOutput() {
  return *getODSResults(0).begin();
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value pred, ::mlir::Value on_true, ::mlir::Value on_false) {
  odsState.addOperands(pred);
  odsState.addOperands(on_true);
  odsState.addOperands(on_false);
  odsState.addTypes(output);
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value pred, ::mlir::Value on_true, ::mlir::Value on_false) {
  odsState.addOperands(pred);
  odsState.addOperands(on_true);
  odsState.addOperands(on_false);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SelectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SelectOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SelectOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void SelectOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add(canonicalize);
}

void SelectOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::SelectOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::SigmoidOp definitions
//===----------------------------------------------------------------------===//

SigmoidOpAdaptor::SigmoidOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.sigmoid", odsAttrs.getContext());
}

SigmoidOpAdaptor::SigmoidOpAdaptor(SigmoidOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> SigmoidOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SigmoidOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SigmoidOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange SigmoidOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr SigmoidOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SigmoidOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SigmoidOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SigmoidOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> SigmoidOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SigmoidOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SigmoidOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SigmoidOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> SigmoidOp::getOutput() {
  return *getODSResults(0).begin();
}

void SigmoidOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(output);
}

void SigmoidOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SigmoidOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SigmoidOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SigmoidOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void SigmoidOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::SigmoidOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::SliceOp definitions
//===----------------------------------------------------------------------===//

SliceOpAdaptor::SliceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.slice", odsAttrs.getContext());
}

SliceOpAdaptor::SliceOpAdaptor(SliceOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> SliceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SliceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SliceOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange SliceOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr SliceOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SliceOpAdaptor::getStartAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SliceOp::getStartAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SliceOpAdaptor::getStart() {
  auto attr = getStartAttr();
  return attr;
}

::mlir::ArrayAttr SliceOpAdaptor::getSizeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SliceOp::getSizeAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SliceOpAdaptor::getSize() {
  auto attr = getSizeAttr();
  return attr;
}

::mlir::LogicalResult SliceOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_size;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.slice' op ""requires attribute 'size'");
    if (namedAttrIt->getName() == SliceOp::getSizeAttrName(*odsOpName)) {
      tblgen_size = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_start;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.slice' op ""requires attribute 'start'");
    if (namedAttrIt->getName() == SliceOp::getStartAttrName(*odsOpName)) {
      tblgen_start = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_start && !(((tblgen_start.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_start.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'tosa.slice' op ""attribute 'start' failed to satisfy constraint: 64-bit integer array attribute");

  if (tblgen_size && !(((tblgen_size.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_size.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'tosa.slice' op ""attribute 'size' failed to satisfy constraint: 64-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SliceOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SliceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SliceOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SliceOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SliceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SliceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SliceOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr SliceOp::getStartAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStartAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SliceOp::getStart() {
  auto attr = getStartAttr();
  return attr;
}

::mlir::ArrayAttr SliceOp::getSizeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getSizeAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SliceOp::getSize() {
  auto attr = getSizeAttr();
  return attr;
}

void SliceOp::setStartAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getStartAttrName(), attr);
}

void SliceOp::setSizeAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getSizeAttrName(), attr);
}

void SliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::ArrayAttr start, ::mlir::ArrayAttr size) {
  odsState.addOperands(input);
  odsState.addAttribute(getStartAttrName(odsState.name), start);
  odsState.addAttribute(getSizeAttrName(odsState.name), size);
  odsState.addTypes(output);
}

void SliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ArrayAttr start, ::mlir::ArrayAttr size) {
  odsState.addOperands(input);
  odsState.addAttribute(getStartAttrName(odsState.name), start);
  odsState.addAttribute(getSizeAttrName(odsState.name), size);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SliceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SliceOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_size;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'size'");
    if (namedAttrIt->getName() == getSizeAttrName()) {
      tblgen_size = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_start;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'start'");
    if (namedAttrIt->getName() == getStartAttrName()) {
      tblgen_start = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps15(*this, tblgen_start, "start")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps15(*this, tblgen_size, "size")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps16(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps16(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SliceOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void SliceOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::SliceOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::SubOp definitions
//===----------------------------------------------------------------------===//

SubOpAdaptor::SubOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.sub", odsAttrs.getContext());
}

SubOpAdaptor::SubOpAdaptor(SubOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> SubOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SubOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubOpAdaptor::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange SubOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr SubOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SubOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SubOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SubOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> SubOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> SubOp::getInput2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SubOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SubOp::getInput2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SubOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> SubOp::getOutput() {
  return *getODSResults(0).begin();
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.addTypes(output);
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SubOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SubOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void SubOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::SubOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::TableOp definitions
//===----------------------------------------------------------------------===//

TableOpAdaptor::TableOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.table", odsAttrs.getContext());
}

TableOpAdaptor::TableOpAdaptor(TableOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> TableOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TableOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TableOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::Value TableOpAdaptor::getTable() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange TableOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr TableOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TableOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TableOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TableOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> TableOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::Value TableOp::getTable() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange TableOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TableOp::getTableMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TableOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TableOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> TableOp::getOutput() {
  return *getODSResults(0).begin();
}

void TableOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value table) {
  odsState.addOperands(input);
  odsState.addOperands(table);
  odsState.addTypes(output);
}

void TableOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value table) {
  odsState.addOperands(input);
  odsState.addOperands(table);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TableOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TableOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TableOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void TableOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::TableOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::TanhOp definitions
//===----------------------------------------------------------------------===//

TanhOpAdaptor::TanhOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.tanh", odsAttrs.getContext());
}

TanhOpAdaptor::TanhOpAdaptor(TanhOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> TanhOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TanhOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TanhOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange TanhOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr TanhOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TanhOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TanhOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TanhOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> TanhOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TanhOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TanhOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TanhOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> TanhOp::getOutput() {
  return *getODSResults(0).begin();
}

void TanhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(output);
}

void TanhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TanhOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TanhOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TanhOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void TanhOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::TanhOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::TileOp definitions
//===----------------------------------------------------------------------===//

TileOpAdaptor::TileOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.tile", odsAttrs.getContext());
}

TileOpAdaptor::TileOpAdaptor(TileOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> TileOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TileOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TileOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange TileOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr TileOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr TileOpAdaptor::getMultiplesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TileOp::getMultiplesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TileOpAdaptor::getMultiples() {
  auto attr = getMultiplesAttr();
  return attr;
}

::mlir::LogicalResult TileOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_multiples;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.tile' op ""requires attribute 'multiples'");
    if (namedAttrIt->getName() == TileOp::getMultiplesAttrName(*odsOpName)) {
      tblgen_multiples = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_multiples && !(((tblgen_multiples.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_multiples.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'tosa.tile' op ""attribute 'multiples' failed to satisfy constraint: 64-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TileOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TileOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TileOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TileOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TileOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TileOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TileOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr TileOp::getMultiplesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getMultiplesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TileOp::getMultiples() {
  auto attr = getMultiplesAttr();
  return attr;
}

void TileOp::setMultiplesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getMultiplesAttrName(), attr);
}

void TileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::ArrayAttr multiples) {
  odsState.addOperands(input1);
  odsState.addAttribute(getMultiplesAttrName(odsState.name), multiples);
  odsState.addTypes(output);
}

void TileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::ArrayAttr multiples) {
  odsState.addOperands(input1);
  odsState.addAttribute(getMultiplesAttrName(odsState.name), multiples);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TileOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TileOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_multiples;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'multiples'");
    if (namedAttrIt->getName() == getMultiplesAttrName()) {
      tblgen_multiples = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps15(*this, tblgen_multiples, "multiples")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TileOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void TileOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::TileOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::TransposeConv2DOp definitions
//===----------------------------------------------------------------------===//

TransposeConv2DOpAdaptor::TransposeConv2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.transpose_conv2d", odsAttrs.getContext());
}

TransposeConv2DOpAdaptor::TransposeConv2DOpAdaptor(TransposeConv2DOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> TransposeConv2DOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TransposeConv2DOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeConv2DOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransposeConv2DOpAdaptor::getFilter() {
  return *getODSOperands(1).begin();
}

::mlir::Value TransposeConv2DOpAdaptor::getBias() {
  return *getODSOperands(2).begin();
}

::mlir::ValueRange TransposeConv2DOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr TransposeConv2DOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr TransposeConv2DOpAdaptor::getOutPadAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, TransposeConv2DOp::getOutPadAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TransposeConv2DOpAdaptor::getOutPad() {
  auto attr = getOutPadAttr();
  return attr;
}

::mlir::ArrayAttr TransposeConv2DOpAdaptor::getStrideAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, TransposeConv2DOp::getStrideAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TransposeConv2DOpAdaptor::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::ArrayAttr TransposeConv2DOpAdaptor::getOutShapeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, TransposeConv2DOp::getOutShapeAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TransposeConv2DOpAdaptor::getOutShape() {
  auto attr = getOutShapeAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr TransposeConv2DOpAdaptor::getQuantizationInfoAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, TransposeConv2DOp::getQuantizationInfoAttrName(*odsOpName)).dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> TransposeConv2DOpAdaptor::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(attr.cast<mlir::tosa::ConvOpQuantizationAttr>()) : (::std::nullopt);
}

::mlir::LogicalResult TransposeConv2DOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_out_pad;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.transpose_conv2d' op ""requires attribute 'out_pad'");
    if (namedAttrIt->getName() == TransposeConv2DOp::getOutPadAttrName(*odsOpName)) {
      tblgen_out_pad = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_out_shape;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.transpose_conv2d' op ""requires attribute 'out_shape'");
    if (namedAttrIt->getName() == TransposeConv2DOp::getOutShapeAttrName(*odsOpName)) {
      tblgen_out_shape = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_stride;
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tosa.transpose_conv2d' op ""requires attribute 'stride'");
    if (namedAttrIt->getName() == TransposeConv2DOp::getStrideAttrName(*odsOpName)) {
      tblgen_stride = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == TransposeConv2DOp::getQuantizationInfoAttrName(*odsOpName)) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_out_pad && !((((tblgen_out_pad.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_out_pad.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_out_pad.cast<::mlir::ArrayAttr>().size() == 4))))
    return emitError(loc, "'tosa.transpose_conv2d' op ""attribute 'out_pad' failed to satisfy constraint: 64-bit integer array attribute with exactly 4 elements");

  if (tblgen_stride && !((((tblgen_stride.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_stride.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_stride.cast<::mlir::ArrayAttr>().size() == 2))))
    return emitError(loc, "'tosa.transpose_conv2d' op ""attribute 'stride' failed to satisfy constraint: 64-bit integer array attribute with exactly 2 elements");

  if (tblgen_out_shape && !((((tblgen_out_shape.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_out_shape.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))) && ((tblgen_out_shape.cast<::mlir::ArrayAttr>().size() <= 4))))
    return emitError(loc, "'tosa.transpose_conv2d' op ""attribute 'out_shape' failed to satisfy constraint: 64-bit integer array attribute with at least 4 elements");

  if (tblgen_quantization_info && !((tblgen_quantization_info.isa<mlir::tosa::ConvOpQuantizationAttr>())))
    return emitError(loc, "'tosa.transpose_conv2d' op ""attribute 'quantization_info' failed to satisfy constraint: Attribute for Conv type op quantization information.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TransposeConv2DOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TransposeConv2DOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeConv2DOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransposeConv2DOp::getFilter() {
  return *getODSOperands(1).begin();
}

::mlir::Value TransposeConv2DOp::getBias() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange TransposeConv2DOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TransposeConv2DOp::getFilterMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TransposeConv2DOp::getBiasMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TransposeConv2DOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TransposeConv2DOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeConv2DOp::getOutput() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr TransposeConv2DOp::getOutPadAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getOutPadAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TransposeConv2DOp::getOutPad() {
  auto attr = getOutPadAttr();
  return attr;
}

::mlir::ArrayAttr TransposeConv2DOp::getStrideAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getStrideAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TransposeConv2DOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::ArrayAttr TransposeConv2DOp::getOutShapeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOutShapeAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TransposeConv2DOp::getOutShape() {
  auto attr = getOutShapeAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr TransposeConv2DOp::getQuantizationInfoAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getQuantizationInfoAttrName()).dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> TransposeConv2DOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(attr.cast<mlir::tosa::ConvOpQuantizationAttr>()) : (::std::nullopt);
}

void TransposeConv2DOp::setOutPadAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOutPadAttrName(), attr);
}

void TransposeConv2DOp::setStrideAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getStrideAttrName(), attr);
}

void TransposeConv2DOp::setOutShapeAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOutShapeAttrName(), attr);
}

void TransposeConv2DOp::setQuantizationInfoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute TransposeConv2DOp::removeQuantizationInfoAttr() {
  return (*this)->removeAttr(getQuantizationInfoAttrName());
}

void TransposeConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input, Value weight, Value bias, ArrayAttr outpad, ArrayAttr stride, ArrayAttr outputShape) {
    buildTransConvOpWithQuantInfo(odsBuilder, odsState, outputType,
                                  input, weight, bias,
                                  outpad, stride,
                                  outputShape);
  
}

void TransposeConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::ArrayAttr out_pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr out_shape, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(bias);
  odsState.addAttribute(getOutPadAttrName(odsState.name), out_pad);
  odsState.addAttribute(getStrideAttrName(odsState.name), stride);
  odsState.addAttribute(getOutShapeAttrName(odsState.name), out_shape);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  odsState.addTypes(output);
}

void TransposeConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::ArrayAttr out_pad, ::mlir::ArrayAttr stride, ::mlir::ArrayAttr out_shape, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(bias);
  odsState.addAttribute(getOutPadAttrName(odsState.name), out_pad);
  odsState.addAttribute(getStrideAttrName(odsState.name), stride);
  odsState.addAttribute(getOutShapeAttrName(odsState.name), out_shape);
  if (quantization_info) {
    odsState.addAttribute(getQuantizationInfoAttrName(odsState.name), quantization_info);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeConv2DOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TransposeConv2DOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_out_pad;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'out_pad'");
    if (namedAttrIt->getName() == getOutPadAttrName()) {
      tblgen_out_pad = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_out_shape;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'out_shape'");
    if (namedAttrIt->getName() == getOutShapeAttrName()) {
      tblgen_out_shape = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_stride;
  ::mlir::Attribute tblgen_quantization_info;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'stride'");
    if (namedAttrIt->getName() == getStrideAttrName()) {
      tblgen_stride = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getQuantizationInfoAttrName()) {
      tblgen_quantization_info = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(*this, tblgen_out_pad, "out_pad")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_stride, "stride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps17(*this, tblgen_out_shape, "out_shape")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps7(*this, tblgen_quantization_info, "quantization_info")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TransposeConv2DOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void TransposeConv2DOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::TransposeConv2DOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::TransposeOp definitions
//===----------------------------------------------------------------------===//

TransposeOpAdaptor::TransposeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.transpose", odsAttrs.getContext());
}

TransposeOpAdaptor::TransposeOpAdaptor(TransposeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> TransposeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TransposeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeOpAdaptor::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransposeOpAdaptor::getPerms() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange TransposeOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr TransposeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TransposeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TransposeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TransposeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeOp::getInput1() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> TransposeOp::getPerms() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange TransposeOp::getInput1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TransposeOp::getPermsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TransposeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TransposeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeOp::getOutput() {
  return *getODSResults(0).begin();
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value perms) {
  odsState.addOperands(input1);
  odsState.addOperands(perms);
  odsState.addTypes(output);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value perms) {
  odsState.addOperands(input1);
  odsState.addOperands(perms);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TransposeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps16(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps16(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TransposeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void TransposeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::TransposeOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::WhileOp definitions
//===----------------------------------------------------------------------===//

WhileOpAdaptor::WhileOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.while_loop", odsAttrs.getContext());
}

WhileOpAdaptor::WhileOpAdaptor(WhileOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> WhileOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange WhileOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WhileOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange WhileOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr WhileOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::Region &WhileOpAdaptor::getCond() {
  return *odsRegions[0];
}

::mlir::Region &WhileOpAdaptor::getBody() {
  return *odsRegions[1];
}

::mlir::RegionRange WhileOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::LogicalResult WhileOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WhileOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range WhileOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WhileOp::getInputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange WhileOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WhileOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range WhileOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range WhileOp::getOutput() {
  return getODSResults(0);
}

::mlir::Region &WhileOp::getCond() {
  return (*this)->getRegion(0);
}

::mlir::Region &WhileOp::getBody() {
  return (*this)->getRegion(1);
}

void WhileOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WhileOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_TosaOps0(*this, region, "cond", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_TosaOps0(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult WhileOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::WhileOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::YieldOp definitions
//===----------------------------------------------------------------------===//

YieldOpAdaptor::YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.yield", odsAttrs.getContext());
}

YieldOpAdaptor::YieldOpAdaptor(YieldOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> YieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange YieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange YieldOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange YieldOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr YieldOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult YieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> YieldOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range YieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range YieldOp::getInputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange YieldOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> YieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range YieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
}

void YieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult YieldOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult YieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void YieldOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::YieldOp)


#endif  // GET_OP_CLASSES

