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
mlir::tosa::FFT2dOp,
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
mlir::tosa::RFFT2dOp,
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
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isSignlessInteger(1))) || ((type.isUnsignedInteger(8))) || ((type.isUnsignedInteger(16))) || (((type.isSignlessInteger(8))) || ((type.isSignlessInteger(16))) || ((type.isSignlessInteger(32))) || ((type.isSignlessInteger(48))) || ((type.isSignlessInteger(64))))) || ((((::llvm::isa<::mlir::VectorType>(type))) && ((::llvm::cast<::mlir::VectorType>(type).getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64)))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || (((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64)))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be signless-integer-like, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isSignlessInteger(8))) || ((((::llvm::isa<::mlir::VectorType>(type))) && ((::llvm::cast<::mlir::VectorType>(type).getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || (((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be signless-integer-8-bit-like, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::UnrankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((((::llvm::isa<::mlir::RankedTensorType>(type))) && (((::llvm::cast<::mlir::ShapedType>(type).hasRank())) && (((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 1)) || ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 2)) || ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 3)) || ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 4))))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked tensor of number values or 1D/2D/3D/4D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::UnrankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((((::llvm::isa<::mlir::RankedTensorType>(type))) && (((::llvm::cast<::mlir::ShapedType>(type).hasRank())) && (((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 0)) || ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 1)) || ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 2)) || ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 3)) || ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 4))))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked tensor of number values or 0D/1D/2D/3D/4D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::UnrankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((((::llvm::isa<::mlir::RankedTensorType>(type))) && (((::llvm::cast<::mlir::ShapedType>(type).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 4)))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked tensor of number values or 4D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))) || ((elementType.isF64())); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of number_plus_f64 values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::UnrankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((((::llvm::isa<::mlir::RankedTensorType>(type))) && (((::llvm::cast<::mlir::ShapedType>(type).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 1)))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked tensor of number values or 1D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::UnrankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((((::llvm::isa<::mlir::RankedTensorType>(type))) && (((::llvm::cast<::mlir::ShapedType>(type).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 5)))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked tensor of number values or 5D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 32-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps10(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 1-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps11(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::UnrankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((((::llvm::isa<::mlir::RankedTensorType>(type))) && (((::llvm::cast<::mlir::ShapedType>(type).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 3)))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked tensor of number values or 3D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps12(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::UnrankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((((::llvm::isa<::mlir::RankedTensorType>(type))) && (((::llvm::cast<::mlir::ShapedType>(type).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 2)))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked tensor of number values or 2D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps13(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::RankedTensorType>(type))) && (((::llvm::cast<::mlir::ShapedType>(type).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 2)))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 2D tensor of 32-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps14(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps15(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 32-bit signless integer or 64-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps16(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::RankedTensorType>(type))) && (((::llvm::cast<::mlir::ShapedType>(type).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 0)))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 0D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TosaOps17(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::UnrankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((((::llvm::isa<::mlir::RankedTensorType>(type))) && (((::llvm::cast<::mlir::ShapedType>(type).hasRank())) && (((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 1)) || ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 2)) || ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 3)) || ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 4)) || ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 5)) || ((::llvm::cast<::mlir::ShapedType>(type).getRank()
                         == 6))))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(1))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || (((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(48))) || ((elementType.isSignlessInteger(64))))) || ((((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 4))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 8))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 16))) || (((::llvm::isa<mlir::quant::QuantizedType>(elementType))) && ((::llvm::cast<mlir::quant::QuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)))) || (((elementType.isF32())) || ((elementType.isF16())) || ((elementType.isBF16()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked tensor of number values or 1D/2D/3D/4D/5D/6D tensor of number values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps0(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<::mlir::BoolAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: bool attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps0(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(attr))) && ((::llvm::cast<::mlir::DenseArrayAttr>(attr).size() == 2))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: i64 dense array attribute with exactly 2 elements";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(attr))) && ((::llvm::cast<::mlir::DenseArrayAttr>(attr).size() == 4))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: i64 dense array attribute with exactly 4 elements";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && (((::llvm::cast<::mlir::TypeAttr>(attr).getValue().isSignlessInteger(32))) || ((::llvm::cast<::mlir::TypeAttr>(attr).getValue().isSignedInteger(32))) || ((::llvm::cast<::mlir::TypeAttr>(attr).getValue().isF16())) || ((::llvm::cast<::mlir::TypeAttr>(attr).getValue().isF32())))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: type attribute of 32-bit signless integer or 32-bit signed integer or 16-bit float or 32-bit float";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps5(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<mlir::tosa::UnaryOpQuantizationAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: Attribute for UnaryOp quantization information.";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps5(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps6(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::FloatAttr>(attr))) && ((::llvm::cast<::mlir::FloatAttr>(attr).getType().isF32()))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit float attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps6(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps7(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<::mlir::ElementsAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: constant vector/tensor attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps7(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps8(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<mlir::tosa::ConvOpQuantizationAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: Attribute for Conv type op quantization information.";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps8(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps9(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(attr))) && ((::llvm::cast<::mlir::DenseArrayAttr>(attr).size() == 6))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: i64 dense array attribute with exactly 6 elements";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps9(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps10(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(attr))) && ((::llvm::cast<::mlir::DenseArrayAttr>(attr).size() == 3))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: i64 dense array attribute with exactly 3 elements";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps10(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps11(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps11(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps12(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<mlir::tosa::MatMulOpQuantizationAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: Attribute for MatMulOp quantization information.";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps12(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps12(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps13(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps13(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps13(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps14(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<mlir::tosa::PadOpQuantizationAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: Attribute for PadOp quantization information.";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps14(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps14(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps15(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<::mlir::DenseI32ArrayAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: i32 dense array attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps15(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps15(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps16(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<::mlir::DenseI64ArrayAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: i64 dense array attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps16(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps16(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps17(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::cast<StringAttr>(attr).getValue() == "BILINEAR"  || ::llvm::cast<StringAttr>(attr).getValue() == "NEAREST_NEIGHBOR")))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: Supported resize/upsampling strategies";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps17(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps17(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps18(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(attr))) && ((::llvm::cast<::mlir::DenseArrayAttr>(attr).size() <= 4))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: i64 dense array attribute with at least 4 elements";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TosaOps18(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TosaOps18(attr, attrName, [op]() {
    return op->emitOpError();
  });
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

namespace detail {
AbsOpGenericAdaptorBase::AbsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.abs", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AbsOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AbsOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AbsOpAdaptor::AbsOpAdaptor(AbsOp op) : AbsOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
AddOpGenericAdaptorBase::AddOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.add", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AddOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AddOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AddOpAdaptor::AddOpAdaptor(AddOp op) : AddOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> AddOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
ApplyScaleOpGenericAdaptorBase::ApplyScaleOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.apply_scale", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ApplyScaleOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ApplyScaleOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr ApplyScaleOpGenericAdaptorBase::getDoubleRoundAttr() {
  auto attr = getProperties().double_round.cast<::mlir::BoolAttr>();
  return attr;
}

bool ApplyScaleOpGenericAdaptorBase::getDoubleRound() {
  auto attr = getDoubleRoundAttr();
  return attr.getValue();
}

} // namespace detail
ApplyScaleOpAdaptor::ApplyScaleOpAdaptor(ApplyScaleOp op) : ApplyScaleOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ApplyScaleOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_double_round = getProperties().double_round; (void)tblgen_double_round;
  if (!tblgen_double_round) return emitError(loc, "'tosa.apply_scale' op ""requires attribute 'double_round'");

  if (tblgen_double_round && !((::llvm::isa<::mlir::BoolAttr>(tblgen_double_round))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value ApplyScaleOp::getMultiplier() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::Value ApplyScaleOp::getShift() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(2).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult ApplyScaleOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.double_round;
    auto attr = dict.get("double_round");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for double_round in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `double_round` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ApplyScaleOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.double_round;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("double_round",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ApplyScaleOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.double_round.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ApplyScaleOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "double_round")
      return prop.double_round;
  return std::nullopt;
}

void ApplyScaleOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "double_round") {
       prop.double_round = dyn_cast_or_null<std::remove_reference_t<decltype(prop.double_round)>>(value);
       return;
    }
}

void ApplyScaleOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.double_round) attrs.append("double_round", prop.double_round);
}

::mlir::LogicalResult ApplyScaleOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getDoubleRoundAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps0(attr, "double_round", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::BoolAttr ApplyScaleOp::getDoubleRoundAttr() {
  return getProperties().double_round.cast<::mlir::BoolAttr>();
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
  odsState.getOrAddProperties<Properties>().double_round = double_round;
  odsState.addTypes(output);
}

void ApplyScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value multiplier, ::mlir::Value shift, ::mlir::BoolAttr double_round) {
  odsState.addOperands(value);
  odsState.addOperands(multiplier);
  odsState.addOperands(shift);
  odsState.getOrAddProperties<Properties>().double_round = double_round;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value value, ::mlir::Value multiplier, ::mlir::Value shift, bool double_round) {
  odsState.addOperands(value);
  odsState.addOperands(multiplier);
  odsState.addOperands(shift);
  odsState.getOrAddProperties<Properties>().double_round = odsBuilder.getBoolAttr(double_round);
  odsState.addTypes(output);
}

void ApplyScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value multiplier, ::mlir::Value shift, bool double_round) {
  odsState.addOperands(value);
  odsState.addOperands(multiplier);
  odsState.addOperands(shift);
  odsState.getOrAddProperties<Properties>().double_round = odsBuilder.getBoolAttr(double_round);
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
  auto tblgen_double_round = getProperties().double_round; (void)tblgen_double_round;
  if (!tblgen_double_round) return emitOpError("requires attribute 'double_round'");

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

namespace detail {
ArgMaxOpGenericAdaptorBase::ArgMaxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.argmax", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ArgMaxOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ArgMaxOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ArgMaxOpGenericAdaptorBase::getAxisAttr() {
  auto attr = getProperties().axis.cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ArgMaxOpGenericAdaptorBase::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ArgMaxOpAdaptor::ArgMaxOpAdaptor(ArgMaxOp op) : ArgMaxOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ArgMaxOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitError(loc, "'tosa.argmax' op ""requires attribute 'axis'");

  if (tblgen_axis && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_axis))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_axis).getType().isSignlessInteger(64)))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult ArgMaxOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.axis;
    auto attr = dict.get("axis");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for axis in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `axis` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ArgMaxOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.axis;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("axis",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ArgMaxOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.axis.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ArgMaxOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "axis")
      return prop.axis;
  return std::nullopt;
}

void ArgMaxOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "axis") {
       prop.axis = dyn_cast_or_null<std::remove_reference_t<decltype(prop.axis)>>(value);
       return;
    }
}

void ArgMaxOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.axis) attrs.append("axis", prop.axis);
}

::mlir::LogicalResult ArgMaxOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getAxisAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(attr, "axis", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr ArgMaxOp::getAxisAttr() {
  return getProperties().axis.cast<::mlir::IntegerAttr>();
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
  odsState.getOrAddProperties<Properties>().axis = axis;
  odsState.addTypes(output);
}

void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = axis;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
  odsState.addTypes(output);
}

void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
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
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitOpError("requires attribute 'axis'");

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

namespace detail {
ArithmeticRightShiftOpGenericAdaptorBase::ArithmeticRightShiftOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.arithmetic_right_shift", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ArithmeticRightShiftOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ArithmeticRightShiftOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr ArithmeticRightShiftOpGenericAdaptorBase::getRoundAttr() {
  auto attr = getProperties().round.cast<::mlir::BoolAttr>();
  return attr;
}

bool ArithmeticRightShiftOpGenericAdaptorBase::getRound() {
  auto attr = getRoundAttr();
  return attr.getValue();
}

} // namespace detail
ArithmeticRightShiftOpAdaptor::ArithmeticRightShiftOpAdaptor(ArithmeticRightShiftOp op) : ArithmeticRightShiftOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ArithmeticRightShiftOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_round = getProperties().round; (void)tblgen_round;
  if (!tblgen_round) return emitError(loc, "'tosa.arithmetic_right_shift' op ""requires attribute 'round'");

  if (tblgen_round && !((::llvm::isa<::mlir::BoolAttr>(tblgen_round))))
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> ArithmeticRightShiftOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult ArithmeticRightShiftOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.round;
    auto attr = dict.get("round");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for round in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `round` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ArithmeticRightShiftOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.round;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("round",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ArithmeticRightShiftOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.round.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ArithmeticRightShiftOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "round")
      return prop.round;
  return std::nullopt;
}

void ArithmeticRightShiftOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "round") {
       prop.round = dyn_cast_or_null<std::remove_reference_t<decltype(prop.round)>>(value);
       return;
    }
}

void ArithmeticRightShiftOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.round) attrs.append("round", prop.round);
}

::mlir::LogicalResult ArithmeticRightShiftOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getRoundAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps0(attr, "round", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::BoolAttr ArithmeticRightShiftOp::getRoundAttr() {
  return getProperties().round.cast<::mlir::BoolAttr>();
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
  odsState.getOrAddProperties<Properties>().round = round;
  odsState.addTypes(output);
}

void ArithmeticRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::BoolAttr round) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.getOrAddProperties<Properties>().round = round;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ArithmeticRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, bool round) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.getOrAddProperties<Properties>().round = odsBuilder.getBoolAttr(round);
  odsState.addTypes(output);
}

void ArithmeticRightShiftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, bool round) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.getOrAddProperties<Properties>().round = odsBuilder.getBoolAttr(round);
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
  auto tblgen_round = getProperties().round; (void)tblgen_round;
  if (!tblgen_round) return emitOpError("requires attribute 'round'");

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

namespace detail {
AvgPool2dOpGenericAdaptorBase::AvgPool2dOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.avg_pool2d", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AvgPool2dOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AvgPool2dOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr AvgPool2dOpGenericAdaptorBase::getKernelAttr() {
  auto attr = getProperties().kernel.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> AvgPool2dOpGenericAdaptorBase::getKernel() {
  auto attr = getKernelAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr AvgPool2dOpGenericAdaptorBase::getStrideAttr() {
  auto attr = getProperties().stride.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> AvgPool2dOpGenericAdaptorBase::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr AvgPool2dOpGenericAdaptorBase::getPadAttr() {
  auto attr = getProperties().pad.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> AvgPool2dOpGenericAdaptorBase::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::TypeAttr AvgPool2dOpGenericAdaptorBase::getAccTypeAttr() {
  auto attr = getProperties().acc_type.cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::Type AvgPool2dOpGenericAdaptorBase::getAccType() {
  auto attr = getAccTypeAttr();
  return ::llvm::cast<::mlir::Type>(attr.getValue());
}

mlir::tosa::UnaryOpQuantizationAttr AvgPool2dOpGenericAdaptorBase::getQuantizationInfoAttr() {
  auto attr = getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::UnaryOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::UnaryOpQuantizationAttr> AvgPool2dOpGenericAdaptorBase::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::UnaryOpQuantizationAttr>(::llvm::cast<mlir::tosa::UnaryOpQuantizationAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
AvgPool2dOpAdaptor::AvgPool2dOpAdaptor(AvgPool2dOp op) : AvgPool2dOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult AvgPool2dOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_acc_type = getProperties().acc_type; (void)tblgen_acc_type;
  if (!tblgen_acc_type) return emitError(loc, "'tosa.avg_pool2d' op ""requires attribute 'acc_type'");
  auto tblgen_kernel = getProperties().kernel; (void)tblgen_kernel;
  if (!tblgen_kernel) return emitError(loc, "'tosa.avg_pool2d' op ""requires attribute 'kernel'");
  auto tblgen_pad = getProperties().pad; (void)tblgen_pad;
  if (!tblgen_pad) return emitError(loc, "'tosa.avg_pool2d' op ""requires attribute 'pad'");
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitError(loc, "'tosa.avg_pool2d' op ""requires attribute 'stride'");

  if (tblgen_kernel && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_kernel))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_kernel).size() == 2))))
    return emitError(loc, "'tosa.avg_pool2d' op ""attribute 'kernel' failed to satisfy constraint: i64 dense array attribute with exactly 2 elements");

  if (tblgen_stride && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_stride))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_stride).size() == 2))))
    return emitError(loc, "'tosa.avg_pool2d' op ""attribute 'stride' failed to satisfy constraint: i64 dense array attribute with exactly 2 elements");

  if (tblgen_pad && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_pad))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_pad).size() == 4))))
    return emitError(loc, "'tosa.avg_pool2d' op ""attribute 'pad' failed to satisfy constraint: i64 dense array attribute with exactly 4 elements");

  if (tblgen_acc_type && !(((::llvm::isa<::mlir::TypeAttr>(tblgen_acc_type))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(tblgen_acc_type).getValue()))) && (((::llvm::cast<::mlir::TypeAttr>(tblgen_acc_type).getValue().isSignlessInteger(32))) || ((::llvm::cast<::mlir::TypeAttr>(tblgen_acc_type).getValue().isSignedInteger(32))) || ((::llvm::cast<::mlir::TypeAttr>(tblgen_acc_type).getValue().isF16())) || ((::llvm::cast<::mlir::TypeAttr>(tblgen_acc_type).getValue().isF32())))))
    return emitError(loc, "'tosa.avg_pool2d' op ""attribute 'acc_type' failed to satisfy constraint: type attribute of 32-bit signless integer or 32-bit signed integer or 16-bit float or 32-bit float");

  if (tblgen_quantization_info && !((::llvm::isa<mlir::tosa::UnaryOpQuantizationAttr>(tblgen_quantization_info))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult AvgPool2dOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.acc_type;
    auto attr = dict.get("acc_type");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for acc_type in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `acc_type` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.kernel;
    auto attr = dict.get("kernel");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for kernel in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `kernel` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.pad;
    auto attr = dict.get("pad");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for pad in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `pad` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.quantization_info;
    auto attr = dict.get("quantization_info");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for quantization_info in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `quantization_info` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.stride;
    auto attr = dict.get("stride");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for stride in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `stride` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute AvgPool2dOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.acc_type;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("acc_type",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.kernel;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("kernel",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.pad;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("pad",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.quantization_info;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("quantization_info",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.stride;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("stride",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code AvgPool2dOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.acc_type.getAsOpaquePointer()), 
    llvm::hash_value(prop.kernel.getAsOpaquePointer()), 
    llvm::hash_value(prop.pad.getAsOpaquePointer()), 
    llvm::hash_value(prop.quantization_info.getAsOpaquePointer()), 
    llvm::hash_value(prop.stride.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AvgPool2dOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "acc_type")
      return prop.acc_type;

    if (name == "kernel")
      return prop.kernel;

    if (name == "pad")
      return prop.pad;

    if (name == "quantization_info")
      return prop.quantization_info;

    if (name == "stride")
      return prop.stride;
  return std::nullopt;
}

void AvgPool2dOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "acc_type") {
       prop.acc_type = dyn_cast_or_null<std::remove_reference_t<decltype(prop.acc_type)>>(value);
       return;
    }

    if (name == "kernel") {
       prop.kernel = dyn_cast_or_null<std::remove_reference_t<decltype(prop.kernel)>>(value);
       return;
    }

    if (name == "pad") {
       prop.pad = dyn_cast_or_null<std::remove_reference_t<decltype(prop.pad)>>(value);
       return;
    }

    if (name == "quantization_info") {
       prop.quantization_info = dyn_cast_or_null<std::remove_reference_t<decltype(prop.quantization_info)>>(value);
       return;
    }

    if (name == "stride") {
       prop.stride = dyn_cast_or_null<std::remove_reference_t<decltype(prop.stride)>>(value);
       return;
    }
}

void AvgPool2dOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.acc_type) attrs.append("acc_type", prop.acc_type);

    if (prop.kernel) attrs.append("kernel", prop.kernel);

    if (prop.pad) attrs.append("pad", prop.pad);

    if (prop.quantization_info) attrs.append("quantization_info", prop.quantization_info);

    if (prop.stride) attrs.append("stride", prop.stride);
}

::mlir::LogicalResult AvgPool2dOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getAccTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps4(attr, "acc_type", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getKernelAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(attr, "kernel", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPadAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(attr, "pad", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getQuantizationInfoAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps5(attr, "quantization_info", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStrideAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(attr, "stride", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::DenseI64ArrayAttr AvgPool2dOp::getKernelAttr() {
  return getProperties().kernel.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> AvgPool2dOp::getKernel() {
  auto attr = getKernelAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr AvgPool2dOp::getStrideAttr() {
  return getProperties().stride.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> AvgPool2dOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr AvgPool2dOp::getPadAttr() {
  return getProperties().pad.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> AvgPool2dOp::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::TypeAttr AvgPool2dOp::getAccTypeAttr() {
  return getProperties().acc_type.cast<::mlir::TypeAttr>();
}

::mlir::Type AvgPool2dOp::getAccType() {
  auto attr = getAccTypeAttr();
  return ::llvm::cast<::mlir::Type>(attr.getValue());
}

mlir::tosa::UnaryOpQuantizationAttr AvgPool2dOp::getQuantizationInfoAttr() {
  return getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::UnaryOpQuantizationAttr>();
}

::std::optional<mlir::tosa::UnaryOpQuantizationAttr> AvgPool2dOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::UnaryOpQuantizationAttr>(::llvm::cast<mlir::tosa::UnaryOpQuantizationAttr>(attr)) : (::std::nullopt);
}

void AvgPool2dOp::setKernelAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getKernelAttrName(), attr);
}

void AvgPool2dOp::setKernel(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getKernelAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void AvgPool2dOp::setStrideAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStrideAttrName(), attr);
}

void AvgPool2dOp::setStride(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStrideAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void AvgPool2dOp::setPadAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getPadAttrName(), attr);
}

void AvgPool2dOp::setPad(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getPadAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void AvgPool2dOp::setAccTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getAccTypeAttrName(), attr);
}

void AvgPool2dOp::setAccType(::mlir::Type attrValue) {
  (*this)->setAttr(getAccTypeAttrName(), ::mlir::TypeAttr::get(attrValue));
}

void AvgPool2dOp::setQuantizationInfoAttr(mlir::tosa::UnaryOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute AvgPool2dOp::removeQuantizationInfoAttr() {
    auto &attr = getProperties().quantization_info;
    attr = {};
    return attr;
}

void AvgPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputType, ::mlir::Value input, ::mlir::DenseI64ArrayAttr kernel, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr pad, ::mlir::TypeAttr acc_type) {
    buildAvgPool2dOpWithQuantInfo(odsBuilder, odsState, outputType,
                                  input, kernel, stride, pad, acc_type);
  
}

void AvgPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::DenseI64ArrayAttr kernel, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr pad, ::mlir::TypeAttr acc_type, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().kernel = kernel;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().pad = pad;
  odsState.getOrAddProperties<Properties>().acc_type = acc_type;
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  odsState.addTypes(output);
}

void AvgPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::DenseI64ArrayAttr kernel, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr pad, ::mlir::TypeAttr acc_type, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().kernel = kernel;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().pad = pad;
  odsState.getOrAddProperties<Properties>().acc_type = acc_type;
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AvgPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::llvm::ArrayRef<int64_t> kernel, ::llvm::ArrayRef<int64_t> stride, ::llvm::ArrayRef<int64_t> pad, ::mlir::Type acc_type, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().kernel = odsBuilder.getDenseI64ArrayAttr(kernel);
  odsState.getOrAddProperties<Properties>().stride = odsBuilder.getDenseI64ArrayAttr(stride);
  odsState.getOrAddProperties<Properties>().pad = odsBuilder.getDenseI64ArrayAttr(pad);
  odsState.getOrAddProperties<Properties>().acc_type = ::mlir::TypeAttr::get(acc_type);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  odsState.addTypes(output);
}

void AvgPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::llvm::ArrayRef<int64_t> kernel, ::llvm::ArrayRef<int64_t> stride, ::llvm::ArrayRef<int64_t> pad, ::mlir::Type acc_type, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().kernel = odsBuilder.getDenseI64ArrayAttr(kernel);
  odsState.getOrAddProperties<Properties>().stride = odsBuilder.getDenseI64ArrayAttr(stride);
  odsState.getOrAddProperties<Properties>().pad = odsBuilder.getDenseI64ArrayAttr(pad);
  odsState.getOrAddProperties<Properties>().acc_type = ::mlir::TypeAttr::get(acc_type);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
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
  auto tblgen_acc_type = getProperties().acc_type; (void)tblgen_acc_type;
  if (!tblgen_acc_type) return emitOpError("requires attribute 'acc_type'");
  auto tblgen_kernel = getProperties().kernel; (void)tblgen_kernel;
  if (!tblgen_kernel) return emitOpError("requires attribute 'kernel'");
  auto tblgen_pad = getProperties().pad; (void)tblgen_pad;
  if (!tblgen_pad) return emitOpError("requires attribute 'pad'");
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitOpError("requires attribute 'stride'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_kernel, "kernel")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_stride, "stride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(*this, tblgen_pad, "pad")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps4(*this, tblgen_acc_type, "acc_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps5(*this, tblgen_quantization_info, "quantization_info")))
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

namespace detail {
BitwiseAndOpGenericAdaptorBase::BitwiseAndOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.bitwise_and", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BitwiseAndOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr BitwiseAndOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
BitwiseAndOpAdaptor::BitwiseAndOpAdaptor(BitwiseAndOp op) : BitwiseAndOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> BitwiseAndOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
BitwiseNotOpGenericAdaptorBase::BitwiseNotOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.bitwise_not", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BitwiseNotOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr BitwiseNotOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
BitwiseNotOpAdaptor::BitwiseNotOpAdaptor(BitwiseNotOp op) : BitwiseNotOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
BitwiseOrOpGenericAdaptorBase::BitwiseOrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.bitwise_or", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BitwiseOrOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr BitwiseOrOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
BitwiseOrOpAdaptor::BitwiseOrOpAdaptor(BitwiseOrOp op) : BitwiseOrOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> BitwiseOrOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
BitwiseXorOpGenericAdaptorBase::BitwiseXorOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.bitwise_xor", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BitwiseXorOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr BitwiseXorOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
BitwiseXorOpAdaptor::BitwiseXorOpAdaptor(BitwiseXorOp op) : BitwiseXorOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> BitwiseXorOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
CastOpGenericAdaptorBase::CastOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.cast", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CastOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CastOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
CastOpAdaptor::CastOpAdaptor(CastOp op) : CastOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(*this, v.getType(), "result", index++)))
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

namespace detail {
CeilOpGenericAdaptorBase::CeilOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.ceil", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CeilOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CeilOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
CeilOpAdaptor::CeilOpAdaptor(CeilOp op) : CeilOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
ClampOpGenericAdaptorBase::ClampOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.clamp", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ClampOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ClampOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ClampOpGenericAdaptorBase::getMinIntAttr() {
  auto attr = getProperties().min_int.cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ClampOpGenericAdaptorBase::getMinInt() {
  auto attr = getMinIntAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ClampOpGenericAdaptorBase::getMaxIntAttr() {
  auto attr = getProperties().max_int.cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ClampOpGenericAdaptorBase::getMaxInt() {
  auto attr = getMaxIntAttr();
  return attr.getValue().getZExtValue();
}

::mlir::FloatAttr ClampOpGenericAdaptorBase::getMinFpAttr() {
  auto attr = getProperties().min_fp.cast<::mlir::FloatAttr>();
  return attr;
}

::llvm::APFloat ClampOpGenericAdaptorBase::getMinFp() {
  auto attr = getMinFpAttr();
  return attr.getValue();
}

::mlir::FloatAttr ClampOpGenericAdaptorBase::getMaxFpAttr() {
  auto attr = getProperties().max_fp.cast<::mlir::FloatAttr>();
  return attr;
}

::llvm::APFloat ClampOpGenericAdaptorBase::getMaxFp() {
  auto attr = getMaxFpAttr();
  return attr.getValue();
}

} // namespace detail
ClampOpAdaptor::ClampOpAdaptor(ClampOp op) : ClampOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ClampOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_max_fp = getProperties().max_fp; (void)tblgen_max_fp;
  if (!tblgen_max_fp) return emitError(loc, "'tosa.clamp' op ""requires attribute 'max_fp'");
  auto tblgen_max_int = getProperties().max_int; (void)tblgen_max_int;
  if (!tblgen_max_int) return emitError(loc, "'tosa.clamp' op ""requires attribute 'max_int'");
  auto tblgen_min_fp = getProperties().min_fp; (void)tblgen_min_fp;
  if (!tblgen_min_fp) return emitError(loc, "'tosa.clamp' op ""requires attribute 'min_fp'");
  auto tblgen_min_int = getProperties().min_int; (void)tblgen_min_int;
  if (!tblgen_min_int) return emitError(loc, "'tosa.clamp' op ""requires attribute 'min_int'");

  if (tblgen_min_int && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_min_int))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_min_int).getType().isSignlessInteger(64)))))
    return emitError(loc, "'tosa.clamp' op ""attribute 'min_int' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_max_int && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_max_int))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_max_int).getType().isSignlessInteger(64)))))
    return emitError(loc, "'tosa.clamp' op ""attribute 'max_int' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_min_fp && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_min_fp))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_min_fp).getType().isF32()))))
    return emitError(loc, "'tosa.clamp' op ""attribute 'min_fp' failed to satisfy constraint: 32-bit float attribute");

  if (tblgen_max_fp && !(((::llvm::isa<::mlir::FloatAttr>(tblgen_max_fp))) && ((::llvm::cast<::mlir::FloatAttr>(tblgen_max_fp).getType().isF32()))))
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult ClampOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.max_fp;
    auto attr = dict.get("max_fp");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for max_fp in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `max_fp` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.max_int;
    auto attr = dict.get("max_int");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for max_int in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `max_int` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.min_fp;
    auto attr = dict.get("min_fp");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for min_fp in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `min_fp` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.min_int;
    auto attr = dict.get("min_int");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for min_int in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `min_int` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ClampOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.max_fp;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("max_fp",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.max_int;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("max_int",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.min_fp;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("min_fp",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.min_int;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("min_int",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ClampOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.max_fp.getAsOpaquePointer()), 
    llvm::hash_value(prop.max_int.getAsOpaquePointer()), 
    llvm::hash_value(prop.min_fp.getAsOpaquePointer()), 
    llvm::hash_value(prop.min_int.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ClampOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "max_fp")
      return prop.max_fp;

    if (name == "max_int")
      return prop.max_int;

    if (name == "min_fp")
      return prop.min_fp;

    if (name == "min_int")
      return prop.min_int;
  return std::nullopt;
}

void ClampOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "max_fp") {
       prop.max_fp = dyn_cast_or_null<std::remove_reference_t<decltype(prop.max_fp)>>(value);
       return;
    }

    if (name == "max_int") {
       prop.max_int = dyn_cast_or_null<std::remove_reference_t<decltype(prop.max_int)>>(value);
       return;
    }

    if (name == "min_fp") {
       prop.min_fp = dyn_cast_or_null<std::remove_reference_t<decltype(prop.min_fp)>>(value);
       return;
    }

    if (name == "min_int") {
       prop.min_int = dyn_cast_or_null<std::remove_reference_t<decltype(prop.min_int)>>(value);
       return;
    }
}

void ClampOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.max_fp) attrs.append("max_fp", prop.max_fp);

    if (prop.max_int) attrs.append("max_int", prop.max_int);

    if (prop.min_fp) attrs.append("min_fp", prop.min_fp);

    if (prop.min_int) attrs.append("min_int", prop.min_int);
}

::mlir::LogicalResult ClampOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getMaxFpAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps6(attr, "max_fp", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getMaxIntAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(attr, "max_int", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getMinFpAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps6(attr, "min_fp", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getMinIntAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(attr, "min_int", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr ClampOp::getMinIntAttr() {
  return getProperties().min_int.cast<::mlir::IntegerAttr>();
}

uint64_t ClampOp::getMinInt() {
  auto attr = getMinIntAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ClampOp::getMaxIntAttr() {
  return getProperties().max_int.cast<::mlir::IntegerAttr>();
}

uint64_t ClampOp::getMaxInt() {
  auto attr = getMaxIntAttr();
  return attr.getValue().getZExtValue();
}

::mlir::FloatAttr ClampOp::getMinFpAttr() {
  return getProperties().min_fp.cast<::mlir::FloatAttr>();
}

::llvm::APFloat ClampOp::getMinFp() {
  auto attr = getMinFpAttr();
  return attr.getValue();
}

::mlir::FloatAttr ClampOp::getMaxFpAttr() {
  return getProperties().max_fp.cast<::mlir::FloatAttr>();
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
  odsState.getOrAddProperties<Properties>().min_int = min_int;
  odsState.getOrAddProperties<Properties>().max_int = max_int;
  odsState.getOrAddProperties<Properties>().min_fp = min_fp;
  odsState.getOrAddProperties<Properties>().max_fp = max_fp;
  odsState.addTypes(output);
}

void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr min_int, ::mlir::IntegerAttr max_int, ::mlir::FloatAttr min_fp, ::mlir::FloatAttr max_fp) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().min_int = min_int;
  odsState.getOrAddProperties<Properties>().max_int = max_int;
  odsState.getOrAddProperties<Properties>().min_fp = min_fp;
  odsState.getOrAddProperties<Properties>().max_fp = max_fp;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t min_int, uint64_t max_int, ::llvm::APFloat min_fp, ::llvm::APFloat max_fp) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().min_int = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), min_int);
  odsState.getOrAddProperties<Properties>().max_int = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), max_int);
  odsState.getOrAddProperties<Properties>().min_fp = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), min_fp);
  odsState.getOrAddProperties<Properties>().max_fp = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max_fp);
  odsState.addTypes(output);
}

void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t min_int, uint64_t max_int, ::llvm::APFloat min_fp, ::llvm::APFloat max_fp) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().min_int = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), min_int);
  odsState.getOrAddProperties<Properties>().max_int = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), max_int);
  odsState.getOrAddProperties<Properties>().min_fp = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), min_fp);
  odsState.getOrAddProperties<Properties>().max_fp = odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max_fp);
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
  auto tblgen_max_fp = getProperties().max_fp; (void)tblgen_max_fp;
  if (!tblgen_max_fp) return emitOpError("requires attribute 'max_fp'");
  auto tblgen_max_int = getProperties().max_int; (void)tblgen_max_int;
  if (!tblgen_max_int) return emitOpError("requires attribute 'max_int'");
  auto tblgen_min_fp = getProperties().min_fp; (void)tblgen_min_fp;
  if (!tblgen_min_fp) return emitOpError("requires attribute 'min_fp'");
  auto tblgen_min_int = getProperties().min_int; (void)tblgen_min_int;
  if (!tblgen_min_int) return emitOpError("requires attribute 'min_int'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(*this, tblgen_min_int, "min_int")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(*this, tblgen_max_int, "max_int")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps6(*this, tblgen_min_fp, "min_fp")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps6(*this, tblgen_max_fp, "max_fp")))
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

namespace detail {
ClzOpGenericAdaptorBase::ClzOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.clz", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ClzOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ClzOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ClzOpAdaptor::ClzOpAdaptor(ClzOp op) : ClzOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
ConcatOpGenericAdaptorBase::ConcatOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.concat", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ConcatOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr ConcatOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ConcatOpGenericAdaptorBase::getAxisAttr() {
  auto attr = getProperties().axis.cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ConcatOpGenericAdaptorBase::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ConcatOpAdaptor::ConcatOpAdaptor(ConcatOp op) : ConcatOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ConcatOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitError(loc, "'tosa.concat' op ""requires attribute 'axis'");

  if (tblgen_axis && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_axis))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_axis).getType().isSignlessInteger(64)))))
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult ConcatOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.axis;
    auto attr = dict.get("axis");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for axis in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `axis` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ConcatOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.axis;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("axis",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ConcatOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.axis.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ConcatOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "axis")
      return prop.axis;
  return std::nullopt;
}

void ConcatOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "axis") {
       prop.axis = dyn_cast_or_null<std::remove_reference_t<decltype(prop.axis)>>(value);
       return;
    }
}

void ConcatOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.axis) attrs.append("axis", prop.axis);
}

::mlir::LogicalResult ConcatOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getAxisAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(attr, "axis", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr ConcatOp::getAxisAttr() {
  return getProperties().axis.cast<::mlir::IntegerAttr>();
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
  odsState.getOrAddProperties<Properties>().axis = axis;
  odsState.addTypes(output);
}

void ConcatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange input1, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().axis = axis;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ConcatOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ConcatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange input1, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().axis = axis;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConcatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::ValueRange input1, uint64_t axis) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
  odsState.addTypes(output);
}

void ConcatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange input1, uint64_t axis) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ConcatOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ConcatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange input1, uint64_t axis) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConcatOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ConcatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ConcatOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ConcatOp::verifyInvariantsImpl() {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitOpError("requires attribute 'axis'");

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

namespace detail {
ConstOpGenericAdaptorBase::ConstOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.const", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ConstOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ConstOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ElementsAttr ConstOpGenericAdaptorBase::getValueAttr() {
  auto attr = getProperties().value.cast<::mlir::ElementsAttr>();
  return attr;
}

::mlir::ElementsAttr ConstOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr;
}

} // namespace detail
ConstOpAdaptor::ConstOpAdaptor(ConstOp op) : ConstOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ConstOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitError(loc, "'tosa.const' op ""requires attribute 'value'");

  if (tblgen_value && !((::llvm::isa<::mlir::ElementsAttr>(tblgen_value))))
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult ConstOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.value;
    auto attr = dict.get("value");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for value in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `value` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ConstOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.value;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("value",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ConstOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.value.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ConstOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "value")
      return prop.value;
  return std::nullopt;
}

void ConstOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "value") {
       prop.value = dyn_cast_or_null<std::remove_reference_t<decltype(prop.value)>>(value);
       return;
    }
}

void ConstOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.value) attrs.append("value", prop.value);
}

::mlir::LogicalResult ConstOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getValueAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps7(attr, "value", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::ElementsAttr ConstOp::getValueAttr() {
  return getProperties().value.cast<::mlir::ElementsAttr>();
}

::mlir::ElementsAttr ConstOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void ConstOp::setValueAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void ConstOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::ElementsAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  odsState.addTypes(output);
}

void ConstOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ElementsAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
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
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitOpError("requires attribute 'value'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps7(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps6(*this, v.getType(), "result", index++)))
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

namespace detail {
Conv2DOpGenericAdaptorBase::Conv2DOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.conv2d", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Conv2DOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr Conv2DOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr Conv2DOpGenericAdaptorBase::getPadAttr() {
  auto attr = getProperties().pad.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> Conv2DOpGenericAdaptorBase::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr Conv2DOpGenericAdaptorBase::getStrideAttr() {
  auto attr = getProperties().stride.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> Conv2DOpGenericAdaptorBase::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr Conv2DOpGenericAdaptorBase::getDilationAttr() {
  auto attr = getProperties().dilation.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> Conv2DOpGenericAdaptorBase::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr Conv2DOpGenericAdaptorBase::getQuantizationInfoAttr() {
  auto attr = getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> Conv2DOpGenericAdaptorBase::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(::llvm::cast<mlir::tosa::ConvOpQuantizationAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
Conv2DOpAdaptor::Conv2DOpAdaptor(Conv2DOp op) : Conv2DOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult Conv2DOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_dilation = getProperties().dilation; (void)tblgen_dilation;
  if (!tblgen_dilation) return emitError(loc, "'tosa.conv2d' op ""requires attribute 'dilation'");
  auto tblgen_pad = getProperties().pad; (void)tblgen_pad;
  if (!tblgen_pad) return emitError(loc, "'tosa.conv2d' op ""requires attribute 'pad'");
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitError(loc, "'tosa.conv2d' op ""requires attribute 'stride'");

  if (tblgen_pad && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_pad))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_pad).size() == 4))))
    return emitError(loc, "'tosa.conv2d' op ""attribute 'pad' failed to satisfy constraint: i64 dense array attribute with exactly 4 elements");

  if (tblgen_stride && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_stride))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_stride).size() == 2))))
    return emitError(loc, "'tosa.conv2d' op ""attribute 'stride' failed to satisfy constraint: i64 dense array attribute with exactly 2 elements");

  if (tblgen_dilation && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_dilation))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_dilation).size() == 2))))
    return emitError(loc, "'tosa.conv2d' op ""attribute 'dilation' failed to satisfy constraint: i64 dense array attribute with exactly 2 elements");

  if (tblgen_quantization_info && !((::llvm::isa<mlir::tosa::ConvOpQuantizationAttr>(tblgen_quantization_info))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value Conv2DOp::getWeight() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::Value Conv2DOp::getBias() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(2).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult Conv2DOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.dilation;
    auto attr = dict.get("dilation");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for dilation in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `dilation` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.pad;
    auto attr = dict.get("pad");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for pad in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `pad` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.quantization_info;
    auto attr = dict.get("quantization_info");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for quantization_info in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `quantization_info` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.stride;
    auto attr = dict.get("stride");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for stride in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `stride` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute Conv2DOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.dilation;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("dilation",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.pad;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("pad",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.quantization_info;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("quantization_info",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.stride;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("stride",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code Conv2DOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.dilation.getAsOpaquePointer()), 
    llvm::hash_value(prop.pad.getAsOpaquePointer()), 
    llvm::hash_value(prop.quantization_info.getAsOpaquePointer()), 
    llvm::hash_value(prop.stride.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> Conv2DOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "dilation")
      return prop.dilation;

    if (name == "pad")
      return prop.pad;

    if (name == "quantization_info")
      return prop.quantization_info;

    if (name == "stride")
      return prop.stride;
  return std::nullopt;
}

void Conv2DOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "dilation") {
       prop.dilation = dyn_cast_or_null<std::remove_reference_t<decltype(prop.dilation)>>(value);
       return;
    }

    if (name == "pad") {
       prop.pad = dyn_cast_or_null<std::remove_reference_t<decltype(prop.pad)>>(value);
       return;
    }

    if (name == "quantization_info") {
       prop.quantization_info = dyn_cast_or_null<std::remove_reference_t<decltype(prop.quantization_info)>>(value);
       return;
    }

    if (name == "stride") {
       prop.stride = dyn_cast_or_null<std::remove_reference_t<decltype(prop.stride)>>(value);
       return;
    }
}

void Conv2DOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.dilation) attrs.append("dilation", prop.dilation);

    if (prop.pad) attrs.append("pad", prop.pad);

    if (prop.quantization_info) attrs.append("quantization_info", prop.quantization_info);

    if (prop.stride) attrs.append("stride", prop.stride);
}

::mlir::LogicalResult Conv2DOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getDilationAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(attr, "dilation", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPadAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(attr, "pad", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getQuantizationInfoAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps8(attr, "quantization_info", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStrideAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(attr, "stride", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::DenseI64ArrayAttr Conv2DOp::getPadAttr() {
  return getProperties().pad.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> Conv2DOp::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr Conv2DOp::getStrideAttr() {
  return getProperties().stride.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> Conv2DOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr Conv2DOp::getDilationAttr() {
  return getProperties().dilation.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> Conv2DOp::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr Conv2DOp::getQuantizationInfoAttr() {
  return getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> Conv2DOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(::llvm::cast<mlir::tosa::ConvOpQuantizationAttr>(attr)) : (::std::nullopt);
}

void Conv2DOp::setPadAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getPadAttrName(), attr);
}

void Conv2DOp::setPad(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getPadAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void Conv2DOp::setStrideAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStrideAttrName(), attr);
}

void Conv2DOp::setStride(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStrideAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void Conv2DOp::setDilationAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getDilationAttrName(), attr);
}

void Conv2DOp::setDilation(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getDilationAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void Conv2DOp::setQuantizationInfoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute Conv2DOp::removeQuantizationInfoAttr() {
    auto &attr = getProperties().quantization_info;
    attr = {};
    return attr;
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputType, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::DenseI64ArrayAttr pad, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr dilation) {
    buildConvOpWithQuantInfo(odsBuilder, odsState, outputType,
                             input, weight, bias,
                             pad, stride, dilation);
  
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::DenseI64ArrayAttr pad, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().pad = pad;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().dilation = dilation;
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  odsState.addTypes(output);
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::DenseI64ArrayAttr pad, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().pad = pad;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().dilation = dilation;
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::llvm::ArrayRef<int64_t> pad, ::llvm::ArrayRef<int64_t> stride, ::llvm::ArrayRef<int64_t> dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().pad = odsBuilder.getDenseI64ArrayAttr(pad);
  odsState.getOrAddProperties<Properties>().stride = odsBuilder.getDenseI64ArrayAttr(stride);
  odsState.getOrAddProperties<Properties>().dilation = odsBuilder.getDenseI64ArrayAttr(dilation);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  odsState.addTypes(output);
}

void Conv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::llvm::ArrayRef<int64_t> pad, ::llvm::ArrayRef<int64_t> stride, ::llvm::ArrayRef<int64_t> dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().pad = odsBuilder.getDenseI64ArrayAttr(pad);
  odsState.getOrAddProperties<Properties>().stride = odsBuilder.getDenseI64ArrayAttr(stride);
  odsState.getOrAddProperties<Properties>().dilation = odsBuilder.getDenseI64ArrayAttr(dilation);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
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
  auto tblgen_dilation = getProperties().dilation; (void)tblgen_dilation;
  if (!tblgen_dilation) return emitOpError("requires attribute 'dilation'");
  auto tblgen_pad = getProperties().pad; (void)tblgen_pad;
  if (!tblgen_pad) return emitOpError("requires attribute 'pad'");
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitOpError("requires attribute 'stride'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(*this, tblgen_pad, "pad")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_stride, "stride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_dilation, "dilation")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps8(*this, tblgen_quantization_info, "quantization_info")))
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps7(*this, v.getType(), "operand", index++)))
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

namespace detail {
Conv3DOpGenericAdaptorBase::Conv3DOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.conv3d", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> Conv3DOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr Conv3DOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr Conv3DOpGenericAdaptorBase::getPadAttr() {
  auto attr = getProperties().pad.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> Conv3DOpGenericAdaptorBase::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr Conv3DOpGenericAdaptorBase::getStrideAttr() {
  auto attr = getProperties().stride.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> Conv3DOpGenericAdaptorBase::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr Conv3DOpGenericAdaptorBase::getDilationAttr() {
  auto attr = getProperties().dilation.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> Conv3DOpGenericAdaptorBase::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr Conv3DOpGenericAdaptorBase::getQuantizationInfoAttr() {
  auto attr = getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> Conv3DOpGenericAdaptorBase::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(::llvm::cast<mlir::tosa::ConvOpQuantizationAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
Conv3DOpAdaptor::Conv3DOpAdaptor(Conv3DOp op) : Conv3DOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult Conv3DOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_dilation = getProperties().dilation; (void)tblgen_dilation;
  if (!tblgen_dilation) return emitError(loc, "'tosa.conv3d' op ""requires attribute 'dilation'");
  auto tblgen_pad = getProperties().pad; (void)tblgen_pad;
  if (!tblgen_pad) return emitError(loc, "'tosa.conv3d' op ""requires attribute 'pad'");
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitError(loc, "'tosa.conv3d' op ""requires attribute 'stride'");

  if (tblgen_pad && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_pad))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_pad).size() == 6))))
    return emitError(loc, "'tosa.conv3d' op ""attribute 'pad' failed to satisfy constraint: i64 dense array attribute with exactly 6 elements");

  if (tblgen_stride && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_stride))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_stride).size() == 3))))
    return emitError(loc, "'tosa.conv3d' op ""attribute 'stride' failed to satisfy constraint: i64 dense array attribute with exactly 3 elements");

  if (tblgen_dilation && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_dilation))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_dilation).size() == 3))))
    return emitError(loc, "'tosa.conv3d' op ""attribute 'dilation' failed to satisfy constraint: i64 dense array attribute with exactly 3 elements");

  if (tblgen_quantization_info && !((::llvm::isa<mlir::tosa::ConvOpQuantizationAttr>(tblgen_quantization_info))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value Conv3DOp::getWeight() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::Value Conv3DOp::getBias() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(2).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult Conv3DOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.dilation;
    auto attr = dict.get("dilation");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for dilation in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `dilation` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.pad;
    auto attr = dict.get("pad");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for pad in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `pad` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.quantization_info;
    auto attr = dict.get("quantization_info");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for quantization_info in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `quantization_info` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.stride;
    auto attr = dict.get("stride");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for stride in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `stride` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute Conv3DOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.dilation;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("dilation",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.pad;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("pad",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.quantization_info;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("quantization_info",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.stride;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("stride",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code Conv3DOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.dilation.getAsOpaquePointer()), 
    llvm::hash_value(prop.pad.getAsOpaquePointer()), 
    llvm::hash_value(prop.quantization_info.getAsOpaquePointer()), 
    llvm::hash_value(prop.stride.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> Conv3DOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "dilation")
      return prop.dilation;

    if (name == "pad")
      return prop.pad;

    if (name == "quantization_info")
      return prop.quantization_info;

    if (name == "stride")
      return prop.stride;
  return std::nullopt;
}

void Conv3DOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "dilation") {
       prop.dilation = dyn_cast_or_null<std::remove_reference_t<decltype(prop.dilation)>>(value);
       return;
    }

    if (name == "pad") {
       prop.pad = dyn_cast_or_null<std::remove_reference_t<decltype(prop.pad)>>(value);
       return;
    }

    if (name == "quantization_info") {
       prop.quantization_info = dyn_cast_or_null<std::remove_reference_t<decltype(prop.quantization_info)>>(value);
       return;
    }

    if (name == "stride") {
       prop.stride = dyn_cast_or_null<std::remove_reference_t<decltype(prop.stride)>>(value);
       return;
    }
}

void Conv3DOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.dilation) attrs.append("dilation", prop.dilation);

    if (prop.pad) attrs.append("pad", prop.pad);

    if (prop.quantization_info) attrs.append("quantization_info", prop.quantization_info);

    if (prop.stride) attrs.append("stride", prop.stride);
}

::mlir::LogicalResult Conv3DOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getDilationAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps10(attr, "dilation", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPadAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps9(attr, "pad", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getQuantizationInfoAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps8(attr, "quantization_info", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStrideAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps10(attr, "stride", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::DenseI64ArrayAttr Conv3DOp::getPadAttr() {
  return getProperties().pad.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> Conv3DOp::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr Conv3DOp::getStrideAttr() {
  return getProperties().stride.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> Conv3DOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr Conv3DOp::getDilationAttr() {
  return getProperties().dilation.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> Conv3DOp::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr Conv3DOp::getQuantizationInfoAttr() {
  return getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> Conv3DOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(::llvm::cast<mlir::tosa::ConvOpQuantizationAttr>(attr)) : (::std::nullopt);
}

void Conv3DOp::setPadAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getPadAttrName(), attr);
}

void Conv3DOp::setPad(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getPadAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void Conv3DOp::setStrideAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStrideAttrName(), attr);
}

void Conv3DOp::setStride(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStrideAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void Conv3DOp::setDilationAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getDilationAttrName(), attr);
}

void Conv3DOp::setDilation(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getDilationAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void Conv3DOp::setQuantizationInfoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute Conv3DOp::removeQuantizationInfoAttr() {
    auto &attr = getProperties().quantization_info;
    attr = {};
    return attr;
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputType, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::DenseI64ArrayAttr pad, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr dilation) {
    buildConvOpWithQuantInfo(odsBuilder, odsState, outputType,
                             input, weight, bias,
                             pad, stride, dilation);
  
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::DenseI64ArrayAttr pad, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().pad = pad;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().dilation = dilation;
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  odsState.addTypes(output);
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::DenseI64ArrayAttr pad, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().pad = pad;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().dilation = dilation;
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::llvm::ArrayRef<int64_t> pad, ::llvm::ArrayRef<int64_t> stride, ::llvm::ArrayRef<int64_t> dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().pad = odsBuilder.getDenseI64ArrayAttr(pad);
  odsState.getOrAddProperties<Properties>().stride = odsBuilder.getDenseI64ArrayAttr(stride);
  odsState.getOrAddProperties<Properties>().dilation = odsBuilder.getDenseI64ArrayAttr(dilation);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  odsState.addTypes(output);
}

void Conv3DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::llvm::ArrayRef<int64_t> pad, ::llvm::ArrayRef<int64_t> stride, ::llvm::ArrayRef<int64_t> dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().pad = odsBuilder.getDenseI64ArrayAttr(pad);
  odsState.getOrAddProperties<Properties>().stride = odsBuilder.getDenseI64ArrayAttr(stride);
  odsState.getOrAddProperties<Properties>().dilation = odsBuilder.getDenseI64ArrayAttr(dilation);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
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
  auto tblgen_dilation = getProperties().dilation; (void)tblgen_dilation;
  if (!tblgen_dilation) return emitOpError("requires attribute 'dilation'");
  auto tblgen_pad = getProperties().pad; (void)tblgen_pad;
  if (!tblgen_pad) return emitOpError("requires attribute 'pad'");
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitOpError("requires attribute 'stride'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps9(*this, tblgen_pad, "pad")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps10(*this, tblgen_stride, "stride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps10(*this, tblgen_dilation, "dilation")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps8(*this, tblgen_quantization_info, "quantization_info")))
    return ::mlir::failure();
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
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps7(*this, v.getType(), "operand", index++)))
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

namespace detail {
CustomOpGenericAdaptorBase::CustomOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.custom", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CustomOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr CustomOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CustomOpGenericAdaptorBase::getIdentifierAttr() {
  auto attr = getProperties().identifier.cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CustomOpGenericAdaptorBase::getIdentifier() {
  auto attr = getIdentifierAttr();
  return attr.getValue();
}

::mlir::StringAttr CustomOpGenericAdaptorBase::getConfigAttr() {
  auto attr = getProperties().config.cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CustomOpGenericAdaptorBase::getConfig() {
  auto attr = getConfigAttr();
  return attr.getValue();
}

::mlir::StringAttr CustomOpGenericAdaptorBase::getImplementationAttrsAttr() {
  auto attr = getProperties().implementation_attrs.cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CustomOpGenericAdaptorBase::getImplementationAttrs() {
  auto attr = getImplementationAttrsAttr();
  return attr.getValue();
}

} // namespace detail
CustomOpAdaptor::CustomOpAdaptor(CustomOp op) : CustomOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult CustomOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_config = getProperties().config; (void)tblgen_config;
  if (!tblgen_config) return emitError(loc, "'tosa.custom' op ""requires attribute 'config'");
  auto tblgen_identifier = getProperties().identifier; (void)tblgen_identifier;
  if (!tblgen_identifier) return emitError(loc, "'tosa.custom' op ""requires attribute 'identifier'");
  auto tblgen_implementation_attrs = getProperties().implementation_attrs; (void)tblgen_implementation_attrs;
  if (!tblgen_implementation_attrs) return emitError(loc, "'tosa.custom' op ""requires attribute 'implementation_attrs'");

  if (tblgen_identifier && !((::llvm::isa<::mlir::StringAttr>(tblgen_identifier))))
    return emitError(loc, "'tosa.custom' op ""attribute 'identifier' failed to satisfy constraint: string attribute");

  if (tblgen_config && !((::llvm::isa<::mlir::StringAttr>(tblgen_config))))
    return emitError(loc, "'tosa.custom' op ""attribute 'config' failed to satisfy constraint: string attribute");

  if (tblgen_implementation_attrs && !((::llvm::isa<::mlir::StringAttr>(tblgen_implementation_attrs))))
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

::mlir::LogicalResult CustomOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.config;
    auto attr = dict.get("config");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for config in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `config` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.identifier;
    auto attr = dict.get("identifier");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for identifier in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `identifier` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.implementation_attrs;
    auto attr = dict.get("implementation_attrs");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for implementation_attrs in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `implementation_attrs` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CustomOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.config;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("config",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.identifier;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("identifier",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.implementation_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("implementation_attrs",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code CustomOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.config.getAsOpaquePointer()), 
    llvm::hash_value(prop.identifier.getAsOpaquePointer()), 
    llvm::hash_value(prop.implementation_attrs.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CustomOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "config")
      return prop.config;

    if (name == "identifier")
      return prop.identifier;

    if (name == "implementation_attrs")
      return prop.implementation_attrs;
  return std::nullopt;
}

void CustomOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "config") {
       prop.config = dyn_cast_or_null<std::remove_reference_t<decltype(prop.config)>>(value);
       return;
    }

    if (name == "identifier") {
       prop.identifier = dyn_cast_or_null<std::remove_reference_t<decltype(prop.identifier)>>(value);
       return;
    }

    if (name == "implementation_attrs") {
       prop.implementation_attrs = dyn_cast_or_null<std::remove_reference_t<decltype(prop.implementation_attrs)>>(value);
       return;
    }
}

void CustomOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.config) attrs.append("config", prop.config);

    if (prop.identifier) attrs.append("identifier", prop.identifier);

    if (prop.implementation_attrs) attrs.append("implementation_attrs", prop.implementation_attrs);
}

::mlir::LogicalResult CustomOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getConfigAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps11(attr, "config", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getIdentifierAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps11(attr, "identifier", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getImplementationAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps11(attr, "implementation_attrs", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::StringAttr CustomOp::getIdentifierAttr() {
  return getProperties().identifier.cast<::mlir::StringAttr>();
}

::llvm::StringRef CustomOp::getIdentifier() {
  auto attr = getIdentifierAttr();
  return attr.getValue();
}

::mlir::StringAttr CustomOp::getConfigAttr() {
  return getProperties().config.cast<::mlir::StringAttr>();
}

::llvm::StringRef CustomOp::getConfig() {
  auto attr = getConfigAttr();
  return attr.getValue();
}

::mlir::StringAttr CustomOp::getImplementationAttrsAttr() {
  return getProperties().implementation_attrs.cast<::mlir::StringAttr>();
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
  odsState.getOrAddProperties<Properties>().identifier = identifier;
  odsState.getOrAddProperties<Properties>().config = config;
  odsState.getOrAddProperties<Properties>().implementation_attrs = implementation_attrs;
  odsState.addTypes(outputs);
}

void CustomOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::llvm::StringRef identifier, ::llvm::StringRef config, ::llvm::StringRef implementation_attrs, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.getOrAddProperties<Properties>().identifier = odsBuilder.getStringAttr(identifier);
  odsState.getOrAddProperties<Properties>().config = odsBuilder.getStringAttr(config);
  odsState.getOrAddProperties<Properties>().implementation_attrs = odsBuilder.getStringAttr(implementation_attrs);
  odsState.addTypes(outputs);
}

void CustomOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CustomOp::verifyInvariantsImpl() {
  auto tblgen_config = getProperties().config; (void)tblgen_config;
  if (!tblgen_config) return emitOpError("requires attribute 'config'");
  auto tblgen_identifier = getProperties().identifier; (void)tblgen_identifier;
  if (!tblgen_identifier) return emitOpError("requires attribute 'identifier'");
  auto tblgen_implementation_attrs = getProperties().implementation_attrs; (void)tblgen_implementation_attrs;
  if (!tblgen_implementation_attrs) return emitOpError("requires attribute 'implementation_attrs'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps11(*this, tblgen_identifier, "identifier")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps11(*this, tblgen_config, "config")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps11(*this, tblgen_implementation_attrs, "implementation_attrs")))
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

namespace detail {
DepthwiseConv2DOpGenericAdaptorBase::DepthwiseConv2DOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.depthwise_conv2d", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DepthwiseConv2DOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr DepthwiseConv2DOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr DepthwiseConv2DOpGenericAdaptorBase::getPadAttr() {
  auto attr = getProperties().pad.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> DepthwiseConv2DOpGenericAdaptorBase::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr DepthwiseConv2DOpGenericAdaptorBase::getStrideAttr() {
  auto attr = getProperties().stride.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> DepthwiseConv2DOpGenericAdaptorBase::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr DepthwiseConv2DOpGenericAdaptorBase::getDilationAttr() {
  auto attr = getProperties().dilation.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> DepthwiseConv2DOpGenericAdaptorBase::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr DepthwiseConv2DOpGenericAdaptorBase::getQuantizationInfoAttr() {
  auto attr = getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> DepthwiseConv2DOpGenericAdaptorBase::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(::llvm::cast<mlir::tosa::ConvOpQuantizationAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
DepthwiseConv2DOpAdaptor::DepthwiseConv2DOpAdaptor(DepthwiseConv2DOp op) : DepthwiseConv2DOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult DepthwiseConv2DOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_dilation = getProperties().dilation; (void)tblgen_dilation;
  if (!tblgen_dilation) return emitError(loc, "'tosa.depthwise_conv2d' op ""requires attribute 'dilation'");
  auto tblgen_pad = getProperties().pad; (void)tblgen_pad;
  if (!tblgen_pad) return emitError(loc, "'tosa.depthwise_conv2d' op ""requires attribute 'pad'");
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitError(loc, "'tosa.depthwise_conv2d' op ""requires attribute 'stride'");

  if (tblgen_pad && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_pad))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_pad).size() == 4))))
    return emitError(loc, "'tosa.depthwise_conv2d' op ""attribute 'pad' failed to satisfy constraint: i64 dense array attribute with exactly 4 elements");

  if (tblgen_stride && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_stride))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_stride).size() == 2))))
    return emitError(loc, "'tosa.depthwise_conv2d' op ""attribute 'stride' failed to satisfy constraint: i64 dense array attribute with exactly 2 elements");

  if (tblgen_dilation && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_dilation))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_dilation).size() == 2))))
    return emitError(loc, "'tosa.depthwise_conv2d' op ""attribute 'dilation' failed to satisfy constraint: i64 dense array attribute with exactly 2 elements");

  if (tblgen_quantization_info && !((::llvm::isa<mlir::tosa::ConvOpQuantizationAttr>(tblgen_quantization_info))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value DepthwiseConv2DOp::getWeight() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::Value DepthwiseConv2DOp::getBias() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(2).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult DepthwiseConv2DOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.dilation;
    auto attr = dict.get("dilation");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for dilation in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `dilation` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.pad;
    auto attr = dict.get("pad");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for pad in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `pad` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.quantization_info;
    auto attr = dict.get("quantization_info");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for quantization_info in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `quantization_info` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.stride;
    auto attr = dict.get("stride");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for stride in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `stride` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute DepthwiseConv2DOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.dilation;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("dilation",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.pad;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("pad",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.quantization_info;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("quantization_info",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.stride;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("stride",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code DepthwiseConv2DOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.dilation.getAsOpaquePointer()), 
    llvm::hash_value(prop.pad.getAsOpaquePointer()), 
    llvm::hash_value(prop.quantization_info.getAsOpaquePointer()), 
    llvm::hash_value(prop.stride.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> DepthwiseConv2DOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "dilation")
      return prop.dilation;

    if (name == "pad")
      return prop.pad;

    if (name == "quantization_info")
      return prop.quantization_info;

    if (name == "stride")
      return prop.stride;
  return std::nullopt;
}

void DepthwiseConv2DOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "dilation") {
       prop.dilation = dyn_cast_or_null<std::remove_reference_t<decltype(prop.dilation)>>(value);
       return;
    }

    if (name == "pad") {
       prop.pad = dyn_cast_or_null<std::remove_reference_t<decltype(prop.pad)>>(value);
       return;
    }

    if (name == "quantization_info") {
       prop.quantization_info = dyn_cast_or_null<std::remove_reference_t<decltype(prop.quantization_info)>>(value);
       return;
    }

    if (name == "stride") {
       prop.stride = dyn_cast_or_null<std::remove_reference_t<decltype(prop.stride)>>(value);
       return;
    }
}

void DepthwiseConv2DOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.dilation) attrs.append("dilation", prop.dilation);

    if (prop.pad) attrs.append("pad", prop.pad);

    if (prop.quantization_info) attrs.append("quantization_info", prop.quantization_info);

    if (prop.stride) attrs.append("stride", prop.stride);
}

::mlir::LogicalResult DepthwiseConv2DOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getDilationAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(attr, "dilation", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPadAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(attr, "pad", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getQuantizationInfoAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps8(attr, "quantization_info", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStrideAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(attr, "stride", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::DenseI64ArrayAttr DepthwiseConv2DOp::getPadAttr() {
  return getProperties().pad.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> DepthwiseConv2DOp::getPad() {
  auto attr = getPadAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr DepthwiseConv2DOp::getStrideAttr() {
  return getProperties().stride.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> DepthwiseConv2DOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr DepthwiseConv2DOp::getDilationAttr() {
  return getProperties().dilation.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> DepthwiseConv2DOp::getDilation() {
  auto attr = getDilationAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr DepthwiseConv2DOp::getQuantizationInfoAttr() {
  return getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> DepthwiseConv2DOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(::llvm::cast<mlir::tosa::ConvOpQuantizationAttr>(attr)) : (::std::nullopt);
}

void DepthwiseConv2DOp::setPadAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getPadAttrName(), attr);
}

void DepthwiseConv2DOp::setPad(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getPadAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void DepthwiseConv2DOp::setStrideAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStrideAttrName(), attr);
}

void DepthwiseConv2DOp::setStride(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStrideAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void DepthwiseConv2DOp::setDilationAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getDilationAttrName(), attr);
}

void DepthwiseConv2DOp::setDilation(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getDilationAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void DepthwiseConv2DOp::setQuantizationInfoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute DepthwiseConv2DOp::removeQuantizationInfoAttr() {
    auto &attr = getProperties().quantization_info;
    attr = {};
    return attr;
}

void DepthwiseConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputType, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::DenseI64ArrayAttr pad, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr dilation) {
    buildConvOpWithQuantInfo(odsBuilder, odsState, outputType,
                             input, weight, bias,
                             pad, stride, dilation);
  
}

void DepthwiseConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::DenseI64ArrayAttr pad, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().pad = pad;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().dilation = dilation;
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  odsState.addTypes(output);
}

void DepthwiseConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::mlir::DenseI64ArrayAttr pad, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().pad = pad;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().dilation = dilation;
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DepthwiseConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::llvm::ArrayRef<int64_t> pad, ::llvm::ArrayRef<int64_t> stride, ::llvm::ArrayRef<int64_t> dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().pad = odsBuilder.getDenseI64ArrayAttr(pad);
  odsState.getOrAddProperties<Properties>().stride = odsBuilder.getDenseI64ArrayAttr(stride);
  odsState.getOrAddProperties<Properties>().dilation = odsBuilder.getDenseI64ArrayAttr(dilation);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  odsState.addTypes(output);
}

void DepthwiseConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, ::llvm::ArrayRef<int64_t> pad, ::llvm::ArrayRef<int64_t> stride, ::llvm::ArrayRef<int64_t> dilation, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().pad = odsBuilder.getDenseI64ArrayAttr(pad);
  odsState.getOrAddProperties<Properties>().stride = odsBuilder.getDenseI64ArrayAttr(stride);
  odsState.getOrAddProperties<Properties>().dilation = odsBuilder.getDenseI64ArrayAttr(dilation);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
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
  auto tblgen_dilation = getProperties().dilation; (void)tblgen_dilation;
  if (!tblgen_dilation) return emitOpError("requires attribute 'dilation'");
  auto tblgen_pad = getProperties().pad; (void)tblgen_pad;
  if (!tblgen_pad) return emitOpError("requires attribute 'pad'");
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitOpError("requires attribute 'stride'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(*this, tblgen_pad, "pad")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_stride, "stride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_dilation, "dilation")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps8(*this, tblgen_quantization_info, "quantization_info")))
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps7(*this, v.getType(), "operand", index++)))
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

namespace detail {
DivOpGenericAdaptorBase::DivOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.div", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DivOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr DivOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
DivOpAdaptor::DivOpAdaptor(DivOp op) : DivOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> DivOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
EqualOpGenericAdaptorBase::EqualOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.equal", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> EqualOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr EqualOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
EqualOpAdaptor::EqualOpAdaptor(EqualOp op) : EqualOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> EqualOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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
                      odsState.getRawProperties(),
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
          odsState.getRawProperties(),
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "result", index++)))
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

namespace detail {
ExpOpGenericAdaptorBase::ExpOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.exp", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ExpOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ExpOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ExpOpAdaptor::ExpOpAdaptor(ExpOp op) : ExpOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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
// mlir::tosa::FFT2dOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FFT2dOpGenericAdaptorBase::FFT2dOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.fft2d", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FFT2dOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FFT2dOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr FFT2dOpGenericAdaptorBase::getInverseAttr() {
  auto attr = getProperties().inverse.cast<::mlir::BoolAttr>();
  return attr;
}

bool FFT2dOpGenericAdaptorBase::getInverse() {
  auto attr = getInverseAttr();
  return attr.getValue();
}

} // namespace detail
FFT2dOpAdaptor::FFT2dOpAdaptor(FFT2dOp op) : FFT2dOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult FFT2dOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_inverse = getProperties().inverse; (void)tblgen_inverse;
  if (!tblgen_inverse) return emitError(loc, "'tosa.fft2d' op ""requires attribute 'inverse'");

  if (tblgen_inverse && !((::llvm::isa<::mlir::BoolAttr>(tblgen_inverse))))
    return emitError(loc, "'tosa.fft2d' op ""attribute 'inverse' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

void FFT2dOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "output_real");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "output_imag");
}

std::pair<unsigned, unsigned> FFT2dOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FFT2dOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FFT2dOp::getInputReal() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value FFT2dOp::getInputImag() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange FFT2dOp::getInputRealMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FFT2dOp::getInputImagMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FFT2dOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FFT2dOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FFT2dOp::getOutputReal() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::Value FFT2dOp::getOutputImag() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(1).begin());
}

::mlir::LogicalResult FFT2dOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.inverse;
    auto attr = dict.get("inverse");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for inverse in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `inverse` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute FFT2dOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.inverse;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("inverse",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code FFT2dOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.inverse.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> FFT2dOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "inverse")
      return prop.inverse;
  return std::nullopt;
}

void FFT2dOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "inverse") {
       prop.inverse = dyn_cast_or_null<std::remove_reference_t<decltype(prop.inverse)>>(value);
       return;
    }
}

void FFT2dOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.inverse) attrs.append("inverse", prop.inverse);
}

::mlir::LogicalResult FFT2dOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getInverseAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps0(attr, "inverse", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::BoolAttr FFT2dOp::getInverseAttr() {
  return getProperties().inverse.cast<::mlir::BoolAttr>();
}

bool FFT2dOp::getInverse() {
  auto attr = getInverseAttr();
  return attr.getValue();
}

void FFT2dOp::setInverseAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getInverseAttrName(), attr);
}

void FFT2dOp::setInverse(bool attrValue) {
  (*this)->setAttr(getInverseAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void FFT2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output_real, ::mlir::Type output_imag, ::mlir::Value input_real, ::mlir::Value input_imag, ::mlir::BoolAttr inverse) {
  odsState.addOperands(input_real);
  odsState.addOperands(input_imag);
  odsState.getOrAddProperties<Properties>().inverse = inverse;
  odsState.addTypes(output_real);
  odsState.addTypes(output_imag);
}

void FFT2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input_real, ::mlir::Value input_imag, ::mlir::BoolAttr inverse) {
  odsState.addOperands(input_real);
  odsState.addOperands(input_imag);
  odsState.getOrAddProperties<Properties>().inverse = inverse;
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FFT2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output_real, ::mlir::Type output_imag, ::mlir::Value input_real, ::mlir::Value input_imag, bool inverse) {
  odsState.addOperands(input_real);
  odsState.addOperands(input_imag);
  odsState.getOrAddProperties<Properties>().inverse = odsBuilder.getBoolAttr(inverse);
  odsState.addTypes(output_real);
  odsState.addTypes(output_imag);
}

void FFT2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input_real, ::mlir::Value input_imag, bool inverse) {
  odsState.addOperands(input_real);
  odsState.addOperands(input_imag);
  odsState.getOrAddProperties<Properties>().inverse = odsBuilder.getBoolAttr(inverse);
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FFT2dOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FFT2dOp::verifyInvariantsImpl() {
  auto tblgen_inverse = getProperties().inverse; (void)tblgen_inverse;
  if (!tblgen_inverse) return emitOpError("requires attribute 'inverse'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps0(*this, tblgen_inverse, "inverse")))
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
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps11(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FFT2dOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void FFT2dOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::FFT2dOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::FloorOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FloorOpGenericAdaptorBase::FloorOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.floor", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FloorOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FloorOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
FloorOpAdaptor::FloorOpAdaptor(FloorOp op) : FloorOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
FullyConnectedOpGenericAdaptorBase::FullyConnectedOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.fully_connected", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FullyConnectedOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FullyConnectedOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

mlir::tosa::ConvOpQuantizationAttr FullyConnectedOpGenericAdaptorBase::getQuantizationInfoAttr() {
  auto attr = getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> FullyConnectedOpGenericAdaptorBase::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(::llvm::cast<mlir::tosa::ConvOpQuantizationAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
FullyConnectedOpAdaptor::FullyConnectedOpAdaptor(FullyConnectedOp op) : FullyConnectedOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult FullyConnectedOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;

  if (tblgen_quantization_info && !((::llvm::isa<mlir::tosa::ConvOpQuantizationAttr>(tblgen_quantization_info))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value FullyConnectedOp::getWeight() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::Value FullyConnectedOp::getBias() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(2).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult FullyConnectedOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.quantization_info;
    auto attr = dict.get("quantization_info");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for quantization_info in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `quantization_info` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute FullyConnectedOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.quantization_info;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("quantization_info",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code FullyConnectedOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.quantization_info.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> FullyConnectedOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "quantization_info")
      return prop.quantization_info;
  return std::nullopt;
}

void FullyConnectedOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "quantization_info") {
       prop.quantization_info = dyn_cast_or_null<std::remove_reference_t<decltype(prop.quantization_info)>>(value);
       return;
    }
}

void FullyConnectedOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.quantization_info) attrs.append("quantization_info", prop.quantization_info);
}

::mlir::LogicalResult FullyConnectedOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getQuantizationInfoAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps8(attr, "quantization_info", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

mlir::tosa::ConvOpQuantizationAttr FullyConnectedOp::getQuantizationInfoAttr() {
  return getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> FullyConnectedOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(::llvm::cast<mlir::tosa::ConvOpQuantizationAttr>(attr)) : (::std::nullopt);
}

void FullyConnectedOp::setQuantizationInfoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute FullyConnectedOp::removeQuantizationInfoAttr() {
    auto &attr = getProperties().quantization_info;
    attr = {};
    return attr;
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
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  odsState.addTypes(output);
}

void FullyConnectedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value weight, ::mlir::Value bias, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(weight);
  odsState.addOperands(bias);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
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
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps8(*this, tblgen_quantization_info, "quantization_info")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps12(*this, v.getType(), "result", index++)))
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

namespace detail {
GatherOpGenericAdaptorBase::GatherOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.gather", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GatherOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GatherOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
GatherOpAdaptor::GatherOpAdaptor(GatherOp op) : GatherOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::RankedTensorType> GatherOp::getIndices() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps13(*this, v.getType(), "operand", index++)))
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

namespace detail {
GreaterEqualOpGenericAdaptorBase::GreaterEqualOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.greater_equal", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GreaterEqualOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GreaterEqualOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
GreaterEqualOpAdaptor::GreaterEqualOpAdaptor(GreaterEqualOp op) : GreaterEqualOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> GreaterEqualOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "result", index++)))
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

namespace detail {
GreaterOpGenericAdaptorBase::GreaterOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.greater", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GreaterOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GreaterOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
GreaterOpAdaptor::GreaterOpAdaptor(GreaterOp op) : GreaterOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> GreaterOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "result", index++)))
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

namespace detail {
IdentityOpGenericAdaptorBase::IdentityOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.identity", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> IdentityOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr IdentityOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
IdentityOpAdaptor::IdentityOpAdaptor(IdentityOp op) : IdentityOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
IfOpGenericAdaptorBase::IfOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.cond_if", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> IfOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr IfOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &IfOpGenericAdaptorBase::getThenBranch() {
  return *odsRegions[0];
}

::mlir::Region &IfOpGenericAdaptorBase::getElseBranch() {
  return *odsRegions[1];
}

::mlir::RegionRange IfOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
IfOpAdaptor::IfOpAdaptor(IfOp op) : IfOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "operand", index++)))
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

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_TosaOps0(*this, region, "then_branch", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(1)))
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

namespace detail {
LogOpGenericAdaptorBase::LogOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.log", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LogOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LogOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
LogOpAdaptor::LogOpAdaptor(LogOp op) : LogOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
LogicalAndOpGenericAdaptorBase::LogicalAndOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.logical_and", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LogicalAndOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LogicalAndOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
LogicalAndOpAdaptor::LogicalAndOpAdaptor(LogicalAndOp op) : LogicalAndOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> LogicalAndOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "operand", index++)))
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

namespace detail {
LogicalLeftShiftOpGenericAdaptorBase::LogicalLeftShiftOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.logical_left_shift", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LogicalLeftShiftOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LogicalLeftShiftOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
LogicalLeftShiftOpAdaptor::LogicalLeftShiftOpAdaptor(LogicalLeftShiftOp op) : LogicalLeftShiftOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> LogicalLeftShiftOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
LogicalNotOpGenericAdaptorBase::LogicalNotOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.logical_not", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LogicalNotOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LogicalNotOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
LogicalNotOpAdaptor::LogicalNotOpAdaptor(LogicalNotOp op) : LogicalNotOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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
                      odsState.getRawProperties(),
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
          odsState.getRawProperties(),
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "operand", index++)))
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

::mlir::LogicalResult LogicalNotOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult LogicalNotOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
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

namespace detail {
LogicalOrOpGenericAdaptorBase::LogicalOrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.logical_or", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LogicalOrOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LogicalOrOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
LogicalOrOpAdaptor::LogicalOrOpAdaptor(LogicalOrOp op) : LogicalOrOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> LogicalOrOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "operand", index++)))
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

namespace detail {
LogicalRightShiftOpGenericAdaptorBase::LogicalRightShiftOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.logical_right_shift", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LogicalRightShiftOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LogicalRightShiftOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
LogicalRightShiftOpAdaptor::LogicalRightShiftOpAdaptor(LogicalRightShiftOp op) : LogicalRightShiftOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> LogicalRightShiftOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
LogicalXorOpGenericAdaptorBase::LogicalXorOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.logical_xor", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LogicalXorOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LogicalXorOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
LogicalXorOpAdaptor::LogicalXorOpAdaptor(LogicalXorOp op) : LogicalXorOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> LogicalXorOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "operand", index++)))
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

namespace detail {
MatMulOpGenericAdaptorBase::MatMulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.matmul", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> MatMulOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr MatMulOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

mlir::tosa::MatMulOpQuantizationAttr MatMulOpGenericAdaptorBase::getQuantizationInfoAttr() {
  auto attr = getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::MatMulOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::MatMulOpQuantizationAttr> MatMulOpGenericAdaptorBase::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::MatMulOpQuantizationAttr>(::llvm::cast<mlir::tosa::MatMulOpQuantizationAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
MatMulOpAdaptor::MatMulOpAdaptor(MatMulOp op) : MatMulOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult MatMulOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;

  if (tblgen_quantization_info && !((::llvm::isa<mlir::tosa::MatMulOpQuantizationAttr>(tblgen_quantization_info))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value MatMulOp::getB() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult MatMulOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.quantization_info;
    auto attr = dict.get("quantization_info");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for quantization_info in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `quantization_info` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute MatMulOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.quantization_info;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("quantization_info",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code MatMulOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.quantization_info.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> MatMulOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "quantization_info")
      return prop.quantization_info;
  return std::nullopt;
}

void MatMulOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "quantization_info") {
       prop.quantization_info = dyn_cast_or_null<std::remove_reference_t<decltype(prop.quantization_info)>>(value);
       return;
    }
}

void MatMulOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.quantization_info) attrs.append("quantization_info", prop.quantization_info);
}

::mlir::LogicalResult MatMulOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getQuantizationInfoAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps12(attr, "quantization_info", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

mlir::tosa::MatMulOpQuantizationAttr MatMulOp::getQuantizationInfoAttr() {
  return getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::MatMulOpQuantizationAttr>();
}

::std::optional<mlir::tosa::MatMulOpQuantizationAttr> MatMulOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::MatMulOpQuantizationAttr>(::llvm::cast<mlir::tosa::MatMulOpQuantizationAttr>(attr)) : (::std::nullopt);
}

void MatMulOp::setQuantizationInfoAttr(mlir::tosa::MatMulOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute MatMulOp::removeQuantizationInfoAttr() {
    auto &attr = getProperties().quantization_info;
    attr = {};
    return attr;
}

void MatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value a, Value b) {
    buildMatMulOpWithQuantInfo(odsBuilder, odsState, outputType,
                               a, b);
  
}

void MatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type c, ::mlir::Value a, ::mlir::Value b, /*optional*/mlir::tosa::MatMulOpQuantizationAttr quantization_info) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  odsState.addTypes(c);
}

void MatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, /*optional*/mlir::tosa::MatMulOpQuantizationAttr quantization_info) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
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
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps12(*this, tblgen_quantization_info, "quantization_info")))
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

namespace detail {
MaxPool2dOpGenericAdaptorBase::MaxPool2dOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.max_pool2d", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> MaxPool2dOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr MaxPool2dOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr MaxPool2dOpGenericAdaptorBase::getKernelAttr() {
  auto attr = getProperties().kernel.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> MaxPool2dOpGenericAdaptorBase::getKernel() {
  auto attr = getKernelAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr MaxPool2dOpGenericAdaptorBase::getStrideAttr() {
  auto attr = getProperties().stride.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> MaxPool2dOpGenericAdaptorBase::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr MaxPool2dOpGenericAdaptorBase::getPadAttr() {
  auto attr = getProperties().pad.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> MaxPool2dOpGenericAdaptorBase::getPad() {
  auto attr = getPadAttr();
  return attr;
}

} // namespace detail
MaxPool2dOpAdaptor::MaxPool2dOpAdaptor(MaxPool2dOp op) : MaxPool2dOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult MaxPool2dOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_kernel = getProperties().kernel; (void)tblgen_kernel;
  if (!tblgen_kernel) return emitError(loc, "'tosa.max_pool2d' op ""requires attribute 'kernel'");
  auto tblgen_pad = getProperties().pad; (void)tblgen_pad;
  if (!tblgen_pad) return emitError(loc, "'tosa.max_pool2d' op ""requires attribute 'pad'");
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitError(loc, "'tosa.max_pool2d' op ""requires attribute 'stride'");

  if (tblgen_kernel && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_kernel))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_kernel).size() == 2))))
    return emitError(loc, "'tosa.max_pool2d' op ""attribute 'kernel' failed to satisfy constraint: i64 dense array attribute with exactly 2 elements");

  if (tblgen_stride && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_stride))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_stride).size() == 2))))
    return emitError(loc, "'tosa.max_pool2d' op ""attribute 'stride' failed to satisfy constraint: i64 dense array attribute with exactly 2 elements");

  if (tblgen_pad && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_pad))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_pad).size() == 4))))
    return emitError(loc, "'tosa.max_pool2d' op ""attribute 'pad' failed to satisfy constraint: i64 dense array attribute with exactly 4 elements");
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult MaxPool2dOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.kernel;
    auto attr = dict.get("kernel");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for kernel in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `kernel` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.pad;
    auto attr = dict.get("pad");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for pad in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `pad` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.stride;
    auto attr = dict.get("stride");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for stride in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `stride` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute MaxPool2dOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.kernel;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("kernel",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.pad;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("pad",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.stride;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("stride",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code MaxPool2dOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.kernel.getAsOpaquePointer()), 
    llvm::hash_value(prop.pad.getAsOpaquePointer()), 
    llvm::hash_value(prop.stride.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> MaxPool2dOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "kernel")
      return prop.kernel;

    if (name == "pad")
      return prop.pad;

    if (name == "stride")
      return prop.stride;
  return std::nullopt;
}

void MaxPool2dOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "kernel") {
       prop.kernel = dyn_cast_or_null<std::remove_reference_t<decltype(prop.kernel)>>(value);
       return;
    }

    if (name == "pad") {
       prop.pad = dyn_cast_or_null<std::remove_reference_t<decltype(prop.pad)>>(value);
       return;
    }

    if (name == "stride") {
       prop.stride = dyn_cast_or_null<std::remove_reference_t<decltype(prop.stride)>>(value);
       return;
    }
}

void MaxPool2dOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.kernel) attrs.append("kernel", prop.kernel);

    if (prop.pad) attrs.append("pad", prop.pad);

    if (prop.stride) attrs.append("stride", prop.stride);
}

::mlir::LogicalResult MaxPool2dOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getKernelAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(attr, "kernel", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPadAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(attr, "pad", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStrideAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(attr, "stride", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::DenseI64ArrayAttr MaxPool2dOp::getKernelAttr() {
  return getProperties().kernel.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> MaxPool2dOp::getKernel() {
  auto attr = getKernelAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr MaxPool2dOp::getStrideAttr() {
  return getProperties().stride.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> MaxPool2dOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr MaxPool2dOp::getPadAttr() {
  return getProperties().pad.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> MaxPool2dOp::getPad() {
  auto attr = getPadAttr();
  return attr;
}

void MaxPool2dOp::setKernelAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getKernelAttrName(), attr);
}

void MaxPool2dOp::setKernel(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getKernelAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void MaxPool2dOp::setStrideAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStrideAttrName(), attr);
}

void MaxPool2dOp::setStride(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStrideAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void MaxPool2dOp::setPadAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getPadAttrName(), attr);
}

void MaxPool2dOp::setPad(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getPadAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void MaxPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::DenseI64ArrayAttr kernel, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr pad) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().kernel = kernel;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().pad = pad;
  odsState.addTypes(output);
}

void MaxPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::DenseI64ArrayAttr kernel, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr pad) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().kernel = kernel;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().pad = pad;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaxPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::llvm::ArrayRef<int64_t> kernel, ::llvm::ArrayRef<int64_t> stride, ::llvm::ArrayRef<int64_t> pad) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().kernel = odsBuilder.getDenseI64ArrayAttr(kernel);
  odsState.getOrAddProperties<Properties>().stride = odsBuilder.getDenseI64ArrayAttr(stride);
  odsState.getOrAddProperties<Properties>().pad = odsBuilder.getDenseI64ArrayAttr(pad);
  odsState.addTypes(output);
}

void MaxPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::llvm::ArrayRef<int64_t> kernel, ::llvm::ArrayRef<int64_t> stride, ::llvm::ArrayRef<int64_t> pad) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().kernel = odsBuilder.getDenseI64ArrayAttr(kernel);
  odsState.getOrAddProperties<Properties>().stride = odsBuilder.getDenseI64ArrayAttr(stride);
  odsState.getOrAddProperties<Properties>().pad = odsBuilder.getDenseI64ArrayAttr(pad);
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
  auto tblgen_kernel = getProperties().kernel; (void)tblgen_kernel;
  if (!tblgen_kernel) return emitOpError("requires attribute 'kernel'");
  auto tblgen_pad = getProperties().pad; (void)tblgen_pad;
  if (!tblgen_pad) return emitOpError("requires attribute 'pad'");
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitOpError("requires attribute 'stride'");

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

namespace detail {
MaximumOpGenericAdaptorBase::MaximumOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.maximum", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> MaximumOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr MaximumOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
MaximumOpAdaptor::MaximumOpAdaptor(MaximumOp op) : MaximumOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> MaximumOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
MinimumOpGenericAdaptorBase::MinimumOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.minimum", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> MinimumOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr MinimumOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
MinimumOpAdaptor::MinimumOpAdaptor(MinimumOp op) : MinimumOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> MinimumOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
MulOpGenericAdaptorBase::MulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.mul", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> MulOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr MulOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MulOpGenericAdaptorBase::getShiftAttr() {
  auto attr = getProperties().shift.cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t MulOpGenericAdaptorBase::getShift() {
  auto attr = getShiftAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
MulOpAdaptor::MulOpAdaptor(MulOp op) : MulOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult MulOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_shift = getProperties().shift; (void)tblgen_shift;
  if (!tblgen_shift) return emitError(loc, "'tosa.mul' op ""requires attribute 'shift'");

  if (tblgen_shift && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_shift))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_shift).getType().isSignlessInteger(32)))))
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> MulOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult MulOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.shift;
    auto attr = dict.get("shift");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for shift in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `shift` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute MulOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.shift;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("shift",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code MulOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.shift.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> MulOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "shift")
      return prop.shift;
  return std::nullopt;
}

void MulOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "shift") {
       prop.shift = dyn_cast_or_null<std::remove_reference_t<decltype(prop.shift)>>(value);
       return;
    }
}

void MulOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.shift) attrs.append("shift", prop.shift);
}

::mlir::LogicalResult MulOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getShiftAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps13(attr, "shift", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr MulOp::getShiftAttr() {
  return getProperties().shift.cast<::mlir::IntegerAttr>();
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
  odsState.getOrAddProperties<Properties>().shift = shift;
  odsState.addTypes(output);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, ::mlir::IntegerAttr shift) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.getOrAddProperties<Properties>().shift = shift;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, uint32_t shift) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.getOrAddProperties<Properties>().shift = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), shift);
  odsState.addTypes(output);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value input2, uint32_t shift) {
  odsState.addOperands(input1);
  odsState.addOperands(input2);
  odsState.getOrAddProperties<Properties>().shift = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), shift);
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
  auto tblgen_shift = getProperties().shift; (void)tblgen_shift;
  if (!tblgen_shift) return emitOpError("requires attribute 'shift'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps13(*this, tblgen_shift, "shift")))
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

namespace detail {
NegateOpGenericAdaptorBase::NegateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.negate", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> NegateOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr NegateOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

mlir::tosa::UnaryOpQuantizationAttr NegateOpGenericAdaptorBase::getQuantizationInfoAttr() {
  auto attr = getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::UnaryOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::UnaryOpQuantizationAttr> NegateOpGenericAdaptorBase::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::UnaryOpQuantizationAttr>(::llvm::cast<mlir::tosa::UnaryOpQuantizationAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
NegateOpAdaptor::NegateOpAdaptor(NegateOp op) : NegateOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult NegateOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;

  if (tblgen_quantization_info && !((::llvm::isa<mlir::tosa::UnaryOpQuantizationAttr>(tblgen_quantization_info))))
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult NegateOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.quantization_info;
    auto attr = dict.get("quantization_info");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for quantization_info in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `quantization_info` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute NegateOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.quantization_info;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("quantization_info",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code NegateOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.quantization_info.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> NegateOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "quantization_info")
      return prop.quantization_info;
  return std::nullopt;
}

void NegateOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "quantization_info") {
       prop.quantization_info = dyn_cast_or_null<std::remove_reference_t<decltype(prop.quantization_info)>>(value);
       return;
    }
}

void NegateOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.quantization_info) attrs.append("quantization_info", prop.quantization_info);
}

::mlir::LogicalResult NegateOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getQuantizationInfoAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps5(attr, "quantization_info", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

mlir::tosa::UnaryOpQuantizationAttr NegateOp::getQuantizationInfoAttr() {
  return getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::UnaryOpQuantizationAttr>();
}

::std::optional<mlir::tosa::UnaryOpQuantizationAttr> NegateOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::UnaryOpQuantizationAttr>(::llvm::cast<mlir::tosa::UnaryOpQuantizationAttr>(attr)) : (::std::nullopt);
}

void NegateOp::setQuantizationInfoAttr(mlir::tosa::UnaryOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute NegateOp::removeQuantizationInfoAttr() {
    auto &attr = getProperties().quantization_info;
    attr = {};
    return attr;
}

void NegateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type outputType, Value input) {
    buildUnaryOpWithQuantInfo(odsBuilder, odsState, outputType, input);
  
}

void NegateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input1);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  odsState.addTypes(output);
}

void NegateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, /*optional*/mlir::tosa::UnaryOpQuantizationAttr quantization_info) {
  odsState.addOperands(input1);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
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
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps5(*this, tblgen_quantization_info, "quantization_info")))
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

namespace detail {
PadOpGenericAdaptorBase::PadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.pad", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PadOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr PadOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

mlir::tosa::PadOpQuantizationAttr PadOpGenericAdaptorBase::getQuantizationInfoAttr() {
  auto attr = getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::PadOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::PadOpQuantizationAttr> PadOpGenericAdaptorBase::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::PadOpQuantizationAttr>(::llvm::cast<mlir::tosa::PadOpQuantizationAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
PadOpAdaptor::PadOpAdaptor(PadOp op) : PadOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult PadOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;

  if (tblgen_quantization_info && !((::llvm::isa<mlir::tosa::PadOpQuantizationAttr>(tblgen_quantization_info))))
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

::mlir::TypedValue<::mlir::RankedTensorType> PadOp::getInput1() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> PadOp::getPadding() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
}

::mlir::TypedValue<::mlir::RankedTensorType> PadOp::getPadConst() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::TypedValue<::mlir::RankedTensorType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*operands.begin());
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

::mlir::TypedValue<::mlir::RankedTensorType> PadOp::getOutput() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult PadOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.quantization_info;
    auto attr = dict.get("quantization_info");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for quantization_info in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `quantization_info` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute PadOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.quantization_info;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("quantization_info",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code PadOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.quantization_info.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> PadOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "quantization_info")
      return prop.quantization_info;
  return std::nullopt;
}

void PadOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "quantization_info") {
       prop.quantization_info = dyn_cast_or_null<std::remove_reference_t<decltype(prop.quantization_info)>>(value);
       return;
    }
}

void PadOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.quantization_info) attrs.append("quantization_info", prop.quantization_info);
}

::mlir::LogicalResult PadOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getQuantizationInfoAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps14(attr, "quantization_info", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

mlir::tosa::PadOpQuantizationAttr PadOp::getQuantizationInfoAttr() {
  return getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::PadOpQuantizationAttr>();
}

::std::optional<mlir::tosa::PadOpQuantizationAttr> PadOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::PadOpQuantizationAttr>(::llvm::cast<mlir::tosa::PadOpQuantizationAttr>(attr)) : (::std::nullopt);
}

void PadOp::setQuantizationInfoAttr(mlir::tosa::PadOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute PadOp::removeQuantizationInfoAttr() {
    auto &attr = getProperties().quantization_info;
    attr = {};
    return attr;
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
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  odsState.addTypes(output);
}

void PadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::Value padding, /*optional*/::mlir::Value pad_const, /*optional*/mlir::tosa::PadOpQuantizationAttr quantization_info) {
  odsState.addOperands(input1);
  odsState.addOperands(padding);
  if (pad_const)
    odsState.addOperands(pad_const);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
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
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps14(*this, tblgen_quantization_info, "quantization_info")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps15(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps16(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps14(*this, v.getType(), "result", index++)))
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

namespace detail {
PowOpGenericAdaptorBase::PowOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.pow", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PowOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr PowOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
PowOpAdaptor::PowOpAdaptor(PowOp op) : PowOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> PowOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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
// mlir::tosa::RFFT2dOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
RFFT2dOpGenericAdaptorBase::RFFT2dOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.rfft2d", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> RFFT2dOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr RFFT2dOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
RFFT2dOpAdaptor::RFFT2dOpAdaptor(RFFT2dOp op) : RFFT2dOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult RFFT2dOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void RFFT2dOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "output_real");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "output_imag");
}

std::pair<unsigned, unsigned> RFFT2dOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range RFFT2dOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RFFT2dOp::getInput() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange RFFT2dOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> RFFT2dOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RFFT2dOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RFFT2dOp::getOutputReal() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::Value RFFT2dOp::getOutputImag() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(1).begin());
}

void RFFT2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output_real, ::mlir::Type output_imag, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(output_real);
  odsState.addTypes(output_imag);
}

void RFFT2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RFFT2dOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RFFT2dOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
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
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps11(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult RFFT2dOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void RFFT2dOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tosa
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::RFFT2dOp)

namespace mlir {
namespace tosa {

//===----------------------------------------------------------------------===//
// mlir::tosa::ReciprocalOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReciprocalOpGenericAdaptorBase::ReciprocalOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reciprocal", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReciprocalOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReciprocalOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ReciprocalOpAdaptor::ReciprocalOpAdaptor(ReciprocalOp op) : ReciprocalOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
ReduceAllOpGenericAdaptorBase::ReduceAllOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reduce_all", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReduceAllOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReduceAllOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceAllOpGenericAdaptorBase::getAxisAttr() {
  auto attr = getProperties().axis.cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ReduceAllOpGenericAdaptorBase::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ReduceAllOpAdaptor::ReduceAllOpAdaptor(ReduceAllOp op) : ReduceAllOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ReduceAllOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitError(loc, "'tosa.reduce_all' op ""requires attribute 'axis'");

  if (tblgen_axis && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_axis))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_axis).getType().isSignlessInteger(64)))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult ReduceAllOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.axis;
    auto attr = dict.get("axis");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for axis in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `axis` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ReduceAllOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.axis;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("axis",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ReduceAllOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.axis.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ReduceAllOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "axis")
      return prop.axis;
  return std::nullopt;
}

void ReduceAllOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "axis") {
       prop.axis = dyn_cast_or_null<std::remove_reference_t<decltype(prop.axis)>>(value);
       return;
    }
}

void ReduceAllOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.axis) attrs.append("axis", prop.axis);
}

::mlir::LogicalResult ReduceAllOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getAxisAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(attr, "axis", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr ReduceAllOp::getAxisAttr() {
  return getProperties().axis.cast<::mlir::IntegerAttr>();
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
  odsState.getOrAddProperties<Properties>().axis = axis;
  odsState.addTypes(output);
}

void ReduceAllOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = axis;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReduceAllOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReduceAllOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = axis;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceAllOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
  odsState.addTypes(output);
}

void ReduceAllOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReduceAllOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReduceAllOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
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

void ReduceAllOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ReduceAllOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ReduceAllOp::verifyInvariantsImpl() {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitOpError("requires attribute 'axis'");

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

namespace detail {
ReduceAnyOpGenericAdaptorBase::ReduceAnyOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reduce_any", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReduceAnyOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReduceAnyOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceAnyOpGenericAdaptorBase::getAxisAttr() {
  auto attr = getProperties().axis.cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ReduceAnyOpGenericAdaptorBase::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ReduceAnyOpAdaptor::ReduceAnyOpAdaptor(ReduceAnyOp op) : ReduceAnyOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ReduceAnyOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitError(loc, "'tosa.reduce_any' op ""requires attribute 'axis'");

  if (tblgen_axis && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_axis))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_axis).getType().isSignlessInteger(64)))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult ReduceAnyOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.axis;
    auto attr = dict.get("axis");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for axis in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `axis` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ReduceAnyOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.axis;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("axis",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ReduceAnyOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.axis.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ReduceAnyOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "axis")
      return prop.axis;
  return std::nullopt;
}

void ReduceAnyOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "axis") {
       prop.axis = dyn_cast_or_null<std::remove_reference_t<decltype(prop.axis)>>(value);
       return;
    }
}

void ReduceAnyOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.axis) attrs.append("axis", prop.axis);
}

::mlir::LogicalResult ReduceAnyOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getAxisAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(attr, "axis", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr ReduceAnyOp::getAxisAttr() {
  return getProperties().axis.cast<::mlir::IntegerAttr>();
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
  odsState.getOrAddProperties<Properties>().axis = axis;
  odsState.addTypes(output);
}

void ReduceAnyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = axis;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReduceAnyOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReduceAnyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = axis;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceAnyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
  odsState.addTypes(output);
}

void ReduceAnyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReduceAnyOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReduceAnyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
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

void ReduceAnyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ReduceAnyOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ReduceAnyOp::verifyInvariantsImpl() {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitOpError("requires attribute 'axis'");

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

namespace detail {
ReduceMaxOpGenericAdaptorBase::ReduceMaxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reduce_max", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReduceMaxOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReduceMaxOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceMaxOpGenericAdaptorBase::getAxisAttr() {
  auto attr = getProperties().axis.cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ReduceMaxOpGenericAdaptorBase::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ReduceMaxOpAdaptor::ReduceMaxOpAdaptor(ReduceMaxOp op) : ReduceMaxOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ReduceMaxOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitError(loc, "'tosa.reduce_max' op ""requires attribute 'axis'");

  if (tblgen_axis && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_axis))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_axis).getType().isSignlessInteger(64)))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult ReduceMaxOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.axis;
    auto attr = dict.get("axis");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for axis in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `axis` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ReduceMaxOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.axis;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("axis",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ReduceMaxOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.axis.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ReduceMaxOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "axis")
      return prop.axis;
  return std::nullopt;
}

void ReduceMaxOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "axis") {
       prop.axis = dyn_cast_or_null<std::remove_reference_t<decltype(prop.axis)>>(value);
       return;
    }
}

void ReduceMaxOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.axis) attrs.append("axis", prop.axis);
}

::mlir::LogicalResult ReduceMaxOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getAxisAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(attr, "axis", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr ReduceMaxOp::getAxisAttr() {
  return getProperties().axis.cast<::mlir::IntegerAttr>();
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
  odsState.getOrAddProperties<Properties>().axis = axis;
  odsState.addTypes(output);
}

void ReduceMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = axis;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReduceMaxOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReduceMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = axis;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
  odsState.addTypes(output);
}

void ReduceMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReduceMaxOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReduceMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
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

void ReduceMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ReduceMaxOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ReduceMaxOp::verifyInvariantsImpl() {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitOpError("requires attribute 'axis'");

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

namespace detail {
ReduceMinOpGenericAdaptorBase::ReduceMinOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reduce_min", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReduceMinOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReduceMinOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceMinOpGenericAdaptorBase::getAxisAttr() {
  auto attr = getProperties().axis.cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ReduceMinOpGenericAdaptorBase::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ReduceMinOpAdaptor::ReduceMinOpAdaptor(ReduceMinOp op) : ReduceMinOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ReduceMinOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitError(loc, "'tosa.reduce_min' op ""requires attribute 'axis'");

  if (tblgen_axis && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_axis))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_axis).getType().isSignlessInteger(64)))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult ReduceMinOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.axis;
    auto attr = dict.get("axis");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for axis in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `axis` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ReduceMinOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.axis;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("axis",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ReduceMinOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.axis.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ReduceMinOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "axis")
      return prop.axis;
  return std::nullopt;
}

void ReduceMinOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "axis") {
       prop.axis = dyn_cast_or_null<std::remove_reference_t<decltype(prop.axis)>>(value);
       return;
    }
}

void ReduceMinOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.axis) attrs.append("axis", prop.axis);
}

::mlir::LogicalResult ReduceMinOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getAxisAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(attr, "axis", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr ReduceMinOp::getAxisAttr() {
  return getProperties().axis.cast<::mlir::IntegerAttr>();
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
  odsState.getOrAddProperties<Properties>().axis = axis;
  odsState.addTypes(output);
}

void ReduceMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = axis;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReduceMinOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReduceMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = axis;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
  odsState.addTypes(output);
}

void ReduceMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReduceMinOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReduceMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
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

void ReduceMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ReduceMinOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ReduceMinOp::verifyInvariantsImpl() {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitOpError("requires attribute 'axis'");

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

namespace detail {
ReduceProdOpGenericAdaptorBase::ReduceProdOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reduce_prod", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReduceProdOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReduceProdOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceProdOpGenericAdaptorBase::getAxisAttr() {
  auto attr = getProperties().axis.cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ReduceProdOpGenericAdaptorBase::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ReduceProdOpAdaptor::ReduceProdOpAdaptor(ReduceProdOp op) : ReduceProdOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ReduceProdOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitError(loc, "'tosa.reduce_prod' op ""requires attribute 'axis'");

  if (tblgen_axis && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_axis))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_axis).getType().isSignlessInteger(64)))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult ReduceProdOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.axis;
    auto attr = dict.get("axis");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for axis in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `axis` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ReduceProdOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.axis;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("axis",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ReduceProdOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.axis.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ReduceProdOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "axis")
      return prop.axis;
  return std::nullopt;
}

void ReduceProdOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "axis") {
       prop.axis = dyn_cast_or_null<std::remove_reference_t<decltype(prop.axis)>>(value);
       return;
    }
}

void ReduceProdOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.axis) attrs.append("axis", prop.axis);
}

::mlir::LogicalResult ReduceProdOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getAxisAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(attr, "axis", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr ReduceProdOp::getAxisAttr() {
  return getProperties().axis.cast<::mlir::IntegerAttr>();
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
  odsState.getOrAddProperties<Properties>().axis = axis;
  odsState.addTypes(output);
}

void ReduceProdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = axis;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReduceProdOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReduceProdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = axis;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceProdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
  odsState.addTypes(output);
}

void ReduceProdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReduceProdOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReduceProdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
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

void ReduceProdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ReduceProdOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ReduceProdOp::verifyInvariantsImpl() {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitOpError("requires attribute 'axis'");

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

namespace detail {
ReduceSumOpGenericAdaptorBase::ReduceSumOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reduce_sum", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReduceSumOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReduceSumOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReduceSumOpGenericAdaptorBase::getAxisAttr() {
  auto attr = getProperties().axis.cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ReduceSumOpGenericAdaptorBase::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ReduceSumOpAdaptor::ReduceSumOpAdaptor(ReduceSumOp op) : ReduceSumOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ReduceSumOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitError(loc, "'tosa.reduce_sum' op ""requires attribute 'axis'");

  if (tblgen_axis && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_axis))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_axis).getType().isSignlessInteger(64)))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult ReduceSumOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.axis;
    auto attr = dict.get("axis");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for axis in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `axis` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ReduceSumOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.axis;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("axis",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ReduceSumOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.axis.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ReduceSumOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "axis")
      return prop.axis;
  return std::nullopt;
}

void ReduceSumOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "axis") {
       prop.axis = dyn_cast_or_null<std::remove_reference_t<decltype(prop.axis)>>(value);
       return;
    }
}

void ReduceSumOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.axis) attrs.append("axis", prop.axis);
}

::mlir::LogicalResult ReduceSumOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getAxisAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(attr, "axis", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr ReduceSumOp::getAxisAttr() {
  return getProperties().axis.cast<::mlir::IntegerAttr>();
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
  odsState.getOrAddProperties<Properties>().axis = axis;
  odsState.addTypes(output);
}

void ReduceSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = axis;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReduceSumOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReduceSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = axis;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
  odsState.addTypes(output);
}

void ReduceSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReduceSumOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReduceSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
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

void ReduceSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ReduceSumOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ReduceSumOp::verifyInvariantsImpl() {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitOpError("requires attribute 'axis'");

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

namespace detail {
RescaleOpGenericAdaptorBase::RescaleOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.rescale", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> RescaleOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr RescaleOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr RescaleOpGenericAdaptorBase::getInputZpAttr() {
  auto attr = getProperties().input_zp.cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t RescaleOpGenericAdaptorBase::getInputZp() {
  auto attr = getInputZpAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr RescaleOpGenericAdaptorBase::getOutputZpAttr() {
  auto attr = getProperties().output_zp.cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t RescaleOpGenericAdaptorBase::getOutputZp() {
  auto attr = getOutputZpAttr();
  return attr.getValue().getZExtValue();
}

::mlir::DenseI32ArrayAttr RescaleOpGenericAdaptorBase::getMultiplierAttr() {
  auto attr = getProperties().multiplier.cast<::mlir::DenseI32ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int32_t> RescaleOpGenericAdaptorBase::getMultiplier() {
  auto attr = getMultiplierAttr();
  return attr;
}

::mlir::DenseI32ArrayAttr RescaleOpGenericAdaptorBase::getShiftAttr() {
  auto attr = getProperties().shift.cast<::mlir::DenseI32ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int32_t> RescaleOpGenericAdaptorBase::getShift() {
  auto attr = getShiftAttr();
  return attr;
}

::mlir::BoolAttr RescaleOpGenericAdaptorBase::getScale32Attr() {
  auto attr = getProperties().scale32.cast<::mlir::BoolAttr>();
  return attr;
}

bool RescaleOpGenericAdaptorBase::getScale32() {
  auto attr = getScale32Attr();
  return attr.getValue();
}

::mlir::BoolAttr RescaleOpGenericAdaptorBase::getDoubleRoundAttr() {
  auto attr = getProperties().double_round.cast<::mlir::BoolAttr>();
  return attr;
}

bool RescaleOpGenericAdaptorBase::getDoubleRound() {
  auto attr = getDoubleRoundAttr();
  return attr.getValue();
}

::mlir::BoolAttr RescaleOpGenericAdaptorBase::getPerChannelAttr() {
  auto attr = getProperties().per_channel.cast<::mlir::BoolAttr>();
  return attr;
}

bool RescaleOpGenericAdaptorBase::getPerChannel() {
  auto attr = getPerChannelAttr();
  return attr.getValue();
}

} // namespace detail
RescaleOpAdaptor::RescaleOpAdaptor(RescaleOp op) : RescaleOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult RescaleOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_double_round = getProperties().double_round; (void)tblgen_double_round;
  if (!tblgen_double_round) return emitError(loc, "'tosa.rescale' op ""requires attribute 'double_round'");
  auto tblgen_input_zp = getProperties().input_zp; (void)tblgen_input_zp;
  if (!tblgen_input_zp) return emitError(loc, "'tosa.rescale' op ""requires attribute 'input_zp'");
  auto tblgen_multiplier = getProperties().multiplier; (void)tblgen_multiplier;
  if (!tblgen_multiplier) return emitError(loc, "'tosa.rescale' op ""requires attribute 'multiplier'");
  auto tblgen_output_zp = getProperties().output_zp; (void)tblgen_output_zp;
  if (!tblgen_output_zp) return emitError(loc, "'tosa.rescale' op ""requires attribute 'output_zp'");
  auto tblgen_per_channel = getProperties().per_channel; (void)tblgen_per_channel;
  if (!tblgen_per_channel) return emitError(loc, "'tosa.rescale' op ""requires attribute 'per_channel'");
  auto tblgen_scale32 = getProperties().scale32; (void)tblgen_scale32;
  if (!tblgen_scale32) return emitError(loc, "'tosa.rescale' op ""requires attribute 'scale32'");
  auto tblgen_shift = getProperties().shift; (void)tblgen_shift;
  if (!tblgen_shift) return emitError(loc, "'tosa.rescale' op ""requires attribute 'shift'");

  if (tblgen_input_zp && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_input_zp))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_input_zp).getType().isSignlessInteger(32)))))
    return emitError(loc, "'tosa.rescale' op ""attribute 'input_zp' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_output_zp && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_output_zp))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_output_zp).getType().isSignlessInteger(32)))))
    return emitError(loc, "'tosa.rescale' op ""attribute 'output_zp' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_multiplier && !((::llvm::isa<::mlir::DenseI32ArrayAttr>(tblgen_multiplier))))
    return emitError(loc, "'tosa.rescale' op ""attribute 'multiplier' failed to satisfy constraint: i32 dense array attribute");

  if (tblgen_shift && !((::llvm::isa<::mlir::DenseI32ArrayAttr>(tblgen_shift))))
    return emitError(loc, "'tosa.rescale' op ""attribute 'shift' failed to satisfy constraint: i32 dense array attribute");

  if (tblgen_scale32 && !((::llvm::isa<::mlir::BoolAttr>(tblgen_scale32))))
    return emitError(loc, "'tosa.rescale' op ""attribute 'scale32' failed to satisfy constraint: bool attribute");

  if (tblgen_double_round && !((::llvm::isa<::mlir::BoolAttr>(tblgen_double_round))))
    return emitError(loc, "'tosa.rescale' op ""attribute 'double_round' failed to satisfy constraint: bool attribute");

  if (tblgen_per_channel && !((::llvm::isa<::mlir::BoolAttr>(tblgen_per_channel))))
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult RescaleOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.double_round;
    auto attr = dict.get("double_round");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for double_round in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `double_round` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.input_zp;
    auto attr = dict.get("input_zp");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for input_zp in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `input_zp` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.multiplier;
    auto attr = dict.get("multiplier");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for multiplier in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `multiplier` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.output_zp;
    auto attr = dict.get("output_zp");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for output_zp in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `output_zp` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.per_channel;
    auto attr = dict.get("per_channel");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for per_channel in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `per_channel` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.scale32;
    auto attr = dict.get("scale32");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for scale32 in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `scale32` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.shift;
    auto attr = dict.get("shift");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for shift in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `shift` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute RescaleOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.double_round;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("double_round",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.input_zp;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("input_zp",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.multiplier;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("multiplier",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.output_zp;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("output_zp",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.per_channel;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("per_channel",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.scale32;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("scale32",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.shift;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("shift",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code RescaleOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.double_round.getAsOpaquePointer()), 
    llvm::hash_value(prop.input_zp.getAsOpaquePointer()), 
    llvm::hash_value(prop.multiplier.getAsOpaquePointer()), 
    llvm::hash_value(prop.output_zp.getAsOpaquePointer()), 
    llvm::hash_value(prop.per_channel.getAsOpaquePointer()), 
    llvm::hash_value(prop.scale32.getAsOpaquePointer()), 
    llvm::hash_value(prop.shift.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> RescaleOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "double_round")
      return prop.double_round;

    if (name == "input_zp")
      return prop.input_zp;

    if (name == "multiplier")
      return prop.multiplier;

    if (name == "output_zp")
      return prop.output_zp;

    if (name == "per_channel")
      return prop.per_channel;

    if (name == "scale32")
      return prop.scale32;

    if (name == "shift")
      return prop.shift;
  return std::nullopt;
}

void RescaleOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "double_round") {
       prop.double_round = dyn_cast_or_null<std::remove_reference_t<decltype(prop.double_round)>>(value);
       return;
    }

    if (name == "input_zp") {
       prop.input_zp = dyn_cast_or_null<std::remove_reference_t<decltype(prop.input_zp)>>(value);
       return;
    }

    if (name == "multiplier") {
       prop.multiplier = dyn_cast_or_null<std::remove_reference_t<decltype(prop.multiplier)>>(value);
       return;
    }

    if (name == "output_zp") {
       prop.output_zp = dyn_cast_or_null<std::remove_reference_t<decltype(prop.output_zp)>>(value);
       return;
    }

    if (name == "per_channel") {
       prop.per_channel = dyn_cast_or_null<std::remove_reference_t<decltype(prop.per_channel)>>(value);
       return;
    }

    if (name == "scale32") {
       prop.scale32 = dyn_cast_or_null<std::remove_reference_t<decltype(prop.scale32)>>(value);
       return;
    }

    if (name == "shift") {
       prop.shift = dyn_cast_or_null<std::remove_reference_t<decltype(prop.shift)>>(value);
       return;
    }
}

void RescaleOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.double_round) attrs.append("double_round", prop.double_round);

    if (prop.input_zp) attrs.append("input_zp", prop.input_zp);

    if (prop.multiplier) attrs.append("multiplier", prop.multiplier);

    if (prop.output_zp) attrs.append("output_zp", prop.output_zp);

    if (prop.per_channel) attrs.append("per_channel", prop.per_channel);

    if (prop.scale32) attrs.append("scale32", prop.scale32);

    if (prop.shift) attrs.append("shift", prop.shift);
}

::mlir::LogicalResult RescaleOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getDoubleRoundAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps0(attr, "double_round", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getInputZpAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps13(attr, "input_zp", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getMultiplierAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps15(attr, "multiplier", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getOutputZpAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps13(attr, "output_zp", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPerChannelAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps0(attr, "per_channel", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getScale32AttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps0(attr, "scale32", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getShiftAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps15(attr, "shift", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr RescaleOp::getInputZpAttr() {
  return getProperties().input_zp.cast<::mlir::IntegerAttr>();
}

uint32_t RescaleOp::getInputZp() {
  auto attr = getInputZpAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr RescaleOp::getOutputZpAttr() {
  return getProperties().output_zp.cast<::mlir::IntegerAttr>();
}

uint32_t RescaleOp::getOutputZp() {
  auto attr = getOutputZpAttr();
  return attr.getValue().getZExtValue();
}

::mlir::DenseI32ArrayAttr RescaleOp::getMultiplierAttr() {
  return getProperties().multiplier.cast<::mlir::DenseI32ArrayAttr>();
}

::llvm::ArrayRef<int32_t> RescaleOp::getMultiplier() {
  auto attr = getMultiplierAttr();
  return attr;
}

::mlir::DenseI32ArrayAttr RescaleOp::getShiftAttr() {
  return getProperties().shift.cast<::mlir::DenseI32ArrayAttr>();
}

::llvm::ArrayRef<int32_t> RescaleOp::getShift() {
  auto attr = getShiftAttr();
  return attr;
}

::mlir::BoolAttr RescaleOp::getScale32Attr() {
  return getProperties().scale32.cast<::mlir::BoolAttr>();
}

bool RescaleOp::getScale32() {
  auto attr = getScale32Attr();
  return attr.getValue();
}

::mlir::BoolAttr RescaleOp::getDoubleRoundAttr() {
  return getProperties().double_round.cast<::mlir::BoolAttr>();
}

bool RescaleOp::getDoubleRound() {
  auto attr = getDoubleRoundAttr();
  return attr.getValue();
}

::mlir::BoolAttr RescaleOp::getPerChannelAttr() {
  return getProperties().per_channel.cast<::mlir::BoolAttr>();
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

void RescaleOp::setMultiplierAttr(::mlir::DenseI32ArrayAttr attr) {
  (*this)->setAttr(getMultiplierAttrName(), attr);
}

void RescaleOp::setMultiplier(::llvm::ArrayRef<int32_t> attrValue) {
  (*this)->setAttr(getMultiplierAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI32ArrayAttr(attrValue));
}

void RescaleOp::setShiftAttr(::mlir::DenseI32ArrayAttr attr) {
  (*this)->setAttr(getShiftAttrName(), attr);
}

void RescaleOp::setShift(::llvm::ArrayRef<int32_t> attrValue) {
  (*this)->setAttr(getShiftAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI32ArrayAttr(attrValue));
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

void RescaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr input_zp, ::mlir::IntegerAttr output_zp, ::mlir::DenseI32ArrayAttr multiplier, ::mlir::DenseI32ArrayAttr shift, ::mlir::BoolAttr scale32, ::mlir::BoolAttr double_round, ::mlir::BoolAttr per_channel) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().input_zp = input_zp;
  odsState.getOrAddProperties<Properties>().output_zp = output_zp;
  odsState.getOrAddProperties<Properties>().multiplier = multiplier;
  odsState.getOrAddProperties<Properties>().shift = shift;
  odsState.getOrAddProperties<Properties>().scale32 = scale32;
  odsState.getOrAddProperties<Properties>().double_round = double_round;
  odsState.getOrAddProperties<Properties>().per_channel = per_channel;
  odsState.addTypes(output);
}

void RescaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr input_zp, ::mlir::IntegerAttr output_zp, ::mlir::DenseI32ArrayAttr multiplier, ::mlir::DenseI32ArrayAttr shift, ::mlir::BoolAttr scale32, ::mlir::BoolAttr double_round, ::mlir::BoolAttr per_channel) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().input_zp = input_zp;
  odsState.getOrAddProperties<Properties>().output_zp = output_zp;
  odsState.getOrAddProperties<Properties>().multiplier = multiplier;
  odsState.getOrAddProperties<Properties>().shift = shift;
  odsState.getOrAddProperties<Properties>().scale32 = scale32;
  odsState.getOrAddProperties<Properties>().double_round = double_round;
  odsState.getOrAddProperties<Properties>().per_channel = per_channel;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RescaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint32_t input_zp, uint32_t output_zp, ::llvm::ArrayRef<int32_t> multiplier, ::llvm::ArrayRef<int32_t> shift, bool scale32, bool double_round, bool per_channel) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().input_zp = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), input_zp);
  odsState.getOrAddProperties<Properties>().output_zp = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), output_zp);
  odsState.getOrAddProperties<Properties>().multiplier = odsBuilder.getDenseI32ArrayAttr(multiplier);
  odsState.getOrAddProperties<Properties>().shift = odsBuilder.getDenseI32ArrayAttr(shift);
  odsState.getOrAddProperties<Properties>().scale32 = odsBuilder.getBoolAttr(scale32);
  odsState.getOrAddProperties<Properties>().double_round = odsBuilder.getBoolAttr(double_round);
  odsState.getOrAddProperties<Properties>().per_channel = odsBuilder.getBoolAttr(per_channel);
  odsState.addTypes(output);
}

void RescaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t input_zp, uint32_t output_zp, ::llvm::ArrayRef<int32_t> multiplier, ::llvm::ArrayRef<int32_t> shift, bool scale32, bool double_round, bool per_channel) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().input_zp = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), input_zp);
  odsState.getOrAddProperties<Properties>().output_zp = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), output_zp);
  odsState.getOrAddProperties<Properties>().multiplier = odsBuilder.getDenseI32ArrayAttr(multiplier);
  odsState.getOrAddProperties<Properties>().shift = odsBuilder.getDenseI32ArrayAttr(shift);
  odsState.getOrAddProperties<Properties>().scale32 = odsBuilder.getBoolAttr(scale32);
  odsState.getOrAddProperties<Properties>().double_round = odsBuilder.getBoolAttr(double_round);
  odsState.getOrAddProperties<Properties>().per_channel = odsBuilder.getBoolAttr(per_channel);
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
  auto tblgen_double_round = getProperties().double_round; (void)tblgen_double_round;
  if (!tblgen_double_round) return emitOpError("requires attribute 'double_round'");
  auto tblgen_input_zp = getProperties().input_zp; (void)tblgen_input_zp;
  if (!tblgen_input_zp) return emitOpError("requires attribute 'input_zp'");
  auto tblgen_multiplier = getProperties().multiplier; (void)tblgen_multiplier;
  if (!tblgen_multiplier) return emitOpError("requires attribute 'multiplier'");
  auto tblgen_output_zp = getProperties().output_zp; (void)tblgen_output_zp;
  if (!tblgen_output_zp) return emitOpError("requires attribute 'output_zp'");
  auto tblgen_per_channel = getProperties().per_channel; (void)tblgen_per_channel;
  if (!tblgen_per_channel) return emitOpError("requires attribute 'per_channel'");
  auto tblgen_scale32 = getProperties().scale32; (void)tblgen_scale32;
  if (!tblgen_scale32) return emitOpError("requires attribute 'scale32'");
  auto tblgen_shift = getProperties().shift; (void)tblgen_shift;
  if (!tblgen_shift) return emitOpError("requires attribute 'shift'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps13(*this, tblgen_input_zp, "input_zp")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps13(*this, tblgen_output_zp, "output_zp")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps15(*this, tblgen_multiplier, "multiplier")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps15(*this, tblgen_shift, "shift")))
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

namespace detail {
ReshapeOpGenericAdaptorBase::ReshapeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reshape", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReshapeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReshapeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr ReshapeOpGenericAdaptorBase::getNewShapeAttr() {
  auto attr = getProperties().new_shape.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ReshapeOpGenericAdaptorBase::getNewShape() {
  auto attr = getNewShapeAttr();
  return attr;
}

} // namespace detail
ReshapeOpAdaptor::ReshapeOpAdaptor(ReshapeOp op) : ReshapeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ReshapeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_new_shape = getProperties().new_shape; (void)tblgen_new_shape;
  if (!tblgen_new_shape) return emitError(loc, "'tosa.reshape' op ""requires attribute 'new_shape'");

  if (tblgen_new_shape && !((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_new_shape))))
    return emitError(loc, "'tosa.reshape' op ""attribute 'new_shape' failed to satisfy constraint: i64 dense array attribute");
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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

::mlir::TypedValue<::mlir::RankedTensorType> ReshapeOp::getOutput() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult ReshapeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.new_shape;
    auto attr = dict.get("new_shape");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for new_shape in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `new_shape` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ReshapeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.new_shape;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("new_shape",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ReshapeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.new_shape.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ReshapeOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "new_shape")
      return prop.new_shape;
  return std::nullopt;
}

void ReshapeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "new_shape") {
       prop.new_shape = dyn_cast_or_null<std::remove_reference_t<decltype(prop.new_shape)>>(value);
       return;
    }
}

void ReshapeOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.new_shape) attrs.append("new_shape", prop.new_shape);
}

::mlir::LogicalResult ReshapeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getNewShapeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps16(attr, "new_shape", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::DenseI64ArrayAttr ReshapeOp::getNewShapeAttr() {
  return getProperties().new_shape.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ReshapeOp::getNewShape() {
  auto attr = getNewShapeAttr();
  return attr;
}

void ReshapeOp::setNewShapeAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getNewShapeAttrName(), attr);
}

void ReshapeOp::setNewShape(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getNewShapeAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::DenseI64ArrayAttr new_shape) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().new_shape = new_shape;
  odsState.addTypes(output);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1, ::mlir::DenseI64ArrayAttr new_shape) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().new_shape = new_shape;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReshapeOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::DenseI64ArrayAttr new_shape) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().new_shape = new_shape;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::llvm::ArrayRef<int64_t> new_shape) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().new_shape = odsBuilder.getDenseI64ArrayAttr(new_shape);
  odsState.addTypes(output);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input1, ::llvm::ArrayRef<int64_t> new_shape) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().new_shape = odsBuilder.getDenseI64ArrayAttr(new_shape);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReshapeOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::llvm::ArrayRef<int64_t> new_shape) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().new_shape = odsBuilder.getDenseI64ArrayAttr(new_shape);
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

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ReshapeOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ReshapeOp::verifyInvariantsImpl() {
  auto tblgen_new_shape = getProperties().new_shape; (void)tblgen_new_shape;
  if (!tblgen_new_shape) return emitOpError("requires attribute 'new_shape'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps16(*this, tblgen_new_shape, "new_shape")))
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReshapeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
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

namespace detail {
ResizeOpGenericAdaptorBase::ResizeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.resize", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ResizeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ResizeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr ResizeOpGenericAdaptorBase::getScaleAttr() {
  auto attr = getProperties().scale.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ResizeOpGenericAdaptorBase::getScale() {
  auto attr = getScaleAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ResizeOpGenericAdaptorBase::getOffsetAttr() {
  auto attr = getProperties().offset.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ResizeOpGenericAdaptorBase::getOffset() {
  auto attr = getOffsetAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ResizeOpGenericAdaptorBase::getBorderAttr() {
  auto attr = getProperties().border.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ResizeOpGenericAdaptorBase::getBorder() {
  auto attr = getBorderAttr();
  return attr;
}

::mlir::StringAttr ResizeOpGenericAdaptorBase::getModeAttr() {
  auto attr = getProperties().mode.cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef ResizeOpGenericAdaptorBase::getMode() {
  auto attr = getModeAttr();
  return attr.getValue();
}

} // namespace detail
ResizeOpAdaptor::ResizeOpAdaptor(ResizeOp op) : ResizeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ResizeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_border = getProperties().border; (void)tblgen_border;
  if (!tblgen_border) return emitError(loc, "'tosa.resize' op ""requires attribute 'border'");
  auto tblgen_mode = getProperties().mode; (void)tblgen_mode;
  if (!tblgen_mode) return emitError(loc, "'tosa.resize' op ""requires attribute 'mode'");
  auto tblgen_offset = getProperties().offset; (void)tblgen_offset;
  if (!tblgen_offset) return emitError(loc, "'tosa.resize' op ""requires attribute 'offset'");
  auto tblgen_scale = getProperties().scale; (void)tblgen_scale;
  if (!tblgen_scale) return emitError(loc, "'tosa.resize' op ""requires attribute 'scale'");

  if (tblgen_scale && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_scale))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_scale).size() == 4))))
    return emitError(loc, "'tosa.resize' op ""attribute 'scale' failed to satisfy constraint: i64 dense array attribute with exactly 4 elements");

  if (tblgen_offset && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_offset))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_offset).size() == 2))))
    return emitError(loc, "'tosa.resize' op ""attribute 'offset' failed to satisfy constraint: i64 dense array attribute with exactly 2 elements");

  if (tblgen_border && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_border))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_border).size() == 2))))
    return emitError(loc, "'tosa.resize' op ""attribute 'border' failed to satisfy constraint: i64 dense array attribute with exactly 2 elements");

  if (tblgen_mode && !((::llvm::cast<StringAttr>(tblgen_mode).getValue() == "BILINEAR"  || ::llvm::cast<StringAttr>(tblgen_mode).getValue() == "NEAREST_NEIGHBOR")))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult ResizeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.border;
    auto attr = dict.get("border");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for border in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `border` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.mode;
    auto attr = dict.get("mode");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for mode in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `mode` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.offset;
    auto attr = dict.get("offset");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for offset in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `offset` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.scale;
    auto attr = dict.get("scale");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for scale in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `scale` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ResizeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.border;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("border",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.mode;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("mode",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.offset;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("offset",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.scale;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("scale",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ResizeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.border.getAsOpaquePointer()), 
    llvm::hash_value(prop.mode.getAsOpaquePointer()), 
    llvm::hash_value(prop.offset.getAsOpaquePointer()), 
    llvm::hash_value(prop.scale.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ResizeOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "border")
      return prop.border;

    if (name == "mode")
      return prop.mode;

    if (name == "offset")
      return prop.offset;

    if (name == "scale")
      return prop.scale;
  return std::nullopt;
}

void ResizeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "border") {
       prop.border = dyn_cast_or_null<std::remove_reference_t<decltype(prop.border)>>(value);
       return;
    }

    if (name == "mode") {
       prop.mode = dyn_cast_or_null<std::remove_reference_t<decltype(prop.mode)>>(value);
       return;
    }

    if (name == "offset") {
       prop.offset = dyn_cast_or_null<std::remove_reference_t<decltype(prop.offset)>>(value);
       return;
    }

    if (name == "scale") {
       prop.scale = dyn_cast_or_null<std::remove_reference_t<decltype(prop.scale)>>(value);
       return;
    }
}

void ResizeOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.border) attrs.append("border", prop.border);

    if (prop.mode) attrs.append("mode", prop.mode);

    if (prop.offset) attrs.append("offset", prop.offset);

    if (prop.scale) attrs.append("scale", prop.scale);
}

::mlir::LogicalResult ResizeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getBorderAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(attr, "border", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getModeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps17(attr, "mode", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getOffsetAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(attr, "offset", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getScaleAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(attr, "scale", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::DenseI64ArrayAttr ResizeOp::getScaleAttr() {
  return getProperties().scale.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ResizeOp::getScale() {
  auto attr = getScaleAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ResizeOp::getOffsetAttr() {
  return getProperties().offset.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ResizeOp::getOffset() {
  auto attr = getOffsetAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ResizeOp::getBorderAttr() {
  return getProperties().border.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ResizeOp::getBorder() {
  auto attr = getBorderAttr();
  return attr;
}

::mlir::StringAttr ResizeOp::getModeAttr() {
  return getProperties().mode.cast<::mlir::StringAttr>();
}

::llvm::StringRef ResizeOp::getMode() {
  auto attr = getModeAttr();
  return attr.getValue();
}

void ResizeOp::setScaleAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getScaleAttrName(), attr);
}

void ResizeOp::setScale(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getScaleAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ResizeOp::setOffsetAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getOffsetAttrName(), attr);
}

void ResizeOp::setOffset(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getOffsetAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ResizeOp::setBorderAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getBorderAttrName(), attr);
}

void ResizeOp::setBorder(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getBorderAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ResizeOp::setModeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getModeAttrName(), attr);
}

void ResizeOp::setMode(::llvm::StringRef attrValue) {
  (*this)->setAttr(getModeAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void ResizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::DenseI64ArrayAttr scale, ::mlir::DenseI64ArrayAttr offset, ::mlir::DenseI64ArrayAttr border, ::mlir::StringAttr mode) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().scale = scale;
  odsState.getOrAddProperties<Properties>().offset = offset;
  odsState.getOrAddProperties<Properties>().border = border;
  odsState.getOrAddProperties<Properties>().mode = mode;
  odsState.addTypes(output);
}

void ResizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::DenseI64ArrayAttr scale, ::mlir::DenseI64ArrayAttr offset, ::mlir::DenseI64ArrayAttr border, ::mlir::StringAttr mode) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().scale = scale;
  odsState.getOrAddProperties<Properties>().offset = offset;
  odsState.getOrAddProperties<Properties>().border = border;
  odsState.getOrAddProperties<Properties>().mode = mode;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::llvm::ArrayRef<int64_t> scale, ::llvm::ArrayRef<int64_t> offset, ::llvm::ArrayRef<int64_t> border, ::llvm::StringRef mode) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().scale = odsBuilder.getDenseI64ArrayAttr(scale);
  odsState.getOrAddProperties<Properties>().offset = odsBuilder.getDenseI64ArrayAttr(offset);
  odsState.getOrAddProperties<Properties>().border = odsBuilder.getDenseI64ArrayAttr(border);
  odsState.getOrAddProperties<Properties>().mode = odsBuilder.getStringAttr(mode);
  odsState.addTypes(output);
}

void ResizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::llvm::ArrayRef<int64_t> scale, ::llvm::ArrayRef<int64_t> offset, ::llvm::ArrayRef<int64_t> border, ::llvm::StringRef mode) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().scale = odsBuilder.getDenseI64ArrayAttr(scale);
  odsState.getOrAddProperties<Properties>().offset = odsBuilder.getDenseI64ArrayAttr(offset);
  odsState.getOrAddProperties<Properties>().border = odsBuilder.getDenseI64ArrayAttr(border);
  odsState.getOrAddProperties<Properties>().mode = odsBuilder.getStringAttr(mode);
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
  auto tblgen_border = getProperties().border; (void)tblgen_border;
  if (!tblgen_border) return emitOpError("requires attribute 'border'");
  auto tblgen_mode = getProperties().mode; (void)tblgen_mode;
  if (!tblgen_mode) return emitOpError("requires attribute 'mode'");
  auto tblgen_offset = getProperties().offset; (void)tblgen_offset;
  if (!tblgen_offset) return emitOpError("requires attribute 'offset'");
  auto tblgen_scale = getProperties().scale; (void)tblgen_scale;
  if (!tblgen_scale) return emitOpError("requires attribute 'scale'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(*this, tblgen_scale, "scale")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_offset, "offset")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_border, "border")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps17(*this, tblgen_mode, "mode")))
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

namespace detail {
ReverseOpGenericAdaptorBase::ReverseOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.reverse", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReverseOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReverseOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReverseOpGenericAdaptorBase::getAxisAttr() {
  auto attr = getProperties().axis.cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ReverseOpGenericAdaptorBase::getAxis() {
  auto attr = getAxisAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ReverseOpAdaptor::ReverseOpAdaptor(ReverseOp op) : ReverseOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ReverseOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitError(loc, "'tosa.reverse' op ""requires attribute 'axis'");

  if (tblgen_axis && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_axis))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_axis).getType().isSignlessInteger(64)))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult ReverseOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.axis;
    auto attr = dict.get("axis");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for axis in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `axis` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ReverseOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.axis;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("axis",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ReverseOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.axis.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ReverseOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "axis")
      return prop.axis;
  return std::nullopt;
}

void ReverseOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "axis") {
       prop.axis = dyn_cast_or_null<std::remove_reference_t<decltype(prop.axis)>>(value);
       return;
    }
}

void ReverseOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.axis) attrs.append("axis", prop.axis);
}

::mlir::LogicalResult ReverseOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getAxisAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps1(attr, "axis", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr ReverseOp::getAxisAttr() {
  return getProperties().axis.cast<::mlir::IntegerAttr>();
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
  odsState.getOrAddProperties<Properties>().axis = axis;
  odsState.addTypes(output);
}

void ReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = axis;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
  odsState.addTypes(output);
}

void ReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint64_t axis) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().axis = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis);
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
  auto tblgen_axis = getProperties().axis; (void)tblgen_axis;
  if (!tblgen_axis) return emitOpError("requires attribute 'axis'");

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

namespace detail {
RsqrtOpGenericAdaptorBase::RsqrtOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.rsqrt", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> RsqrtOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr RsqrtOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
RsqrtOpAdaptor::RsqrtOpAdaptor(RsqrtOp op) : RsqrtOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
ScatterOpGenericAdaptorBase::ScatterOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.scatter", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ScatterOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ScatterOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ScatterOpAdaptor::ScatterOpAdaptor(ScatterOp op) : ScatterOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::RankedTensorType> ScatterOp::getIndices() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(1).begin());
}

::mlir::Value ScatterOp::getInput() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(2).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps13(*this, v.getType(), "operand", index++)))
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

namespace detail {
SelectOpGenericAdaptorBase::SelectOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.select", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SelectOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SelectOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
SelectOpAdaptor::SelectOpAdaptor(SelectOp op) : SelectOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> SelectOp::getOnTrue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
}

::mlir::TypedValue<::mlir::TensorType> SelectOp::getOnFalse() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(2).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps10(*this, v.getType(), "operand", index++)))
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

namespace detail {
SigmoidOpGenericAdaptorBase::SigmoidOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.sigmoid", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SigmoidOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SigmoidOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
SigmoidOpAdaptor::SigmoidOpAdaptor(SigmoidOp op) : SigmoidOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
SliceOpGenericAdaptorBase::SliceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.slice", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SliceOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SliceOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr SliceOpGenericAdaptorBase::getStartAttr() {
  auto attr = getProperties().start.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> SliceOpGenericAdaptorBase::getStart() {
  auto attr = getStartAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr SliceOpGenericAdaptorBase::getSizeAttr() {
  auto attr = getProperties().size.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> SliceOpGenericAdaptorBase::getSize() {
  auto attr = getSizeAttr();
  return attr;
}

} // namespace detail
SliceOpAdaptor::SliceOpAdaptor(SliceOp op) : SliceOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult SliceOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_size = getProperties().size; (void)tblgen_size;
  if (!tblgen_size) return emitError(loc, "'tosa.slice' op ""requires attribute 'size'");
  auto tblgen_start = getProperties().start; (void)tblgen_start;
  if (!tblgen_start) return emitError(loc, "'tosa.slice' op ""requires attribute 'start'");

  if (tblgen_start && !((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_start))))
    return emitError(loc, "'tosa.slice' op ""attribute 'start' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_size && !((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_size))))
    return emitError(loc, "'tosa.slice' op ""attribute 'size' failed to satisfy constraint: i64 dense array attribute");
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult SliceOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.size;
    auto attr = dict.get("size");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for size in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `size` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.start;
    auto attr = dict.get("start");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for start in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `start` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SliceOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.size;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("size",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.start;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("start",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code SliceOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.size.getAsOpaquePointer()), 
    llvm::hash_value(prop.start.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SliceOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "size")
      return prop.size;

    if (name == "start")
      return prop.start;
  return std::nullopt;
}

void SliceOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "size") {
       prop.size = dyn_cast_or_null<std::remove_reference_t<decltype(prop.size)>>(value);
       return;
    }

    if (name == "start") {
       prop.start = dyn_cast_or_null<std::remove_reference_t<decltype(prop.start)>>(value);
       return;
    }
}

void SliceOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.size) attrs.append("size", prop.size);

    if (prop.start) attrs.append("start", prop.start);
}

::mlir::LogicalResult SliceOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getSizeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps16(attr, "size", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStartAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps16(attr, "start", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::DenseI64ArrayAttr SliceOp::getStartAttr() {
  return getProperties().start.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> SliceOp::getStart() {
  auto attr = getStartAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr SliceOp::getSizeAttr() {
  return getProperties().size.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> SliceOp::getSize() {
  auto attr = getSizeAttr();
  return attr;
}

void SliceOp::setStartAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStartAttrName(), attr);
}

void SliceOp::setStart(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStartAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void SliceOp::setSizeAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getSizeAttrName(), attr);
}

void SliceOp::setSize(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getSizeAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void SliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::DenseI64ArrayAttr start, ::mlir::DenseI64ArrayAttr size) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().start = start;
  odsState.getOrAddProperties<Properties>().size = size;
  odsState.addTypes(output);
}

void SliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::DenseI64ArrayAttr start, ::mlir::DenseI64ArrayAttr size) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().start = start;
  odsState.getOrAddProperties<Properties>().size = size;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::llvm::ArrayRef<int64_t> start, ::llvm::ArrayRef<int64_t> size) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().start = odsBuilder.getDenseI64ArrayAttr(start);
  odsState.getOrAddProperties<Properties>().size = odsBuilder.getDenseI64ArrayAttr(size);
  odsState.addTypes(output);
}

void SliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::llvm::ArrayRef<int64_t> start, ::llvm::ArrayRef<int64_t> size) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().start = odsBuilder.getDenseI64ArrayAttr(start);
  odsState.getOrAddProperties<Properties>().size = odsBuilder.getDenseI64ArrayAttr(size);
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
  auto tblgen_size = getProperties().size; (void)tblgen_size;
  if (!tblgen_size) return emitOpError("requires attribute 'size'");
  auto tblgen_start = getProperties().start; (void)tblgen_start;
  if (!tblgen_start) return emitOpError("requires attribute 'start'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps16(*this, tblgen_start, "start")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps16(*this, tblgen_size, "size")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps17(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps17(*this, v.getType(), "result", index++)))
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

namespace detail {
SubOpGenericAdaptorBase::SubOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.sub", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SubOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SubOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
SubOpAdaptor::SubOpAdaptor(SubOp op) : SubOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> SubOp::getInput2() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
TableOpGenericAdaptorBase::TableOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.table", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TableOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TableOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
TableOpAdaptor::TableOpAdaptor(TableOp op) : TableOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::Value TableOp::getTable() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps7(*this, v.getType(), "operand", index++)))
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

namespace detail {
TanhOpGenericAdaptorBase::TanhOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.tanh", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TanhOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TanhOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
TanhOpAdaptor::TanhOpAdaptor(TanhOp op) : TanhOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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

namespace detail {
TileOpGenericAdaptorBase::TileOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.tile", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TileOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TileOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr TileOpGenericAdaptorBase::getMultiplesAttr() {
  auto attr = getProperties().multiples.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> TileOpGenericAdaptorBase::getMultiples() {
  auto attr = getMultiplesAttr();
  return attr;
}

} // namespace detail
TileOpAdaptor::TileOpAdaptor(TileOp op) : TileOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult TileOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_multiples = getProperties().multiples; (void)tblgen_multiples;
  if (!tblgen_multiples) return emitError(loc, "'tosa.tile' op ""requires attribute 'multiples'");

  if (tblgen_multiples && !((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_multiples))))
    return emitError(loc, "'tosa.tile' op ""attribute 'multiples' failed to satisfy constraint: i64 dense array attribute");
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult TileOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.multiples;
    auto attr = dict.get("multiples");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for multiples in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `multiples` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TileOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.multiples;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("multiples",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code TileOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.multiples.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TileOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "multiples")
      return prop.multiples;
  return std::nullopt;
}

void TileOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "multiples") {
       prop.multiples = dyn_cast_or_null<std::remove_reference_t<decltype(prop.multiples)>>(value);
       return;
    }
}

void TileOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.multiples) attrs.append("multiples", prop.multiples);
}

::mlir::LogicalResult TileOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getMultiplesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps16(attr, "multiples", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::DenseI64ArrayAttr TileOp::getMultiplesAttr() {
  return getProperties().multiples.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> TileOp::getMultiples() {
  auto attr = getMultiplesAttr();
  return attr;
}

void TileOp::setMultiplesAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getMultiplesAttrName(), attr);
}

void TileOp::setMultiples(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getMultiplesAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void TileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::DenseI64ArrayAttr multiples) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().multiples = multiples;
  odsState.addTypes(output);
}

void TileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::mlir::DenseI64ArrayAttr multiples) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().multiples = multiples;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input1, ::llvm::ArrayRef<int64_t> multiples) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().multiples = odsBuilder.getDenseI64ArrayAttr(multiples);
  odsState.addTypes(output);
}

void TileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input1, ::llvm::ArrayRef<int64_t> multiples) {
  odsState.addOperands(input1);
  odsState.getOrAddProperties<Properties>().multiples = odsBuilder.getDenseI64ArrayAttr(multiples);
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
  auto tblgen_multiples = getProperties().multiples; (void)tblgen_multiples;
  if (!tblgen_multiples) return emitOpError("requires attribute 'multiples'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps16(*this, tblgen_multiples, "multiples")))
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

namespace detail {
TransposeConv2DOpGenericAdaptorBase::TransposeConv2DOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.transpose_conv2d", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TransposeConv2DOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TransposeConv2DOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr TransposeConv2DOpGenericAdaptorBase::getOutPadAttr() {
  auto attr = getProperties().out_pad.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> TransposeConv2DOpGenericAdaptorBase::getOutPad() {
  auto attr = getOutPadAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr TransposeConv2DOpGenericAdaptorBase::getStrideAttr() {
  auto attr = getProperties().stride.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> TransposeConv2DOpGenericAdaptorBase::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr TransposeConv2DOpGenericAdaptorBase::getOutShapeAttr() {
  auto attr = getProperties().out_shape.cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> TransposeConv2DOpGenericAdaptorBase::getOutShape() {
  auto attr = getOutShapeAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr TransposeConv2DOpGenericAdaptorBase::getQuantizationInfoAttr() {
  auto attr = getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
  return attr;
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> TransposeConv2DOpGenericAdaptorBase::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(::llvm::cast<mlir::tosa::ConvOpQuantizationAttr>(attr)) : (::std::nullopt);
}

} // namespace detail
TransposeConv2DOpAdaptor::TransposeConv2DOpAdaptor(TransposeConv2DOp op) : TransposeConv2DOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult TransposeConv2DOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_out_pad = getProperties().out_pad; (void)tblgen_out_pad;
  if (!tblgen_out_pad) return emitError(loc, "'tosa.transpose_conv2d' op ""requires attribute 'out_pad'");
  auto tblgen_out_shape = getProperties().out_shape; (void)tblgen_out_shape;
  if (!tblgen_out_shape) return emitError(loc, "'tosa.transpose_conv2d' op ""requires attribute 'out_shape'");
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitError(loc, "'tosa.transpose_conv2d' op ""requires attribute 'stride'");

  if (tblgen_out_pad && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_out_pad))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_out_pad).size() == 4))))
    return emitError(loc, "'tosa.transpose_conv2d' op ""attribute 'out_pad' failed to satisfy constraint: i64 dense array attribute with exactly 4 elements");

  if (tblgen_stride && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_stride))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_stride).size() == 2))))
    return emitError(loc, "'tosa.transpose_conv2d' op ""attribute 'stride' failed to satisfy constraint: i64 dense array attribute with exactly 2 elements");

  if (tblgen_out_shape && !(((::llvm::isa<::mlir::DenseI64ArrayAttr>(tblgen_out_shape))) && ((::llvm::cast<::mlir::DenseArrayAttr>(tblgen_out_shape).size() <= 4))))
    return emitError(loc, "'tosa.transpose_conv2d' op ""attribute 'out_shape' failed to satisfy constraint: i64 dense array attribute with at least 4 elements");

  if (tblgen_quantization_info && !((::llvm::isa<mlir::tosa::ConvOpQuantizationAttr>(tblgen_quantization_info))))
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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value TransposeConv2DOp::getFilter() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::Value TransposeConv2DOp::getBias() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(2).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::LogicalResult TransposeConv2DOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.out_pad;
    auto attr = dict.get("out_pad");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for out_pad in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `out_pad` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.out_shape;
    auto attr = dict.get("out_shape");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for out_shape in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `out_shape` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.quantization_info;
    auto attr = dict.get("quantization_info");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for quantization_info in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `quantization_info` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.stride;
    auto attr = dict.get("stride");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for stride in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `stride` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute TransposeConv2DOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.out_pad;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("out_pad",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.out_shape;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("out_shape",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.quantization_info;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("quantization_info",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.stride;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("stride",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code TransposeConv2DOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.out_pad.getAsOpaquePointer()), 
    llvm::hash_value(prop.out_shape.getAsOpaquePointer()), 
    llvm::hash_value(prop.quantization_info.getAsOpaquePointer()), 
    llvm::hash_value(prop.stride.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> TransposeConv2DOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "out_pad")
      return prop.out_pad;

    if (name == "out_shape")
      return prop.out_shape;

    if (name == "quantization_info")
      return prop.quantization_info;

    if (name == "stride")
      return prop.stride;
  return std::nullopt;
}

void TransposeConv2DOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "out_pad") {
       prop.out_pad = dyn_cast_or_null<std::remove_reference_t<decltype(prop.out_pad)>>(value);
       return;
    }

    if (name == "out_shape") {
       prop.out_shape = dyn_cast_or_null<std::remove_reference_t<decltype(prop.out_shape)>>(value);
       return;
    }

    if (name == "quantization_info") {
       prop.quantization_info = dyn_cast_or_null<std::remove_reference_t<decltype(prop.quantization_info)>>(value);
       return;
    }

    if (name == "stride") {
       prop.stride = dyn_cast_or_null<std::remove_reference_t<decltype(prop.stride)>>(value);
       return;
    }
}

void TransposeConv2DOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.out_pad) attrs.append("out_pad", prop.out_pad);

    if (prop.out_shape) attrs.append("out_shape", prop.out_shape);

    if (prop.quantization_info) attrs.append("quantization_info", prop.quantization_info);

    if (prop.stride) attrs.append("stride", prop.stride);
}

::mlir::LogicalResult TransposeConv2DOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getOutPadAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(attr, "out_pad", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getOutShapeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps18(attr, "out_shape", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getQuantizationInfoAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps8(attr, "quantization_info", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStrideAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(attr, "stride", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::DenseI64ArrayAttr TransposeConv2DOp::getOutPadAttr() {
  return getProperties().out_pad.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> TransposeConv2DOp::getOutPad() {
  auto attr = getOutPadAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr TransposeConv2DOp::getStrideAttr() {
  return getProperties().stride.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> TransposeConv2DOp::getStride() {
  auto attr = getStrideAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr TransposeConv2DOp::getOutShapeAttr() {
  return getProperties().out_shape.cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> TransposeConv2DOp::getOutShape() {
  auto attr = getOutShapeAttr();
  return attr;
}

mlir::tosa::ConvOpQuantizationAttr TransposeConv2DOp::getQuantizationInfoAttr() {
  return getProperties().quantization_info.dyn_cast_or_null<mlir::tosa::ConvOpQuantizationAttr>();
}

::std::optional<mlir::tosa::ConvOpQuantizationAttr> TransposeConv2DOp::getQuantizationInfo() {
  auto attr = getQuantizationInfoAttr();
  return attr ? ::std::optional<mlir::tosa::ConvOpQuantizationAttr>(::llvm::cast<mlir::tosa::ConvOpQuantizationAttr>(attr)) : (::std::nullopt);
}

void TransposeConv2DOp::setOutPadAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getOutPadAttrName(), attr);
}

void TransposeConv2DOp::setOutPad(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getOutPadAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void TransposeConv2DOp::setStrideAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStrideAttrName(), attr);
}

void TransposeConv2DOp::setStride(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStrideAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void TransposeConv2DOp::setOutShapeAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getOutShapeAttrName(), attr);
}

void TransposeConv2DOp::setOutShape(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getOutShapeAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void TransposeConv2DOp::setQuantizationInfoAttr(mlir::tosa::ConvOpQuantizationAttr attr) {
  (*this)->setAttr(getQuantizationInfoAttrName(), attr);
}

::mlir::Attribute TransposeConv2DOp::removeQuantizationInfoAttr() {
    auto &attr = getProperties().quantization_info;
    attr = {};
    return attr;
}

void TransposeConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputType, ::mlir::Value input, ::mlir::Value weight, mlir::Value bias, ::mlir::DenseI64ArrayAttr outpad, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr outputShape) {
    buildTransConvOpWithQuantInfo(odsBuilder, odsState, outputType,
                                  input, weight, bias,
                                  outpad, stride,
                                  outputShape);
  
}

void TransposeConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::DenseI64ArrayAttr out_pad, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr out_shape, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().out_pad = out_pad;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().out_shape = out_shape;
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  odsState.addTypes(output);
}

void TransposeConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::DenseI64ArrayAttr out_pad, ::mlir::DenseI64ArrayAttr stride, ::mlir::DenseI64ArrayAttr out_shape, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().out_pad = out_pad;
  odsState.getOrAddProperties<Properties>().stride = stride;
  odsState.getOrAddProperties<Properties>().out_shape = out_shape;
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::llvm::ArrayRef<int64_t> out_pad, ::llvm::ArrayRef<int64_t> stride, ::llvm::ArrayRef<int64_t> out_shape, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().out_pad = odsBuilder.getDenseI64ArrayAttr(out_pad);
  odsState.getOrAddProperties<Properties>().stride = odsBuilder.getDenseI64ArrayAttr(stride);
  odsState.getOrAddProperties<Properties>().out_shape = odsBuilder.getDenseI64ArrayAttr(out_shape);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
  }
  odsState.addTypes(output);
}

void TransposeConv2DOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::llvm::ArrayRef<int64_t> out_pad, ::llvm::ArrayRef<int64_t> stride, ::llvm::ArrayRef<int64_t> out_shape, /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(bias);
  odsState.getOrAddProperties<Properties>().out_pad = odsBuilder.getDenseI64ArrayAttr(out_pad);
  odsState.getOrAddProperties<Properties>().stride = odsBuilder.getDenseI64ArrayAttr(stride);
  odsState.getOrAddProperties<Properties>().out_shape = odsBuilder.getDenseI64ArrayAttr(out_shape);
  if (quantization_info) {
    odsState.getOrAddProperties<Properties>().quantization_info = quantization_info;
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
  auto tblgen_out_pad = getProperties().out_pad; (void)tblgen_out_pad;
  if (!tblgen_out_pad) return emitOpError("requires attribute 'out_pad'");
  auto tblgen_out_shape = getProperties().out_shape; (void)tblgen_out_shape;
  if (!tblgen_out_shape) return emitOpError("requires attribute 'out_shape'");
  auto tblgen_quantization_info = getProperties().quantization_info; (void)tblgen_quantization_info;
  auto tblgen_stride = getProperties().stride; (void)tblgen_stride;
  if (!tblgen_stride) return emitOpError("requires attribute 'stride'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps3(*this, tblgen_out_pad, "out_pad")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps2(*this, tblgen_stride, "stride")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps18(*this, tblgen_out_shape, "out_shape")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TosaOps8(*this, tblgen_quantization_info, "quantization_info")))
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps7(*this, v.getType(), "operand", index++)))
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

namespace detail {
TransposeOpGenericAdaptorBase::TransposeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.transpose", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TransposeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TransposeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
TransposeOpAdaptor::TransposeOpAdaptor(TransposeOp op) : TransposeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> TransposeOp::getPerms() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
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
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps17(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps15(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TosaOps17(*this, v.getType(), "result", index++)))
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

namespace detail {
WhileOpGenericAdaptorBase::WhileOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.while_loop", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> WhileOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr WhileOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &WhileOpGenericAdaptorBase::getCond() {
  return *odsRegions[0];
}

::mlir::Region &WhileOpGenericAdaptorBase::getBody() {
  return *odsRegions[1];
}

::mlir::RegionRange WhileOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
WhileOpAdaptor::WhileOpAdaptor(WhileOp op) : WhileOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_TosaOps0(*this, region, "cond", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(1)))
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

namespace detail {
YieldOpGenericAdaptorBase::YieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tosa.yield", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> YieldOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr YieldOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
YieldOpAdaptor::YieldOpAdaptor(YieldOp op) : YieldOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

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

