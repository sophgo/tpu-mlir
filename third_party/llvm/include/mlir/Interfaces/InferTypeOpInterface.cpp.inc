/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::LogicalResult mlir::InferShapedTypeOpInterface::inferReturnTypeComponents(::mlir::MLIRContext* context, ::mlir::Optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::mlir::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes) {
      return getImpl()->inferReturnTypeComponents(context, location, operands, attributes, regions, inferredReturnShapes);
  }
::mlir::LogicalResult mlir::InferShapedTypeOpInterface::reifyReturnTypeShapes(::mlir::OpBuilder& builder, ::mlir::ValueRange operands, ::mlir::SmallVectorImpl<::mlir::Value> & reifiedReturnShapes) {
      return getImpl()->reifyReturnTypeShapes(getImpl(), getOperation(), builder, operands, reifiedReturnShapes);
  }
::mlir::LogicalResult mlir::InferTypeOpInterface::inferReturnTypes(::mlir::MLIRContext * context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
      return getImpl()->inferReturnTypes(context, location, operands, attributes, regions, inferredReturnTypes);
  }
bool mlir::InferTypeOpInterface::isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
      return getImpl()->isCompatibleReturnTypes(lhs, rhs);
  }
::mlir::LogicalResult mlir::ReifyRankedShapedTypeOpInterface::reifyResultShapes(::mlir::OpBuilder & builder, ::mlir::ReifiedRankedShapedTypeDims & reifiedReturnShapes) {
      return getImpl()->reifyResultShapes(getImpl(), getOperation(), builder, reifiedReturnShapes);
  }
