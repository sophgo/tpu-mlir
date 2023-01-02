/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Infer the components of return type of shape containter.
/// 
///       The method takes an optional location which, if set, will be used to
///       report errors on. The operands and attributes correspond to those with
///       which an Operation would be created (e.g., as used in Operation::create)
///       and the regions of the op.
/// 
///       Unknown (e.g., unranked) shape and nullptrs for element type and attribute
///       may be returned by this function while returning success. E.g., partial
///       population of components is not error condition.
::mlir::LogicalResult mlir::InferShapedTypeOpInterface::inferReturnTypeComponents(::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes) {
      return getImpl()->inferReturnTypeComponents(context, location, operands, attributes, regions, inferredReturnShapes);
  }
/// Reify the shape computation for the operation.
/// 
///       Insert operations using the given OpBuilder that computes the
///       result shape. This interface is supposed to be workable during dialect
///       conversion (e.g. convert from tensor world to buffer world),
///       where `getOperand` may be invalid. For example, some ops (e.g.
///       dynamic_reshape(input, target_shape)) may depend on their operands
///       to calculate the result shape. When the `matchAndRewrite ` method
///       of a conversion pattern is called, the operands of the op to convert
///       may have been converted into other types, which makes it invalid to
///       call the `getOperand` method of such op directly inside the
///       conversion pattern.  To solve this problem, this interface follows
///       the design of the conversion pattern, that is, accepting passed in
///       operands to avoid calling `getOperand` directly inside the interface
///       implementation.
::mlir::LogicalResult mlir::InferShapedTypeOpInterface::reifyReturnTypeShapes(::mlir::OpBuilder& builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> & reifiedReturnShapes) {
      return getImpl()->reifyReturnTypeShapes(getImpl(), getOperation(), builder, operands, reifiedReturnShapes);
  }
/// Infer the return types that an op would generate.
/// 
///       The method takes an optional location which, if set, will be used to
///       report errors on. The operands and attributes correspond to those with
///       which an Operation would be created (e.g., as used in Operation::create)
///       and the regions of the op. Be aware that this method is supposed to be
///       called with valid arguments, e.g., operands are verified, or it may result
///       in an undefined behavior.
::mlir::LogicalResult mlir::InferTypeOpInterface::inferReturnTypes(::mlir::MLIRContext * context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
      return getImpl()->inferReturnTypes(context, location, operands, attributes, regions, inferredReturnTypes);
  }
/// Refine the return types that an op would generate.
/// 
///       This method computes the return types as `inferReturnTypes` does but
///       additionally takes the existing result types as input. The existing
///       result types can be checked as part of inference to provide more
///       op-specific error messages as well as part of inference to merge
///       additional information, attributes, during inference. It is called during
///       verification for ops implementing this trait with default behavior
///       reporting mismatch with current and inferred types printed.
/// 
///       The operands and attributes correspond to those with which an Operation
///       would be created (e.g., as used in Operation::create) and the regions of
///       the op. The method takes an optional location which, if set, will be used
///       to report errors on.
/// 
///       The return types may be elided or specific elements be null for elements
///       that should just be returned but not verified.
/// 
///       Be aware that this method is supposed to be called with valid arguments,
///       e.g., operands are verified, or it may result in an undefined behavior.
::mlir::LogicalResult mlir::InferTypeOpInterface::refineReturnTypes(::mlir::MLIRContext * context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& returnTypes) {
      return getImpl()->refineReturnTypes(context, location, operands, attributes, regions, returnTypes);
  }
/// Returns whether two array of types are compatible result types for an op.
bool mlir::InferTypeOpInterface::isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
      return getImpl()->isCompatibleReturnTypes(lhs, rhs);
  }
/// Reify the shape of the result of an operation (typically in
/// terms of shape of its operands)
/// 
/// Insert operations using the given `OpBuilder` that computes
/// the result shape. The `reifiedReturnShapes` is expected to be
/// populated with as many vectors as the number of results of the
/// op. Each of these vectors is expected to be of size equal to
/// rank of the corresponding result. If the shape of a particular
/// result cannot be computed it must be empty.
::mlir::LogicalResult mlir::ReifyRankedShapedTypeOpInterface::reifyResultShapes(::mlir::OpBuilder & builder, ::mlir::ReifiedRankedShapedTypeDims & reifiedReturnShapes) {
      return getImpl()->reifyResultShapes(getImpl(), getOperation(), builder, reifiedReturnShapes);
  }
