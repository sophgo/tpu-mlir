/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class InferShapedTypeOpInterface;
namespace detail {
struct InferShapedTypeOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::LogicalResult (*inferReturnTypeComponents)(::mlir::MLIRContext*, ::mlir::Optional<::mlir::Location>, ::mlir::ValueShapeRange, ::mlir::DictionaryAttr, ::mlir::RegionRange, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&);
    ::mlir::LogicalResult (*reifyReturnTypeShapes)(const Concept *impl, ::mlir::Operation *, ::mlir::OpBuilder&, ::mlir::ValueRange, ::llvm::SmallVectorImpl<::mlir::Value> &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::InferShapedTypeOpInterface;
    Model() : Concept{inferReturnTypeComponents, reifyReturnTypeShapes} {}

    static inline ::mlir::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext* context, ::mlir::Optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes);
    static inline ::mlir::LogicalResult reifyReturnTypeShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder& builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> & reifiedReturnShapes);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::InferShapedTypeOpInterface;
    FallbackModel() : Concept{inferReturnTypeComponents, reifyReturnTypeShapes} {}

    static inline ::mlir::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext* context, ::mlir::Optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes);
    static inline ::mlir::LogicalResult reifyReturnTypeShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder& builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> & reifiedReturnShapes);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    static ::mlir::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext*context, ::mlir::Optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes);
    ::mlir::LogicalResult reifyReturnTypeShapes(::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder&builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) const;
  };
};template <typename ConcreteOp>
struct InferShapedTypeOpInterfaceTrait;

} // namespace detail
class InferShapedTypeOpInterface : public ::mlir::OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::InferShapedTypeOpInterfaceTrait<ConcreteOp> {};
  ::mlir::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext* context, ::mlir::Optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes);
  ::mlir::LogicalResult reifyReturnTypeShapes(::mlir::OpBuilder& builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> & reifiedReturnShapes);
};
namespace detail {
  template <typename ConcreteOp>
  struct InferShapedTypeOpInterfaceTrait : public ::mlir::OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::mlir::LogicalResult inferReturnTypeComponents(::mlir::MLIRContext* context, ::mlir::Optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes) {
      return ::mlir::failure();
    }
    ::mlir::LogicalResult reifyReturnTypeShapes(::mlir::OpBuilder& builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> & reifiedReturnShapes) {
      return ::mlir::failure();
    }
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::LogicalResult detail::InferShapedTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferReturnTypeComponents(::mlir::MLIRContext* context, ::mlir::Optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes) {
  return ConcreteOp::inferReturnTypeComponents(context, location, operands, attributes, regions, inferredReturnShapes);
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::InferShapedTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::reifyReturnTypeShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder& builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> & reifiedReturnShapes) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).reifyReturnTypeShapes(builder, operands, reifiedReturnShapes);
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::InferShapedTypeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::inferReturnTypeComponents(::mlir::MLIRContext* context, ::mlir::Optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes) {
  return ConcreteOp::inferReturnTypeComponents(context, location, operands, attributes, regions, inferredReturnShapes);
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::InferShapedTypeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::reifyReturnTypeShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder& builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> & reifiedReturnShapes) {
  return static_cast<const ConcreteOp *>(impl)->reifyReturnTypeShapes(tablegen_opaque_val, builder, operands, reifiedReturnShapes);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::LogicalResult detail::InferShapedTypeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::inferReturnTypeComponents(::mlir::MLIRContext*context, ::mlir::Optional<::mlir::Location> location, ::mlir::ValueShapeRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>&inferredReturnShapes) {
return ::mlir::failure();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::LogicalResult detail::InferShapedTypeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::reifyReturnTypeShapes(::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder&builder, ::mlir::ValueRange operands, ::llvm::SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) const {
return ::mlir::failure();
}
} // namespace mlir
namespace mlir {
class InferTypeOpInterface;
namespace detail {
struct InferTypeOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::LogicalResult (*inferReturnTypes)(::mlir::MLIRContext *, ::llvm::Optional<::mlir::Location>, ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::RegionRange, ::llvm::SmallVectorImpl<::mlir::Type>&);
    bool (*isCompatibleReturnTypes)(::mlir::TypeRange, ::mlir::TypeRange);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::InferTypeOpInterface;
    Model() : Concept{inferReturnTypes, isCompatibleReturnTypes} {}

    static inline ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext * context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes);
    static inline bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::InferTypeOpInterface;
    FallbackModel() : Concept{inferReturnTypes, isCompatibleReturnTypes} {}

    static inline ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext * context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes);
    static inline bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
  };
};template <typename ConcreteOp>
struct InferTypeOpInterfaceTrait;

} // namespace detail
class InferTypeOpInterface : public ::mlir::OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::InferTypeOpInterfaceTrait<ConcreteOp> {};
  ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext * context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes);
  bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
};
namespace detail {
  template <typename ConcreteOp>
  struct InferTypeOpInterfaceTrait : public ::mlir::OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
      /// Returns whether two arrays are equal as strongest check for
        /// compatibility by default.
        return lhs == rhs;
    }
    static ::mlir::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
      return detail::verifyInferredResultTypes(op);
    }
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::LogicalResult detail::InferTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::inferReturnTypes(::mlir::MLIRContext * context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  return ConcreteOp::inferReturnTypes(context, location, operands, attributes, regions, inferredReturnTypes);
}
template<typename ConcreteOp>
bool detail::InferTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
  return ConcreteOp::isCompatibleReturnTypes(lhs, rhs);
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::InferTypeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::inferReturnTypes(::mlir::MLIRContext * context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  return ConcreteOp::inferReturnTypes(context, location, operands, attributes, regions, inferredReturnTypes);
}
template<typename ConcreteOp>
bool detail::InferTypeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
  return ConcreteOp::isCompatibleReturnTypes(lhs, rhs);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::InferTypeOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
/// Returns whether two arrays are equal as strongest check for
        /// compatibility by default.
        return lhs == rhs;
}
} // namespace mlir
namespace mlir {
class ReifyRankedShapedTypeOpInterface;
namespace detail {
struct ReifyRankedShapedTypeOpInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::LogicalResult (*reifyResultShapes)(const Concept *impl, ::mlir::Operation *, ::mlir::OpBuilder &, ::mlir::ReifiedRankedShapedTypeDims &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ReifyRankedShapedTypeOpInterface;
    Model() : Concept{reifyResultShapes} {}

    static inline ::mlir::LogicalResult reifyResultShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::ReifiedRankedShapedTypeDims & reifiedReturnShapes);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ReifyRankedShapedTypeOpInterface;
    FallbackModel() : Concept{reifyResultShapes} {}

    static inline ::mlir::LogicalResult reifyResultShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::ReifiedRankedShapedTypeDims & reifiedReturnShapes);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
  };
};template <typename ConcreteOp>
struct ReifyRankedShapedTypeOpInterfaceTrait;

} // namespace detail
class ReifyRankedShapedTypeOpInterface : public ::mlir::OpInterface<ReifyRankedShapedTypeOpInterface, detail::ReifyRankedShapedTypeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ReifyRankedShapedTypeOpInterface, detail::ReifyRankedShapedTypeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ReifyRankedShapedTypeOpInterfaceTrait<ConcreteOp> {};
  ::mlir::LogicalResult reifyResultShapes(::mlir::OpBuilder & builder, ::mlir::ReifiedRankedShapedTypeDims & reifiedReturnShapes);
};
namespace detail {
  template <typename ConcreteOp>
  struct ReifyRankedShapedTypeOpInterfaceTrait : public ::mlir::OpInterface<ReifyRankedShapedTypeOpInterface, detail::ReifyRankedShapedTypeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::LogicalResult detail::ReifyRankedShapedTypeOpInterfaceInterfaceTraits::Model<ConcreteOp>::reifyResultShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::ReifiedRankedShapedTypeDims & reifiedReturnShapes) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).reifyResultShapes(builder, reifiedReturnShapes);
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::ReifyRankedShapedTypeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::reifyResultShapes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::ReifiedRankedShapedTypeDims & reifiedReturnShapes) {
  return static_cast<const ConcreteOp *>(impl)->reifyResultShapes(tablegen_opaque_val, builder, reifiedReturnShapes);
}
} // namespace mlir
