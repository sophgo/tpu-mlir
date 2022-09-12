/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace gpu {
class AllReduceOperationAttr;
class DimensionAttr;
class ShuffleModeAttr;
class MMAElementwiseOpAttr;
class ParallelLoopDimMappingAttr;
namespace detail {
struct AllReduceOperationAttrStorage;
} // namespace detail
class AllReduceOperationAttr : public ::mlir::Attribute::AttrBase<AllReduceOperationAttr, ::mlir::Attribute, detail::AllReduceOperationAttrStorage> {
public:
  using Base::Base;
public:
  static AllReduceOperationAttr get(::mlir::MLIRContext *context, ::mlir::gpu::AllReduceOperation value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"all_reduce_op"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::AllReduceOperation getValue() const;
};
namespace detail {
struct DimensionAttrStorage;
} // namespace detail
class DimensionAttr : public ::mlir::Attribute::AttrBase<DimensionAttr, ::mlir::Attribute, detail::DimensionAttrStorage> {
public:
  using Base::Base;
public:
  static DimensionAttr get(::mlir::MLIRContext *context, ::mlir::gpu::Dimension value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"dim"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::Dimension getValue() const;
};
namespace detail {
struct ShuffleModeAttrStorage;
} // namespace detail
class ShuffleModeAttr : public ::mlir::Attribute::AttrBase<ShuffleModeAttr, ::mlir::Attribute, detail::ShuffleModeAttrStorage> {
public:
  using Base::Base;
public:
  static ShuffleModeAttr get(::mlir::MLIRContext *context, ::mlir::gpu::ShuffleMode value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shuffle_mode"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::ShuffleMode getValue() const;
};
namespace detail {
struct MMAElementwiseOpAttrStorage;
} // namespace detail
class MMAElementwiseOpAttr : public ::mlir::Attribute::AttrBase<MMAElementwiseOpAttr, ::mlir::Attribute, detail::MMAElementwiseOpAttrStorage> {
public:
  using Base::Base;
public:
  static MMAElementwiseOpAttr get(::mlir::MLIRContext *context, ::mlir::gpu::MMAElementwiseOp value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_element_wise"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::MMAElementwiseOp getValue() const;
};
namespace detail {
struct ParallelLoopDimMappingAttrStorage;
} // namespace detail
class ParallelLoopDimMappingAttr : public ::mlir::Attribute::AttrBase<ParallelLoopDimMappingAttr, ::mlir::Attribute, detail::ParallelLoopDimMappingAttrStorage> {
public:
  using Base::Base;
public:
  static ParallelLoopDimMappingAttr get(::mlir::MLIRContext *context, ::mlir::gpu::Processor processor, AffineMap map, AffineMap bound);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"loop_dim_map"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::gpu::Processor getProcessor() const;
  AffineMap getMap() const;
  AffineMap getBound() const;
};
} // namespace gpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::AllReduceOperationAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::DimensionAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::ShuffleModeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::MMAElementwiseOpAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::ParallelLoopDimMappingAttr)

#endif  // GET_ATTRDEF_CLASSES

