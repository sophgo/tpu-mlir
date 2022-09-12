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
namespace tosa {
class ConvOpQuantizationAttr;
class MatMulOpQuantizationAttr;
class PadOpQuantizationAttr;
class UnaryOpQuantizationAttr;
namespace detail {
struct ConvOpQuantizationAttrStorage;
} // namespace detail
class ConvOpQuantizationAttr : public ::mlir::Attribute::AttrBase<ConvOpQuantizationAttr, ::mlir::Attribute, detail::ConvOpQuantizationAttrStorage> {
public:
  using Base::Base;
public:
  static ConvOpQuantizationAttr get(::mlir::MLIRContext *context, int64_t input_zp, int64_t weight_zp);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"conv_quant"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getInputZp() const;
  int64_t getWeightZp() const;
};
namespace detail {
struct MatMulOpQuantizationAttrStorage;
} // namespace detail
class MatMulOpQuantizationAttr : public ::mlir::Attribute::AttrBase<MatMulOpQuantizationAttr, ::mlir::Attribute, detail::MatMulOpQuantizationAttrStorage> {
public:
  using Base::Base;
public:
  static MatMulOpQuantizationAttr get(::mlir::MLIRContext *context, int64_t a_zp, int64_t b_zp);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"matmul_quant"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getAZp() const;
  int64_t getBZp() const;
};
namespace detail {
struct PadOpQuantizationAttrStorage;
} // namespace detail
class PadOpQuantizationAttr : public ::mlir::Attribute::AttrBase<PadOpQuantizationAttr, ::mlir::Attribute, detail::PadOpQuantizationAttrStorage> {
public:
  using Base::Base;
public:
  static PadOpQuantizationAttr get(::mlir::MLIRContext *context, int64_t input_zp);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"pad_quant"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getInputZp() const;
};
namespace detail {
struct UnaryOpQuantizationAttrStorage;
} // namespace detail
class UnaryOpQuantizationAttr : public ::mlir::Attribute::AttrBase<UnaryOpQuantizationAttr, ::mlir::Attribute, detail::UnaryOpQuantizationAttrStorage> {
public:
  using Base::Base;
public:
  static UnaryOpQuantizationAttr get(::mlir::MLIRContext *context, int64_t input_zp, int64_t output_zp);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"unary_quant"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getInputZp() const;
  int64_t getOutputZp() const;
};
} // namespace tosa
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::ConvOpQuantizationAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::MatMulOpQuantizationAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::PadOpQuantizationAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::UnaryOpQuantizationAttr)

#endif  // GET_ATTRDEF_CLASSES

