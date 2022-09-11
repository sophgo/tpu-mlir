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
namespace amdgpu {
class MFMAPermBAttr;
namespace detail {
struct MFMAPermBAttrStorage;
} // namespace detail
class MFMAPermBAttr : public ::mlir::Attribute::AttrBase<MFMAPermBAttr, ::mlir::Attribute, detail::MFMAPermBAttrStorage> {
public:
  using Base::Base;
public:
  static MFMAPermBAttr get(::mlir::MLIRContext *context, ::mlir::amdgpu::MFMAPermB value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mfma_perm_b"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::amdgpu::MFMAPermB getValue() const;
};
} // namespace amdgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::MFMAPermBAttr)

#endif  // GET_ATTRDEF_CLASSES

