/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ROCDL::ROCDLDialect)
namespace mlir {
namespace ROCDL {

ROCDLDialect::ROCDLDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<ROCDLDialect>()) {
  
    getContext()->getOrLoadDialect<LLVM::LLVMDialect>();

  initialize();
}

ROCDLDialect::~ROCDLDialect() = default;

} // namespace ROCDL
} // namespace mlir
