/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMDialect)
namespace mlir {
namespace LLVM {

LLVMDialect::LLVMDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<LLVMDialect>()) {
  
  initialize();
}

LLVMDialect::~LLVMDialect() = default;

} // namespace LLVM
} // namespace mlir
