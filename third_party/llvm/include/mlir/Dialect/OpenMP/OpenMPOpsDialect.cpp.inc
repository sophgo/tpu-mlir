/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::OpenMPDialect)
namespace mlir {
namespace omp {

OpenMPDialect::OpenMPDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<OpenMPDialect>()) {
  
    getContext()->getOrLoadDialect<::mlir::LLVM::LLVMDialect>();

  initialize();
}

OpenMPDialect::~OpenMPDialect() = default;

} // namespace omp
} // namespace mlir
