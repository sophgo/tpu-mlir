/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUDialect)
namespace mlir {
namespace gpu {

GPUDialect::GPUDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<GPUDialect>()) {
  
    getContext()->getOrLoadDialect<arith::ArithmeticDialect>();

  initialize();
}

GPUDialect::~GPUDialect() = default;

} // namespace gpu
} // namespace mlir
