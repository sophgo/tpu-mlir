/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::amdgpu::AMDGPUDialect)
namespace mlir {
namespace amdgpu {

AMDGPUDialect::AMDGPUDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<AMDGPUDialect>()) {
  
  initialize();
}

AMDGPUDialect::~AMDGPUDialect() = default;

} // namespace amdgpu
} // namespace mlir
