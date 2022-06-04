/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::EmitCDialect)
namespace mlir {
namespace emitc {

EmitCDialect::EmitCDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<EmitCDialect>()) {
  
  initialize();
}

EmitCDialect::~EmitCDialect() = default;

} // namespace emitc
} // namespace mlir
