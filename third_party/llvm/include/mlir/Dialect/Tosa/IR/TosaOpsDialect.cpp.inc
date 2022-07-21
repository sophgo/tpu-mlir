/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::TosaDialect)
namespace mlir {
namespace tosa {

TosaDialect::TosaDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<TosaDialect>()) {
  
    getContext()->getOrLoadDialect<tensor::TensorDialect>();

  initialize();
}

TosaDialect::~TosaDialect() = default;

} // namespace tosa
} // namespace mlir
