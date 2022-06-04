/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::TensorDialect)
namespace mlir {
namespace tensor {

TensorDialect::TensorDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<TensorDialect>()) {
  
    getContext()->getOrLoadDialect<arith::ArithmeticDialect>();

    getContext()->getOrLoadDialect<complex::ComplexDialect>();

  initialize();
}

TensorDialect::~TensorDialect() = default;

} // namespace tensor
} // namespace mlir
