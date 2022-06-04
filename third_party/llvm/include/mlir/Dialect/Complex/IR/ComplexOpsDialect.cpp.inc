/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::complex::ComplexDialect)
namespace mlir {
namespace complex {

ComplexDialect::ComplexDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<ComplexDialect>()) {
  
    getContext()->getOrLoadDialect<arith::ArithmeticDialect>();

  initialize();
}

ComplexDialect::~ComplexDialect() = default;

} // namespace complex
} // namespace mlir
