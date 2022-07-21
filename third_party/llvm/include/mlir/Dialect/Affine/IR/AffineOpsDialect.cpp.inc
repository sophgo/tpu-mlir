/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::AffineDialect)
namespace mlir {

AffineDialect::AffineDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<AffineDialect>()) {
  
    getContext()->getOrLoadDialect<arith::ArithmeticDialect>();

  initialize();
}

AffineDialect::~AffineDialect() = default;

} // namespace mlir
