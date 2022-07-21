/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::arith::ArithmeticDialect)
namespace mlir {
namespace arith {

ArithmeticDialect::ArithmeticDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<ArithmeticDialect>()) {
  
  initialize();
}

ArithmeticDialect::~ArithmeticDialect() = default;

} // namespace arith
} // namespace mlir
