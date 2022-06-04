/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::LinalgDialect)
namespace mlir {
namespace linalg {

LinalgDialect::LinalgDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<LinalgDialect>()) {
  
    getContext()->getOrLoadDialect<arith::ArithmeticDialect>();

    getContext()->getOrLoadDialect<AffineDialect>();

    getContext()->getOrLoadDialect<math::MathDialect>();

    getContext()->getOrLoadDialect<memref::MemRefDialect>();

    getContext()->getOrLoadDialect<tensor::TensorDialect>();

  initialize();
}

LinalgDialect::~LinalgDialect() = default;

} // namespace linalg
} // namespace mlir
