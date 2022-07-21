/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::bufferization::BufferizationDialect)
namespace mlir {
namespace bufferization {

BufferizationDialect::BufferizationDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<BufferizationDialect>()) {
  
    getContext()->getOrLoadDialect<AffineDialect>();

    getContext()->getOrLoadDialect<memref::MemRefDialect>();

    getContext()->getOrLoadDialect<tensor::TensorDialect>();

  initialize();
}

BufferizationDialect::~BufferizationDialect() = default;

} // namespace bufferization
} // namespace mlir
