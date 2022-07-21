/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::async::AsyncDialect)
namespace mlir {
namespace async {

AsyncDialect::AsyncDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<AsyncDialect>()) {
  
  initialize();
}

AsyncDialect::~AsyncDialect() = default;

} // namespace async
} // namespace mlir
