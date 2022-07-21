/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::spirv::SPIRVDialect)
namespace mlir {
namespace spirv {

SPIRVDialect::SPIRVDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<SPIRVDialect>()) {
  
  initialize();
}

SPIRVDialect::~SPIRVDialect() = default;

} // namespace spirv
} // namespace mlir
