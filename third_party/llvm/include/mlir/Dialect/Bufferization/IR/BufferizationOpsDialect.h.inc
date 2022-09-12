/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace bufferization {

class BufferizationDialect : public ::mlir::Dialect {
  explicit BufferizationDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~BufferizationDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("bufferization");
  }

    /// Provides a hook for verifying dialect attributes attached to the given
    /// op.
    ::mlir::LogicalResult verifyOperationAttribute(
        ::mlir::Operation *op, ::mlir::NamedAttribute attribute) override;

    /// An attribute that can override writability of buffers of tensor function
    /// arguments during One-Shot Module Bufferize.
    constexpr const static ::llvm::StringLiteral
        kWritableAttrName = "bufferization.writable";

    /// Attribute name used to mark the bufferization layout for region
    /// arguments during One-Shot Module Bufferize.
    constexpr const static ::llvm::StringLiteral
        kBufferLayoutAttrName = "bufferization.buffer_layout";

    /// Attribute name used to mark escaping behavior of buffer allocations.
    /// Escaping allocations cannot be deallocated in the same block and must
    /// be treated specially: They are currently deallocated with the
    /// BufferDeallocation pass.
    ///
    /// Note: Only ops with at least one OpResult that bufferizes to a buffer
    /// allocation (as per BufferizableOpInterface) may have this attribute.
    constexpr const static ::llvm::StringLiteral
        kEscapeAttrName = "bufferization.escape";
  };
} // namespace bufferization
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::bufferization::BufferizationDialect)
