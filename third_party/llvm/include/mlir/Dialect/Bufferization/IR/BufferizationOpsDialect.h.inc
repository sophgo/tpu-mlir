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

    /// Verify an attribute from this dialect on the argument at 'argIndex' for
    /// the region at 'regionIndex' on the given operation. Returns failure if
    /// the verification failed, success otherwise. This hook may optionally be
    /// invoked from any operation containing a region.
    LogicalResult verifyRegionArgAttribute(Operation *,
                                           unsigned regionIndex,
                                           unsigned argIndex,
                                           NamedAttribute) override;

    /// An attribute that can override writability of buffers of tensor function
    /// arguments during One-Shot Module Bufferize.
    constexpr const static ::llvm::StringLiteral
        kWritableAttrName = "bufferization.writable";

    /// An attribute for function arguments that describes how the function
    /// accesses the buffer. Can be one "none", "read", "write" or "read-write".
    ///
    /// When no attribute is specified, the analysis tries to infer the access
    /// behavior from its body. In case of external functions, for which no
    /// function body is available, "read-write" is assumed by default.
    constexpr const static ::llvm::StringLiteral
        kBufferAccessAttrName = "bufferization.access";

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
