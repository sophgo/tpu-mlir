/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace spirv {

class SPIRVDialect : public ::mlir::Dialect {
  explicit SPIRVDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~SPIRVDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("spv");
  }

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;

  /// Materialize a single constant operation from a given attribute value with
  /// the desired resultant type.
  ::mlir::Operation *materializeConstant(::mlir::OpBuilder &builder,
                                         ::mlir::Attribute value,
                                         ::mlir::Type type,
                                         ::mlir::Location loc) override;

    /// Provides a hook for verifying dialect attributes attached to the given
    /// op.
    ::mlir::LogicalResult verifyOperationAttribute(
        ::mlir::Operation *op, ::mlir::NamedAttribute attribute) override;

    /// Provides a hook for verifying dialect attributes attached to the given
    /// op's region argument.
    ::mlir::LogicalResult verifyRegionArgAttribute(
        ::mlir::Operation *op, unsigned regionIndex, unsigned argIndex,
        ::mlir::NamedAttribute attribute) override;

    /// Provides a hook for verifying dialect attributes attached to the given
    /// op's region result.
    ::mlir::LogicalResult verifyRegionResultAttribute(
        ::mlir::Operation *op, unsigned regionIndex, unsigned resultIndex,
        ::mlir::NamedAttribute attribute) override;

    void registerAttributes();
    void registerTypes();

    //===------------------------------------------------------------------===//
    // Attribute
    //===------------------------------------------------------------------===//

    /// Returns the attribute name to use when specifying decorations on results
    /// of operations.
    static std::string getAttributeName(Decoration decoration);

    /// Dialect attribute parsing hook.
    Attribute parseAttribute(
        DialectAsmParser &parser, Type type) const override;
    /// Dialect attribute printing hook.
    void printAttribute(
        Attribute attr, DialectAsmPrinter &printer) const override;
  };
} // namespace spirv
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::spirv::SPIRVDialect)
