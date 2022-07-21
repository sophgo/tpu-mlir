/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace transform {

class TransformDialect : public ::mlir::Dialect {
  explicit TransformDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~TransformDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("transform");
  }

      /// Returns the named PDL constraint functions available in the dialect
      /// as a map from their name to the function.
      const ::llvm::StringMap<::mlir::PDLConstraintFunction> &
      getPDLConstraintHooks() const;

    private:
      template <typename OpTy>
      void addOperationIfNotRegistered() {
        Optional<RegisteredOperationName> opName =
            RegisteredOperationName::lookup(OpTy::getOperationName(),
                                            getContext());
        if (!opName)
          return addOperations<OpTy>();

        if (opName->getTypeID() == TypeID::get<OpTy>())
          return;

        llvm::errs() << "error: extensible dialect operation '"
                     << OpTy::getOperationName()
                     << "' is already registered with a mismatching TypeID";
        abort();
      }

      /// Registers operations specified as template parameters with this
      /// dialect. Checks that they implement the required interfaces.
      template <typename... OpTys>
      void addOperationsChecked() {
        (void)std::initializer_list<int>{(addOperationIfNotRegistered<OpTys>(),
                                          0)...};

        #ifndef NDEBUG
        (void)std::initializer_list<int>{
          (detail::checkImplementsTransformInterface<OpTys>(getContext()),
           0)...};
        #endif // NDEBUG
      }

      template <typename, typename...>
      friend class TransformDialectExtension;

      /// Takes ownership of the named PDL constraint function from the given
      /// map and makes them available for use by the operations in the dialect.
      void mergeInPDLMatchHooks(
          ::llvm::StringMap<::mlir::PDLConstraintFunction> &&constraintFns);

      /// A container for PDL constraint function that can be used by
      /// operations in this dialect.
      PDLPatternModule pdlMatchHooks;
  };
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::TransformDialect)
