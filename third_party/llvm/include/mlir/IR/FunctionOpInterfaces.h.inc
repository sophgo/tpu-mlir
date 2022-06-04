/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class FunctionOpInterface;
namespace detail {
struct FunctionOpInterfaceInterfaceTraits {
  struct Concept {
    ::llvm::ArrayRef<::mlir::Type> (*getArgumentTypes)(const Concept *impl, ::mlir::Operation *);
    ::llvm::ArrayRef<::mlir::Type> (*getResultTypes)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Type (*cloneTypeWith)(const Concept *impl, ::mlir::Operation *, ::mlir::TypeRange, ::mlir::TypeRange);
    ::mlir::LogicalResult (*verifyBody)(const Concept *impl, ::mlir::Operation *);
    ::mlir::LogicalResult (*verifyType)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::FunctionOpInterface;
    Model() : Concept{getArgumentTypes, getResultTypes, cloneTypeWith, verifyBody, verifyType} {}

    static inline ::llvm::ArrayRef<::mlir::Type> getArgumentTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::ArrayRef<::mlir::Type> getResultTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Type cloneTypeWith(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results);
    static inline ::mlir::LogicalResult verifyBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::LogicalResult verifyType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::FunctionOpInterface;
    FallbackModel() : Concept{getArgumentTypes, getResultTypes, cloneTypeWith, verifyBody, verifyType} {}

    static inline ::llvm::ArrayRef<::mlir::Type> getArgumentTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::ArrayRef<::mlir::Type> getResultTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Type cloneTypeWith(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results);
    static inline ::mlir::LogicalResult verifyBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::LogicalResult verifyType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    ::mlir::Type cloneTypeWith(::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results) const;
    ::mlir::LogicalResult verifyBody(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::LogicalResult verifyType(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct FunctionOpInterfaceTrait;

} // namespace detail
class FunctionOpInterface : public ::mlir::OpInterface<FunctionOpInterface, detail::FunctionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<FunctionOpInterface, detail::FunctionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::FunctionOpInterfaceTrait<ConcreteOp> {};
  ::llvm::ArrayRef<::mlir::Type> getArgumentTypes();
  ::llvm::ArrayRef<::mlir::Type> getResultTypes();
  ::mlir::Type cloneTypeWith(::mlir::TypeRange inputs, ::mlir::TypeRange results);
  ::mlir::LogicalResult verifyBody();
  ::mlir::LogicalResult verifyType();

    //===------------------------------------------------------------------===//
    // Name
    //===------------------------------------------------------------------===//

    /// Return the name of the function.
    StringRef getName() { return SymbolTable::getSymbolName(*this); }
  

    /// Block list iterator types.
    using BlockListType = Region::BlockListType;
    using iterator = BlockListType::iterator;
    using reverse_iterator = BlockListType::reverse_iterator;

    /// Block argument iterator types.
    using BlockArgListType = Region::BlockArgListType;
    using args_iterator = BlockArgListType::iterator;

    //===------------------------------------------------------------------===//
    // Body Handling
    //===------------------------------------------------------------------===//

    /// Returns true if this function is external, i.e. it has no body.
    bool isExternal() { return empty(); }

    /// Return the region containing the body of this function.
    Region &getBody() { return (*this)->getRegion(0); }

    /// Delete all blocks from this function.
    void eraseBody() {
      getBody().dropAllReferences();
      getBody().getBlocks().clear();
    }

    /// Return the list of blocks within the function body.
    BlockListType &getBlocks() { return getBody().getBlocks(); }

    iterator begin() { return getBody().begin(); }
    iterator end() { return getBody().end(); }
    reverse_iterator rbegin() { return getBody().rbegin(); }
    reverse_iterator rend() { return getBody().rend(); }

    /// Returns true if this function has no blocks within the body.
    bool empty() { return getBody().empty(); }

    /// Push a new block to the back of the body region.
    void push_back(Block *block) { getBody().push_back(block); }

    /// Push a new block to the front of the body region.
    void push_front(Block *block) { getBody().push_front(block); }

    /// Return the last block in the body region.
    Block &back() { return getBody().back(); }

    /// Return the first block in the body region.
    Block &front() { return getBody().front(); }

    /// Add an entry block to an empty function, and set up the block arguments
    /// to match the signature of the function. The newly inserted entry block
    /// is returned.
    Block *addEntryBlock() {
      assert(empty() && "function already has an entry block");
      Block *entry = new Block();
      push_back(entry);

      // FIXME: Allow for passing in locations for these arguments instead of using
      // the operations location.
      ArrayRef<Type> inputTypes = (*this).getArgumentTypes();
      SmallVector<Location> locations(inputTypes.size(),
                                      (*this).getOperation()->getLoc());
      entry->addArguments(inputTypes, locations);
      return entry;
    }

    /// Add a normal block to the end of the function's block list. The function
    /// should at least already have an entry block.
    Block *addBlock() {
      assert(!empty() && "function should at least have an entry block");
      push_back(new Block());
      return &back();
    }

    //===------------------------------------------------------------------===//
    // Type Attribute Handling
    //===------------------------------------------------------------------===//

    /// Change the type of this function in place. This is an extremely dangerous
    /// operation and it is up to the caller to ensure that this is legal for
    /// this function, and to restore invariants:
    ///  - the entry block args must be updated to match the function params.
    ///  - the argument/result attributes may need an update: if the new type
    ///    has less parameters we drop the extra attributes, if there are more
    ///    parameters they won't have any attributes.
    void setType(Type newType) {
      function_interface_impl::setFunctionType(this->getOperation(), newType);
    }

    // FIXME: These functions should be removed in favor of just forwarding to
    // the derived operation, which should already have these defined
    // (via ODS).

    /// Returns the name of the attribute used for function types.
    static StringRef getTypeAttrName() {
      return function_interface_impl::getTypeAttrName();
    }

    /// Returns the name of the attribute used for function argument attributes.
    static StringRef getArgDictAttrName() {
      return function_interface_impl::getArgDictAttrName();
    }

    /// Returns the name of the attribute used for function argument attributes.
    static StringRef getResultDictAttrName() {
      return function_interface_impl::getResultDictAttrName();
    }

    /// Return the attribute containing the type of this function.
    TypeAttr getFunctionTypeAttr() {
      return this->getOperation()->template getAttrOfType<TypeAttr>(
          getTypeAttrName());
    }

    /// Return the type of this function.
    Type getFunctionType() { return getFunctionTypeAttr().getValue(); }

    //===------------------------------------------------------------------===//
    // Argument and Result Handling
    //===------------------------------------------------------------------===//

    /// Returns the number of function arguments.
    unsigned getNumArguments() { return (*this).getArgumentTypes().size(); }

    /// Returns the number of function results.
    unsigned getNumResults() { return (*this).getResultTypes().size(); }

    /// Returns the entry block function argument at the given index.
    BlockArgument getArgument(unsigned idx) {
      return getBody().getArgument(idx);
    }

    /// Support argument iteration.
    args_iterator args_begin() { return getBody().args_begin(); }
    args_iterator args_end() { return getBody().args_end(); }
    BlockArgListType getArguments() { return getBody().getArguments(); }

    /// Insert a single argument of type `argType` with attributes `argAttrs` and
    /// location `argLoc` at `argIndex`.
    void insertArgument(unsigned argIndex, Type argType, DictionaryAttr argAttrs,
                        Location argLoc) {
      insertArguments({argIndex}, {argType}, {argAttrs}, {argLoc});
    }

    /// Inserts arguments with the listed types, attributes, and locations at the
    /// listed indices. `argIndices` must be sorted. Arguments are inserted in the
    /// order they are listed, such that arguments with identical index will
    /// appear in the same order that they were listed here.
    void insertArguments(ArrayRef<unsigned> argIndices, TypeRange argTypes,
                        ArrayRef<DictionaryAttr> argAttrs,
                        ArrayRef<Location> argLocs) {
      unsigned originalNumArgs = (*this).getNumArguments();
      Type newType = (*this).getTypeWithArgsAndResults(
          argIndices, argTypes, /*resultIndices=*/{}, /*resultTypes=*/{});
      function_interface_impl::insertFunctionArguments(
          this->getOperation(), argIndices, argTypes, argAttrs, argLocs,
          originalNumArgs, newType);
    }

    /// Insert a single result of type `resultType` at `resultIndex`.
    void insertResult(unsigned resultIndex, Type resultType,
                      DictionaryAttr resultAttrs) {
      insertResults({resultIndex}, {resultType}, {resultAttrs});
    }

    /// Inserts results with the listed types at the listed indices.
    /// `resultIndices` must be sorted. Results are inserted in the order they are
    /// listed, such that results with identical index will appear in the same
    /// order that they were listed here.
    void insertResults(ArrayRef<unsigned> resultIndices, TypeRange resultTypes,
                      ArrayRef<DictionaryAttr> resultAttrs) {
      unsigned originalNumResults = (*this).getNumResults();
      Type newType = (*this).getTypeWithArgsAndResults(
        /*argIndices=*/{}, /*argTypes=*/{}, resultIndices, resultTypes);
      function_interface_impl::insertFunctionResults(
          this->getOperation(), resultIndices, resultTypes, resultAttrs,
          originalNumResults, newType);
    }

    /// Erase a single argument at `argIndex`.
    void eraseArgument(unsigned argIndex) {
      BitVector argsToErase((*this).getNumArguments());
      argsToErase.set(argIndex);
      eraseArguments(argsToErase);
    }

    /// Erases the arguments listed in `argIndices`.
    void eraseArguments(const BitVector &argIndices) {
      Type newType = (*this).getTypeWithoutArgs(argIndices);
      function_interface_impl::eraseFunctionArguments(
        this->getOperation(), argIndices, newType);
    }

    /// Erase a single result at `resultIndex`.
    void eraseResult(unsigned resultIndex) {
      BitVector resultsToErase((*this).getNumResults());
      resultsToErase.set(resultIndex);
      eraseResults(resultsToErase);
    }

    /// Erases the results listed in `resultIndices`.
    void eraseResults(const BitVector &resultIndices) {
      Type newType = (*this).getTypeWithoutResults(resultIndices);
      function_interface_impl::eraseFunctionResults(
          this->getOperation(), resultIndices, newType);
    }

    /// Return the type of this function with the specified arguments and
    /// results inserted. This is used to update the function's signature in
    /// the `insertArguments` and `insertResults` methods. The arrays must be
    /// sorted by increasing index.
    Type getTypeWithArgsAndResults(
      ArrayRef<unsigned> argIndices, TypeRange argTypes,
      ArrayRef<unsigned> resultIndices, TypeRange resultTypes) {
      SmallVector<Type> argStorage, resultStorage;
      TypeRange newArgTypes = function_interface_impl::insertTypesInto(
          (*this).getArgumentTypes(), argIndices, argTypes, argStorage);
      TypeRange newResultTypes = function_interface_impl::insertTypesInto(
          (*this).getResultTypes(), resultIndices, resultTypes, resultStorage);
      return (*this).cloneTypeWith(newArgTypes, newResultTypes);
    }

    /// Return the type of this function without the specified arguments and
    /// results. This is used to update the function's signature in the
    /// `eraseArguments` and `eraseResults` methods.
    Type getTypeWithoutArgsAndResults(
      const BitVector &argIndices, const BitVector &resultIndices) {
      SmallVector<Type> argStorage, resultStorage;
      TypeRange newArgTypes = function_interface_impl::filterTypesOut(
          (*this).getArgumentTypes(), argIndices, argStorage);
      TypeRange newResultTypes = function_interface_impl::filterTypesOut(
          (*this).getResultTypes(), resultIndices, resultStorage);
      return (*this).cloneTypeWith(newArgTypes, newResultTypes);
    }
    Type getTypeWithoutArgs(const BitVector &argIndices) {
      SmallVector<Type> argStorage;
      TypeRange newArgTypes = function_interface_impl::filterTypesOut(
          (*this).getArgumentTypes(), argIndices, argStorage);
      return (*this).cloneTypeWith(newArgTypes, (*this).getResultTypes());
    }
    Type getTypeWithoutResults(const BitVector &resultIndices) {
      SmallVector<Type> resultStorage;
      TypeRange newResultTypes = function_interface_impl::filterTypesOut(
          (*this).getResultTypes(), resultIndices, resultStorage);
      return (*this).cloneTypeWith((*this).getArgumentTypes(), newResultTypes);
    }

    //===------------------------------------------------------------------===//
    // Argument Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the argument at 'index'.
    ArrayRef<NamedAttribute> getArgAttrs(unsigned index) {
      return function_interface_impl::getArgAttrs(this->getOperation(), index);
    }

    /// Return an ArrayAttr containing all argument attribute dictionaries of
    /// this function, or nullptr if no arguments have attributes.
    ArrayAttr getAllArgAttrs() {
      return this->getOperation()->template getAttrOfType<ArrayAttr>(
          getArgDictAttrName());
    }
    /// Return all argument attributes of this function.
    void getAllArgAttrs(SmallVectorImpl<DictionaryAttr> &result) {
      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append((*this).getNumArguments(),
                      DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the argument at 'index',
    /// null otherwise.
    Attribute getArgAttr(unsigned index, StringAttr name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    Attribute getArgAttr(unsigned index, StringRef name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, StringAttr name) {
      return getArgAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }
    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, StringRef name) {
      return getArgAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }

    /// Set the attributes held by the argument at 'index'.
    void setArgAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
      function_interface_impl::setArgAttrs((*this), index, attributes);
    }

    /// Set the attributes held by the argument at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setArgAttrs(unsigned index, DictionaryAttr attributes) {
      function_interface_impl::setArgAttrs((*this), index, attributes);
    }
    void setAllArgAttrs(ArrayRef<DictionaryAttr> attributes) {
      assert(attributes.size() == (*this).getNumArguments());
      function_interface_impl::setAllArgAttrDicts(this->getOperation(), attributes);
    }
    void setAllArgAttrs(ArrayRef<Attribute> attributes) {
      assert(attributes.size() == (*this).getNumArguments());
      function_interface_impl::setAllArgAttrDicts(this->getOperation(), attributes);
    }
    void setAllArgAttrs(ArrayAttr attributes) {
      assert(attributes.size() == (*this).getNumArguments());
      this->getOperation()->setAttr(getArgDictAttrName(), attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setArgAttr(unsigned index, StringAttr name, Attribute value) {
      function_interface_impl::setArgAttr((*this), index, name, value);
    }
    void setArgAttr(unsigned index, StringRef name, Attribute value) {
      setArgAttr(index,
                 StringAttr::get(this->getOperation()->getContext(), name),
                 value);
    }

    /// Remove the attribute 'name' from the argument at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    Attribute removeArgAttr(unsigned index, StringAttr name) {
      return function_interface_impl::removeArgAttr((*this), index, name);
    }
    Attribute removeArgAttr(unsigned index, StringRef name) {
      return removeArgAttr(
          index, StringAttr::get(this->getOperation()->getContext(), name));
    }

    //===------------------------------------------------------------------===//
    // Result Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the result at 'index'.
    ArrayRef<NamedAttribute> getResultAttrs(unsigned index) {
      return function_interface_impl::getResultAttrs(this->getOperation(), index);
    }

    /// Return an ArrayAttr containing all result attribute dictionaries of this
    /// function, or nullptr if no result have attributes.
    ArrayAttr getAllResultAttrs() {
      return this->getOperation()->template getAttrOfType<ArrayAttr>(
          getResultDictAttrName());
    }
    /// Return all result attributes of this function.
    void getAllResultAttrs(SmallVectorImpl<DictionaryAttr> &result) {
      if (ArrayAttr argAttrs = getAllResultAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append((*this).getNumResults(),
                      DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the result at 'index',
    /// null otherwise.
    Attribute getResultAttr(unsigned index, StringAttr name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    Attribute getResultAttr(unsigned index, StringRef name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, StringAttr name) {
      return getResultAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }
    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, StringRef name) {
      return getResultAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }

    /// Set the attributes held by the result at 'index'.
    void setResultAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
      function_interface_impl::setResultAttrs((*this), index, attributes);
    }

    /// Set the attributes held by the result at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setResultAttrs(unsigned index, DictionaryAttr attributes) {
      function_interface_impl::setResultAttrs((*this), index, attributes);
    }
    void setAllResultAttrs(ArrayRef<DictionaryAttr> attributes) {
      assert(attributes.size() == (*this).getNumResults());
      function_interface_impl::setAllResultAttrDicts(
        this->getOperation(), attributes);
    }
    void setAllResultAttrs(ArrayRef<Attribute> attributes) {
      assert(attributes.size() == (*this).getNumResults());
      function_interface_impl::setAllResultAttrDicts(
        this->getOperation(), attributes);
    }
    void setAllResultAttrs(ArrayAttr attributes) {
      assert(attributes.size() == (*this).getNumResults());
      this->getOperation()->setAttr(getResultDictAttrName(), attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setResultAttr(unsigned index, StringAttr name, Attribute value) {
      function_interface_impl::setResultAttr((*this), index, name, value);
    }
    void setResultAttr(unsigned index, StringRef name, Attribute value) {
      setResultAttr(index,
                    StringAttr::get(this->getOperation()->getContext(), name),
                    value);
    }

    /// Remove the attribute 'name' from the result at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    Attribute removeResultAttr(unsigned index, StringAttr name) {
      return function_interface_impl::removeResultAttr((*this), index, name);
    }

    /// Returns the dictionary attribute corresponding to the argument at
    /// 'index'. If there are no argument attributes at 'index', a null
    /// attribute is returned.
    DictionaryAttr getArgAttrDict(unsigned index) {
      assert(index < (*this).getNumArguments() && "invalid argument number");
      return function_interface_impl::getArgAttrDict(this->getOperation(), index);
    }

    /// Returns the dictionary attribute corresponding to the result at 'index'.
    /// If there are no result attributes at 'index', a null attribute is
    /// returned.
    DictionaryAttr getResultAttrDict(unsigned index) {
      assert(index < (*this).getNumResults() && "invalid result number");
      return function_interface_impl::getResultAttrDict(this->getOperation(), index);
    }
  };
namespace detail {
  template <typename ConcreteOp>
  struct FunctionOpInterfaceTrait : public ::mlir::OpInterface<FunctionOpInterface, detail::FunctionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    ::mlir::Type cloneTypeWith(::mlir::TypeRange inputs, ::mlir::TypeRange results) {
      return (*static_cast<ConcreteOp *>(this)).getFunctionType().clone(inputs, results);
    }
    ::mlir::LogicalResult verifyBody() {
      if ((*static_cast<ConcreteOp *>(this)).isExternal())
        return success();
      ArrayRef<Type> fnInputTypes = (*static_cast<ConcreteOp *>(this)).getArgumentTypes();
      // NOTE: This should just be (*static_cast<ConcreteOp *>(this)).front() but access generically
      // because the interface methods defined here may be shadowed in
      // arbitrary ways. https://github.com/llvm/llvm-project/issues/54807
      Block &entryBlock = (*static_cast<ConcreteOp *>(this))->getRegion(0).front();

      unsigned numArguments = fnInputTypes.size();
      if (entryBlock.getNumArguments() != numArguments)
        return (*static_cast<ConcreteOp *>(this)).emitOpError("entry block must have ")
              << numArguments << " arguments to match function signature";

      for (unsigned i = 0, e = fnInputTypes.size(); i != e; ++i) {
        Type argType = entryBlock.getArgument(i).getType();
        if (fnInputTypes[i] != argType) {
          return (*static_cast<ConcreteOp *>(this)).emitOpError("type of entry block argument #")
                << i << '(' << argType
                << ") must match the type of the corresponding argument in "
                << "function signature(" << fnInputTypes[i] << ')';
        }
      }

      return success();
    }
    ::mlir::LogicalResult verifyType() {
      return success();
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return function_interface_impl::verifyTrait(cast<ConcreteOp>(op));
    }

    //===------------------------------------------------------------------===//
    // Builders
    //===------------------------------------------------------------------===//

    /// Build the function with the given name, attributes, and type. This
    /// builder also inserts an entry block into the function body with the
    /// given argument types.
    static void buildWithEntryBlock(
        OpBuilder &builder, OperationState &state, StringRef name, Type type,
        ArrayRef<NamedAttribute> attrs, ArrayRef<Type> inputTypes) {
      state.addAttribute(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(name));
      state.addAttribute(function_interface_impl::getTypeAttrName(),
                        TypeAttr::get(type));
      state.attributes.append(attrs.begin(), attrs.end());

      // Add the function body.
      Region *bodyRegion = state.addRegion();
      Block *body = new Block();
      bodyRegion->push_back(body);
      for (Type input : inputTypes)
        body->addArgument(input, state.location);
    }
  

    /// Block list iterator types.
    using BlockListType = Region::BlockListType;
    using iterator = BlockListType::iterator;
    using reverse_iterator = BlockListType::reverse_iterator;

    /// Block argument iterator types.
    using BlockArgListType = Region::BlockArgListType;
    using args_iterator = BlockArgListType::iterator;

    //===------------------------------------------------------------------===//
    // Body Handling
    //===------------------------------------------------------------------===//

    /// Returns true if this function is external, i.e. it has no body.
    bool isExternal() { return empty(); }

    /// Return the region containing the body of this function.
    Region &getBody() { return (*static_cast<ConcreteOp *>(this))->getRegion(0); }

    /// Delete all blocks from this function.
    void eraseBody() {
      getBody().dropAllReferences();
      getBody().getBlocks().clear();
    }

    /// Return the list of blocks within the function body.
    BlockListType &getBlocks() { return getBody().getBlocks(); }

    iterator begin() { return getBody().begin(); }
    iterator end() { return getBody().end(); }
    reverse_iterator rbegin() { return getBody().rbegin(); }
    reverse_iterator rend() { return getBody().rend(); }

    /// Returns true if this function has no blocks within the body.
    bool empty() { return getBody().empty(); }

    /// Push a new block to the back of the body region.
    void push_back(Block *block) { getBody().push_back(block); }

    /// Push a new block to the front of the body region.
    void push_front(Block *block) { getBody().push_front(block); }

    /// Return the last block in the body region.
    Block &back() { return getBody().back(); }

    /// Return the first block in the body region.
    Block &front() { return getBody().front(); }

    /// Add an entry block to an empty function, and set up the block arguments
    /// to match the signature of the function. The newly inserted entry block
    /// is returned.
    Block *addEntryBlock() {
      assert(empty() && "function already has an entry block");
      Block *entry = new Block();
      push_back(entry);

      // FIXME: Allow for passing in locations for these arguments instead of using
      // the operations location.
      ArrayRef<Type> inputTypes = (*static_cast<ConcreteOp *>(this)).getArgumentTypes();
      SmallVector<Location> locations(inputTypes.size(),
                                      (*static_cast<ConcreteOp *>(this)).getOperation()->getLoc());
      entry->addArguments(inputTypes, locations);
      return entry;
    }

    /// Add a normal block to the end of the function's block list. The function
    /// should at least already have an entry block.
    Block *addBlock() {
      assert(!empty() && "function should at least have an entry block");
      push_back(new Block());
      return &back();
    }

    //===------------------------------------------------------------------===//
    // Type Attribute Handling
    //===------------------------------------------------------------------===//

    /// Change the type of this function in place. This is an extremely dangerous
    /// operation and it is up to the caller to ensure that this is legal for
    /// this function, and to restore invariants:
    ///  - the entry block args must be updated to match the function params.
    ///  - the argument/result attributes may need an update: if the new type
    ///    has less parameters we drop the extra attributes, if there are more
    ///    parameters they won't have any attributes.
    void setType(Type newType) {
      function_interface_impl::setFunctionType(this->getOperation(), newType);
    }

    // FIXME: These functions should be removed in favor of just forwarding to
    // the derived operation, which should already have these defined
    // (via ODS).

    /// Returns the name of the attribute used for function types.
    static StringRef getTypeAttrName() {
      return function_interface_impl::getTypeAttrName();
    }

    /// Returns the name of the attribute used for function argument attributes.
    static StringRef getArgDictAttrName() {
      return function_interface_impl::getArgDictAttrName();
    }

    /// Returns the name of the attribute used for function argument attributes.
    static StringRef getResultDictAttrName() {
      return function_interface_impl::getResultDictAttrName();
    }

    /// Return the attribute containing the type of this function.
    TypeAttr getFunctionTypeAttr() {
      return this->getOperation()->template getAttrOfType<TypeAttr>(
          getTypeAttrName());
    }

    /// Return the type of this function.
    Type getFunctionType() { return getFunctionTypeAttr().getValue(); }

    //===------------------------------------------------------------------===//
    // Argument and Result Handling
    //===------------------------------------------------------------------===//

    /// Returns the number of function arguments.
    unsigned getNumArguments() { return (*static_cast<ConcreteOp *>(this)).getArgumentTypes().size(); }

    /// Returns the number of function results.
    unsigned getNumResults() { return (*static_cast<ConcreteOp *>(this)).getResultTypes().size(); }

    /// Returns the entry block function argument at the given index.
    BlockArgument getArgument(unsigned idx) {
      return getBody().getArgument(idx);
    }

    /// Support argument iteration.
    args_iterator args_begin() { return getBody().args_begin(); }
    args_iterator args_end() { return getBody().args_end(); }
    BlockArgListType getArguments() { return getBody().getArguments(); }

    /// Insert a single argument of type `argType` with attributes `argAttrs` and
    /// location `argLoc` at `argIndex`.
    void insertArgument(unsigned argIndex, Type argType, DictionaryAttr argAttrs,
                        Location argLoc) {
      insertArguments({argIndex}, {argType}, {argAttrs}, {argLoc});
    }

    /// Inserts arguments with the listed types, attributes, and locations at the
    /// listed indices. `argIndices` must be sorted. Arguments are inserted in the
    /// order they are listed, such that arguments with identical index will
    /// appear in the same order that they were listed here.
    void insertArguments(ArrayRef<unsigned> argIndices, TypeRange argTypes,
                        ArrayRef<DictionaryAttr> argAttrs,
                        ArrayRef<Location> argLocs) {
      unsigned originalNumArgs = (*static_cast<ConcreteOp *>(this)).getNumArguments();
      Type newType = (*static_cast<ConcreteOp *>(this)).getTypeWithArgsAndResults(
          argIndices, argTypes, /*resultIndices=*/{}, /*resultTypes=*/{});
      function_interface_impl::insertFunctionArguments(
          this->getOperation(), argIndices, argTypes, argAttrs, argLocs,
          originalNumArgs, newType);
    }

    /// Insert a single result of type `resultType` at `resultIndex`.
    void insertResult(unsigned resultIndex, Type resultType,
                      DictionaryAttr resultAttrs) {
      insertResults({resultIndex}, {resultType}, {resultAttrs});
    }

    /// Inserts results with the listed types at the listed indices.
    /// `resultIndices` must be sorted. Results are inserted in the order they are
    /// listed, such that results with identical index will appear in the same
    /// order that they were listed here.
    void insertResults(ArrayRef<unsigned> resultIndices, TypeRange resultTypes,
                      ArrayRef<DictionaryAttr> resultAttrs) {
      unsigned originalNumResults = (*static_cast<ConcreteOp *>(this)).getNumResults();
      Type newType = (*static_cast<ConcreteOp *>(this)).getTypeWithArgsAndResults(
        /*argIndices=*/{}, /*argTypes=*/{}, resultIndices, resultTypes);
      function_interface_impl::insertFunctionResults(
          this->getOperation(), resultIndices, resultTypes, resultAttrs,
          originalNumResults, newType);
    }

    /// Erase a single argument at `argIndex`.
    void eraseArgument(unsigned argIndex) {
      BitVector argsToErase((*static_cast<ConcreteOp *>(this)).getNumArguments());
      argsToErase.set(argIndex);
      eraseArguments(argsToErase);
    }

    /// Erases the arguments listed in `argIndices`.
    void eraseArguments(const BitVector &argIndices) {
      Type newType = (*static_cast<ConcreteOp *>(this)).getTypeWithoutArgs(argIndices);
      function_interface_impl::eraseFunctionArguments(
        this->getOperation(), argIndices, newType);
    }

    /// Erase a single result at `resultIndex`.
    void eraseResult(unsigned resultIndex) {
      BitVector resultsToErase((*static_cast<ConcreteOp *>(this)).getNumResults());
      resultsToErase.set(resultIndex);
      eraseResults(resultsToErase);
    }

    /// Erases the results listed in `resultIndices`.
    void eraseResults(const BitVector &resultIndices) {
      Type newType = (*static_cast<ConcreteOp *>(this)).getTypeWithoutResults(resultIndices);
      function_interface_impl::eraseFunctionResults(
          this->getOperation(), resultIndices, newType);
    }

    /// Return the type of this function with the specified arguments and
    /// results inserted. This is used to update the function's signature in
    /// the `insertArguments` and `insertResults` methods. The arrays must be
    /// sorted by increasing index.
    Type getTypeWithArgsAndResults(
      ArrayRef<unsigned> argIndices, TypeRange argTypes,
      ArrayRef<unsigned> resultIndices, TypeRange resultTypes) {
      SmallVector<Type> argStorage, resultStorage;
      TypeRange newArgTypes = function_interface_impl::insertTypesInto(
          (*static_cast<ConcreteOp *>(this)).getArgumentTypes(), argIndices, argTypes, argStorage);
      TypeRange newResultTypes = function_interface_impl::insertTypesInto(
          (*static_cast<ConcreteOp *>(this)).getResultTypes(), resultIndices, resultTypes, resultStorage);
      return (*static_cast<ConcreteOp *>(this)).cloneTypeWith(newArgTypes, newResultTypes);
    }

    /// Return the type of this function without the specified arguments and
    /// results. This is used to update the function's signature in the
    /// `eraseArguments` and `eraseResults` methods.
    Type getTypeWithoutArgsAndResults(
      const BitVector &argIndices, const BitVector &resultIndices) {
      SmallVector<Type> argStorage, resultStorage;
      TypeRange newArgTypes = function_interface_impl::filterTypesOut(
          (*static_cast<ConcreteOp *>(this)).getArgumentTypes(), argIndices, argStorage);
      TypeRange newResultTypes = function_interface_impl::filterTypesOut(
          (*static_cast<ConcreteOp *>(this)).getResultTypes(), resultIndices, resultStorage);
      return (*static_cast<ConcreteOp *>(this)).cloneTypeWith(newArgTypes, newResultTypes);
    }
    Type getTypeWithoutArgs(const BitVector &argIndices) {
      SmallVector<Type> argStorage;
      TypeRange newArgTypes = function_interface_impl::filterTypesOut(
          (*static_cast<ConcreteOp *>(this)).getArgumentTypes(), argIndices, argStorage);
      return (*static_cast<ConcreteOp *>(this)).cloneTypeWith(newArgTypes, (*static_cast<ConcreteOp *>(this)).getResultTypes());
    }
    Type getTypeWithoutResults(const BitVector &resultIndices) {
      SmallVector<Type> resultStorage;
      TypeRange newResultTypes = function_interface_impl::filterTypesOut(
          (*static_cast<ConcreteOp *>(this)).getResultTypes(), resultIndices, resultStorage);
      return (*static_cast<ConcreteOp *>(this)).cloneTypeWith((*static_cast<ConcreteOp *>(this)).getArgumentTypes(), newResultTypes);
    }

    //===------------------------------------------------------------------===//
    // Argument Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the argument at 'index'.
    ArrayRef<NamedAttribute> getArgAttrs(unsigned index) {
      return function_interface_impl::getArgAttrs(this->getOperation(), index);
    }

    /// Return an ArrayAttr containing all argument attribute dictionaries of
    /// this function, or nullptr if no arguments have attributes.
    ArrayAttr getAllArgAttrs() {
      return this->getOperation()->template getAttrOfType<ArrayAttr>(
          getArgDictAttrName());
    }
    /// Return all argument attributes of this function.
    void getAllArgAttrs(SmallVectorImpl<DictionaryAttr> &result) {
      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append((*static_cast<ConcreteOp *>(this)).getNumArguments(),
                      DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the argument at 'index',
    /// null otherwise.
    Attribute getArgAttr(unsigned index, StringAttr name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    Attribute getArgAttr(unsigned index, StringRef name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, StringAttr name) {
      return getArgAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }
    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, StringRef name) {
      return getArgAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }

    /// Set the attributes held by the argument at 'index'.
    void setArgAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
      function_interface_impl::setArgAttrs((*static_cast<ConcreteOp *>(this)), index, attributes);
    }

    /// Set the attributes held by the argument at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setArgAttrs(unsigned index, DictionaryAttr attributes) {
      function_interface_impl::setArgAttrs((*static_cast<ConcreteOp *>(this)), index, attributes);
    }
    void setAllArgAttrs(ArrayRef<DictionaryAttr> attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumArguments());
      function_interface_impl::setAllArgAttrDicts(this->getOperation(), attributes);
    }
    void setAllArgAttrs(ArrayRef<Attribute> attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumArguments());
      function_interface_impl::setAllArgAttrDicts(this->getOperation(), attributes);
    }
    void setAllArgAttrs(ArrayAttr attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumArguments());
      this->getOperation()->setAttr(getArgDictAttrName(), attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setArgAttr(unsigned index, StringAttr name, Attribute value) {
      function_interface_impl::setArgAttr((*static_cast<ConcreteOp *>(this)), index, name, value);
    }
    void setArgAttr(unsigned index, StringRef name, Attribute value) {
      setArgAttr(index,
                 StringAttr::get(this->getOperation()->getContext(), name),
                 value);
    }

    /// Remove the attribute 'name' from the argument at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    Attribute removeArgAttr(unsigned index, StringAttr name) {
      return function_interface_impl::removeArgAttr((*static_cast<ConcreteOp *>(this)), index, name);
    }
    Attribute removeArgAttr(unsigned index, StringRef name) {
      return removeArgAttr(
          index, StringAttr::get(this->getOperation()->getContext(), name));
    }

    //===------------------------------------------------------------------===//
    // Result Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the result at 'index'.
    ArrayRef<NamedAttribute> getResultAttrs(unsigned index) {
      return function_interface_impl::getResultAttrs(this->getOperation(), index);
    }

    /// Return an ArrayAttr containing all result attribute dictionaries of this
    /// function, or nullptr if no result have attributes.
    ArrayAttr getAllResultAttrs() {
      return this->getOperation()->template getAttrOfType<ArrayAttr>(
          getResultDictAttrName());
    }
    /// Return all result attributes of this function.
    void getAllResultAttrs(SmallVectorImpl<DictionaryAttr> &result) {
      if (ArrayAttr argAttrs = getAllResultAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append((*static_cast<ConcreteOp *>(this)).getNumResults(),
                      DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the result at 'index',
    /// null otherwise.
    Attribute getResultAttr(unsigned index, StringAttr name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    Attribute getResultAttr(unsigned index, StringRef name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, StringAttr name) {
      return getResultAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }
    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, StringRef name) {
      return getResultAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }

    /// Set the attributes held by the result at 'index'.
    void setResultAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
      function_interface_impl::setResultAttrs((*static_cast<ConcreteOp *>(this)), index, attributes);
    }

    /// Set the attributes held by the result at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setResultAttrs(unsigned index, DictionaryAttr attributes) {
      function_interface_impl::setResultAttrs((*static_cast<ConcreteOp *>(this)), index, attributes);
    }
    void setAllResultAttrs(ArrayRef<DictionaryAttr> attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumResults());
      function_interface_impl::setAllResultAttrDicts(
        this->getOperation(), attributes);
    }
    void setAllResultAttrs(ArrayRef<Attribute> attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumResults());
      function_interface_impl::setAllResultAttrDicts(
        this->getOperation(), attributes);
    }
    void setAllResultAttrs(ArrayAttr attributes) {
      assert(attributes.size() == (*static_cast<ConcreteOp *>(this)).getNumResults());
      this->getOperation()->setAttr(getResultDictAttrName(), attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setResultAttr(unsigned index, StringAttr name, Attribute value) {
      function_interface_impl::setResultAttr((*static_cast<ConcreteOp *>(this)), index, name, value);
    }
    void setResultAttr(unsigned index, StringRef name, Attribute value) {
      setResultAttr(index,
                    StringAttr::get(this->getOperation()->getContext(), name),
                    value);
    }

    /// Remove the attribute 'name' from the result at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    Attribute removeResultAttr(unsigned index, StringAttr name) {
      return function_interface_impl::removeResultAttr((*static_cast<ConcreteOp *>(this)), index, name);
    }

    /// Returns the dictionary attribute corresponding to the argument at
    /// 'index'. If there are no argument attributes at 'index', a null
    /// attribute is returned.
    DictionaryAttr getArgAttrDict(unsigned index) {
      assert(index < (*static_cast<ConcreteOp *>(this)).getNumArguments() && "invalid argument number");
      return function_interface_impl::getArgAttrDict(this->getOperation(), index);
    }

    /// Returns the dictionary attribute corresponding to the result at 'index'.
    /// If there are no result attributes at 'index', a null attribute is
    /// returned.
    DictionaryAttr getResultAttrDict(unsigned index) {
      assert(index < (*static_cast<ConcreteOp *>(this)).getNumResults() && "invalid result number");
      return function_interface_impl::getResultAttrDict(this->getOperation(), index);
    }
  
  };
}// namespace detail
template<typename ConcreteOp>
::llvm::ArrayRef<::mlir::Type> detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::getArgumentTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getArgumentTypes();
}
template<typename ConcreteOp>
::llvm::ArrayRef<::mlir::Type> detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::getResultTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getResultTypes();
}
template<typename ConcreteOp>
::mlir::Type detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::cloneTypeWith(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).cloneTypeWith(inputs, results);
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifyBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyBody();
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::Model<ConcreteOp>::verifyType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyType();
}
template<typename ConcreteOp>
::llvm::ArrayRef<::mlir::Type> detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getArgumentTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getArgumentTypes(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::ArrayRef<::mlir::Type> detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getResultTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getResultTypes(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Type detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::cloneTypeWith(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results) {
  return static_cast<const ConcreteOp *>(impl)->cloneTypeWith(tablegen_opaque_val, inputs, results);
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifyBody(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyBody(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::verifyType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyType(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Type detail::FunctionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::cloneTypeWith(::mlir::Operation *tablegen_opaque_val, ::mlir::TypeRange inputs, ::mlir::TypeRange results) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getFunctionType().clone(inputs, results);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyBody(::mlir::Operation *tablegen_opaque_val) const {
if ((llvm::cast<ConcreteOp>(tablegen_opaque_val)).isExternal())
        return success();
      ArrayRef<Type> fnInputTypes = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getArgumentTypes();
      // NOTE: This should just be (llvm::cast<ConcreteOp>(tablegen_opaque_val)).front() but access generically
      // because the interface methods defined here may be shadowed in
      // arbitrary ways. https://github.com/llvm/llvm-project/issues/54807
      Block &entryBlock = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getRegion(0).front();

      unsigned numArguments = fnInputTypes.size();
      if (entryBlock.getNumArguments() != numArguments)
        return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).emitOpError("entry block must have ")
              << numArguments << " arguments to match function signature";

      for (unsigned i = 0, e = fnInputTypes.size(); i != e; ++i) {
        Type argType = entryBlock.getArgument(i).getType();
        if (fnInputTypes[i] != argType) {
          return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).emitOpError("type of entry block argument #")
                << i << '(' << argType
                << ") must match the type of the corresponding argument in "
                << "function signature(" << fnInputTypes[i] << ')';
        }
      }

      return success();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::LogicalResult detail::FunctionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyType(::mlir::Operation *tablegen_opaque_val) const {
return success();
}
} // namespace mlir
