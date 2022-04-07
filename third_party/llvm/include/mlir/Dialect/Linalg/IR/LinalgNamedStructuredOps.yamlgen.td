
//===----------------------------------------------------------------------===//
// Op definition for CopyOp
//===----------------------------------------------------------------------===//

def CopyOp : LinalgStructuredBase_Op<"copy", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{ Copies the tensor elementwise. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<TypeFnAttr, "TypeFn::cast_signed">:$cast
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<CopyOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<CopyOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$cast,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("cast", cast);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<CopyOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for ElemwiseUnaryOp
//===----------------------------------------------------------------------===//

def ElemwiseUnaryOp : LinalgStructuredBase_Op<"elemwise_unary", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{ Applies the unary function fun elementwise. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<UnaryFnAttr, "UnaryFn::exp">:$fun,
DefaultValuedAttr<TypeFnAttr, "TypeFn::cast_signed">:$cast
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<ElemwiseUnaryOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ElemwiseUnaryOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$fun, "Attribute":$cast,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("fun", fun);
$_state.addAttribute("cast", cast);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<ElemwiseUnaryOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for ElemwiseBinaryOp
//===----------------------------------------------------------------------===//

def ElemwiseBinaryOp : LinalgStructuredBase_Op<"elemwise_binary", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{ Applies the binary function fun elementwise. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<BinaryFnAttr, "BinaryFn::add">:$fun,
DefaultValuedAttr<TypeFnAttr, "TypeFn::cast_signed">:$cast
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<ElemwiseBinaryOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<ElemwiseBinaryOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$fun, "Attribute":$cast,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("fun", fun);
$_state.addAttribute("cast", cast);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<ElemwiseBinaryOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for MatmulOp
//===----------------------------------------------------------------------===//

def MatmulOp : LinalgStructuredBase_Op<"matmul", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs a matrix multiplication of two 2D inputs. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<TypeFnAttr, "TypeFn::cast_signed">:$cast
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<MatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$cast,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("cast", cast);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<MatmulOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for MatmulUnsignedOp
//===----------------------------------------------------------------------===//

def MatmulUnsignedOp : LinalgStructuredBase_Op<"matmul_unsigned", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs an unsigned matrix multiplication of two 2D inputs. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<MatmulUnsignedOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatmulUnsignedOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for QuantizedMatmulOp
//===----------------------------------------------------------------------===//

def QuantizedMatmulOp : LinalgStructuredBase_Op<"quantized_matmul", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{ Performs a matrix multiplication of two 2D inputs. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. The quantized variant
includes zero-point adjustments for the left and right operands of the
matmul.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<QuantizedMatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<QuantizedMatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Mmt4DOp
//===----------------------------------------------------------------------===//

def Mmt4DOp : LinalgStructuredBase_Op<"mmt4d", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs a matrix-matrix-transpose multiplication of two 4D inputs. }];
  let description = [{
    Differences from linalg.matmul:
* The right hand side is transposed, whence the 't' in 'mmt'.
* The input and output tensors have a 4D shape instead of a 2D shape. They
  are interpreted as 2D matrices with one level of 2D tile subdivision,
  whence the 2+2=4 dimensions. The inner tile dimensions are identified with
  '0' suffixes below, for instance the LHS matrix shape (M, K, M0, K0) reads
  as: MxK tiles, each of shape M0xK0.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Mmt4DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Mmt4DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for BatchMatmulOp
//===----------------------------------------------------------------------===//

def BatchMatmulOp : LinalgStructuredBase_Op<"batch_matmul", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs a batched matrix multiplication of two 3D inputs. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<BatchMatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<BatchMatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for QuantizedBatchMatmulOp
//===----------------------------------------------------------------------===//

def QuantizedBatchMatmulOp : LinalgStructuredBase_Op<"quantized_batch_matmul", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{ Performs a batched matrix multiplication of two 3D inputs. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. The quantized variant
includes zero-point adjustments for the left and right operands of the
matmul.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<QuantizedBatchMatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<QuantizedBatchMatmulOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for MatvecOp
//===----------------------------------------------------------------------===//

def MatvecOp : LinalgStructuredBase_Op<"matvec", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs a matrix-vector multiplication. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<MatvecOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<MatvecOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for VecmatOp
//===----------------------------------------------------------------------===//

def VecmatOp : LinalgStructuredBase_Op<"vecmat", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs a vector-matrix multiplication. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<VecmatOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<VecmatOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for BatchMatvecOp
//===----------------------------------------------------------------------===//

def BatchMatvecOp : LinalgStructuredBase_Op<"batch_matvec", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs a batched matrix-vector multiplication. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<BatchMatvecOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<BatchMatvecOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DotOp
//===----------------------------------------------------------------------===//

def DotOp : LinalgStructuredBase_Op<"dot", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgContractionOpInterface])> {
    
  let summary = [{ Performs a dot product of two vectors to a scalar result. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<DotOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DotOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv1DOp
//===----------------------------------------------------------------------===//

def Conv1DOp : LinalgStructuredBase_Op<"conv_1d", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 1-D convolution with no channels. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv1DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv1DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DOp
//===----------------------------------------------------------------------===//

def Conv2DOp : LinalgStructuredBase_Op<"conv_2d", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 2-D convolution with no channels. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv3DOp
//===----------------------------------------------------------------------===//

def Conv3DOp : LinalgStructuredBase_Op<"conv_3d", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 3-D convolution with no channels. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv3DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv3DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv1DNwcWcfOp
//===----------------------------------------------------------------------===//

def Conv1DNwcWcfOp : LinalgStructuredBase_Op<"conv_1d_nwc_wcf", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 1-D convolution. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv1DNwcWcfOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv1DNwcWcfOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv1DNwcWcfOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DNhwcHwcfOp
//===----------------------------------------------------------------------===//

def Conv2DNhwcHwcfOp : LinalgStructuredBase_Op<"conv_2d_nhwc_hwcf", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 2-D convolution. }];
  let description = [{
    Layout:
  * Input: NHWC.
  * Kernel: HWCF.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv2DNhwcHwcfOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DNhwcHwcfQOp
//===----------------------------------------------------------------------===//

def Conv2DNhwcHwcfQOp : LinalgStructuredBase_Op<"conv_2d_nhwc_hwcf_q", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 2-D convolution with zero point offsets. }];
  let description = [{
    Layout:
  * Input: NHWC.
  * Kernel: HWCF.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. This includes the zero
point offsets common to quantized operations.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfQOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DNhwcHwcfQOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv2DNhwcHwcfQOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv2DNchwFchwOp
//===----------------------------------------------------------------------===//

def Conv2DNchwFchwOp : LinalgStructuredBase_Op<"conv_2d_nchw_fchw", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 2-D convolution. }];
  let description = [{
    Layout:
  * Input: NCHW.
  * Kernel: FCHW.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv2DNchwFchwOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv2DNchwFchwOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv2DNchwFchwOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for Conv3DNdhwcDhwcfOp
//===----------------------------------------------------------------------===//

def Conv3DNdhwcDhwcfOp : LinalgStructuredBase_Op<"conv_3d_ndhwc_dhwcf", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 3-D convolution. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<Conv3DNdhwcDhwcfOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<Conv3DNdhwcDhwcfOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<Conv3DNdhwcDhwcfOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv1DNwcWcOp
//===----------------------------------------------------------------------===//

def DepthwiseConv1DNwcWcOp : LinalgStructuredBase_Op<"depthwise_conv_1d_nwc_wc", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs depth-wise 1-D convolution. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[1]>, "{ static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv1DNwcWcOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv1DNwcWcOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv1DNwcWcOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv2DNhwcHwcOp
//===----------------------------------------------------------------------===//

def DepthwiseConv2DNhwcHwcOp : LinalgStructuredBase_Op<"depthwise_conv_2d_nhwc_hwc", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs depth-wise 2-D convolution. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output. Multiplier is set to 1
which is a special case for most depthwise convolutions.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv2DNhwcHwcQOp
//===----------------------------------------------------------------------===//

def DepthwiseConv2DNhwcHwcQOp : LinalgStructuredBase_Op<"depthwise_conv_2d_nhwc_hwc_q", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs depth-wise 2-D convolution. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcQOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcQOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcQOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv2DNhwcHwcmOp
//===----------------------------------------------------------------------===//

def DepthwiseConv2DNhwcHwcmOp : LinalgStructuredBase_Op<"depthwise_conv_2d_nhwc_hwcm", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs depth-wise 2-D convolution. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcmOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcmOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcmOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for DepthwiseConv2DNhwcHwcmQOp
//===----------------------------------------------------------------------===//

def DepthwiseConv2DNhwcHwcmQOp : LinalgStructuredBase_Op<"depthwise_conv_2d_nhwc_hwcm_q", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs depth-wise 2-D convolution. }];
  let description = [{
    Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcmQOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcmQOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<DepthwiseConv2DNhwcHwcmQOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNhwcSumOp
//===----------------------------------------------------------------------===//

def PoolingNhwcSumOp : LinalgStructuredBase_Op<"pooling_nhwc_sum", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs sum pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcSumOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcSumOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcSumOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNhwcMaxOp
//===----------------------------------------------------------------------===//

def PoolingNhwcMaxOp : LinalgStructuredBase_Op<"pooling_nhwc_max", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs max pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcMaxOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcMaxOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcMaxOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNhwcMaxUnsignedOp
//===----------------------------------------------------------------------===//

def PoolingNhwcMaxUnsignedOp : LinalgStructuredBase_Op<"pooling_nhwc_max_unsigned", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs unsigned max pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcMaxUnsignedOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcMaxUnsignedOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcMaxUnsignedOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNchwMaxOp
//===----------------------------------------------------------------------===//

def PoolingNchwMaxOp : LinalgStructuredBase_Op<"pooling_nchw_max", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs max pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNchwMaxOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNchwMaxOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNchwMaxOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNhwcMinOp
//===----------------------------------------------------------------------===//

def PoolingNhwcMinOp : LinalgStructuredBase_Op<"pooling_nhwc_min", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs min pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcMinOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcMinOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcMinOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNhwcMinUnsignedOp
//===----------------------------------------------------------------------===//

def PoolingNhwcMinUnsignedOp : LinalgStructuredBase_Op<"pooling_nhwc_min_unsigned", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs unsigned min pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[2]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNhwcMinUnsignedOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNhwcMinUnsignedOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNhwcMinUnsignedOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNdhwcSumOp
//===----------------------------------------------------------------------===//

def PoolingNdhwcSumOp : LinalgStructuredBase_Op<"pooling_ndhwc_sum", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 3D sum pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNdhwcSumOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNdhwcSumOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNdhwcSumOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNdhwcMaxOp
//===----------------------------------------------------------------------===//

def PoolingNdhwcMaxOp : LinalgStructuredBase_Op<"pooling_ndhwc_max", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 3D max pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNdhwcMaxOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNdhwcMaxOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNdhwcMaxOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for PoolingNdhwcMinOp
//===----------------------------------------------------------------------===//

def PoolingNdhwcMinOp : LinalgStructuredBase_Op<"pooling_ndhwc_min", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {
    
  let summary = [{ Performs 3D min pooling. }];
  let description = [{
    Numeric casting is performed on the input operand, promoting it to the same
data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
DefaultValuedAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$strides,
DefaultValuedAttr<RankedI64ElementsAttr<[3]>, "{ static_cast<int64_t>(1), static_cast<int64_t>(1), static_cast<int64_t>(1) }">:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<PoolingNdhwcMinOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<PoolingNdhwcMinOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<PoolingNdhwcMinOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
  }]>

    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for FillOp
//===----------------------------------------------------------------------===//

def FillOp : LinalgStructuredBase_Op<"fill", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgFillOpInterface])> {
    
  let summary = [{ Fills the output tensor with the given value. }];
  let description = [{
    Works for arbitrary ranked output tensors since the operation performs scalar
accesses only and is thus rank polymorphic. Numeric casting is performed on
the value operand, promoting it to the same data type as the output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<FillOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<FillOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    let hasCanonicalizer = 1;


    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}

//===----------------------------------------------------------------------===//
// Op definition for FillRng2DOp
//===----------------------------------------------------------------------===//

def FillRng2DOp : LinalgStructuredBase_Op<"fill_rng_2d", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[])> {
    
  let summary = [{ Fills the output tensor with pseudo random numbers. }];
  let description = [{
    The operation generations pseudo random numbers using a linear congruential
generator. It provides no guarantees regarding the distribution of the
generated random numbers. Instead of generating the random numbers
sequentially, it instantiates one random number generator per data element
and runs them in parallel. The seed operand and the indices of the data
element seed the random number generation. The min and max operands limit
the range of the generated random numbers.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<FillRng2DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<FillRng2DOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      
    }];
}
