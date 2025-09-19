// RUN: tpuc-opt --struct-optimize -split-input-file %s | FileCheck %s

// CHECK-LABEL: module @ConvertFuseCorrelationPattern
// CHECK:      %[[INPUT1:.*]] = "top.Input"(%arg0)
// CHECK:      %[[INPUT2:.*]] = "top.Input"(%arg1)
// CHECK:      %[[CORR:.*]] = "top.Correlation"(%[[INPUT1]], %[[INPUT2]]) {max_disp = 3 : i64, num_groups = 2 : i64} : (tensor<1x4x6x8xf16>, tensor<1x4x6x8xf16>) -> tensor<*xf16>
// CHECK:      %[[UNSQUEEZE:.*]] = "top.Unsqueeze"(%[[CORR]]) {axes = [0]} : (tensor<*xf16>) -> tensor<*xf16>
// CHECK:      return %[[UNSQUEEZE]] : tensor<*xf16>

#loc = loc(unknown)
module @ConvertFuseCorrelationPattern attributes {module.chip = "ALL", module.platform = "ONNX", module.state = "TOP_F16", module.top_run_mode = "STATIC", module.weight_file = "test_weight.npz"} {
  func.func @main(%arg0: tensor<1x4x6x8xf16>, %arg1: tensor<1x4x6x8xf16>) -> tensor<*xf16> {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "top.Input"(%arg0) {do_preprocess = false} : (tensor<1x4x6x8xf16>) -> tensor<1x4x6x8xf16> loc(#loc1)
    %2 = "top.Input"(%arg1) {do_preprocess = false} : (tensor<1x4x6x8xf16>) -> tensor<1x4x6x8xf16> loc(#loc2)
    
    // Offset 0: Direct Mul -> Reshape -> ReduceMean -> Reshape -> ScatterND  
    %3 = "top.Mul"(%1, %2) {do_relu = false, is_scalar = false, relu_limit = -1.000000e+00 : f64} : (tensor<1x4x6x8xf16>, tensor<1x4x6x8xf16>) -> tensor<*xf16> loc(#loc3)
    %4 = "top.Reshape"(%3) {flatten_start_dim = -1 : i64, shape = [1, 2, 2, 6, 8]} : (tensor<*xf16>) -> tensor<*xf16> loc(#loc4)
    %5 = "top.Reduce"(%4) {axes = [2], is_scalar = false, keepdims = false, mode = "ReduceMean"} : (tensor<*xf16>) -> tensor<*xf16> loc(#loc5)
    %6 = "top.Reshape"(%5) {flatten_start_dim = -1 : i64, shape = [1, 2, 1, 6, 8]} : (tensor<*xf16>) -> tensor<*xf16> loc(#loc6)
    %7 = "top.Weight"() : () -> tensor<1x2x3x6x8xf16> loc(#loc7)
    %8 = "top.Weight"() : () -> tensor<1x2x1x6x8x5xf16> loc(#loc8)
    %9 = "top.ScatterND"(%7, %8, %6) {reduction = 0 : i32} : (tensor<1x2x3x6x8xf16>, tensor<1x2x1x6x8x5xf16>, tensor<*xf16>) -> tensor<*xf16> loc(#loc9)
    
    // Offset 1: Slice -> Slice -> Mul -> Reshape -> ReduceMean -> Reshape -> ScatterND
    %10 = "top.Slice"(%1, %0, %0, %0) {axes = [3], ends = [2147482624], hasparamConvert_axes = [], offset = [1], steps = [1]} : (tensor<1x4x6x8xf16>, none, none, none) -> tensor<*xf16> loc(#loc10)
    %11 = "top.Slice"(%2, %0, %0, %0) {axes = [3], ends = [-1], hasparamConvert_axes = [], offset = [0], steps = [1]} : (tensor<1x4x6x8xf16>, none, none, none) -> tensor<*xf16> loc(#loc11)
    %12 = "top.Mul"(%10, %11) {do_relu = false, is_scalar = false, relu_limit = -1.000000e+00 : f64} : (tensor<*xf16>, tensor<*xf16>) -> tensor<*xf16> loc(#loc12)
    %13 = "top.Reshape"(%12) {flatten_start_dim = -1 : i64, shape = [1, 2, 2, 6, 7]} : (tensor<*xf16>) -> tensor<*xf16> loc(#loc13)
    %14 = "top.Reduce"(%13) {axes = [2], is_scalar = false, keepdims = false, mode = "ReduceMean"} : (tensor<*xf16>) -> tensor<*xf16> loc(#loc14)
    %15 = "top.Reshape"(%14) {flatten_start_dim = -1 : i64, shape = [1, 2, 1, 6, 7]} : (tensor<*xf16>) -> tensor<*xf16> loc(#loc15)
    %16 = "top.Weight"() : () -> tensor<1x2x1x6x7x5xf16> loc(#loc16)
    %17 = "top.ScatterND"(%9, %16, %15) {reduction = 0 : i32} : (tensor<*xf16>, tensor<1x2x1x6x7x5xf16>, tensor<*xf16>) -> tensor<*xf16> loc(#loc17)
    
    // Offset 2: Slice -> Slice -> Mul -> Reshape -> ReduceMean -> Reshape -> ScatterND (final)
    %18 = "top.Slice"(%1, %0, %0, %0) {axes = [3], ends = [2147482624], hasparamConvert_axes = [], offset = [2], steps = [1]} : (tensor<1x4x6x8xf16>, none, none, none) -> tensor<*xf16> loc(#loc18)
    %19 = "top.Slice"(%2, %0, %0, %0) {axes = [3], ends = [-2], hasparamConvert_axes = [], offset = [0], steps = [1]} : (tensor<1x4x6x8xf16>, none, none, none) -> tensor<*xf16> loc(#loc19)
    %20 = "top.Mul"(%18, %19) {do_relu = false, is_scalar = false, relu_limit = -1.000000e+00 : f64} : (tensor<*xf16>, tensor<*xf16>) -> tensor<*xf16> loc(#loc20)
    %21 = "top.Reshape"(%20) {flatten_start_dim = -1 : i64, shape = [1, 2, 2, 6, 6]} : (tensor<*xf16>) -> tensor<*xf16> loc(#loc21)
    %22 = "top.Reduce"(%21) {axes = [2], is_scalar = false, keepdims = false, mode = "ReduceMean"} : (tensor<*xf16>) -> tensor<*xf16> loc(#loc22)
    %23 = "top.Reshape"(%22) {flatten_start_dim = -1 : i64, shape = [1, 2, 1, 6, 6]} : (tensor<*xf16>) -> tensor<*xf16> loc(#loc23)
    %24 = "top.Weight"() : () -> tensor<1x2x1x6x6x5xf16> loc(#loc24)
    %25 = "top.ScatterND"(%17, %24, %23) {reduction = 0 : i32} : (tensor<*xf16>, tensor<1x2x1x6x6x5xf16>, tensor<*xf16>) -> tensor<*xf16> loc(#loc25)
    
    return %25 : tensor<*xf16> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("in_0")
#loc2 = loc("in_1")
#loc3 = loc("/Mul_output_0_Mul")
#loc4 = loc("/Reshape_output_0_Reshape")
#loc5 = loc("/ReduceMean_output_0_ReduceMean")
#loc6 = loc("/Reshape_1_output_0_Reshape")
#loc7 = loc("/Constant_3_output_0")
#loc8 = loc("/Concat_output_0")
#loc9 = loc("/ScatterND_output_0_ScatterND")
#loc10 = loc("/Slice_output_0_Slice")
#loc11 = loc("/Slice_1_output_0_Slice")
#loc12 = loc("/Mul_6_output_0_Mul")
#loc13 = loc("/Reshape_2_output_0_Reshape")
#loc14 = loc("/ReduceMean_1_output_0_ReduceMean")
#loc15 = loc("/Reshape_3_output_0_Reshape")
#loc16 = loc("/Concat_1_output_0")
#loc17 = loc("/ScatterND_1_output_0_ScatterND")
#loc18 = loc("/Slice_4_output_0_Slice")
#loc19 = loc("/Slice_5_output_0_Slice")
#loc20 = loc("/Mul_12_output_0_Mul")
#loc21 = loc("/Reshape_4_output_0_Reshape")
#loc22 = loc("/ReduceMean_2_output_0_ReduceMean")
#loc23 = loc("/Reshape_5_output_0_Reshape")
#loc24 = loc("/Concat_3_output_0")
#loc25 = loc("/ScatterND_2_output_0_ScatterND")
