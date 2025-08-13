// RUN: env TOP_DIALECT_REWRITER_CONFIG=%S/struct_optimize.json tpuc-opt --struct-optimize -split-input-file %s | FileCheck %s

// CHECK-LABEL: module @FuseAttentionSlicePattern
// CHECK:      %[[RESHAPE:.*]] = "top.Reshape"(%[[INPUT:.*]]) {{{.*}}}
// CHECK:      %[[W0:.*]] = "top.Weight"() : () -> tensor<1xf32>
// CHECK:      %[[G:.*]] = "top.Gather"(%[[RESHAPE]], %[[W0]]) {{{.*}}} : (tensor<1x77x3x512xf32>, tensor<1xf32>) -> tensor<77x1x512xf32>
// CHECK:      %[[S1:.*]] = "top.Squeeze"(%[[G]]) {{{.*}}} : (tensor<77x1x512xf32>) -> tensor<77x512xf32>
// CHECK:      %[[U1:.*]] = "top.Unsqueeze"(%[[S1]]) {{{.*}}} : (tensor<77x512xf32>) -> tensor<1x77x512xf32>
#loc = loc(unknown)
module @FuseAttentionSlicePattern attributes {module.chip = "ALL", module.platform = "ONNX", module.state = "TOP_F32", module.top_run_mode = "STATIC", module.weight_file = "test_weight.npz"} {
  func.func @main(%arg0: tensor<1x77x512xf32>) -> tensor<1x77x512xf32> {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "top.Input"(%arg0) {channel_format = "nchw", do_preprocess = true, keep_aspect_ratio = false, keep_ratio_mode = "letterbox", mean = [0.000000e+00, 0.000000e+00, 0.000000e+00], pad_type = "center", pad_value = 0 : i64, pixel_format = "bgr", scale = [1.000000e+00, 1.000000e+00, 1.000000e+00], yuv_type = ""} : (tensor<1x77x512xf32>) -> tensor<1x77x512xf32>
    %2 = "top.Weight"() : () -> tensor<512x1536xf32>
    %3 = "top.MatMul"(%1, %2, %0) {do_relu = false, hdim_is_batch = false, keep_dims = true, left_transpose = false, output_transpose = false, relu_limit = -1.000000e+00 : f64, right_transpose = false} : (tensor<1x77x512xf32>, tensor<512x1536xf32>, none) -> tensor<1x77x1536xf32>
    %4 = "top.Weight"() : () -> tensor<1x1x1536xf32>
    %5 = "top.Add"(%3, %4) {do_relu = false, is_scalar = false, relu_limit = -1.000000e+00 : f64} : (tensor<1x77x1536xf32>, tensor<1x1x1536xf32>) -> tensor<1x77x1536xf32>
    // reshape to [77, 1, 3, 512] to satisfy pattern precondition (second dim == batch)
    %6 = "top.Reshape"(%5) {flatten_start_dim = -1 : i64, shape = [77, 1, 3, 512]} : (tensor<1x77x1536xf32>) -> tensor<77x1x3x512xf32>
    %7 = "top.Unsqueeze"(%6) {axes = [0]} : (tensor<77x1x3x512xf32>) -> tensor<1x77x1x3x512xf32>
    %8 = "top.Permute"(%7) {order = [3, 1, 2, 0, 4]} : (tensor<1x77x1x3x512xf32>) -> tensor<3x77x1x1x512xf32>
    %9 = "top.Squeeze"(%8) {axes = [3], is_scalar = false} : (tensor<3x77x1x1x512xf32>) -> tensor<3x77x1x512xf32>
    %10 = "top.Weight"() : () -> tensor<1xf32>
    %11 = "top.Gather"(%9, %10) {axis = 0 : si32, is_scalar = false, keepdims = false} : (tensor<3x77x1x512xf32>, tensor<1xf32>) -> tensor<77x1x512xf32>
    %12 = "top.Weight"() : () -> tensor<1xf32>
    %13 = "top.Gather"(%9, %12) {axis = 0 : si32, is_scalar = false, keepdims = false} : (tensor<3x77x1x512xf32>, tensor<1xf32>) -> tensor<77x1x512xf32>
    %14 = "top.Weight"() : () -> tensor<1xf32>
    %15 = "top.Gather"(%9, %14) {axis = 0 : si32, is_scalar = false, keepdims = false} : (tensor<3x77x1x512xf32>, tensor<1xf32>) -> tensor<77x1x512xf32>
    // fold back to 1x77x512 just to return
    %16 = "top.Squeeze"(%11) {axes = [1], is_scalar = false} : (tensor<77x1x512xf32>) -> tensor<77x512xf32>
    %17 = "top.Unsqueeze"(%16) {axes = [0]} : (tensor<77x512xf32>) -> tensor<1x77x512xf32>
    return %17 : tensor<1x77x512xf32>
  }
}

// -----


// CHECK-LABEL: module @RemovePermuteBeforeLayerNorm
// CHECK:      %[[LN:.*]] = "top.LayerNorm"(%[[INPUT:.*]], %[[WEIGHT1:.*]], %[[WEIGHT2:.*]]) {{{.*}}} : (tensor<1x77x512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<77x1x512xf32> loc({{.*}})
// CHECK:      %[[PERMUTE2:.*]] = "top.Permute"(%[[LN]]) {{{.*}}} : (tensor<77x1x512xf32>) -> tensor<1x77x512xf32> loc({{.*}})

#loc = loc(unknown)
module @RemovePermuteBeforeLayerNorm attributes {module.chip = "ALL", module.platform = "ONNX", module.state = "TOP_F32", module.top_run_mode = "STATIC", module.weight_file = "test_weight.npz"} {
  func.func @main(%arg0: tensor<1x77x512xf32>) -> tensor<1x77x512xf32> {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "top.Input"(%arg0) {channel_format = "nchw", do_preprocess = true, keep_aspect_ratio = false, keep_ratio_mode = "letterbox", mean = [0.000000e+00, 0.000000e+00, 0.000000e+00], pad_type = "center", pad_value = 0 : i64, pixel_format = "bgr", scale = [1.000000e+00, 1.000000e+00, 1.000000e+00], yuv_type = ""} : (tensor<1x77x512xf32>) -> tensor<1x77x512xf32> loc(#loc1)
    %2 = "top.Permute"(%1) {order = [1, 0, 2]} : (tensor<1x77x512xf32>) -> tensor<77x1x512xf32> loc(#loc2)
    %3 = "top.Weight"() : () -> tensor<512xf32> loc(#loc3)
    %4 = "top.Weight"() : () -> tensor<512xf32> loc(#loc4)
    %5 = "top.LayerNorm"(%2, %3, %4) {axis = -1 : si32, eps = 1.000000e-05 : f64, normalized_shape = [512]} : (tensor<77x1x512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<77x1x512xf32> loc(#loc5)
    %6 = "top.Permute"(%5) {order = [1, 0, 2]} : (tensor<77x1x512xf32>) -> tensor<1x77x512xf32> loc(#loc6)
    return %6 : tensor<1x77x512xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("input")
#loc2 = loc("permute1")
#loc3 = loc("weight1")
#loc4 = loc("weight2")
#loc5 = loc("layernorm")
#loc6 = loc("permute2")

// -----

// CHECK-LABEL: module @RemovePermuteBetweenAddGather
// CHECK:      %[[ADD:.*]] = "top.Add"({{.*}})
// CHECK:      %[[G:.*]] = "top.Gather"(%[[ADD]], {{.*}})
#loc = loc(unknown)
module @RemovePermuteBetweenAddGather attributes {module.chip = "ALL", module.platform = "ONNX", module.state = "TOP_F32", module.top_run_mode = "STATIC", module.weight_file = "test_weight.npz"} {
  func.func @main(%arg0: tensor<77x512xf32>) -> tensor<77x512xf32> {
    %0 = "top.Input"(%arg0) {channel_format = "nchw", do_preprocess = true, keep_aspect_ratio = false, keep_ratio_mode = "letterbox", mean = [0.0, 0.0, 0.0], pad_type = "center", pad_value = 0 : i64, pixel_format = "bgr", scale = [1.0, 1.0, 1.0], yuv_type = ""} : (tensor<77x512xf32>) -> tensor<77x512xf32>
    %w = "top.Weight"() : () -> tensor<77x512xf32>
    %a = "top.Add"(%0, %w) {do_relu = false, is_scalar = false, relu_limit = -1.0 : f64} : (tensor<77x512xf32>, tensor<77x512xf32>) -> tensor<77x512xf32>
    %p = "top.Permute"(%a) {order = [0, 1]} : (tensor<77x512xf32>) -> tensor<77x512xf32>
    %idx = "top.Weight"() : () -> tensor<1xf32>
    %g = "top.Gather"(%p, %idx) {axis = 0 : si32, is_scalar = false, keepdims = false} : (tensor<77x512xf32>, tensor<1xf32>) -> tensor<77x512xf32>
    return %g : tensor<77x512xf32>
  }
}
