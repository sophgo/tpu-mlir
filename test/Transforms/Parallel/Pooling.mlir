// RUN: tpuc-opt --core-parallel -split-input-file %s | FileCheck %s

// CHECK-LABEL: module @AvgPool1d
// CHECK:      %[[POOLING0:.*]] = "tpu.Pool1D"(%[[INPUT:.*]]#0) {{{.*}}}  : (tensor<1x16x128xf32>) -> tensor<1x16x64xf32> loc({{.*}})
// CHECK:      %[[POOLING1:.*]] = "tpu.Pool1D"(%[[INPUT]]#1) {{{.*}}}  : (tensor<1x16x128xf32>) -> tensor<1x16x64xf32> loc({{.*}})
// CHECK:      %[[JOIN:.*]] = "tpu.Join"(%[[POOLING0]], %[[POOLING1]]) : (tensor<1x16x64xf32>, tensor<1x16x64xf32>) -> tensor<1x32x64xf32> loc({{.*}})

#loc = loc(unknown)
module @AvgPool1d attributes {module.FLOPs = 6144 : i64, module.asymmetric = false, module.chip = "bm1688", module.cores = 2 : i64, module.devices = 1 : i64, module.mode = "F16", module.platform = "ONNX", module.q_group_size = 0 : i64, module.state = "TPU_LOWERED", module.weight_file = "avgpool1d_tpu_lowered_bm1688_f16_weight.npz"} {
  func.func @main(%arg0: tensor<1x32x128xf32> loc(unknown)) -> tensor<1x32x64xf32> {
    %0 = "top.Input"(%arg0) {do_preprocess = false} : (tensor<1x32x128xf32>) -> tensor<1x32x128xf32> loc(#loc1)
    %1 = "tpu.Pool1D"(%0) {count_include_pad = false, do_relu = false, keepdims = true, kernel_shape = [2], pad_value = 0 : i64, pads = [0, 0], pool_mode = #tpu<pool_mode Avg>, relu_limit = -1.000000e+00 : f64, strides = [2]} : (tensor<1x32x128xf32>) -> tensor<1x32x64xf32> loc(#loc2)
    %2 = "tpu.Cast"(%1) {with_scale = true} : (tensor<1x32x64xf32>) -> tensor<1x32x64xf16> loc(#loc3)
    %3 = "tpu.MulConst"(%2) {const_val = 2.000000e+00 : f64, do_relu = false, multiplier = 1 : si32, relu_limit = -1.000000e+00 : f64, rshift = 0 : si32} : (tensor<1x32x64xf16>) -> tensor<1x32x64xf16> loc(#loc4)
    %4 = "tpu.Cast"(%3) {with_scale = true} : (tensor<1x32x64xf16>) -> tensor<1x32x64xf32> loc(#loc5)
    return %4 : tensor<1x32x64xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("input")
#loc2 = loc("pool_output_AveragePool")
#loc3 = loc("pool_output_AveragePooloutput_Mul_f16")
#loc4 = loc("output_Mul")
#loc5 = loc("output_Mul_f32")

// -----

// CHECK-LABEL: module @AvgPool2d
// CHECK:      %[[POOLING0:.*]] = "tpu.Pool2D"(%[[INPUT:.*]]#0) {{{.*}}}  : (tensor<1x16x128x128xf32>) -> tensor<1x16x64x64xf32> loc({{.*}})
// CHECK:      %[[POOLING1:.*]] = "tpu.Pool2D"(%[[INPUT]]#1) {{{.*}}}  : (tensor<1x16x128x128xf32>) -> tensor<1x16x64x64xf32> loc({{.*}})
// CHECK:      %[[JOIN:.*]] = "tpu.Join"(%[[POOLING0]], %[[POOLING1]]) : (tensor<1x16x64x64xf32>, tensor<1x16x64x64xf32>) -> tensor<1x32x64x64xf32> loc({{.*}})
#loc = loc(unknown)
module @AvgPool2d attributes {module.FLOPs = 524288 : i64, module.asymmetric = false, module.chip = "bm1688", module.cores = 2 : i64, module.devices = 1 : i64, module.mode = "F16", module.platform = "ONNX", module.q_group_size = 0 : i64, module.state = "TPU_LOWERED", module.weight_file = "avgpool2d_tpu_lowered_bm1688_f16_weight.npz"} {
  func.func @main(%arg0: tensor<1x32x128x128xf32> loc(unknown)) -> tensor<1x32x64x64xf32> {
    %0 = "top.Input"(%arg0) {do_preprocess = false} : (tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> loc(#loc1)
    %1 = "tpu.Pool2D"(%0) {count_include_pad = false, do_relu = false, keepdims = true, kernel_shape = [2, 2], pad_value = 0 : i64, pads = [0, 0, 0, 0], pool_mode = #tpu<pool_mode Avg>, relu_limit = -1.000000e+00 : f64, strides = [2, 2]} : (tensor<1x32x128x128xf32>) -> tensor<1x32x64x64xf32> loc(#loc2)
    return %1 : tensor<1x32x64x64xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("input")
#loc2 = loc("output_AveragePool")

// -----

// CHECK-LABEL: module @AvgPool3d
// CHECK:      %[[POOLING0:.*]] = "tpu.Pool3D"(%[[INPUT:.*]]#0, %0) {{{.*}}}  : (tensor<1x8x16x32x64xf32>, none) -> tensor<1x8x8x16x32xf32> loc({{.*}})
// CHECK:      %[[POOLING1:.*]] = "tpu.Pool3D"(%[[INPUT]]#1, %0) {{{.*}}}  : (tensor<1x8x16x32x64xf32>, none) -> tensor<1x8x8x16x32xf32> loc({{.*}})
// CHECK:      %[[POOLING2:.*]] = "tpu.Pool3D"(%[[INPUT]]#2, %0) {{{.*}}}  : (tensor<1x8x16x32x64xf32>, none) -> tensor<1x8x8x16x32xf32> loc({{.*}})
// CHECK:      %[[POOLING3:.*]] = "tpu.Pool3D"(%[[INPUT]]#3, %0) {{{.*}}}  : (tensor<1x8x16x32x64xf32>, none) -> tensor<1x8x8x16x32xf32> loc({{.*}})
// CHECK:      %[[JOIN:.*]] = "tpu.Join"(%[[POOLING0]], %[[POOLING1]], %[[POOLING2]], %[[POOLING3]], %[[V0:.*]], %[[V1:.*]], %[[V2:.*]], %[[V3:.*]]) : (tensor<1x8x8x16x32xf32>, tensor<1x8x8x16x32xf32>, tensor<1x8x8x16x32xf32>, tensor<1x8x8x16x32xf32>, tensor<1x8x8x16x32xf32>, tensor<1x8x8x16x32xf32>, tensor<1x8x8x16x32xf32>, tensor<1x8x8x16x32xf32>) -> tensor<2x32x8x16x32xf32> loc({{.*}})
#loc = loc(unknown)
module @AvgPool3d attributes {module.FLOPs = 2097152 : i64, module.asymmetric = false, module.chip = "bm1690", module.cores = 8 : i64, module.devices = 1 : i64, module.mode = "F16", module.platform = "ONNX", module.q_group_size = 0 : i64, module.state = "TPU_LOWERED", module.weight_file = "avgpool3d_tpu_lowered_bm1688_f16_weight.npz"} {
  func.func @main(%arg0: tensor<2x32x16x32x64xf32> loc(unknown)) -> tensor<2x32x8x16x32xf32> {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "top.Input"(%arg0) {do_preprocess = false} : (tensor<2x32x16x32x64xf32>) -> tensor<2x32x16x32x64xf32> loc(#loc1)
    %2 = "tpu.Pool3D"(%1, %0) {count_include_pad = false, do_relu = false, keepdims = true, kernel_shape = [2, 2, 2], pad_value = 0 : i64, pads = [0, 0, 0, 0, 0, 0], pool_mode = #tpu<pool_mode Avg>, relu_limit = -1.000000e+00 : f64, strides = [2, 2, 2]} : (tensor<2x32x16x32x64xf32>, none) -> tensor<2x32x8x16x32xf32> loc(#loc2)
    return %2 : tensor<2x32x8x16x32xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("input")
#loc2 = loc("output_AveragePool")
