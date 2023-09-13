// RUN: tpuc-opt --parallel='num_core=2' %s | FileCheck %s

// CHECK-LABEL:     "tpu.Parallel"(%1, %0) ({
// CHECK:           %[[SPLIT:.*]]:2 = "tpu.Split"(%1) : (tensor<4x8x32x32xf32>) -> (tensor<2x8x32x32xf32>, tensor<2x8x32x32xf32>) loc({{.*}})
// CHECK:           %[[PERMUTE0:.*]] = "tpu.Permute"(%[[SPLIT]]#0, %0) {order = [0, 3, 2, 1]} : (tensor<2x8x32x32xf32>, none) -> tensor<2x32x32x8xf32> loc({{.*}})
// CHECK:           %[[PERMUTE1:.*]] = "tpu.Permute"(%[[SPLIT]]#1, %0) {order = [0, 3, 2, 1]} : (tensor<2x8x32x32xf32>, none) -> tensor<2x32x32x8xf32> loc({{.*}})
// CHECK:           %[[JOIN:.*]] = "tpu.Join"(%[[PERMUTE0]], %[[PERMUTE1]]) : (tensor<2x32x32x8xf32>, tensor<2x32x32x8xf32>) -> tensor<4x32x32x8xf32> loc({{.*}})
// CHECK:           "tpu.Yield"(%[[JOIN]]) : (tensor<4x32x32x8xf32>) -> () loc({{.*}})
// CHECK:               }) : (tensor<4x8x32x32xf32>, none) -> tensor<4x32x32x8xf32> loc({{.*}})
#loc = loc(unknown)
module @PermuteBinaryAdd attributes {module.FLOPs = 32768 : i64, module.asymmetric = false, module.chip = "bm1686", module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.w8a16_linear = false, module.weight_file = "permutebinaryadd_tpu_lowered_bm1686_f32_weight.npz"} {
  func.func @main(%arg0: tensor<4x8x32x32xf32> loc(unknown), %arg1: tensor<4x1x32x32xf32> loc(unknown)) -> tensor<4x32x32x8xf32> {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "top.Input"(%arg0) : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc1)
    %2 = "top.Input"(%arg1) : (tensor<4x1x32x32xf32>) -> tensor<4x1x32x32xf32> loc(#loc2)
    %3 = "tpu.Permute"(%1, %0) {order = [0, 3, 2, 1]} : (tensor<4x8x32x32xf32>, none) -> tensor<4x32x32x8xf32> loc(#loc3)
    %4 = "tpu.Permute"(%2, %0) {order = [0, 3, 2, 1]} : (tensor<4x1x32x32xf32>, none) -> tensor<4x32x32x1xf32> loc(#loc4)
    %5 = "tpu.Add"(%3, %4) {do_relu = false, relu_limit = -1.000000e+00 : f64} : (tensor<4x32x32x8xf32>, tensor<4x32x32x1xf32>) -> tensor<4x32x32x8xf32> loc(#loc5)
    return %5 : tensor<4x32x32x8xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("in_0")
#loc2 = loc("in_1")
#loc3 = loc("/Transpose_output_0_Transpose")
#loc4 = loc("/Transpose_1_output_0_Transpose")
#loc5 = loc("4_Add")
