// RUN: tpuc-opt --core-parallel -split-input-file %s | FileCheck %s

#loc = loc(unknown)
module @CompareConstInversedLocal attributes {module.FLOPs = 32768 : i64, module.asymmetric = false, module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.w8a16_linear = false, module.weight_file = "permutebinaryadd_tpu_lowered_bm1686_f32_weight.npz"} {
  func.func @main(%arg0: tensor<4x8x32x32xf32> loc(unknown)) -> tensor<4x8x32x32xf32> {
    %0 = "top.Input"(%arg0) : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc1)
    %1 = "tpu.CompareConst"(%0) {const_val = 1.000000e+00 : f64, inversed = true, mode = "Greater"} : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc2)
    return %1 : tensor<4x8x32x32xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("in_0")
#loc2 = loc("y")

// CHECK-LABEL:     "tpu.CoreParallel"(%0) ({
// CHECK:           %[[SPLIT:.*]]:2 = "tpu.{{(Core)?}}Split"(%0)
// CHECK:           %[[COMPARECONST0:.*]] = "tpu.CompareConst"(%[[SPLIT]]#0) {{{.*}} inversed = true{{.*}} mode = "Greater"{{.*}}}
// CHECK:           %[[COMPARECONST1:.*]] = "tpu.CompareConst"(%[[SPLIT]]#1) {{{.*}} inversed = true{{.*}} mode = "Greater"{{.*}}}
