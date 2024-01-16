// RUN: tpuc-opt --core-parallel -split-input-file %s | FileCheck %s

#loc = loc(unknown)
module @AddConst attributes {module.FLOPs = 32768 : i64, module.asymmetric = false, module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.w8a16_linear = false, module.weight_file = "permutebinaryadd_tpu_lowered_bm1686_f32_weight.npz"} {
  func.func @main(%arg0: tensor<4x8x32x32xf32> loc(unknown)) -> tensor<4x8x32x32xf32> {
    %0 = "top.Input"(%arg0) : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc1)
    %1 = "tpu.AddConst"(%0) {const_val = 3.000000e+00 : f64, do_relu = false, f8_scale = 1.000000e+00 : f64, multiplier = 1 : si32, relu_limit = -1.000000e+00 : f64, rshift = 0 : si32} : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc2)
    return %1 : tensor<4x8x32x32xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("in_0")
#loc2 = loc("y")

// CHECK-LABEL:     "tpu.CoreParallel"(%0) ({
// CHECK:           %[[SPLIT:.*]]:2 = "tpu.Split"(%0) : (tensor<4x8x32x32xf32>) -> (tensor<2x8x32x32xf32>, tensor<2x8x32x32xf32>) loc({{.*}})
// CHECK:           %[[ADDCONST0:.*]] = "tpu.AddConst"(%[[SPLIT]]#0) {{{.*}}} : (tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32> loc({{.*}})
// CHECK:           %[[ADDCONST1:.*]] = "tpu.AddConst"(%[[SPLIT]]#1) {{{.*}}} : (tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32> loc({{.*}})
// CHECK:           %[[JOIN:.*]] = "tpu.Join"(%[[ADDCONST0]], %[[ADDCONST1]]) : (tensor<2x8x32x32xf32>, tensor<2x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc({{.*}})
// CHECK:           "tpu.Yield"(%[[JOIN]]) : (tensor<4x8x32x32xf32>) -> () loc({{.*}})
// CHECK:             }) {{.*}} : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc({{.*}})

// -----

#loc = loc(unknown)
module @SubConst attributes {module.FLOPs = 32768 : i64, module.asymmetric = false, module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.w8a16_linear = false, module.weight_file = "permutebinaryadd_tpu_lowered_bm1686_f32_weight.npz"} {
  func.func @main(%arg0: tensor<4x8x32x32xf32> loc(unknown)) -> tensor<4x8x32x32xf32> {
    %0 = "top.Input"(%arg0) : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc1)
    %1 = "tpu.SubConst"(%0) {const_val = 3.000000e+00 : f64, do_relu = false, f8_scale = 1.000000e+00 : f64, is_reverse = true, multiplier = 1 : si32, relu_limit = -1.000000e+00 : f64, rshift = 0 : si32} : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc2)
    return %1 : tensor<4x8x32x32xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("in_0")
#loc2 = loc("y")

// CHECK-LABEL:     "tpu.CoreParallel"(%0) ({
// CHECK:           %[[SPLIT:.*]]:2 = "tpu.Split"(%0) : (tensor<4x8x32x32xf32>) -> (tensor<2x8x32x32xf32>, tensor<2x8x32x32xf32>) loc({{.*}})
// CHECK:           %[[SUBCONST0:.*]] = "tpu.SubConst"(%[[SPLIT]]#0) {{{.*}}} : (tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32> loc({{.*}})
// CHECK:           %[[SUBCONST1:.*]] = "tpu.SubConst"(%[[SPLIT]]#1) {{{.*}}} : (tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32> loc({{.*}})
// CHECK:           %[[JOIN:.*]] = "tpu.Join"(%[[SUBCONST0]], %[[SUBCONST1]]) : (tensor<2x8x32x32xf32>, tensor<2x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc({{.*}})
// CHECK:           "tpu.Yield"(%[[JOIN]]) : (tensor<4x8x32x32xf32>) -> () loc({{.*}})
// CHECK:             }) {{.*}} : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc({{.*}})

// -----

#loc = loc(unknown)
module @MulConst attributes {module.FLOPs = 32768 : i64, module.asymmetric = false, module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.w8a16_linear = false, module.weight_file = "permutebinaryadd_tpu_lowered_bm1686_f32_weight.npz"} {
  func.func @main(%arg0: tensor<4x8x32x32xf32> loc(unknown)) -> tensor<4x8x32x32xf32> {
    %0 = "top.Input"(%arg0) : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc1)
    %1 = "tpu.MulConst"(%0) {const_val = 3.000000e+00 : f64, do_relu = false, multiplier = 1 : si32, relu_limit = -1.000000e+00 : f64, rshift = 0 : si32} : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc2)
    return %1 : tensor<4x8x32x32xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("in_0")
#loc2 = loc("y")

// CHECK-LABEL:     "tpu.CoreParallel"(%0) ({
// CHECK:           %[[SPLIT:.*]]:2 = "tpu.Split"(%0) : (tensor<4x8x32x32xf32>) -> (tensor<2x8x32x32xf32>, tensor<2x8x32x32xf32>) loc({{.*}})
// CHECK:           %[[MULCONST0:.*]] = "tpu.MulConst"(%[[SPLIT]]#0) {{{.*}}} : (tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32> loc({{.*}})
// CHECK:           %[[MULCONST1:.*]] = "tpu.MulConst"(%[[SPLIT]]#1) {{{.*}}} : (tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32> loc({{.*}})
// CHECK:           %[[JOIN:.*]] = "tpu.Join"(%[[MULCONST0]], %[[MULCONST1]]) : (tensor<2x8x32x32xf32>, tensor<2x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc({{.*}})
// CHECK:           "tpu.Yield"(%[[JOIN]]) : (tensor<4x8x32x32xf32>) -> () loc({{.*}})
// CHECK:             }) {{.*}} : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc({{.*}})

// -----

#loc = loc(unknown)
module @MaxConst attributes {module.FLOPs = 32768 : i64, module.asymmetric = false, module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.w8a16_linear = false, module.weight_file = "permutebinaryadd_tpu_lowered_bm1686_f32_weight.npz"} {
  func.func @main(%arg0: tensor<4x8x32x32xf32> loc(unknown)) -> tensor<4x8x32x32xf32> {
    %0 = "top.Input"(%arg0) : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc1)
    %1 = "tpu.MaxConst"(%0) {const_val = 3.000000e+00 : f64, do_relu = false, multiplier = 1 : si32, relu_limit = -1.000000e+00 : f64, rshift = 0 : si32} : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc2)
    return %1 : tensor<4x8x32x32xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("in_0")
#loc2 = loc("y")

// CHECK-LABEL:     "tpu.CoreParallel"(%0) ({
// CHECK:           %[[SPLIT:.*]]:2 = "tpu.Split"(%0) : (tensor<4x8x32x32xf32>) -> (tensor<2x8x32x32xf32>, tensor<2x8x32x32xf32>) loc({{.*}})
// CHECK:           %[[MAXCONST0:.*]] = "tpu.MaxConst"(%[[SPLIT]]#0) {{{.*}}} : (tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32> loc({{.*}})
// CHECK:           %[[MAXCONST1:.*]] = "tpu.MaxConst"(%[[SPLIT]]#1) {{{.*}}} : (tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32> loc({{.*}})
// CHECK:           %[[JOIN:.*]] = "tpu.Join"(%[[MAXCONST0]], %[[MAXCONST1]]) : (tensor<2x8x32x32xf32>, tensor<2x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc({{.*}})
// CHECK:           "tpu.Yield"(%[[JOIN]]) : (tensor<4x8x32x32xf32>) -> () loc({{.*}})
// CHECK:             }) {{.*}} : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc({{.*}})

// -----

#loc = loc(unknown)
module @MinConst attributes {module.FLOPs = 32768 : i64, module.asymmetric = false, module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.w8a16_linear = false, module.weight_file = "permutebinaryadd_tpu_lowered_bm1686_f32_weight.npz"} {
  func.func @main(%arg0: tensor<4x8x32x32xf32> loc(unknown)) -> tensor<4x8x32x32xf32> {
    %0 = "top.Input"(%arg0) : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc1)
    %1 = "tpu.MinConst"(%0) {const_val = 3.000000e+00 : f64, do_relu = false, multiplier = 1 : si32, relu_limit = -1.000000e+00 : f64, rshift = 0 : si32} : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc2)
    return %1 : tensor<4x8x32x32xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("in_0")
#loc2 = loc("y")

// CHECK-LABEL:     "tpu.CoreParallel"(%0) ({
// CHECK:           %[[SPLIT:.*]]:2 = "tpu.Split"(%0) : (tensor<4x8x32x32xf32>) -> (tensor<2x8x32x32xf32>, tensor<2x8x32x32xf32>) loc({{.*}})
// CHECK:           %[[MINCONST0:.*]] = "tpu.MinConst"(%[[SPLIT]]#0) {{{.*}}} : (tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32> loc({{.*}})
// CHECK:           %[[MINCONST1:.*]] = "tpu.MinConst"(%[[SPLIT]]#1) {{{.*}}} : (tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32> loc({{.*}})
// CHECK:           %[[JOIN:.*]] = "tpu.Join"(%[[MINCONST0]], %[[MINCONST1]]) : (tensor<2x8x32x32xf32>, tensor<2x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc({{.*}})
// CHECK:           "tpu.Yield"(%[[JOIN]]) : (tensor<4x8x32x32xf32>) -> () loc({{.*}})
// CHECK:             }) {{.*}} : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc({{.*}})

// -----

#loc = loc(unknown)
module @CompareConst attributes {module.FLOPs = 32768 : i64, module.asymmetric = false, module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.w8a16_linear = false, module.weight_file = "permutebinaryadd_tpu_lowered_bm1686_f32_weight.npz"} {
  func.func @main(%arg0: tensor<4x8x32x32xf32> loc(unknown)) -> tensor<4x8x32x32xf32> {
    %0 = "top.Input"(%arg0) : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc1)
    %1 = "tpu.CompareConst"(%0) {const_val = 0.000000e+00 : f64, inversed = false, mode = "Greater"} : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc2)
    return %1 : tensor<4x8x32x32xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("in_0")
#loc2 = loc("y")

// CHECK-LABEL:     "tpu.CoreParallel"(%0) ({
// CHECK:           %[[SPLIT:.*]]:2 = "tpu.Split"(%0) : (tensor<4x8x32x32xf32>) -> (tensor<2x8x32x32xf32>, tensor<2x8x32x32xf32>) loc({{.*}})
// CHECK:           %[[COMPARECONST0:.*]] = "tpu.CompareConst"(%[[SPLIT]]#0) {{{.*}} mode = "Greater"{{.*}}} : (tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32> loc({{.*}})
// CHECK:           %[[COMPARECONST1:.*]] = "tpu.CompareConst"(%[[SPLIT]]#1) {{{.*}} mode = "Greater"{{.*}}} : (tensor<2x8x32x32xf32>) -> tensor<2x8x32x32xf32> loc({{.*}})
// CHECK:           %[[JOIN:.*]] = "tpu.Join"(%[[COMPARECONST0]], %[[COMPARECONST1]]) : (tensor<2x8x32x32xf32>, tensor<2x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc({{.*}})
// CHECK:           "tpu.Yield"(%[[JOIN]]) : (tensor<4x8x32x32xf32>) -> () loc({{.*}})
// CHECK:             }) {{.*}} : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc({{.*}})

// -----

#loc = loc(unknown)
module @Activation_0 attributes {module.FLOPs = 15360 : i64, module.asymmetric = false, module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "TORCH", module.state = "TPU_LOWERED", module.weight_file = "activation_0_tpu_lowered_bm1688_f32_weight.npz"} {
  func.func @main(%arg0: tensor<1x3x32x32xf32> loc(unknown)) -> tensor<1x3x32x32xf32> {
    %0 = "top.Input"(%arg0) {do_preprocess = false} : (tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32> loc(#loc1)
    %1 = "tpu.Active"(%0) {mode = #tpu<active_mode GELU>} : (tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32> loc(#loc2)
    return %1 : tensor<1x3x32x32xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("x.1")
#loc2 = loc("5")

// CHECK-LABEL:     "tpu.CoreParallel"(%0) ({
// CHECK:            %[[SPLIT:.*]]:2 = "tpu.Split"(%0) : (tensor<1x3x32x32xf32>) -> (tensor<1x2x32x32xf32>, tensor<1x1x32x32xf32>) loc({{.*}})
// CHECK:            %[[ACTIVE0:.*]] = "tpu.Active"(%[[SPLIT]]#0) {mode = #tpu<active_mode GELU>} : (tensor<1x2x32x32xf32>) -> tensor<1x2x32x32xf32> loc({{.*}})
// CHECK:            %[[ACTIVE1:.*]] = "tpu.Active"(%[[SPLIT]]#1) {mode = #tpu<active_mode GELU>} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32> loc({{.*}})
// CHECK:            %[[JOIN:.*]] = "tpu.Join"(%[[ACTIVE0]], %[[ACTIVE1]]) : (tensor<1x2x32x32xf32>, tensor<1x1x32x32xf32>) -> tensor<1x3x32x32xf32> loc({{.*}})
// CHECK:            "tpu.Yield"(%[[JOIN]]) : (tensor<1x3x32x32xf32>) -> () loc({{.*}})
// CHECK:                 }) {{.*}} : (tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32> loc({{.*}})
