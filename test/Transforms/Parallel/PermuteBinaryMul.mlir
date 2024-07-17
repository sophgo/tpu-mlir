// RUN: tpuc-opt --core-parallel -split-input-file %s | FileCheck %s

// CHECK-LABEL:     "tpu.CoreParallel"(%1, %0) ({
// CHECK:           %[[SPLIT:.*]]:2 = "tpu.Split"(%1) : (tensor<4x8x32x32xf32>) -> (tensor<2x8x32x32xf32>, tensor<2x8x32x32xf32>) loc({{.*}})
// CHECK:           %[[PERMUTE0:.*]] = "tpu.Permute"(%[[SPLIT]]#0, %0) {order = [0, 3, 2, 1]} : (tensor<2x8x32x32xf32>, none) -> tensor<2x32x32x8xf32> loc({{.*}})
// CHECK:           %[[PERMUTE1:.*]] = "tpu.Permute"(%[[SPLIT]]#1, %0) {order = [0, 3, 2, 1]} : (tensor<2x8x32x32xf32>, none) -> tensor<2x32x32x8xf32> loc({{.*}})
// CHECK:           %[[JOIN:.*]] = "tpu.Join"(%[[PERMUTE0]], %[[PERMUTE1]]) : (tensor<2x32x32x8xf32>, tensor<2x32x32x8xf32>) -> tensor<4x32x32x8xf32> loc({{.*}})
// CHECK:           "tpu.Yield"(%[[JOIN]]) : (tensor<4x32x32x8xf32>) -> () loc({{.*}})
// CHECK:               }) {{.*}} : (tensor<4x8x32x32xf32>, none) -> tensor<4x32x32x8xf32> loc({{.*}})
#loc = loc(unknown)
module @PermuteBinaryAdd attributes {module.FLOPs = 32768 : i64, module.asymmetric = false, module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.w8a16_linear = false, module.weight_file = "permutebinaryadd_tpu_lowered_bm1686_f32_weight.npz"} {
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

// -----

#loc = loc(unknown)
module @PermuteBroadcastAdd attributes {module.FLOPs = 32768 : i64, module.asymmetric = false, module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.w8a16_linear = false, module.weight_file = "permutebinaryadd_tpu_lowered_bm1686_f32_weight.npz"} {
  func.func @main(%arg0: tensor<4x8x32x32xf32> loc(unknown), %arg1: tensor<1x1x32x32xf32> loc(unknown)) -> tensor<4x32x32x8xf32> {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "top.Input"(%arg0) : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc1)
    %2 = "top.Input"(%arg1) : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32> loc(#loc2)
    %3 = "tpu.Permute"(%1, %0) {order = [0, 3, 2, 1]} : (tensor<4x8x32x32xf32>, none) -> tensor<4x32x32x8xf32> loc(#loc3)
    %4 = "tpu.Permute"(%2, %0) {order = [0, 3, 2, 1]} : (tensor<1x1x32x32xf32>, none) -> tensor<1x32x32x1xf32> loc(#loc4)
    %5 = "tpu.Add"(%3, %4) {do_relu = false, relu_limit = -1.000000e+00 : f64} : (tensor<4x32x32x8xf32>, tensor<1x32x32x1xf32>) -> tensor<4x32x32x8xf32> loc(#loc5)
    return %5 : tensor<4x32x32x8xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("in_0")
#loc2 = loc("in_1")
#loc3 = loc("/Transpose_output_0_Transpose")
#loc4 = loc("/Transpose_1_output_0_Transpose")
#loc5 = loc("4_Add")

// CHECK-LABEL:     "tpu.CoreParallel"(%3, %4) ({
// CHECK:           %[[SPLIT:.*]]:2 = "tpu.Split"(%3) : (tensor<4x32x32x8xf32>) -> (tensor<2x32x32x8xf32>, tensor<2x32x32x8xf32>) loc({{.*}})
// CHECK:           %[[ADD0:.*]] = "tpu.Add"(%[[SPLIT]]#0, %4) {{{.*}}} : (tensor<2x32x32x8xf32>, tensor<1x32x32x1xf32>) -> tensor<2x32x32x8xf32> loc({{.*}})
// CHECK:           %[[ADD1:.*]] = "tpu.Add"(%[[SPLIT]]#1, %4) {{{.*}}} : (tensor<2x32x32x8xf32>, tensor<1x32x32x1xf32>) -> tensor<2x32x32x8xf32> loc({{.*}})
// CHECK:           %[[JOIN:.*]] = "tpu.Join"(%[[ADD0]], %[[ADD1]]) : (tensor<2x32x32x8xf32>, tensor<2x32x32x8xf32>) -> tensor<4x32x32x8xf32> loc({{.*}})
// CHECK:           "tpu.Yield"(%[[JOIN]]) : (tensor<4x32x32x8xf32>) -> () loc({{.*}})
// CHECK:               }) {{.*}} : (tensor<4x32x32x8xf32>, tensor<1x32x32x1xf32>) -> tensor<4x32x32x8xf32> loc({{.*}})

// -----

#loc = loc(unknown)
module @PermuteBinaryAdd attributes {module.FLOPs = 32768 : i64, module.asymmetric = false, module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.w8a16_linear = false, module.weight_file = "permutebinaryadd_tpu_lowered_bm1686_f32_weight.npz"} {
  func.func @main(%arg0: tensor<4x8x32x32xf32> loc(unknown), %arg1: tensor<1x32x8xf32> loc(unknown)) -> tensor<4x32x32x8xf32> {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "top.Input"(%arg0) : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc1)
    %2 = "top.Input"(%arg1) : (tensor<1x32x8xf32>) -> tensor<1x32x8xf32> loc(#loc2)
    %3 = "tpu.Permute"(%1, %0) {order = [0, 3, 2, 1]} : (tensor<4x8x32x32xf32>, none) -> tensor<4x32x32x8xf32> loc(#loc3)
    %5 = "tpu.Add"(%3, %2) {do_relu = false, relu_limit = -1.000000e+00 : f64} : (tensor<4x32x32x8xf32>, tensor<1x32x8xf32>) -> tensor<4x32x32x8xf32> loc(#loc5)
    return %5 : tensor<4x32x32x8xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("in_0")
#loc2 = loc("in_1")
#loc3 = loc("/Transpose_output_0_Transpose")
#loc4 = loc("/Transpose_1_output_0_Transpose")
#loc5 = loc("4_Add")

// CHECK-LABEL:     "tpu.CoreParallel"(%3, %2) ({
// CHECK:           %[[SPLIT:.*]]:2 = "tpu.Split"(%3) : (tensor<4x32x32x8xf32>) -> (tensor<2x32x32x8xf32>, tensor<2x32x32x8xf32>) loc({{.*}})
// CHECK:           %[[ADD0:.*]] = "tpu.Add"(%[[SPLIT]]#0, %2) {{{.*}}} : (tensor<2x32x32x8xf32>, tensor<1x32x8xf32>) -> tensor<2x32x32x8xf32> loc({{.*}})
// CHECK:           %[[ADD1:.*]] = "tpu.Add"(%[[SPLIT]]#1, %2) {{{.*}}} : (tensor<2x32x32x8xf32>, tensor<1x32x8xf32>) -> tensor<2x32x32x8xf32> loc({{.*}})
// CHECK:           %[[JOIN:.*]] = "tpu.Join"(%[[ADD0]], %[[ADD1]]) : (tensor<2x32x32x8xf32>, tensor<2x32x32x8xf32>) -> tensor<4x32x32x8xf32> loc({{.*}})
// CHECK:           "tpu.Yield"(%[[JOIN]]) : (tensor<4x32x32x8xf32>) -> () loc({{.*}})
// CHECK:               }) {{.*}} : (tensor<4x32x32x8xf32>, tensor<1x32x8xf32>) -> tensor<4x32x32x8xf32> loc({{.*}})

// -----

#loc = loc(unknown)
module @PermuteBinaryAdd attributes {module.FLOPs = 32768 : i64, module.asymmetric = false, module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.w8a16_linear = false, module.weight_file = "permutebinaryadd_tpu_lowered_bm1686_f32_weight.npz"} {
  func.func @main(%arg0: tensor<4x8x32x32xf32> loc(unknown), %arg1: tensor<32x32x8xf32> loc(unknown)) -> tensor<4x32x32x8xf32> {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "top.Input"(%arg0) : (tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> loc(#loc1)
    %2 = "top.Input"(%arg1) : (tensor<32x32x8xf32>) -> tensor<32x32x8xf32> loc(#loc2)
    %3 = "tpu.Permute"(%1, %0) {order = [0, 3, 2, 1]} : (tensor<4x8x32x32xf32>, none) -> tensor<4x32x32x8xf32> loc(#loc3)
    %5 = "tpu.Add"(%3, %2) {do_relu = false, relu_limit = -1.000000e+00 : f64} : (tensor<4x32x32x8xf32>, tensor<32x32x8xf32>) -> tensor<4x32x32x8xf32> loc(#loc5)
    return %5 : tensor<4x32x32x8xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("in_0")
#loc2 = loc("in_1")
#loc3 = loc("/Transpose_output_0_Transpose")
#loc4 = loc("/Transpose_1_output_0_Transpose")
#loc5 = loc("4_Add")

// CHECK-LABEL:     "tpu.CoreParallel"(%3, %2) ({
// CHECK:           %[[SPLIT:.*]]:2 = "tpu.Split"(%3) : (tensor<4x32x32x8xf32>) -> (tensor<2x32x32x8xf32>, tensor<2x32x32x8xf32>) loc({{.*}})
// CHECK:           %[[ADD0:.*]] = "tpu.Add"(%[[SPLIT]]#0, %2) {{{.*}}} : (tensor<2x32x32x8xf32>, tensor<32x32x8xf32>) -> tensor<2x32x32x8xf32> loc({{.*}})
// CHECK:           %[[ADD1:.*]] = "tpu.Add"(%[[SPLIT]]#1, %2) {{{.*}}} : (tensor<2x32x32x8xf32>, tensor<32x32x8xf32>) -> tensor<2x32x32x8xf32> loc({{.*}})
// CHECK:           %[[JOIN:.*]] = "tpu.Join"(%[[ADD0]], %[[ADD1]]) : (tensor<2x32x32x8xf32>, tensor<2x32x32x8xf32>) -> tensor<4x32x32x8xf32> loc({{.*}})
// CHECK:           "tpu.Yield"(%[[JOIN]]) : (tensor<4x32x32x8xf32>) -> () loc({{.*}})
// CHECK:               }) {{.*}} : (tensor<4x32x32x8xf32>, tensor<32x32x8xf32>) -> tensor<4x32x32x8xf32> loc({{.*}})


// -----

// case 1: [5, 6] * [6, 7] = [5, 7] => batch = 1, M = 5, K = 6, N = 7
// CHECK-LABEL:     module @MatMul1 {{.*}}
// CHECK:           %[[SPLIT:.*]]:2 = "tpu.Split"(%arg0) : (tensor<5x6xf32>) -> (tensor<3x6xf32>, tensor<2x6xf32>) loc({{.*}})
// CHECK:           %[[MAMUL1:.*]] = "tpu.MatMul"(%[[SPLIT]]#0, %arg1, %0, %0, %0) {{{.*}}} : (tensor<3x6xf32>, tensor<6x7xf32>, none, none, none) -> tensor<3x7xf32> loc({{.*}})
// CHECK:           %[[MAMUL2:.*]] = "tpu.MatMul"(%[[SPLIT]]#1, %arg1, %0, %0, %0) {{{.*}}} : (tensor<2x6xf32>, tensor<6x7xf32>, none, none, none) -> tensor<2x7xf32> loc({{.*}})
module @MatMul1 attributes {module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.weight_file = ""} {
  func.func @main(%arg0: tensor<5x6xf32> loc("a"), %arg1: tensor<6x7xf32> loc("b")) -> tensor<5x7xf32> {
    %0 = "top.None"() : () -> none loc(unknown)
    %1 = "tpu.MatMul"(%arg0, %arg1, %0, %0, %0) : (tensor<5x6xf32>, tensor<6x7xf32>, none, none, none) -> tensor<5x7xf32> loc("c")
    return %1 : tensor<5x7xf32> loc("d")
  }
}


// -----

// case 2.1: [1, 512, 7, 7] * [25088, 4096] = [1, 4096] => batch = 1, M = 1, K = 25088, N = 4096
// CHECK-LABEL:     module @MatMul2.1 {{.*}}
// CHECK:           %[[MAMUL1:.*]] = "tpu.MatMul"(%arg0, %arg1, %0, %0, %0) {{{.*}}} : (tensor<1x512x7x7xf32>, tensor<25088x4096xf32>, none, none, none) -> tensor<1x4096xf32> loc({{.*}})
module @MatMul2.1 attributes {module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.weight_file = ""} {
  func.func @main(%arg0: tensor<1x512x7x7xf32> loc("a"), %arg1: tensor<25088x4096xf32> loc("b")) -> tensor<1x4096xf32> {
    %0 = "top.None"() : () -> none loc(unknown)
    %1 = "tpu.MatMul"(%arg0, %arg1, %0, %0, %0) : (tensor<1x512x7x7xf32>, tensor<25088x4096xf32>, none, none, none) -> tensor<1x4096xf32> loc("c")
    return %1 : tensor<1x4096xf32> loc("d")
  }
}


// -----

// case 2.2: [3, 512, 7, 7] * [25088, 4096] = [3, 4096] => batch = 1, M = 1, K = 25088, N = 4096
// CHECK-LABEL:     module @MatMul2.2 {{.*}}
// CHECK:           %[[SPLIT:.*]]:2 = "tpu.Split"(%arg0) : (tensor<3x512x7x7xf32>) -> (tensor<2x512x7x7xf32>, tensor<1x512x7x7xf32>) loc({{.*}})
// CHECK:           %[[MAMUL1:.*]] = "tpu.MatMul"(%[[SPLIT]]#0, %arg1, %0, %0, %0) {{{.*}}} : (tensor<2x512x7x7xf32>, tensor<25088x4096xf32>, none, none, none) -> tensor<2x4096xf32> loc({{.*}})
// CHECK:           %[[MAMUL2:.*]] = "tpu.MatMul"(%[[SPLIT]]#1, %arg1, %0, %0, %0) {{{.*}}} : (tensor<1x512x7x7xf32>, tensor<25088x4096xf32>, none, none, none) -> tensor<1x4096xf32> loc({{.*}})
module @MatMul2.2 attributes {module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.weight_file = ""} {
  func.func @main(%arg0: tensor<3x512x7x7xf32> loc("a"), %arg1: tensor<25088x4096xf32> loc("b")) -> tensor<3x4096xf32> {
    %0 = "top.None"() : () -> none loc(unknown)
    %1 = "tpu.MatMul"(%arg0, %arg1, %0, %0, %0) : (tensor<3x512x7x7xf32>, tensor<25088x4096xf32>, none, none, none) -> tensor<3x4096xf32> loc("c")
    return %1 : tensor<3x4096xf32> loc("d")
  }
}

// -----

// case 2.3: [3, 512, 7, 7] * [25088, 4096] = [3, 4096] => batch = 1, M = 1, K = 25088, N = 4096
// CHECK-LABEL:     module @MatMul2.3 {{.*}}
// CHECK:           %[[MAMUL2:.*]] = "tpu.MatMul"(%arg0, %arg1, %0, %0, %0) {{{.*}}} : (tensor<3x512x7x7xf32>, tensor<25088x4096xf32>, none, none, none) -> tensor<3x4096xf32> loc({{.*}})
module @MatMul2.3 attributes {module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.weight_file = ""} {
  func.func @main(%arg0: tensor<3x512x7x7xf32> loc("a"), %arg1: tensor<25088x4096xf32> loc("b")) -> tensor<3x4096xf32> {
    %0 = "top.None"() : () -> none loc(unknown)
    %1 = "tpu.MatMul"(%arg0, %arg1, %0, %0, %0) {left_transpose=true}: (tensor<3x512x7x7xf32>, tensor<25088x4096xf32>, none, none, none) -> tensor<3x4096xf32> loc("c")
    return %1 : tensor<3x4096xf32> loc("d")
  }
}



// -----

// case 3.1: [1, 4, 5, 6] * [1, 4, 6, 7] = [1, 4, 5, 7] => batch = 4, M = 5, K = 6, N = 7
// CHECK-LABEL:     module @MatMul3.1 {{.*}}
// CHECK:           %[[SPLIT1:.*]]:2 = "tpu.Split"(%arg0) : (tensor<1x4x5x6xf32>) -> (tensor<1x2x5x6xf32>, tensor<1x2x5x6xf32>) loc({{.*}})
// CHECK:           %[[SPLIT2:.*]]:2 = "tpu.Split"(%arg1) : (tensor<1x4x6x7xf32>) -> (tensor<1x2x6x7xf32>, tensor<1x2x6x7xf32>) loc({{.*}})
// CHECK:           %[[MAMUL1:.*]] = "tpu.MatMul"(%[[SPLIT1]]#0, %[[SPLIT2]]#0, %0, %0, %0) {{{.*}}} : (tensor<1x2x5x6xf32>, tensor<1x2x6x7xf32>, none, none, none) -> tensor<1x2x6x7xf32> loc({{.*}})
// CHECK:           %[[MAMUL2:.*]] = "tpu.MatMul"(%[[SPLIT1]]#1, %[[SPLIT2]]#1, %0, %0, %0) {{{.*}}} : (tensor<1x2x5x6xf32>, tensor<1x2x6x7xf32>, none, none, none) -> tensor<1x2x6x7xf32> loc({{.*}})
module @MatMul3.1 attributes {module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.weight_file = ""} {
  func.func @main(%arg0: tensor<1x4x5x6xf32> loc("a"), %arg1: tensor<1x4x6x7xf32> loc("b")) -> tensor<1x4x6x7xf32> {
    %0 = "top.None"() : () -> none loc(unknown)
    %1 = "tpu.MatMul"(%arg0, %arg1, %0, %0, %0) : (tensor<1x4x5x6xf32>, tensor<1x4x6x7xf32>, none, none, none) -> tensor<1x4x6x7xf32> loc("c")
    return %1 : tensor<1x4x6x7xf32> loc("d")
  }
}

// -----

// case 3.2: [3, 5, 4, 6] * [3, 6, 4, 7] = [3, 5, 4, 7] => batch = 12, M = 5, K = 6, N = 7 (hdim_is_batch=true)
// CHECK-LABEL:     module @MatMul3.2 {{.*}}
// CHECK:           %[[SPLIT1:.*]]:2 = "tpu.Split"(%arg0) : (tensor<3x5x4x6xf32>) -> (tensor<2x5x4x6xf32>, tensor<1x5x4x6xf32>) loc({{.*}})
// CHECK:           %[[SPLIT2:.*]]:2 = "tpu.Split"(%arg1) : (tensor<3x6x4x7xf32>) -> (tensor<2x6x4x7xf32>, tensor<1x6x4x7xf32>) loc({{.*}})
// CHECK:           %[[MAMUL1:.*]] = "tpu.MatMul"(%[[SPLIT1]]#0, %[[SPLIT2]]#0, %0, %0, %0) {{{.*}}} : (tensor<2x5x4x6xf32>, tensor<2x6x4x7xf32>, none, none, none) -> tensor<2x6x4x7xf32> loc({{.*}})
// CHECK:           %[[MAMUL2:.*]] = "tpu.MatMul"(%[[SPLIT1]]#1, %[[SPLIT2]]#1, %0, %0, %0) {{{.*}}} : (tensor<1x5x4x6xf32>, tensor<1x6x4x7xf32>, none, none, none) -> tensor<1x6x4x7xf32> loc({{.*}})
module @MatMul3.2 attributes {module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.weight_file = ""} {
  func.func @main(%arg0: tensor<3x5x4x6xf32> loc("a"), %arg1: tensor<3x6x4x7xf32> loc("b")) -> tensor<3x6x4x7xf32> {
    %0 = "top.None"() : () -> none loc(unknown)
    %1 = "tpu.MatMul"(%arg0, %arg1, %0, %0, %0) {hdim_is_batch=true} : (tensor<3x5x4x6xf32>, tensor<3x6x4x7xf32>, none, none, none) -> tensor<3x6x4x7xf32> loc("c")
    return %1 : tensor<3x6x4x7xf32> loc("d")
  }
}


// -----

// case 4: [4, 5, 6] * [6,7] = [4, 5, 7] => batch =1, M = 20, K = 6, N = 7
// CHECK-LABEL:     module @MatMul4 {{.*}}
// CHECK:           %[[SPLIT1:.*]]:2 = "tpu.Split"(%arg0) : (tensor<4x5x6xf32>) -> (tensor<2x5x6xf32>, tensor<2x5x6xf32>) loc({{.*}})
// CHECK:           %[[MAMUL1:.*]] = "tpu.MatMul"(%[[SPLIT1]]#0, %arg1, %0, %0, %0) {{{.*}}} : (tensor<2x5x6xf32>, tensor<6x7xf32>, none, none, none) -> tensor<2x6x7xf32> loc({{.*}})
// CHECK:           %[[MAMUL2:.*]] = "tpu.MatMul"(%[[SPLIT1]]#1, %arg1, %0, %0, %0) {{{.*}}} : (tensor<2x5x6xf32>, tensor<6x7xf32>, none, none, none) -> tensor<2x6x7xf32> loc({{.*}})
module @MatMul4 attributes {module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.weight_file = ""} {
  func.func @main(%arg0: tensor<4x5x6xf32> loc("a"), %arg1: tensor<6x7xf32> loc("b")) -> tensor<4x6x7xf32> {
    %0 = "top.None"() : () -> none loc(unknown)
    %1 = "tpu.MatMul"(%arg0, %arg1, %0, %0, %0) : (tensor<4x5x6xf32>, tensor<6x7xf32>, none, none, none) -> tensor<4x6x7xf32> loc("c")
    return %1 : tensor<4x6x7xf32> loc("d")
  }
}

// -----

// case 5: [4, 5, 6] * [6] = [4, 5] => batch =1, M = 20, K = 6, N = 1
// CHECK-LABEL:     module @MatMul5 {{.*}}
// CHECK:           %[[SPLIT1:.*]]:2 = "tpu.Split"(%arg0) : (tensor<4x5x6xf32>) -> (tensor<2x5x6xf32>, tensor<2x5x6xf32>) loc({{.*}})
// CHECK:           %[[MAMUL1:.*]] = "tpu.MatMul"(%[[SPLIT1]]#0, %arg1, %0, %0, %0) {{{.*}}} : (tensor<2x5x6xf32>, tensor<6xf32>, none, none, none) -> tensor<2x6xf32> loc({{.*}})
// CHECK:           %[[MAMUL2:.*]] = "tpu.MatMul"(%[[SPLIT1]]#1, %arg1, %0, %0, %0) {{{.*}}} : (tensor<2x5x6xf32>, tensor<6xf32>, none, none, none) -> tensor<2x6xf32> loc({{.*}})
module @MatMul5 attributes {module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.weight_file = ""} {
  func.func @main(%arg0: tensor<4x5x6xf32> loc("a"), %arg1: tensor<6xf32> loc("b")) -> tensor<4x6xf32> {
    %0 = "top.None"() : () -> none loc(unknown)
    %1 = "tpu.MatMul"(%arg0, %arg1, %0, %0, %0) : (tensor<4x5x6xf32>, tensor<6xf32>, none, none, none) -> tensor<4x6xf32> loc("c")
    return %1 : tensor<4x6xf32> loc("d")
  }
}

// -----

// case 6: [4096] * [4096, 12884] = [1,12884] => batch =1, M = 1, K = 4096, N = 12884
// CHECK-LABEL:     module @MatMul6 {{.*}}
// CHECK:           %[[MAMUL1:.*]] = "tpu.MatMul"(%arg0, %arg1, %0, %0, %0) {{{.*}}} : (tensor<4096xf32>, tensor<4096x12884xf32>, none, none, none) -> tensor<1x12884xf32> loc({{.*}})
module @MatMul6 attributes {module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.weight_file = ""} {
  func.func @main(%arg0: tensor<4096xf32> loc("a"), %arg1: tensor<4096x12884xf32> loc("b")) -> tensor<1x12884xf32> {
    %0 = "top.None"() : () -> none loc(unknown)
    %1 = "tpu.MatMul"(%arg0, %arg1, %0, %0, %0) : (tensor<4096xf32>, tensor<4096x12884xf32>, none, none, none) -> tensor<1x12884xf32> loc("c")
    return %1 : tensor<1x12884xf32> loc("d")
  }
}
