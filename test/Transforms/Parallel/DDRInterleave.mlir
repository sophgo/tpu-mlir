// RUN: tpuc-opt --ddr-interleave -split-input-file %s | FileCheck %s

// CHECK-LABEL:   module @MatMul2.0 {{.*}}
// CHECK:    %[[Weight:.*]] = "top.Weight"() : () -> tensor<25088x4096xf32, #tpu.ddr_interleave<{map = (d0, d1) -> (8, d0, d1 ceildiv 8), interleave_dim = [0], address = 0}>> loc(#loc4)
// CHECK:    %[[MatMul:.*]] = "tpu.MatMul"(%arg0, %[[Weight]], %0, %0, %0) {{.*}} : (tensor<256x25088xf32>, tensor<25088x4096xf32, #tpu.ddr_interleave<{map = (d0, d1) -> (8, d0, d1 ceildiv 8), interleave_dim = [0], address = 0}>>, none, none, none) -> tensor<256x4096xf32, #tpu.ddr_interleave<{map = (d0, d1) -> (8, d0, d1 ceildiv 8), interleave_dim = [0], address = 0}>>
// CHECK:    %[[Out:.*]] = "tpu.MatMul"(%[[MatMul]], %[[MatMul]], %0, %0, %0)

module @MatMul2.0 attributes {module.chip = "bm1688", module.cores = 2 : i64, module.mode = "F32", module.platform = "ONNX", module.state = "TPU_LOWERED", module.weight_file = ""} {
  func.func @main(%arg0: tensor<256x25088xf32> loc("a")) -> tensor<256x256xf32> {
    %0 = "top.None"() : () -> none loc(unknown)
    %1 = "top.Weight"() : () -> tensor<25088x4096xf32> loc("b")
    %2 = "tpu.MatMul"(%arg0, %1, %0, %0, %0) : (tensor<256x25088xf32>, tensor<25088x4096xf32>, none, none, none) -> tensor<256x4096xf32> loc("c")
    %3 = "tpu.MatMul"(%2, %2, %0, %0, %0) {right_transpose=true} : (tensor<256x4096xf32>, tensor<256x4096xf32>, none, none, none) -> tensor<256x256xf32> loc("d")
    return %3 : tensor<256x256xf32> loc("e")
  }
}
