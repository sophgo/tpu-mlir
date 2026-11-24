// RUN: not tpuc-opt %s --sensevoice-unpadded 2>&1 | FileCheck %s
// Reject unrelated dynamic graphs even when the four input shapes match.
// CHECK: unexpected SenseVoice graph or mask
module @unrelated attributes {module.chip = "cv84x6", module.mode = "F16", module.cores = 1 : i64, module.devices = 1 : i64, module.platform = "ONNX", module.state = "TOP_F32", module.top_run_mode = "DYNAMIC", module.weight_file = "unused.npz"} {
  func.func @main(%x: tensor<1x256x560xf32>, %len: tensor<1xsi32>,
                  %language: tensor<1xsi32>, %itn: tensor<1xsi32>) -> tensor<1x256x560xf32> {
    return %x : tensor<1x256x560xf32>
  }
}
