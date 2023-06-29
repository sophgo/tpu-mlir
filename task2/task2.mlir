#loc = loc(unknown)
module attributes {module.FLOPs = 25088 : i64, module.chip = "ALL", module.name = "task2", module.platform = "ONNX", module.state = "TOP_F32", module.weight_file = "task2_top_f32_all_weight.npz"} {
  func.func @main(%arg0: tensor<1x16x28x28xf32> loc(unknown)) -> tensor<1x16x28x28xf32> {
    %0 = "top.Input"(%arg0) : (tensor<1x16x28x28xf32>) -> tensor<1x16x28x28xf32> loc(#loc1)
    %1 = "top.AbsAdd"(%0) {b_val = 0.54881352186203003 : f64} : (tensor<1x16x28x28xf32>) -> tensor<1x16x28x28xf32> loc(#loc2)
    return %1 : tensor<1x16x28x28xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("input")
#loc2 = loc("output_Add")

