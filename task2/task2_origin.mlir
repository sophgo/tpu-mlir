#loc = loc(unknown)
module attributes {module.chip = "ALL", module.name = "task2", module.platform = "ONNX", module.state = "TOP_F32", module.weight_file = "task2_top_f32_all_origin_weight.npz"} {
  func.func @main(%arg0: tensor<1x16x28x28xf32> loc(unknown)) -> tensor<1x16x28x28xf32> {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "top.Input"(%arg0) : (tensor<1x16x28x28xf32>) -> tensor<1x16x28x28xf32> loc(#loc1)
    %2 = "top.Abs"(%1) : (tensor<1x16x28x28xf32>) -> tensor<1x16x28x28xf32> loc(#loc2)
    %3 = "top.AddConst"(%2) {const_val = 0.54881352186203003 : f64, do_relu = false} : (tensor<1x16x28x28xf32>) -> tensor<1x16x28x28xf32> loc(#loc3)
    return %3 : tensor<1x16x28x28xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("input")
#loc2 = loc("e_Abs")
#loc3 = loc("output_Add")
