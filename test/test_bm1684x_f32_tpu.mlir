#loc = loc(unknown)
module attributes {module.FLOPs = 2352 : i64, module.asymmetric = true, module.chip = "bm1684x", module.mode = "F32", module.name = "test", module.platform = "ONNX", module.state = "TPU_LOWERED", module.weight_file = "test_tpu_lowered_bm1684x_f32_weight.npz"} {
  func.func @main(%arg0: tensor<1x3x14x14xf32> loc(unknown)) -> tensor<1x3x14x14xf32> {
    %0 = "top.Input"(%arg0) {channel_format = "nchw", keep_aspect_ratio = false, keep_ratio_mode = "letterbox", mean = [0.000000e+00, 0.000000e+00, 0.000000e+00], pad_type = "center", pad_value = 0 : i64, pixel_format = "bgr", resize_dims = [14, 14], scale = [1.000000e+00, 1.000000e+00, 1.000000e+00]} : (tensor<1x3x14x14xf32>) -> tensor<1x3x14x14xf32> loc(#loc1)
    %1 = "top.Weight"() : () -> tensor<3x1x1x1xf32> loc(#loc2)
    %2 = "top.Weight"() : () -> tensor<3xf32> loc(#loc3)
    %3 = "tpu.Conv2D"(%0, %1, %2) {coeff_merged = false, do_relu = false, group = 3 : i64, kernel_shape = [1, 1], kernel_zp = 0 : i64, pads = [0, 0, 0, 0], quant_mode = #tpu<rq_mode MultiplierShift>, relu_limit = -1.000000e+00 : f64, strides = [1, 1], use_3ic_optimize = 0 : i64, with_bias = true} : (tensor<1x3x14x14xf32>, tensor<3x1x1x1xf32>, tensor<3xf32>) -> tensor<1x3x14x14xf32> loc(#loc4)
    %4 = "tpu.AbsAdd"(%3) {b_val = 1.200000e+00 : f64, multiplier = 1 : si32, rshift = 0 : si32} : (tensor<1x3x14x14xf32>) -> tensor<1x3x14x14xf32> loc(#loc5)
    return %4 : tensor<1x3x14x14xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("input0")
#loc2 = loc("input0_bn_merged_to_weight")
#loc3 = loc("input0_bn_merged_to_bias")
#loc4 = loc("input0_bn")
#loc5 = loc("absadd")

