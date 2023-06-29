#loc = loc(unknown)
#loc1 = loc("input0")
module attributes {module.FLOPs = 2352 : i64, module.asymmetric = true, module.chip = "bm1684x", module.coeff_addr = 4294967296 : i64, module.coeff_size = 8192 : i64, module.mode = "F32", module.name = "test", module.neuron_addr = 4294975488 : i64, module.neuron_size = 12288 : i64, module.platform = "ONNX", module.state = "TPU_ADDRESSED", module.weight_file = "test_tpu_addressed_bm1684x_f32_weight.npz"} {
  func.func @main(%arg0: tensor<1x3x14x14xf32> loc(unknown)) -> tensor<1x3x14x14xf32, 4294983680 : i64> {
    %0 = "top.Input"(%arg0) {channel_format = "nchw", keep_aspect_ratio = false, keep_ratio_mode = "letterbox", mean = [0.000000e+00, 0.000000e+00, 0.000000e+00], pad_type = "center", pad_value = 0 : i64, pixel_format = "bgr", resize_dims = [14, 14], scale = [1.000000e+00, 1.000000e+00, 1.000000e+00]} : (tensor<1x3x14x14xf32>) -> tensor<1x3x14x14xf32, 4294975488 : i64> loc(#loc1)
    %1 = call @subfunc_0(%0) : (tensor<1x3x14x14xf32, 4294975488 : i64>) -> tensor<1x3x14x14xf32, 4294983680 : i64> loc(#loc)
    return %1 : tensor<1x3x14x14xf32, 4294983680 : i64> loc(#loc)
  } loc(#loc)
  func.func @subfunc_0(%arg0: tensor<1x3x14x14xf32, 4294975488 : i64> loc("input0")) -> tensor<1x3x14x14xf32, 4294983680 : i64> attributes {id = 0 : i64, mode = #tpu<run_mode TPU_STATIC>} {
    %0 = "top.Weight"() : () -> tensor<1x3x1x1xf32, 4294967296 : i64> loc(#loc2)
    %1 = "top.Weight"() : () -> tensor<1x3x1x1xf32, 4294971392 : i64> loc(#loc3)
    %2 = "tpu.Conv2D"(%arg0, %0, %1) {coeff_merged = false, do_relu = false, group = 3 : i64, kernel_shape = [1, 1], kernel_zp = 0 : i64, pads = [0, 0, 0, 0], quant_mode = #tpu<rq_mode MultiplierShift>, relu_limit = -1.000000e+00 : f64, strides = [1, 1], use_3ic_optimize = 0 : i64, with_bias = true} : (tensor<1x3x14x14xf32, 4294975488 : i64>, tensor<1x3x1x1xf32, 4294967296 : i64>, tensor<1x3x1x1xf32, 4294971392 : i64>) -> tensor<1x3x14x14xf32, 4294979584 : i64> loc(#loc4)
    %3 = "tpu.AbsAdd"(%2) {b_val = 1.200000e+00 : f64, multiplier = 1 : si32, rshift = 0 : si32} : (tensor<1x3x14x14xf32, 4294979584 : i64>) -> tensor<1x3x14x14xf32, 4294983680 : i64> loc(#loc5)
    return %3 : tensor<1x3x14x14xf32, 4294983680 : i64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("input0_bn_merged_to_weight")
#loc3 = loc("input0_bn_merged_to_bias")
#loc4 = loc("input0_bn")
#loc5 = loc("absadd")

