#loc = loc(unknown)
#loc1 = loc("BMTensor0")
module @Yuv2rgb_0 attributes {module.FLOPs = 0 : i64, module.addr_mode = "basic", module.asymmetric = true, module.chip = "bm1684x", module.cores = 1 : i64, module.devices = 1 : i64, module.inputs = ["BMTensor0"], module.mode = "INT8", module.outputs = ["yuv2rgb_d9453665-e20c-47e1-8b68-16ec073c21b4"], module.platform = "TPULANG", module.q_group_size = 0 : i64, module.state = "TPU_ADDRESSED", module.top_run_mode = "STATIC", module.weight_file = "yuv2rgb_0_tpu_addressed_bm1684x_int8_asym_weight.npz"} {
  module @Yuv2rgb_0 attributes {module.coeff_addr = 4294967296 : i64, module.coeff_size = 0 : i64, module.device_id = 0 : i64, module.dynamic_coeff_offset = 0 : i64, module.neuron_addr = 4294967296 : i64, module.neuron_size = 49152 : i64, module.step = 0 : i64} {
    func.func @main(%arg0: tensor<3x90x60xui8> loc(unknown)) -> tensor<3x3x60x60xui8, 4294983680 : i64> {
      %0 = "top.Input"(%arg0) {do_preprocess = false} : (tensor<3x90x60xui8>) -> tensor<3x90x60xui8, 4294967296 : i64> loc(#loc1)
      %1 = call @subfunc_0(%0) : (tensor<3x90x60xui8, 4294967296 : i64>) -> tensor<3x3x60x60xui8, 4294983680 : i64> loc(#loc)
      return %1 : tensor<3x3x60x60xui8, 4294983680 : i64> loc(#loc)
    } loc(#loc)
    func.func @subfunc_0(%arg0: tensor<3x90x60xui8, 4294967296 : i64> loc("BMTensor0")) -> tensor<3x3x60x60xui8, 4294983680 : i64> attributes {id = 0 : i64, mode = #tpu<run_mode TPU_STATIC>, next_index = array<i32: -1>} {
      %0 = "tpu.Yuv2rgbFormula"(%arg0) {dst_format = 5 : ui32, formula_mode = #tpu<yuv2rgb_formula_mode _601_full>, image_format = #tpu<image_out_format UINT8>, round_mode = #tpu<round_mode HalfAwayFromZero>, src_format = 2 : ui32} : (tensor<3x90x60xui8, 4294967296 : i64>) -> tensor<3x3x60x60xui8, 4294983680 : i64> loc(#loc2)
      return %0 : tensor<3x3x60x60xui8, 4294983680 : i64> loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("yuv2rgb_d9453665-e20c-47e1-8b68-16ec073c21b4")

