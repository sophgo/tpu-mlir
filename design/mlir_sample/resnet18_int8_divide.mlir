module attributes {module.chip = "BM1684", module.name = "resnet18", module.state = "TPU_DIVIDED", module.weight_file = "resnet18_tpu_reordered_bm1684_weight.npz"} {
  func @main(%arg0: tensor<4x3x224x224xf32>) -> tensor<4x1000xf32> {
    %0 = "top.Input"(%arg0) {name = "input"} : (tensor<4x3x224x224xf32>) -> tensor<4x3x224x224x!quant.calibrated<f32<-2.6412898999999999:2.6412898999999999>>>
    %1 = call @subfunc_1(%0) : (tensor<4x3x224x224x!quant.calibrated<f32<-2.6412898999999999:2.6412898999999999>>>) -> tensor<4x64x112x112x!quant.uniform<i8:f32, 0.027347358267716535>>
    %2 = call @subfunc_2(%1) : (tensor<4x64x112x112x!quant.uniform<i8:f32, 0.027347358267716535>>) -> tensor<4x64x56x56x!quant.uniform<i8:f32, 0.027347358267716535>>
    %3 = call @subfunc_3(%2) : (tensor<4x64x56x56x!quant.uniform<i8:f32, 0.027347358267716535>>) -> tensor<4x1000xf32>
    return %3 : tensor<4x1000xf32>
  }
  func @subfunc_1(%arg0: tensor<4x3x224x224x!quant.calibrated<f32<-2.6412898999999999:2.6412898999999999>>>) -> tensor<4x64x112x112x!quant.uniform<i8:f32, 0.027347358267716535>> attributes {id = 1 : i64, mode = "TPU"} {
    %0 = "tpu.Cast"(%arg0) {name = "input_to_int8"} : (tensor<4x3x224x224x!quant.calibrated<f32<-2.6412898999999999:2.6412898999999999>>>) -> tensor<4x3x224x224x!quant.uniform<i8:f32, 0.020797558267716534>>
    %1 = "top.Weight"() {name = "125_Relu_filter_int8_reorderd"} : () -> tensor<1x64x196x1xi8>
    %2 = "top.Weight"() {name = "125_Relu_bias_int16"} : () -> tensor<64xi16>
    %3 = "tpu.Conv"(%0, %1, %2) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [7, 7], name = "125_Relu", pads = [3, 3, 3, 3], rshift = 8 : i64, strides = [2, 2]} : (tensor<4x3x224x224x!quant.uniform<i8:f32, 0.020797558267716534>>, tensor<1x64x196x1xi8>, tensor<64xi16>) -> tensor<4x64x112x112x!quant.uniform<i8:f32, 0.027347358267716535>>
    return %3 : tensor<4x64x112x112x!quant.uniform<i8:f32, 0.027347358267716535>>
  }
  func @subfunc_2(%arg0: tensor<4x64x112x112x!quant.uniform<i8:f32, 0.027347358267716535>>) -> tensor<4x64x56x56x!quant.uniform<i8:f32, 0.027347358267716535>> attributes {id = 2 : i64, mode = "CPU"} {
    %0 = "tpu.MaxPool"(%arg0) {do_relu = false, kernel_shape = [3, 3], name = "126_MaxPool", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<4x64x112x112x!quant.uniform<i8:f32, 0.027347358267716535>>) -> tensor<4x64x56x56x!quant.uniform<i8:f32, 0.027347358267716535>>
    return %0 : tensor<4x64x56x56x!quant.uniform<i8:f32, 0.027347358267716535>>
  }
  func @subfunc_3(%arg0: tensor<4x64x56x56x!quant.uniform<i8:f32, 0.027347358267716535>>) -> tensor<4x1000xf32> attributes {id = 3 : i64, mode = "TPU"} {
    %0 = "top.Weight"() {name = "129_Relu_filter_int8_reorderd"} : () -> tensor<1x64x576x1xi8>
    %1 = "top.Weight"() {name = "129_Relu_bias_int16"} : () -> tensor<64xi16>
    %2 = "tpu.Conv"(%arg0, %0, %1) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "129_Relu", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [1, 1]} : (tensor<4x64x56x56x!quant.uniform<i8:f32, 0.027347358267716535>>, tensor<1x64x576x1xi8>, tensor<64xi16>) -> tensor<4x64x56x56x!quant.uniform<i8:f32, 0.016837784251968503>>
    %3 = "top.Weight"() {name = "130_Conv_filter_int8_reorderd"} : () -> tensor<1x64x576x1xi8>
    %4 = "top.Weight"() {name = "130_Conv_bias_int16"} : () -> tensor<64xi16>
    %5 = "tpu.Conv"(%2, %3, %4) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "130_Conv", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [1, 1]} : (tensor<4x64x56x56x!quant.uniform<i8:f32, 0.016837784251968503>>, tensor<1x64x576x1xi8>, tensor<64xi16>) -> tensor<4x64x56x56x!quant.uniform<i8:f32, 0.02463193937007874>>
    %6 = "tpu.Add"(%5, %arg0) {coeff = [1.020000e+02, 1.130000e+02], do_relu = true, name = "133_Relu", rshifts = [7, 7]} : (tensor<4x64x56x56x!quant.uniform<i8:f32, 0.02463193937007874>>, tensor<4x64x56x56x!quant.uniform<i8:f32, 0.027347358267716535>>) -> tensor<4x64x56x56x!quant.uniform<i8:f32, 0.030926265354330709>>
    %7 = "top.Weight"() {name = "136_Relu_filter_int8_reorderd"} : () -> tensor<1x64x576x1xi8>
    %8 = "top.Weight"() {name = "136_Relu_bias_int16"} : () -> tensor<64xi16>
    %9 = "tpu.Conv"(%6, %7, %8) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "136_Relu", pads = [1, 1, 1, 1], rshift = 8 : i64, strides = [1, 1]} : (tensor<4x64x56x56x!quant.uniform<i8:f32, 0.030926265354330709>>, tensor<1x64x576x1xi8>, tensor<64xi16>) -> tensor<4x64x56x56x!quant.uniform<i8:f32, 0.019197859842519684>>
    %10 = "top.Weight"() {name = "137_Conv_filter_int8_reorderd"} : () -> tensor<1x64x576x1xi8>
    %11 = "top.Weight"() {name = "137_Conv_bias_int16"} : () -> tensor<64xi16>
    %12 = "tpu.Conv"(%9, %10, %11) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "137_Conv", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [1, 1]} : (tensor<4x64x56x56x!quant.uniform<i8:f32, 0.019197859842519684>>, tensor<1x64x576x1xi8>, tensor<64xi16>) -> tensor<4x64x56x56x!quant.uniform<i8:f32, 0.027846443307086612>>
    %13 = "tpu.Add"(%12, %6) {coeff = [6.400000e+01, 7.100000e+01], do_relu = true, name = "140_Relu", rshifts = [7, 7]} : (tensor<4x64x56x56x!quant.uniform<i8:f32, 0.027846443307086612>>, tensor<4x64x56x56x!quant.uniform<i8:f32, 0.030926265354330709>>) -> tensor<4x64x56x56x!quant.uniform<i8:f32, 0.055844555118110234>>
    %14 = "top.Weight"() {name = "143_Relu_filter_int8_reorderd"} : () -> tensor<1x128x576x1xi8>
    %15 = "top.Weight"() {name = "143_Relu_bias_int16"} : () -> tensor<128xi16>
    %16 = "tpu.Conv"(%13, %14, %15) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "143_Relu", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [2, 2]} : (tensor<4x64x56x56x!quant.uniform<i8:f32, 0.055844555118110234>>, tensor<1x128x576x1xi8>, tensor<128xi16>) -> tensor<4x128x28x28x!quant.uniform<i8:f32, 0.01627884724409449>>
    %17 = "top.Weight"() {name = "144_Conv_filter_int8_reorderd"} : () -> tensor<1x128x1152x1xi8>
    %18 = "top.Weight"() {name = "144_Conv_bias_int16"} : () -> tensor<128xi16>
    %19 = "tpu.Conv"(%16, %17, %18) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "144_Conv", pads = [1, 1, 1, 1], rshift = 9 : i64, strides = [1, 1]} : (tensor<4x128x28x28x!quant.uniform<i8:f32, 0.01627884724409449>>, tensor<1x128x1152x1xi8>, tensor<128xi16>) -> tensor<4x128x28x28x!quant.uniform<i8:f32, 0.047900481102362204>>
    %20 = "top.Weight"() {name = "146_Conv_filter_int8_reorderd"} : () -> tensor<1x128x64x1xi8>
    %21 = "top.Weight"() {name = "146_Conv_bias_int16"} : () -> tensor<128xi16>
    %22 = "tpu.Conv"(%13, %20, %21) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "146_Conv", pads = [0, 0, 0, 0], rshift = 6 : i64, strides = [2, 2]} : (tensor<4x64x56x56x!quant.uniform<i8:f32, 0.055844555118110234>>, tensor<1x128x64x1xi8>, tensor<128xi16>) -> tensor<4x128x28x28x!quant.uniform<i8:f32, 0.020879310236220472>>
    %23 = "tpu.Add"(%19, %22) {coeff = [1.170000e+02, 1.020000e+02], do_relu = true, name = "149_Relu", rshifts = [7, 8]} : (tensor<4x128x28x28x!quant.uniform<i8:f32, 0.047900481102362204>>, tensor<4x128x28x28x!quant.uniform<i8:f32, 0.020879310236220472>>) -> tensor<4x128x28x28x!quant.uniform<i8:f32, 0.05242758818897638>>
    %24 = "top.Weight"() {name = "152_Relu_filter_int8_reorderd"} : () -> tensor<1x128x1152x1xi8>
    %25 = "top.Weight"() {name = "152_Relu_bias_int16"} : () -> tensor<128xi16>
    %26 = "tpu.Conv"(%23, %24, %25) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "152_Relu", pads = [1, 1, 1, 1], rshift = 6 : i64, strides = [1, 1]} : (tensor<4x128x28x28x!quant.uniform<i8:f32, 0.05242758818897638>>, tensor<1x128x1152x1xi8>, tensor<128xi16>) -> tensor<4x128x28x28x!quant.uniform<i8:f32, 0.012759493700787401>>
    %27 = "top.Weight"() {name = "153_Conv_filter_int8_reorderd"} : () -> tensor<1x128x1152x1xi8>
    %28 = "top.Weight"() {name = "153_Conv_bias_int16"} : () -> tensor<128xi16>
    %29 = "tpu.Conv"(%26, %27, %28) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "153_Conv", pads = [1, 1, 1, 1], rshift = 8 : i64, strides = [1, 1]} : (tensor<4x128x28x28x!quant.uniform<i8:f32, 0.012759493700787401>>, tensor<1x128x1152x1xi8>, tensor<128xi16>) -> tensor<4x128x28x28x!quant.uniform<i8:f32, 0.029383494488188975>>
    %30 = "tpu.Add"(%29, %23) {coeff = [6.700000e+01, 1.200000e+02], do_relu = true, name = "156_Relu", rshifts = [7, 7]} : (tensor<4x128x28x28x!quant.uniform<i8:f32, 0.029383494488188975>>, tensor<4x128x28x28x!quant.uniform<i8:f32, 0.05242758818897638>>) -> tensor<4x128x28x28x!quant.uniform<i8:f32, 0.055967818897637793>>
    %31 = "top.Weight"() {name = "159_Relu_filter_int8_reorderd"} : () -> tensor<1x256x1152x1xi8>
    %32 = "top.Weight"() {name = "159_Relu_bias_int16"} : () -> tensor<256xi16>
    %33 = "tpu.Conv"(%30, %31, %32) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "159_Relu", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [2, 2]} : (tensor<4x128x28x28x!quant.uniform<i8:f32, 0.055967818897637793>>, tensor<1x256x1152x1xi8>, tensor<256xi16>) -> tensor<4x256x14x14x!quant.uniform<i8:f32, 0.026231895275590552>>
    %34 = "top.Weight"() {name = "160_Conv_filter_int8_reorderd"} : () -> tensor<1x256x2304x1xi8>
    %35 = "top.Weight"() {name = "160_Conv_bias_int16"} : () -> tensor<256xi16>
    %36 = "tpu.Conv"(%33, %34, %35) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "160_Conv", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [1, 1]} : (tensor<4x256x14x14x!quant.uniform<i8:f32, 0.026231895275590552>>, tensor<1x256x2304x1xi8>, tensor<256xi16>) -> tensor<4x256x14x14x!quant.uniform<i8:f32, 0.028183774015748033>>
    %37 = "top.Weight"() {name = "162_Conv_filter_int8_reorderd"} : () -> tensor<1x256x128x1xi8>
    %38 = "top.Weight"() {name = "162_Conv_bias_int16"} : () -> tensor<256xi16>
    %39 = "tpu.Conv"(%30, %37, %38) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "162_Conv", pads = [0, 0, 0, 0], rshift = 6 : i64, strides = [2, 2]} : (tensor<4x128x28x28x!quant.uniform<i8:f32, 0.055967818897637793>>, tensor<1x256x128x1xi8>, tensor<256xi16>) -> tensor<4x256x14x14x!quant.uniform<i8:f32, 0.013166514960629922>>
    %40 = "tpu.Add"(%36, %39) {coeff = [1.190000e+02, 1.110000e+02], do_relu = true, name = "165_Relu", rshifts = [7, 8]} : (tensor<4x256x14x14x!quant.uniform<i8:f32, 0.028183774015748033>>, tensor<4x256x14x14x!quant.uniform<i8:f32, 0.013166514960629922>>) -> tensor<4x256x14x14x!quant.uniform<i8:f32, 0.03023116220472441>>
    %41 = "top.Weight"() {name = "168_Relu_filter_int8_reorderd"} : () -> tensor<1x256x2304x1xi8>
    %42 = "top.Weight"() {name = "168_Relu_bias_int16"} : () -> tensor<256xi16>
    %43 = "tpu.Conv"(%40, %41, %42) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "168_Relu", pads = [1, 1, 1, 1], rshift = 8 : i64, strides = [1, 1]} : (tensor<4x256x14x14x!quant.uniform<i8:f32, 0.03023116220472441>>, tensor<1x256x2304x1xi8>, tensor<256xi16>) -> tensor<4x256x14x14x!quant.uniform<i8:f32, 0.024559417322834644>>
    %44 = "top.Weight"() {name = "169_Conv_filter_int8_reorderd"} : () -> tensor<1x256x2304x1xi8>
    %45 = "top.Weight"() {name = "169_Conv_bias_int16"} : () -> tensor<256xi16>
    %46 = "tpu.Conv"(%43, %44, %45) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "169_Conv", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [1, 1]} : (tensor<4x256x14x14x!quant.uniform<i8:f32, 0.024559417322834644>>, tensor<1x256x2304x1xi8>, tensor<256xi16>) -> tensor<4x256x14x14x!quant.uniform<i8:f32, 0.046117151968503939>>
    %47 = "tpu.Add"(%46, %40) {coeff = [9.400000e+01, 1.230000e+02], do_relu = true, name = "172_Relu", rshifts = [6, 7]} : (tensor<4x256x14x14x!quant.uniform<i8:f32, 0.046117151968503939>>, tensor<4x256x14x14x!quant.uniform<i8:f32, 0.03023116220472441>>) -> tensor<4x256x14x14x!quant.uniform<i8:f32, 0.03152902834645669>>
    %48 = "top.Weight"() {name = "175_Relu_filter_int8_reorderd"} : () -> tensor<1x512x2304x1xi8>
    %49 = "top.Weight"() {name = "175_Relu_bias_int16"} : () -> tensor<512xi16>
    %50 = "tpu.Conv"(%47, %48, %49) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "175_Relu", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [2, 2]} : (tensor<4x256x14x14x!quant.uniform<i8:f32, 0.03152902834645669>>, tensor<1x512x2304x1xi8>, tensor<512xi16>) -> tensor<4x512x7x7x!quant.uniform<i8:f32, 0.018364696062992125>>
    %51 = "top.Weight"() {name = "176_Conv_filter_int8_reorderd"} : () -> tensor<1x512x4608x1xi8>
    %52 = "top.Weight"() {name = "176_Conv_bias_int16"} : () -> tensor<512xi16>
    %53 = "tpu.Conv"(%50, %51, %52) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "176_Conv", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [1, 1]} : (tensor<4x512x7x7x!quant.uniform<i8:f32, 0.018364696062992125>>, tensor<1x512x4608x1xi8>, tensor<512xi16>) -> tensor<4x512x7x7x!quant.uniform<i8:f32, 0.028710800787401573>>
    %54 = "top.Weight"() {name = "178_Conv_filter_int8_reorderd"} : () -> tensor<1x512x256x1xi8>
    %55 = "top.Weight"() {name = "178_Conv_bias_int16"} : () -> tensor<512xi16>
    %56 = "tpu.Conv"(%47, %54, %55) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "178_Conv", pads = [0, 0, 0, 0], rshift = 6 : i64, strides = [2, 2]} : (tensor<4x256x14x14x!quant.uniform<i8:f32, 0.03152902834645669>>, tensor<1x512x256x1xi8>, tensor<512xi16>) -> tensor<4x512x7x7x!quant.uniform<i8:f32, 0.016461327559055116>>
    %57 = "tpu.Add"(%53, %56) {coeff = [1.170000e+02, 6.700000e+01], do_relu = true, name = "181_Relu", rshifts = [7, 7]} : (tensor<4x512x7x7x!quant.uniform<i8:f32, 0.028710800787401573>>, tensor<4x512x7x7x!quant.uniform<i8:f32, 0.016461327559055116>>) -> tensor<4x512x7x7x!quant.uniform<i8:f32, 0.03142888503937008>>
    %58 = "top.Weight"() {name = "184_Relu_filter_int8_reorderd"} : () -> tensor<1x512x4608x1xi8>
    %59 = "top.Weight"() {name = "184_Relu_bias_int16"} : () -> tensor<512xi16>
    %60 = "tpu.Conv"(%57, %58, %59) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "184_Relu", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [1, 1]} : (tensor<4x512x7x7x!quant.uniform<i8:f32, 0.03142888503937008>>, tensor<1x512x4608x1xi8>, tensor<512xi16>) -> tensor<4x512x7x7x!quant.uniform<i8:f32, 0.011750085039370079>>
    %61 = "top.Weight"() {name = "185_Conv_filter_int8_reorderd"} : () -> tensor<1x512x4608x1xi8>
    %62 = "top.Weight"() {name = "185_Conv_bias_int16"} : () -> tensor<512xi16>
    %63 = "tpu.Conv"(%60, %61, %62) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "185_Conv", pads = [1, 1, 1, 1], rshift = 8 : i64, strides = [1, 1]} : (tensor<4x512x7x7x!quant.uniform<i8:f32, 0.011750085039370079>>, tensor<1x512x4608x1xi8>, tensor<512xi16>) -> tensor<4x512x7x7x!quant.uniform<i8:f32, 0.099151010236220471>>
    %64 = "tpu.Add"(%63, %57) {coeff = [1.040000e+02, 6.600000e+01], do_relu = true, name = "188_Relu", rshifts = [7, 8]} : (tensor<4x512x7x7x!quant.uniform<i8:f32, 0.099151010236220471>>, tensor<4x512x7x7x!quant.uniform<i8:f32, 0.03142888503937008>>) -> tensor<4x512x7x7x!quant.uniform<i8:f32, 0.12260971653543307>>
    %65 = "tpu.AvgPool"(%64) {do_relu = false, kernel_shape = [7, 7], name = "189_GlobalAveragePool", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<4x512x7x7x!quant.uniform<i8:f32, 0.12260971653543307>>) -> tensor<4x512x1x1x!quant.uniform<i8:f32, 0.12260971653543307>>
    %66 = "tpu.Reshape"(%65) {name = "190_Flatten"} : (tensor<4x512x1x1x!quant.uniform<i8:f32, 0.12260971653543307>>) -> tensor<4x512x!quant.uniform<i8:f32, 0.034969870866141735>>
    %67 = "top.Weight"() {name = "output_Gemm_filter_int8"} : () -> tensor<512x1000xi8>
    %68 = "top.Weight"() {name = "output_Gemm_bias_int16"} : () -> tensor<1000xi16>
    %69 = "tpu.MatMul"(%66, %67, %68) {do_relu = false, name = "output_Gemm_quantized", rshift = 8 : i64} : (tensor<4x512x!quant.uniform<i8:f32, 0.034969870866141735>>, tensor<512x1000xi8>, tensor<1000xi16>) -> tensor<4x1000x!quant.uniform<i8:f32, 0.080851126771653542>>
    %70 = "tpu.Cast"(%69) {name = "output_Gemm"} : (tensor<4x1000x!quant.uniform<i8:f32, 0.080851126771653542>>) -> tensor<4x1000xf32>
    return %70 : tensor<4x1000xf32>
  }
}

