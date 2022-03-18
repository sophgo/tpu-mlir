module attributes {module.chip = "bm1684", module.name = "resnet18", module.state = "TPU_QUANTIED", module.weight_file = "resnet18_tpu_quantied_bm1684_weight.npz"} {
  func @main(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = "top.Input"(%arg0) {name = "input"} : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.calibrated<f32<-2.6412898999999999:2.6412898999999999>>>
    %1 = "tpu.Cast"(%0) {name = "input_to_int8"} : (tensor<1x3x224x224x!quant.calibrated<f32<-2.6412898999999999:2.6412898999999999>>>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 0.020797558267716534>>
    %2 = "top.Weight"() {name = "125_Relu_filter_int8"} : () -> tensor<64x3x7x7xi8>
    %3 = "top.Weight"() {name = "125_Relu_bias_int16"} : () -> tensor<64xi16>
    %4 = "tpu.Conv"(%1, %2, %3) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [7, 7], name = "125_Relu", pads = [3, 3, 3, 3], rshift = 8 : i64, strides = [2, 2]} : (tensor<1x3x224x224x!quant.uniform<i8:f32, 0.020797558267716534>>, tensor<64x3x7x7xi8>, tensor<64xi16>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 0.027347358267716535>>
    %5 = "tpu.MaxPool"(%4) {do_relu = false, kernel_shape = [3, 3], name = "126_MaxPool", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x64x112x112x!quant.uniform<i8:f32, 0.027347358267716535>>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.027347358267716535>>
    %6 = "top.Weight"() {name = "129_Relu_filter_int8"} : () -> tensor<64x64x3x3xi8>
    %7 = "top.Weight"() {name = "129_Relu_bias_int16"} : () -> tensor<64xi16>
    %8 = "tpu.Conv"(%5, %6, %7) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "129_Relu", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [1, 1]} : (tensor<1x64x56x56x!quant.uniform<i8:f32, 0.027347358267716535>>, tensor<64x64x3x3xi8>, tensor<64xi16>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.016837784251968503>>
    %9 = "top.Weight"() {name = "130_Conv_filter_int8"} : () -> tensor<64x64x3x3xi8>
    %10 = "top.Weight"() {name = "130_Conv_bias_int16"} : () -> tensor<64xi16>
    %11 = "tpu.Conv"(%8, %9, %10) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "130_Conv", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [1, 1]} : (tensor<1x64x56x56x!quant.uniform<i8:f32, 0.016837784251968503>>, tensor<64x64x3x3xi8>, tensor<64xi16>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.02463193937007874>>
    %12 = "tpu.Add"(%11, %5) {do_relu = true, name = "133_Relu", rshifts = [7, 7]} : (tensor<1x64x56x56x!quant.uniform<i8:f32, 0.02463193937007874>>, tensor<1x64x56x56x!quant.uniform<i8:f32, 0.027347358267716535>>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.030926265354330709>>
    %13 = "top.Weight"() {name = "136_Relu_filter_int8"} : () -> tensor<64x64x3x3xi8>
    %14 = "top.Weight"() {name = "136_Relu_bias_int16"} : () -> tensor<64xi16>
    %15 = "tpu.Conv"(%12, %13, %14) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "136_Relu", pads = [1, 1, 1, 1], rshift = 8 : i64, strides = [1, 1]} : (tensor<1x64x56x56x!quant.uniform<i8:f32, 0.030926265354330709>>, tensor<64x64x3x3xi8>, tensor<64xi16>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.019197859842519684>>
    %16 = "top.Weight"() {name = "137_Conv_filter_int8"} : () -> tensor<64x64x3x3xi8>
    %17 = "top.Weight"() {name = "137_Conv_bias_int16"} : () -> tensor<64xi16>
    %18 = "tpu.Conv"(%15, %16, %17) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "137_Conv", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [1, 1]} : (tensor<1x64x56x56x!quant.uniform<i8:f32, 0.019197859842519684>>, tensor<64x64x3x3xi8>, tensor<64xi16>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.027846443307086612>>
    %19 = "tpu.Add"(%18, %12) {do_relu = true, name = "140_Relu", rshifts = [7, 7]} : (tensor<1x64x56x56x!quant.uniform<i8:f32, 0.027846443307086612>>, tensor<1x64x56x56x!quant.uniform<i8:f32, 0.030926265354330709>>) -> tensor<1x64x56x56x!quant.uniform<i8:f32, 0.055844555118110234>>
    %20 = "top.Weight"() {name = "143_Relu_filter_int8"} : () -> tensor<128x64x3x3xi8>
    %21 = "top.Weight"() {name = "143_Relu_bias_int16"} : () -> tensor<128xi16>
    %22 = "tpu.Conv"(%19, %20, %21) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "143_Relu", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [2, 2]} : (tensor<1x64x56x56x!quant.uniform<i8:f32, 0.055844555118110234>>, tensor<128x64x3x3xi8>, tensor<128xi16>) -> tensor<1x128x28x28x!quant.uniform<i8:f32, 0.01627884724409449>>
    %23 = "top.Weight"() {name = "144_Conv_filter_int8"} : () -> tensor<128x128x3x3xi8>
    %24 = "top.Weight"() {name = "144_Conv_bias_int16"} : () -> tensor<128xi16>
    %25 = "tpu.Conv"(%22, %23, %24) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "144_Conv", pads = [1, 1, 1, 1], rshift = 9 : i64, strides = [1, 1]} : (tensor<1x128x28x28x!quant.uniform<i8:f32, 0.01627884724409449>>, tensor<128x128x3x3xi8>, tensor<128xi16>) -> tensor<1x128x28x28x!quant.uniform<i8:f32, 0.047900481102362204>>
    %26 = "top.Weight"() {name = "146_Conv_filter_int8"} : () -> tensor<128x64x1x1xi8>
    %27 = "top.Weight"() {name = "146_Conv_bias_int16"} : () -> tensor<128xi16>
    %28 = "tpu.Conv"(%19, %26, %27) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "146_Conv", pads = [0, 0, 0, 0], rshift = 6 : i64, strides = [2, 2]} : (tensor<1x64x56x56x!quant.uniform<i8:f32, 0.055844555118110234>>, tensor<128x64x1x1xi8>, tensor<128xi16>) -> tensor<1x128x28x28x!quant.uniform<i8:f32, 0.020879310236220472>>
    %29 = "tpu.Add"(%25, %28) {do_relu = true, name = "149_Relu", rshifts = [7, 8]} : (tensor<1x128x28x28x!quant.uniform<i8:f32, 0.047900481102362204>>, tensor<1x128x28x28x!quant.uniform<i8:f32, 0.020879310236220472>>) -> tensor<1x128x28x28x!quant.uniform<i8:f32, 0.05242758818897638>>
    %30 = "top.Weight"() {name = "152_Relu_filter_int8"} : () -> tensor<128x128x3x3xi8>
    %31 = "top.Weight"() {name = "152_Relu_bias_int16"} : () -> tensor<128xi16>
    %32 = "tpu.Conv"(%29, %30, %31) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "152_Relu", pads = [1, 1, 1, 1], rshift = 6 : i64, strides = [1, 1]} : (tensor<1x128x28x28x!quant.uniform<i8:f32, 0.05242758818897638>>, tensor<128x128x3x3xi8>, tensor<128xi16>) -> tensor<1x128x28x28x!quant.uniform<i8:f32, 0.012759493700787401>>
    %33 = "top.Weight"() {name = "153_Conv_filter_int8"} : () -> tensor<128x128x3x3xi8>
    %34 = "top.Weight"() {name = "153_Conv_bias_int16"} : () -> tensor<128xi16>
    %35 = "tpu.Conv"(%32, %33, %34) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "153_Conv", pads = [1, 1, 1, 1], rshift = 8 : i64, strides = [1, 1]} : (tensor<1x128x28x28x!quant.uniform<i8:f32, 0.012759493700787401>>, tensor<128x128x3x3xi8>, tensor<128xi16>) -> tensor<1x128x28x28x!quant.uniform<i8:f32, 0.029383494488188975>>
    %36 = "tpu.Add"(%35, %29) {do_relu = true, name = "156_Relu", rshifts = [7, 7]} : (tensor<1x128x28x28x!quant.uniform<i8:f32, 0.029383494488188975>>, tensor<1x128x28x28x!quant.uniform<i8:f32, 0.05242758818897638>>) -> tensor<1x128x28x28x!quant.uniform<i8:f32, 0.055967818897637793>>
    %37 = "top.Weight"() {name = "159_Relu_filter_int8"} : () -> tensor<256x128x3x3xi8>
    %38 = "top.Weight"() {name = "159_Relu_bias_int16"} : () -> tensor<256xi16>
    %39 = "tpu.Conv"(%36, %37, %38) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "159_Relu", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [2, 2]} : (tensor<1x128x28x28x!quant.uniform<i8:f32, 0.055967818897637793>>, tensor<256x128x3x3xi8>, tensor<256xi16>) -> tensor<1x256x14x14x!quant.uniform<i8:f32, 0.026231895275590552>>
    %40 = "top.Weight"() {name = "160_Conv_filter_int8"} : () -> tensor<256x256x3x3xi8>
    %41 = "top.Weight"() {name = "160_Conv_bias_int16"} : () -> tensor<256xi16>
    %42 = "tpu.Conv"(%39, %40, %41) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "160_Conv", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [1, 1]} : (tensor<1x256x14x14x!quant.uniform<i8:f32, 0.026231895275590552>>, tensor<256x256x3x3xi8>, tensor<256xi16>) -> tensor<1x256x14x14x!quant.uniform<i8:f32, 0.028183774015748033>>
    %43 = "top.Weight"() {name = "162_Conv_filter_int8"} : () -> tensor<256x128x1x1xi8>
    %44 = "top.Weight"() {name = "162_Conv_bias_int16"} : () -> tensor<256xi16>
    %45 = "tpu.Conv"(%36, %43, %44) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "162_Conv", pads = [0, 0, 0, 0], rshift = 6 : i64, strides = [2, 2]} : (tensor<1x128x28x28x!quant.uniform<i8:f32, 0.055967818897637793>>, tensor<256x128x1x1xi8>, tensor<256xi16>) -> tensor<1x256x14x14x!quant.uniform<i8:f32, 0.013166514960629922>>
    %46 = "tpu.Add"(%42, %45) {do_relu = true, name = "165_Relu", rshifts = [7, 8]} : (tensor<1x256x14x14x!quant.uniform<i8:f32, 0.028183774015748033>>, tensor<1x256x14x14x!quant.uniform<i8:f32, 0.013166514960629922>>) -> tensor<1x256x14x14x!quant.uniform<i8:f32, 0.03023116220472441>>
    %47 = "top.Weight"() {name = "168_Relu_filter_int8"} : () -> tensor<256x256x3x3xi8>
    %48 = "top.Weight"() {name = "168_Relu_bias_int16"} : () -> tensor<256xi16>
    %49 = "tpu.Conv"(%46, %47, %48) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "168_Relu", pads = [1, 1, 1, 1], rshift = 8 : i64, strides = [1, 1]} : (tensor<1x256x14x14x!quant.uniform<i8:f32, 0.03023116220472441>>, tensor<256x256x3x3xi8>, tensor<256xi16>) -> tensor<1x256x14x14x!quant.uniform<i8:f32, 0.024559417322834644>>
    %50 = "top.Weight"() {name = "169_Conv_filter_int8"} : () -> tensor<256x256x3x3xi8>
    %51 = "top.Weight"() {name = "169_Conv_bias_int16"} : () -> tensor<256xi16>
    %52 = "tpu.Conv"(%49, %50, %51) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "169_Conv", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [1, 1]} : (tensor<1x256x14x14x!quant.uniform<i8:f32, 0.024559417322834644>>, tensor<256x256x3x3xi8>, tensor<256xi16>) -> tensor<1x256x14x14x!quant.uniform<i8:f32, 0.046117151968503939>>
    %53 = "tpu.Add"(%52, %46) {do_relu = true, name = "172_Relu", rshifts = [6, 7]} : (tensor<1x256x14x14x!quant.uniform<i8:f32, 0.046117151968503939>>, tensor<1x256x14x14x!quant.uniform<i8:f32, 0.03023116220472441>>) -> tensor<1x256x14x14x!quant.uniform<i8:f32, 0.03152902834645669>>
    %54 = "top.Weight"() {name = "175_Relu_filter_int8"} : () -> tensor<512x256x3x3xi8>
    %55 = "top.Weight"() {name = "175_Relu_bias_int16"} : () -> tensor<512xi16>
    %56 = "tpu.Conv"(%53, %54, %55) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "175_Relu", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [2, 2]} : (tensor<1x256x14x14x!quant.uniform<i8:f32, 0.03152902834645669>>, tensor<512x256x3x3xi8>, tensor<512xi16>) -> tensor<1x512x7x7x!quant.uniform<i8:f32, 0.018364696062992125>>
    %57 = "top.Weight"() {name = "176_Conv_filter_int8"} : () -> tensor<512x512x3x3xi8>
    %58 = "top.Weight"() {name = "176_Conv_bias_int16"} : () -> tensor<512xi16>
    %59 = "tpu.Conv"(%56, %57, %58) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "176_Conv", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [1, 1]} : (tensor<1x512x7x7x!quant.uniform<i8:f32, 0.018364696062992125>>, tensor<512x512x3x3xi8>, tensor<512xi16>) -> tensor<1x512x7x7x!quant.uniform<i8:f32, 0.028710800787401573>>
    %60 = "top.Weight"() {name = "178_Conv_filter_int8"} : () -> tensor<512x256x1x1xi8>
    %61 = "top.Weight"() {name = "178_Conv_bias_int16"} : () -> tensor<512xi16>
    %62 = "tpu.Conv"(%53, %60, %61) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "178_Conv", pads = [0, 0, 0, 0], rshift = 6 : i64, strides = [2, 2]} : (tensor<1x256x14x14x!quant.uniform<i8:f32, 0.03152902834645669>>, tensor<512x256x1x1xi8>, tensor<512xi16>) -> tensor<1x512x7x7x!quant.uniform<i8:f32, 0.016461327559055116>>
    %63 = "tpu.Add"(%59, %62) {do_relu = true, name = "181_Relu", rshifts = [7, 7]} : (tensor<1x512x7x7x!quant.uniform<i8:f32, 0.028710800787401573>>, tensor<1x512x7x7x!quant.uniform<i8:f32, 0.016461327559055116>>) -> tensor<1x512x7x7x!quant.uniform<i8:f32, 0.03142888503937008>>
    %64 = "top.Weight"() {name = "184_Relu_filter_int8"} : () -> tensor<512x512x3x3xi8>
    %65 = "top.Weight"() {name = "184_Relu_bias_int16"} : () -> tensor<512xi16>
    %66 = "tpu.Conv"(%63, %64, %65) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "184_Relu", pads = [1, 1, 1, 1], rshift = 7 : i64, strides = [1, 1]} : (tensor<1x512x7x7x!quant.uniform<i8:f32, 0.03142888503937008>>, tensor<512x512x3x3xi8>, tensor<512xi16>) -> tensor<1x512x7x7x!quant.uniform<i8:f32, 0.011750085039370079>>
    %67 = "top.Weight"() {name = "185_Conv_filter_int8"} : () -> tensor<512x512x3x3xi8>
    %68 = "top.Weight"() {name = "185_Conv_bias_int16"} : () -> tensor<512xi16>
    %69 = "tpu.Conv"(%66, %67, %68) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "185_Conv", pads = [1, 1, 1, 1], rshift = 8 : i64, strides = [1, 1]} : (tensor<1x512x7x7x!quant.uniform<i8:f32, 0.011750085039370079>>, tensor<512x512x3x3xi8>, tensor<512xi16>) -> tensor<1x512x7x7x!quant.uniform<i8:f32, 0.099151010236220471>>
    %70 = "tpu.Add"(%69, %63) {do_relu = true, name = "188_Relu", rshifts = [7, 8]} : (tensor<1x512x7x7x!quant.uniform<i8:f32, 0.099151010236220471>>, tensor<1x512x7x7x!quant.uniform<i8:f32, 0.03142888503937008>>) -> tensor<1x512x7x7x!quant.uniform<i8:f32, 0.12260971653543307>>
    %71 = "tpu.AvgPool"(%70) {do_relu = false, kernel_shape = [7, 7], name = "189_GlobalAveragePool", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x512x7x7x!quant.uniform<i8:f32, 0.12260971653543307>>) -> tensor<1x512x1x1x!quant.uniform<i8:f32, 0.12260971653543307>>
    %72 = "tpu.Reshape"(%71) {name = "190_Flatten"} : (tensor<1x512x1x1x!quant.uniform<i8:f32, 0.12260971653543307>>) -> tensor<1x512x!quant.uniform<i8:f32, 0.034969870866141735>>
    %73 = "top.Weight"() {name = "output_Gemm_filter_int8"} : () -> tensor<512x1000xi8>
    %74 = "top.Weight"() {name = "output_Gemm_bias_int16"} : () -> tensor<1000xi16>
    %75 = "tpu.MatMul"(%72, %73, %74) {do_relu = false, name = "output_Gemm_quantized", rshift = 8 : i64} : (tensor<1x512x!quant.uniform<i8:f32, 0.034969870866141735>>, tensor<512x1000xi8>, tensor<1000xi16>) -> tensor<1x1000x!quant.uniform<i8:f32, 0.080851164566929146>>
    %76 = "tpu.Cast"(%75) {name = "output_Gemm"} : (tensor<1x1000x!quant.uniform<i8:f32, 0.080851164566929146>>) -> tensor<1x1000xf32>
    return %76 : tensor<1x1000xf32>
  }
}

