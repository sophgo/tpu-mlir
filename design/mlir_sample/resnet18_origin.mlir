module attributes {module.chip = "ALL", module.name = "resnet18", module.state = "TOP_F32", module.weight_file = "resnet18_top_f32_all_weight.npz"} {
  func @main(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = "top.None"() : () -> none
    %1 = "top.Input"(%arg0) {name = "input"} : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
    %2 = "top.Weight"() {name = "194"} : () -> tensor<64x3x7x7xf32>
    %3 = "top.Weight"() {name = "196"} : () -> tensor<64xf32>
    %4 = "top.Conv"(%1, %2, %3) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [7, 7], name = "123_Conv", pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %5 = "top.Relu"(%4) {name = "125_Relu"} : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %6 = "top.MaxPool"(%5) {do_relu = false, kernel_shape = [3, 3], name = "126_MaxPool", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
    %7 = "top.Weight"() {name = "198"} : () -> tensor<64x64x3x3xf32>
    %8 = "top.Weight"() {name = "200"} : () -> tensor<64xf32>
    %9 = "top.Conv"(%6, %7, %8) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "127_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %10 = "top.Relu"(%9) {name = "129_Relu"} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %11 = "top.Weight"() {name = "202"} : () -> tensor<64x64x3x3xf32>
    %12 = "top.Weight"() {name = "204"} : () -> tensor<64xf32>
    %13 = "top.Conv"(%10, %11, %12) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "130_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %14 = "top.Add"(%13, %6) {do_relu = false, name = "132_Add"} : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %15 = "top.Relu"(%14) {name = "133_Relu"} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %16 = "top.Weight"() {name = "206"} : () -> tensor<64x64x3x3xf32>
    %17 = "top.Weight"() {name = "208"} : () -> tensor<64xf32>
    %18 = "top.Conv"(%15, %16, %17) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "134_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %19 = "top.Relu"(%18) {name = "136_Relu"} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %20 = "top.Weight"() {name = "210"} : () -> tensor<64x64x3x3xf32>
    %21 = "top.Weight"() {name = "212"} : () -> tensor<64xf32>
    %22 = "top.Conv"(%19, %20, %21) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "137_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %23 = "top.Add"(%22, %15) {do_relu = false, name = "139_Add"} : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %24 = "top.Relu"(%23) {name = "140_Relu"} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %25 = "top.Weight"() {name = "214"} : () -> tensor<128x64x3x3xf32>
    %26 = "top.Weight"() {name = "216"} : () -> tensor<128xf32>
    %27 = "top.Conv"(%24, %25, %26) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "141_Conv", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x64x56x56xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %28 = "top.Relu"(%27) {name = "143_Relu"} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %29 = "top.Weight"() {name = "218"} : () -> tensor<128x128x3x3xf32>
    %30 = "top.Weight"() {name = "220"} : () -> tensor<128xf32>
    %31 = "top.Conv"(%28, %29, %30) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "144_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %32 = "top.Weight"() {name = "222"} : () -> tensor<128x64x1x1xf32>
    %33 = "top.Weight"() {name = "224"} : () -> tensor<128xf32>
    %34 = "top.Conv"(%24, %32, %33) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "146_Conv", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x64x56x56xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %35 = "top.Add"(%31, %34) {do_relu = false, name = "148_Add"} : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %36 = "top.Relu"(%35) {name = "149_Relu"} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %37 = "top.Weight"() {name = "226"} : () -> tensor<128x128x3x3xf32>
    %38 = "top.Weight"() {name = "228"} : () -> tensor<128xf32>
    %39 = "top.Conv"(%36, %37, %38) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "150_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %40 = "top.Relu"(%39) {name = "152_Relu"} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %41 = "top.Weight"() {name = "230"} : () -> tensor<128x128x3x3xf32>
    %42 = "top.Weight"() {name = "232"} : () -> tensor<128xf32>
    %43 = "top.Conv"(%40, %41, %42) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "153_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %44 = "top.Add"(%43, %36) {do_relu = false, name = "155_Add"} : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %45 = "top.Relu"(%44) {name = "156_Relu"} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %46 = "top.Weight"() {name = "234"} : () -> tensor<256x128x3x3xf32>
    %47 = "top.Weight"() {name = "236"} : () -> tensor<256xf32>
    %48 = "top.Conv"(%45, %46, %47) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "157_Conv", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x128x28x28xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %49 = "top.Relu"(%48) {name = "159_Relu"} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %50 = "top.Weight"() {name = "238"} : () -> tensor<256x256x3x3xf32>
    %51 = "top.Weight"() {name = "240"} : () -> tensor<256xf32>
    %52 = "top.Conv"(%49, %50, %51) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "160_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %53 = "top.Weight"() {name = "242"} : () -> tensor<256x128x1x1xf32>
    %54 = "top.Weight"() {name = "244"} : () -> tensor<256xf32>
    %55 = "top.Conv"(%45, %53, %54) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "162_Conv", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x128x28x28xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %56 = "top.Add"(%52, %55) {do_relu = false, name = "164_Add"} : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %57 = "top.Relu"(%56) {name = "165_Relu"} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %58 = "top.Weight"() {name = "246"} : () -> tensor<256x256x3x3xf32>
    %59 = "top.Weight"() {name = "248"} : () -> tensor<256xf32>
    %60 = "top.Conv"(%57, %58, %59) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "166_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %61 = "top.Relu"(%60) {name = "168_Relu"} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %62 = "top.Weight"() {name = "250"} : () -> tensor<256x256x3x3xf32>
    %63 = "top.Weight"() {name = "252"} : () -> tensor<256xf32>
    %64 = "top.Conv"(%61, %62, %63) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "169_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %65 = "top.Add"(%64, %57) {do_relu = false, name = "171_Add"} : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %66 = "top.Relu"(%65) {name = "172_Relu"} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %67 = "top.Weight"() {name = "254"} : () -> tensor<512x256x3x3xf32>
    %68 = "top.Weight"() {name = "256"} : () -> tensor<512xf32>
    %69 = "top.Conv"(%66, %67, %68) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "173_Conv", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x256x14x14xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %70 = "top.Relu"(%69) {name = "175_Relu"} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %71 = "top.Weight"() {name = "258"} : () -> tensor<512x512x3x3xf32>
    %72 = "top.Weight"() {name = "260"} : () -> tensor<512xf32>
    %73 = "top.Conv"(%70, %71, %72) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "176_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %74 = "top.Weight"() {name = "262"} : () -> tensor<512x256x1x1xf32>
    %75 = "top.Weight"() {name = "264"} : () -> tensor<512xf32>
    %76 = "top.Conv"(%66, %74, %75) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "178_Conv", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x256x14x14xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %77 = "top.Add"(%73, %76) {do_relu = false, name = "180_Add"} : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %78 = "top.Relu"(%77) {name = "181_Relu"} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %79 = "top.Weight"() {name = "266"} : () -> tensor<512x512x3x3xf32>
    %80 = "top.Weight"() {name = "268"} : () -> tensor<512xf32>
    %81 = "top.Conv"(%78, %79, %80) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "182_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %82 = "top.Relu"(%81) {name = "184_Relu"} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %83 = "top.Weight"() {name = "270"} : () -> tensor<512x512x3x3xf32>
    %84 = "top.Weight"() {name = "272"} : () -> tensor<512xf32>
    %85 = "top.Conv"(%82, %83, %84) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "185_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %86 = "top.Add"(%85, %78) {do_relu = false, name = "187_Add"} : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %87 = "top.Relu"(%86) {name = "188_Relu"} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %88 = "top.AvgPool"(%87) {do_relu = false, kernel_shape = [7, 7], name = "189_GlobalAveragePool", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x512x7x7xf32>) -> tensor<1x512x1x1xf32>
    %89 = "top.Reshape"(%88) {name = "190_Flatten"} : (tensor<1x512x1x1xf32>) -> tensor<1x512xf32>
    %90 = "top.Weight"() {name = "fc.weight_fix"} : () -> tensor<512x1000xf32>
    %91 = "top.Weight"() {name = "fc.bias"} : () -> tensor<1000xf32>
    %92 = "top.MatMul"(%89, %90, %91) {do_relu = false, name = "output_Gemm"} : (tensor<1x512xf32>, tensor<512x1000xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
    return %92 : tensor<1x1000xf32>
  }
}
