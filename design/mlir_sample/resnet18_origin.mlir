module attributes {mlir.state = "TOP_F32", mlir.weight_file = "resnet18_topweight.npz"} {
  func @main(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = "topNone"() : () -> none
    %1 = "topInput"(%arg0) {name = "input"} : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
    %2 = "topWeight"() {name = "194"} : () -> tensor<64x3x7x7xf32>
    %3 = "topWeight"() {name = "196"} : () -> tensor<64xf32>
    %4 = "topConv"(%1, %2, %3) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [7, 7], name = "123_Conv", pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %5 = "topRelu"(%4) {name = "125_Relu"} : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %6 = "topMaxPool"(%5) {do_relu = false, kernel_shape = [3, 3], name = "126_MaxPool", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
    %7 = "topWeight"() {name = "198"} : () -> tensor<64x64x3x3xf32>
    %8 = "topWeight"() {name = "200"} : () -> tensor<64xf32>
    %9 = "topConv"(%6, %7, %8) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "127_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %10 = "topRelu"(%9) {name = "129_Relu"} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %11 = "topWeight"() {name = "202"} : () -> tensor<64x64x3x3xf32>
    %12 = "topWeight"() {name = "204"} : () -> tensor<64xf32>
    %13 = "topConv"(%10, %11, %12) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "130_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %14 = "topAdd"(%13, %6) {do_relu = false, name = "132_Add"} : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %15 = "topRelu"(%14) {name = "133_Relu"} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %16 = "topWeight"() {name = "206"} : () -> tensor<64x64x3x3xf32>
    %17 = "topWeight"() {name = "208"} : () -> tensor<64xf32>
    %18 = "topConv"(%15, %16, %17) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "134_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %19 = "topRelu"(%18) {name = "136_Relu"} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %20 = "topWeight"() {name = "210"} : () -> tensor<64x64x3x3xf32>
    %21 = "topWeight"() {name = "212"} : () -> tensor<64xf32>
    %22 = "topConv"(%19, %20, %21) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "137_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %23 = "topAdd"(%22, %15) {do_relu = false, name = "139_Add"} : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %24 = "topRelu"(%23) {name = "140_Relu"} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %25 = "topWeight"() {name = "214"} : () -> tensor<128x64x3x3xf32>
    %26 = "topWeight"() {name = "216"} : () -> tensor<128xf32>
    %27 = "topConv"(%24, %25, %26) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "141_Conv", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x64x56x56xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %28 = "topRelu"(%27) {name = "143_Relu"} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %29 = "topWeight"() {name = "218"} : () -> tensor<128x128x3x3xf32>
    %30 = "topWeight"() {name = "220"} : () -> tensor<128xf32>
    %31 = "topConv"(%28, %29, %30) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "144_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %32 = "topWeight"() {name = "222"} : () -> tensor<128x64x1x1xf32>
    %33 = "topWeight"() {name = "224"} : () -> tensor<128xf32>
    %34 = "topConv"(%24, %32, %33) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "146_Conv", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x64x56x56xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %35 = "topAdd"(%31, %34) {do_relu = false, name = "148_Add"} : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %36 = "topRelu"(%35) {name = "149_Relu"} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %37 = "topWeight"() {name = "226"} : () -> tensor<128x128x3x3xf32>
    %38 = "topWeight"() {name = "228"} : () -> tensor<128xf32>
    %39 = "topConv"(%36, %37, %38) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "150_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %40 = "topRelu"(%39) {name = "152_Relu"} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %41 = "topWeight"() {name = "230"} : () -> tensor<128x128x3x3xf32>
    %42 = "topWeight"() {name = "232"} : () -> tensor<128xf32>
    %43 = "topConv"(%40, %41, %42) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "153_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %44 = "topAdd"(%43, %36) {do_relu = false, name = "155_Add"} : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %45 = "topRelu"(%44) {name = "156_Relu"} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %46 = "topWeight"() {name = "234"} : () -> tensor<256x128x3x3xf32>
    %47 = "topWeight"() {name = "236"} : () -> tensor<256xf32>
    %48 = "topConv"(%45, %46, %47) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "157_Conv", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x128x28x28xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %49 = "topRelu"(%48) {name = "159_Relu"} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %50 = "topWeight"() {name = "238"} : () -> tensor<256x256x3x3xf32>
    %51 = "topWeight"() {name = "240"} : () -> tensor<256xf32>
    %52 = "topConv"(%49, %50, %51) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "160_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %53 = "topWeight"() {name = "242"} : () -> tensor<256x128x1x1xf32>
    %54 = "topWeight"() {name = "244"} : () -> tensor<256xf32>
    %55 = "topConv"(%45, %53, %54) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "162_Conv", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x128x28x28xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %56 = "topAdd"(%52, %55) {do_relu = false, name = "164_Add"} : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %57 = "topRelu"(%56) {name = "165_Relu"} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %58 = "topWeight"() {name = "246"} : () -> tensor<256x256x3x3xf32>
    %59 = "topWeight"() {name = "248"} : () -> tensor<256xf32>
    %60 = "topConv"(%57, %58, %59) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "166_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %61 = "topRelu"(%60) {name = "168_Relu"} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %62 = "topWeight"() {name = "250"} : () -> tensor<256x256x3x3xf32>
    %63 = "topWeight"() {name = "252"} : () -> tensor<256xf32>
    %64 = "topConv"(%61, %62, %63) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "169_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %65 = "topAdd"(%64, %57) {do_relu = false, name = "171_Add"} : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %66 = "topRelu"(%65) {name = "172_Relu"} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %67 = "topWeight"() {name = "254"} : () -> tensor<512x256x3x3xf32>
    %68 = "topWeight"() {name = "256"} : () -> tensor<512xf32>
    %69 = "topConv"(%66, %67, %68) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "173_Conv", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x256x14x14xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %70 = "topRelu"(%69) {name = "175_Relu"} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %71 = "topWeight"() {name = "258"} : () -> tensor<512x512x3x3xf32>
    %72 = "topWeight"() {name = "260"} : () -> tensor<512xf32>
    %73 = "topConv"(%70, %71, %72) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "176_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %74 = "topWeight"() {name = "262"} : () -> tensor<512x256x1x1xf32>
    %75 = "topWeight"() {name = "264"} : () -> tensor<512xf32>
    %76 = "topConv"(%66, %74, %75) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "178_Conv", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x256x14x14xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %77 = "topAdd"(%73, %76) {do_relu = false, name = "180_Add"} : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %78 = "topRelu"(%77) {name = "181_Relu"} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %79 = "topWeight"() {name = "266"} : () -> tensor<512x512x3x3xf32>
    %80 = "topWeight"() {name = "268"} : () -> tensor<512xf32>
    %81 = "topConv"(%78, %79, %80) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "182_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %82 = "topRelu"(%81) {name = "184_Relu"} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %83 = "topWeight"() {name = "270"} : () -> tensor<512x512x3x3xf32>
    %84 = "topWeight"() {name = "272"} : () -> tensor<512xf32>
    %85 = "topConv"(%82, %83, %84) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "185_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %86 = "topAdd"(%85, %78) {do_relu = false, name = "187_Add"} : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %87 = "topRelu"(%86) {name = "188_Relu"} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %88 = "topAvgPool"(%87) {do_relu = false, kernel_shape = [7, 7], name = "189_GlobalAveragePool", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x512x7x7xf32>) -> tensor<1x512x1x1xf32>
    %89 = "topReshape"(%88) {name = "190_Flatten"} : (tensor<1x512x1x1xf32>) -> tensor<1x512xf32>
    %90 = "topWeight"() {name = "fc.weight_fix"} : () -> tensor<512x1000xf32>
    %91 = "topWeight"() {name = "fc.bias"} : () -> tensor<1000xf32>
    %92 = "topMatMul"(%89, %90, %91) {do_relu = false, name = "output_Gemm"} : (tensor<1x512xf32>, tensor<512x1000xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
    return %92 : tensor<1x1000xf32>
  }
}
