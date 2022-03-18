module attributes {module.chip = "ALL", module.name = "resnet18", module.state = "TOP_CALIBRATED", module.weight_file = "resnet18_top_f32_all_weight.npz"} {
  func @main(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000x!quant.calibrated<f32<-10.268097900000001:10.268097900000001>>> {
    %0 = "top.Input"(%arg0) {name = "input"} : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.calibrated<f32<-2.6412898999999999:2.6412898999999999>>>
    %1 = "top.Weight"() {name = "194"} : () -> tensor<64x3x7x7xf32>
    %2 = "top.Weight"() {name = "196"} : () -> tensor<64xf32>
    %3 = "top.Conv"(%0, %1, %2) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [7, 7], name = "125_Relu", pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224x!quant.calibrated<f32<-2.6412898999999999:2.6412898999999999>>>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112x!quant.calibrated<f32<-3.4731144999999999:3.4731144999999999>>>
    %4 = "top.MaxPool"(%3) {do_relu = false, kernel_shape = [3, 3], name = "126_MaxPool", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x64x112x112x!quant.calibrated<f32<-3.4731144999999999:3.4731144999999999>>>) -> tensor<1x64x56x56x!quant.calibrated<f32<-3.4731144999999999:3.4731144999999999>>>
    %5 = "top.Weight"() {name = "198"} : () -> tensor<64x64x3x3xf32>
    %6 = "top.Weight"() {name = "200"} : () -> tensor<64xf32>
    %7 = "top.Conv"(%4, %5, %6) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "129_Relu", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56x!quant.calibrated<f32<-3.4731144999999999:3.4731144999999999>>>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56x!quant.calibrated<f32<-2.1383985999999999:2.1383985999999999>>>
    %8 = "top.Weight"() {name = "202"} : () -> tensor<64x64x3x3xf32>
    %9 = "top.Weight"() {name = "204"} : () -> tensor<64xf32>
    %10 = "top.Conv"(%7, %8, %9) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "130_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56x!quant.calibrated<f32<-2.1383985999999999:2.1383985999999999>>>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56x!quant.calibrated<f32<-3.1282562999999999:3.1282562999999999>>>
    %11 = "top.Add"(%10, %4) {do_relu = true, name = "133_Relu"} : (tensor<1x64x56x56x!quant.calibrated<f32<-3.1282562999999999:3.1282562999999999>>>, tensor<1x64x56x56x!quant.calibrated<f32<-3.4731144999999999:3.4731144999999999>>>) -> tensor<1x64x56x56x!quant.calibrated<f32<-3.9276357000000002:3.9276357000000002>>>
    %12 = "top.Weight"() {name = "206"} : () -> tensor<64x64x3x3xf32>
    %13 = "top.Weight"() {name = "208"} : () -> tensor<64xf32>
    %14 = "top.Conv"(%11, %12, %13) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "136_Relu", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56x!quant.calibrated<f32<-3.9276357000000002:3.9276357000000002>>>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56x!quant.calibrated<f32<-2.4381282:2.4381282>>>
    %15 = "top.Weight"() {name = "210"} : () -> tensor<64x64x3x3xf32>
    %16 = "top.Weight"() {name = "212"} : () -> tensor<64xf32>
    %17 = "top.Conv"(%14, %15, %16) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "137_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56x!quant.calibrated<f32<-2.4381282:2.4381282>>>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56x!quant.calibrated<f32<-3.5364982999999999:3.5364982999999999>>>
    %18 = "top.Add"(%17, %11) {do_relu = true, name = "140_Relu"} : (tensor<1x64x56x56x!quant.calibrated<f32<-3.5364982999999999:3.5364982999999999>>>, tensor<1x64x56x56x!quant.calibrated<f32<-3.9276357000000002:3.9276357000000002>>>) -> tensor<1x64x56x56x!quant.calibrated<f32<-7.0922584999999998:7.0922584999999998>>>
    %19 = "top.Weight"() {name = "214"} : () -> tensor<128x64x3x3xf32>
    %20 = "top.Weight"() {name = "216"} : () -> tensor<128xf32>
    %21 = "top.Conv"(%18, %19, %20) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "143_Relu", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x64x56x56x!quant.calibrated<f32<-7.0922584999999998:7.0922584999999998>>>, tensor<128x64x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28x!quant.calibrated<f32<-2.0674136000000001:2.0674136000000001>>>
    %22 = "top.Weight"() {name = "218"} : () -> tensor<128x128x3x3xf32>
    %23 = "top.Weight"() {name = "220"} : () -> tensor<128xf32>
    %24 = "top.Conv"(%21, %22, %23) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "144_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x28x28x!quant.calibrated<f32<-2.0674136000000001:2.0674136000000001>>>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28x!quant.calibrated<f32<-6.0833611000000003:6.0833611000000003>>>
    %25 = "top.Weight"() {name = "222"} : () -> tensor<128x64x1x1xf32>
    %26 = "top.Weight"() {name = "224"} : () -> tensor<128xf32>
    %27 = "top.Conv"(%18, %25, %26) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "146_Conv", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x64x56x56x!quant.calibrated<f32<-7.0922584999999998:7.0922584999999998>>>, tensor<128x64x1x1xf32>, tensor<128xf32>) -> tensor<1x128x28x28x!quant.calibrated<f32<-2.6516723999999998:2.6516723999999998>>>
    %28 = "top.Add"(%24, %27) {do_relu = true, name = "149_Relu"} : (tensor<1x128x28x28x!quant.calibrated<f32<-6.0833611000000003:6.0833611000000003>>>, tensor<1x128x28x28x!quant.calibrated<f32<-2.6516723999999998:2.6516723999999998>>>) -> tensor<1x128x28x28x!quant.calibrated<f32<-6.6583037000000003:6.6583037000000003>>>
    %29 = "top.Weight"() {name = "226"} : () -> tensor<128x128x3x3xf32>
    %30 = "top.Weight"() {name = "228"} : () -> tensor<128xf32>
    %31 = "top.Conv"(%28, %29, %30) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "152_Relu", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x28x28x!quant.calibrated<f32<-6.6583037000000003:6.6583037000000003>>>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28x!quant.calibrated<f32<-1.6204556999999999:1.6204556999999999>>>
    %32 = "top.Weight"() {name = "230"} : () -> tensor<128x128x3x3xf32>
    %33 = "top.Weight"() {name = "232"} : () -> tensor<128xf32>
    %34 = "top.Conv"(%31, %32, %33) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "153_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x28x28x!quant.calibrated<f32<-1.6204556999999999:1.6204556999999999>>>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28x!quant.calibrated<f32<-3.7317038:3.7317038>>>
    %35 = "top.Add"(%34, %28) {do_relu = true, name = "156_Relu"} : (tensor<1x128x28x28x!quant.calibrated<f32<-3.7317038:3.7317038>>>, tensor<1x128x28x28x!quant.calibrated<f32<-6.6583037000000003:6.6583037000000003>>>) -> tensor<1x128x28x28x!quant.calibrated<f32<-7.1079129999999999:7.1079129999999999>>>
    %36 = "top.Weight"() {name = "234"} : () -> tensor<256x128x3x3xf32>
    %37 = "top.Weight"() {name = "236"} : () -> tensor<256xf32>
    %38 = "top.Conv"(%35, %36, %37) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "159_Relu", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x128x28x28x!quant.calibrated<f32<-7.1079129999999999:7.1079129999999999>>>, tensor<256x128x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14x!quant.calibrated<f32<-3.3314507:3.3314507>>>
    %39 = "top.Weight"() {name = "238"} : () -> tensor<256x256x3x3xf32>
    %40 = "top.Weight"() {name = "240"} : () -> tensor<256xf32>
    %41 = "top.Conv"(%38, %39, %40) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "160_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x256x14x14x!quant.calibrated<f32<-3.3314507:3.3314507>>>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14x!quant.calibrated<f32<-3.5793393:3.5793393>>>
    %42 = "top.Weight"() {name = "242"} : () -> tensor<256x128x1x1xf32>
    %43 = "top.Weight"() {name = "244"} : () -> tensor<256xf32>
    %44 = "top.Conv"(%35, %42, %43) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "162_Conv", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x128x28x28x!quant.calibrated<f32<-7.1079129999999999:7.1079129999999999>>>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14x!quant.calibrated<f32<-1.6721474000000001:1.6721474000000001>>>
    %45 = "top.Add"(%41, %44) {do_relu = true, name = "165_Relu"} : (tensor<1x256x14x14x!quant.calibrated<f32<-3.5793393:3.5793393>>>, tensor<1x256x14x14x!quant.calibrated<f32<-1.6721474000000001:1.6721474000000001>>>) -> tensor<1x256x14x14x!quant.calibrated<f32<-3.8393576:3.8393576>>>
    %46 = "top.Weight"() {name = "246"} : () -> tensor<256x256x3x3xf32>
    %47 = "top.Weight"() {name = "248"} : () -> tensor<256xf32>
    %48 = "top.Conv"(%45, %46, %47) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "168_Relu", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x256x14x14x!quant.calibrated<f32<-3.8393576:3.8393576>>>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14x!quant.calibrated<f32<-3.119046:3.119046>>>
    %49 = "top.Weight"() {name = "250"} : () -> tensor<256x256x3x3xf32>
    %50 = "top.Weight"() {name = "252"} : () -> tensor<256xf32>
    %51 = "top.Conv"(%48, %49, %50) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "169_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x256x14x14x!quant.calibrated<f32<-3.119046:3.119046>>>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14x!quant.calibrated<f32<-5.8568783:5.8568783>>>
    %52 = "top.Add"(%51, %45) {do_relu = true, name = "172_Relu"} : (tensor<1x256x14x14x!quant.calibrated<f32<-5.8568783:5.8568783>>>, tensor<1x256x14x14x!quant.calibrated<f32<-3.8393576:3.8393576>>>) -> tensor<1x256x14x14x!quant.calibrated<f32<-4.0041865999999997:4.0041865999999997>>>
    %53 = "top.Weight"() {name = "254"} : () -> tensor<512x256x3x3xf32>
    %54 = "top.Weight"() {name = "256"} : () -> tensor<512xf32>
    %55 = "top.Conv"(%52, %53, %54) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "175_Relu", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x256x14x14x!quant.calibrated<f32<-4.0041865999999997:4.0041865999999997>>>, tensor<512x256x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7x!quant.calibrated<f32<-2.3323163999999998:2.3323163999999998>>>
    %56 = "top.Weight"() {name = "258"} : () -> tensor<512x512x3x3xf32>
    %57 = "top.Weight"() {name = "260"} : () -> tensor<512xf32>
    %58 = "top.Conv"(%55, %56, %57) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "176_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x7x7x!quant.calibrated<f32<-2.3323163999999998:2.3323163999999998>>>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7x!quant.calibrated<f32<-3.6462716999999998:3.6462716999999998>>>
    %59 = "top.Weight"() {name = "262"} : () -> tensor<512x256x1x1xf32>
    %60 = "top.Weight"() {name = "264"} : () -> tensor<512xf32>
    %61 = "top.Conv"(%52, %59, %60) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "178_Conv", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x256x14x14x!quant.calibrated<f32<-4.0041865999999997:4.0041865999999997>>>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tensor<1x512x7x7x!quant.calibrated<f32<-2.0905885999999998:2.0905885999999998>>>
    %62 = "top.Add"(%58, %61) {do_relu = true, name = "181_Relu"} : (tensor<1x512x7x7x!quant.calibrated<f32<-3.6462716999999998:3.6462716999999998>>>, tensor<1x512x7x7x!quant.calibrated<f32<-2.0905885999999998:2.0905885999999998>>>) -> tensor<1x512x7x7x!quant.calibrated<f32<-3.9914684:3.9914684>>>
    %63 = "top.Weight"() {name = "266"} : () -> tensor<512x512x3x3xf32>
    %64 = "top.Weight"() {name = "268"} : () -> tensor<512xf32>
    %65 = "top.Conv"(%62, %63, %64) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "184_Relu", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x7x7x!quant.calibrated<f32<-3.9914684:3.9914684>>>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7x!quant.calibrated<f32<-1.4922607999999999:1.4922607999999999>>>
    %66 = "top.Weight"() {name = "270"} : () -> tensor<512x512x3x3xf32>
    %67 = "top.Weight"() {name = "272"} : () -> tensor<512xf32>
    %68 = "top.Conv"(%65, %66, %67) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "185_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x7x7x!quant.calibrated<f32<-1.4922607999999999:1.4922607999999999>>>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7x!quant.calibrated<f32<-12.5921783:12.5921783>>>
    %69 = "top.Add"(%68, %62) {do_relu = true, name = "188_Relu"} : (tensor<1x512x7x7x!quant.calibrated<f32<-12.5921783:12.5921783>>>, tensor<1x512x7x7x!quant.calibrated<f32<-3.9914684:3.9914684>>>) -> tensor<1x512x7x7x!quant.calibrated<f32<-15.571434:15.571434>>>
    %70 = "top.AvgPool"(%69) {do_relu = false, kernel_shape = [7, 7], name = "189_GlobalAveragePool", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x512x7x7x!quant.calibrated<f32<-15.571434:15.571434>>>) -> tensor<1x512x1x1x!quant.calibrated<f32<-4.4411735999999999:4.4411735999999999>>>
    %71 = "top.Reshape"(%70) {name = "190_Flatten"} : (tensor<1x512x1x1x!quant.calibrated<f32<-4.4411735999999999:4.4411735999999999>>>) -> tensor<1x512x!quant.calibrated<f32<-4.4411735999999999:4.4411735999999999>>>
    %72 = "top.Weight"() {name = "fc.weight_fix"} : () -> tensor<512x1000xf32>
    %73 = "top.Weight"() {name = "fc.bias"} : () -> tensor<1000xf32>
    %74 = "top.MatMul"(%71, %72, %73) {do_relu = false, name = "output_Gemm"} : (tensor<1x512x!quant.calibrated<f32<-4.4411735999999999:4.4411735999999999>>>, tensor<512x1000xf32>, tensor<1000xf32>) -> tensor<1x1000x!quant.calibrated<f32<-10.268097900000001:10.268097900000001>>>
    return %74 : tensor<1x1000x!quant.calibrated<f32<-10.268097900000001:10.268097900000001>>>
  }
}

