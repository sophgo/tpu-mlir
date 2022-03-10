module attributes {mlir.state = "TOPS_F32", mlir.weight_file = "resnet18_tops_weight.npz"} {
  func @main(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000x!quant.calibrated<f32<-5.8380175000000003:10.2630844>>> {
    %0 = "tops.Input"(%arg0) {name = "input"} : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.calibrated<f32<-2.0665298000000001:2.6400003000000001>>>
    %1 = "tops.Weight"() {name = "194"} : () -> tensor<64x3x7x7xf32>
    %2 = "tops.Weight"() {name = "196"} : () -> tensor<64xf32>
    %3 = "tops.Conv"(%0, %1, %2) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [7, 7], name = "125_Relu", pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224x!quant.calibrated<f32<-2.0665298000000001:2.6400003000000001>>>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112x!quant.calibrated<f32<0.000000e+00:3.4714185999999998>>>
    %4 = "tops.MaxPool"(%3) {do_relu = false, kernel_shape = [3, 3], name = "126_MaxPool", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x64x112x112x!quant.calibrated<f32<0.000000e+00:3.4714185999999998>>>) -> tensor<1x64x56x56x!quant.calibrated<f32<0.000000e+00:3.4714185999999998>>>
    %5 = "tops.Weight"() {name = "198"} : () -> tensor<64x64x3x3xf32>
    %6 = "tops.Weight"() {name = "200"} : () -> tensor<64xf32>
    %7 = "tops.Conv"(%4, %5, %6) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "129_Relu", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56x!quant.calibrated<f32<0.000000e+00:3.4714185999999998>>>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56x!quant.calibrated<f32<0.000000e+00:2.1373546000000001>>>
    %8 = "tops.Weight"() {name = "202"} : () -> tensor<64x64x3x3xf32>
    %9 = "tops.Weight"() {name = "204"} : () -> tensor<64xf32>
    %10 = "tops.Conv"(%7, %8, %9) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "130_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56x!quant.calibrated<f32<0.000000e+00:2.1373546000000001>>>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56x!quant.calibrated<f32<-3.1267288:2.2347237999999998>>>
    %11 = "tops.Add"(%10, %4) {do_relu = true, name = "133_Relu"} : (tensor<1x64x56x56x!quant.calibrated<f32<-3.1267288:2.2347237999999998>>>, tensor<1x64x56x56x!quant.calibrated<f32<0.000000e+00:3.4714185999999998>>>) -> tensor<1x64x56x56x!quant.calibrated<f32<0.000000e+00:3.9257178000000001>>>
    %12 = "tops.Weight"() {name = "206"} : () -> tensor<64x64x3x3xf32>
    %13 = "tops.Weight"() {name = "208"} : () -> tensor<64xf32>
    %14 = "tops.Conv"(%11, %12, %13) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "136_Relu", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56x!quant.calibrated<f32<0.000000e+00:3.9257178000000001>>>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56x!quant.calibrated<f32<0.000000e+00:2.4369377999999999>>>
    %15 = "tops.Weight"() {name = "210"} : () -> tensor<64x64x3x3xf32>
    %16 = "tops.Weight"() {name = "212"} : () -> tensor<64xf32>
    %17 = "tops.Conv"(%14, %15, %16) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "137_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56x!quant.calibrated<f32<0.000000e+00:2.4369377999999999>>>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56x!quant.calibrated<f32<-5.6556344000000003:4.5319881000000004>>>
    %18 = "tops.Add"(%17, %11) {do_relu = true, name = "140_Relu"} : (tensor<1x64x56x56x!quant.calibrated<f32<-5.6556344000000003:4.5319881000000004>>>, tensor<1x64x56x56x!quant.calibrated<f32<0.000000e+00:3.9257178000000001>>>) -> tensor<1x64x56x56x!quant.calibrated<f32<0.000000e+00:7.0887957000000004>>>
    %19 = "tops.Weight"() {name = "214"} : () -> tensor<128x64x3x3xf32>
    %20 = "tops.Weight"() {name = "216"} : () -> tensor<128xf32>
    %21 = "tops.Conv"(%18, %19, %20) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "143_Relu", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x64x56x56x!quant.calibrated<f32<0.000000e+00:7.0887957000000004>>>, tensor<128x64x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28x!quant.calibrated<f32<0.000000e+00:2.0664041000000002>>>
    %22 = "tops.Weight"() {name = "218"} : () -> tensor<128x128x3x3xf32>
    %23 = "tops.Weight"() {name = "220"} : () -> tensor<128xf32>
    %24 = "tops.Conv"(%21, %22, %23) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "144_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x28x28x!quant.calibrated<f32<0.000000e+00:2.0664041000000002>>>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28x!quant.calibrated<f32<-2.3946154000000002:6.0803909000000003>>>
    %25 = "tops.Weight"() {name = "222"} : () -> tensor<128x64x1x1xf32>
    %26 = "tops.Weight"() {name = "224"} : () -> tensor<128xf32>
    %27 = "tops.Conv"(%18, %25, %26) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "146_Conv", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x64x56x56x!quant.calibrated<f32<0.000000e+00:7.0887957000000004>>>, tensor<128x64x1x1xf32>, tensor<128xf32>) -> tensor<1x128x28x28x!quant.calibrated<f32<-2.0705146999999999:2.6503774999999998>>>
    %28 = "tops.Add"(%24, %27) {do_relu = true, name = "149_Relu"} : (tensor<1x128x28x28x!quant.calibrated<f32<-2.3946154000000002:6.0803909000000003>>>, tensor<1x128x28x28x!quant.calibrated<f32<-2.0705146999999999:2.6503774999999998>>>) -> tensor<1x128x28x28x!quant.calibrated<f32<0.000000e+00:6.6550526999999997>>>
    %29 = "tops.Weight"() {name = "226"} : () -> tensor<128x128x3x3xf32>
    %30 = "tops.Weight"() {name = "228"} : () -> tensor<128xf32>
    %31 = "tops.Conv"(%28, %29, %30) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "152_Relu", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x28x28x!quant.calibrated<f32<0.000000e+00:6.6550526999999997>>>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28x!quant.calibrated<f32<0.000000e+00:4.3191050999999998>>>
    %32 = "tops.Weight"() {name = "230"} : () -> tensor<128x128x3x3xf32>
    %33 = "tops.Weight"() {name = "232"} : () -> tensor<128xf32>
    %34 = "tops.Conv"(%31, %32, %33) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "153_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x128x28x28x!quant.calibrated<f32<0.000000e+00:4.3191050999999998>>>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28x!quant.calibrated<f32<-5.4252824999999998:3.4575309999999999>>>
    %35 = "tops.Add"(%34, %28) {do_relu = true, name = "156_Relu"} : (tensor<1x128x28x28x!quant.calibrated<f32<-5.4252824999999998:3.4575309999999999>>>, tensor<1x128x28x28x!quant.calibrated<f32<0.000000e+00:6.6550526999999997>>>) -> tensor<1x128x28x28x!quant.calibrated<f32<0.000000e+00:7.1044421>>>
    %36 = "tops.Weight"() {name = "234"} : () -> tensor<256x128x3x3xf32>
    %37 = "tops.Weight"() {name = "236"} : () -> tensor<256xf32>
    %38 = "tops.Conv"(%35, %36, %37) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "159_Relu", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x128x28x28x!quant.calibrated<f32<0.000000e+00:7.1044421>>>, tensor<256x128x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14x!quant.calibrated<f32<0.000000e+00:3.3298239999999999>>>
    %39 = "tops.Weight"() {name = "238"} : () -> tensor<256x256x3x3xf32>
    %40 = "tops.Weight"() {name = "240"} : () -> tensor<256xf32>
    %41 = "tops.Conv"(%38, %39, %40) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "160_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x256x14x14x!quant.calibrated<f32<0.000000e+00:3.3298239999999999>>>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14x!quant.calibrated<f32<-2.6611403999999999:3.5775917000000002>>>
    %42 = "tops.Weight"() {name = "242"} : () -> tensor<256x128x1x1xf32>
    %43 = "tops.Weight"() {name = "244"} : () -> tensor<256xf32>
    %44 = "tops.Conv"(%35, %42, %43) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "162_Conv", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x128x28x28x!quant.calibrated<f32<0.000000e+00:7.1044421>>>, tensor<256x128x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14x!quant.calibrated<f32<-1.6713309000000001:1.0027511>>>
    %45 = "tops.Add"(%41, %44) {do_relu = true, name = "165_Relu"} : (tensor<1x256x14x14x!quant.calibrated<f32<-2.6611403999999999:3.5775917000000002>>>, tensor<1x256x14x14x!quant.calibrated<f32<-1.6713309000000001:1.0027511>>>) -> tensor<1x256x14x14x!quant.calibrated<f32<0.000000e+00:3.8374828999999999>>>
    %46 = "tops.Weight"() {name = "246"} : () -> tensor<256x256x3x3xf32>
    %47 = "tops.Weight"() {name = "248"} : () -> tensor<256xf32>
    %48 = "tops.Conv"(%45, %46, %47) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "168_Relu", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x256x14x14x!quant.calibrated<f32<0.000000e+00:3.8374828999999999>>>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14x!quant.calibrated<f32<0.000000e+00:3.1175229999999998>>>
    %49 = "tops.Weight"() {name = "250"} : () -> tensor<256x256x3x3xf32>
    %50 = "tops.Weight"() {name = "252"} : () -> tensor<256xf32>
    %51 = "tops.Conv"(%48, %49, %50) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "169_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x256x14x14x!quant.calibrated<f32<0.000000e+00:3.1175229999999998>>>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14x!quant.calibrated<f32<-5.8540187000000001:2.878412>>>
    %52 = "tops.Add"(%51, %45) {do_relu = true, name = "172_Relu"} : (tensor<1x256x14x14x!quant.calibrated<f32<-5.8540187000000001:2.878412>>>, tensor<1x256x14x14x!quant.calibrated<f32<0.000000e+00:3.8374828999999999>>>) -> tensor<1x256x14x14x!quant.calibrated<f32<0.000000e+00:4.0022316>>>
    %53 = "tops.Weight"() {name = "254"} : () -> tensor<512x256x3x3xf32>
    %54 = "tops.Weight"() {name = "256"} : () -> tensor<512xf32>
    %55 = "tops.Conv"(%52, %53, %54) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "175_Relu", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x256x14x14x!quant.calibrated<f32<0.000000e+00:4.0022316>>>, tensor<512x256x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7x!quant.calibrated<f32<0.000000e+00:2.3311774999999999>>>
    %56 = "tops.Weight"() {name = "258"} : () -> tensor<512x512x3x3xf32>
    %57 = "tops.Weight"() {name = "260"} : () -> tensor<512xf32>
    %58 = "tops.Conv"(%55, %56, %57) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "176_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x7x7x!quant.calibrated<f32<0.000000e+00:2.3311774999999999>>>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7x!quant.calibrated<f32<-3.6444912:2.7373536000000001>>>
    %59 = "tops.Weight"() {name = "262"} : () -> tensor<512x256x1x1xf32>
    %60 = "tops.Weight"() {name = "264"} : () -> tensor<512xf32>
    %61 = "tops.Conv"(%52, %59, %60) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [1, 1], name = "178_Conv", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x256x14x14x!quant.calibrated<f32<0.000000e+00:4.0022316>>>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tensor<1x512x7x7x!quant.calibrated<f32<-2.0895676999999999:1.6711008999999999>>>
    %62 = "tops.Add"(%58, %61) {do_relu = true, name = "181_Relu"} : (tensor<1x512x7x7x!quant.calibrated<f32<-3.6444912:2.7373536000000001>>>, tensor<1x512x7x7x!quant.calibrated<f32<-2.0895676999999999:1.6711008999999999>>>) -> tensor<1x512x7x7x!quant.calibrated<f32<0.000000e+00:3.9895193999999998>>>
    %63 = "tops.Weight"() {name = "266"} : () -> tensor<512x512x3x3xf32>
    %64 = "tops.Weight"() {name = "268"} : () -> tensor<512xf32>
    %65 = "tops.Conv"(%62, %63, %64) {dilations = [1, 1], do_relu = true, group = 1 : i64, kernel_shape = [3, 3], name = "184_Relu", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x7x7x!quant.calibrated<f32<0.000000e+00:3.9895193999999998>>>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7x!quant.calibrated<f32<0.000000e+00:1.4915322>>>
    %66 = "tops.Weight"() {name = "270"} : () -> tensor<512x512x3x3xf32>
    %67 = "tops.Weight"() {name = "272"} : () -> tensor<512xf32>
    %68 = "tops.Conv"(%65, %66, %67) {dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], name = "185_Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x512x7x7x!quant.calibrated<f32<0.000000e+00:1.4915322>>>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7x!quant.calibrated<f32<-8.3822545999999996:12.586029999999999>>>
    %69 = "tops.Add"(%68, %62) {do_relu = true, name = "188_Relu"} : (tensor<1x512x7x7x!quant.calibrated<f32<-8.3822545999999996:12.586029999999999>>>, tensor<1x512x7x7x!quant.calibrated<f32<0.000000e+00:3.9895193999999998>>>) -> tensor<1x512x7x7x!quant.calibrated<f32<0.000000e+00:15.563830400000001>>>
    %70 = "tops.AvgPool"(%69) {do_relu = false, kernel_shape = [7, 7], name = "189_GlobalAveragePool", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x512x7x7x!quant.calibrated<f32<0.000000e+00:15.563830400000001>>>) -> tensor<1x512x1x1x!quant.calibrated<f32<0.000000e+00:4.4390048999999996>>>
    %71 = "tops.Reshape"(%70) {name = "190_Flatten"} : (tensor<1x512x1x1x!quant.calibrated<f32<0.000000e+00:4.4390048999999996>>>) -> tensor<1x512x!quant.calibrated<f32<0.000000e+00:4.4390048999999996>>>
    %72 = "tops.Weight"() {name = "fc.weight_fix"} : () -> tensor<512x1000xf32>
    %73 = "tops.Weight"() {name = "fc.bias"} : () -> tensor<1000xf32>
    %74 = "tops.MatMul"(%71, %72, %73) {do_relu = false, name = "output_Gemm"} : (tensor<1x512x!quant.calibrated<f32<0.000000e+00:4.4390048999999996>>>, tensor<512x1000xf32>, tensor<1000xf32>) -> tensor<1x1000x!quant.calibrated<f32<-5.8380175000000003:10.2630844>>>
    return %74 : tensor<1x1000x!quant.calibrated<f32<-5.8380175000000003:10.2630844>>>
  }
}

