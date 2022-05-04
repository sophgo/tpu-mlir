// RUN: sophgo-opt --split-input-file --toptflite-to-tpu --verify-each %s | FileCheck %s

// -----

// CHECK-LABEL: @test_cast
// CHECK: %[[VAR0:.*]] = "tpu.Cast"(%arg0) {name = ""} : (tensor<?x3x224x224xf32>) -> tensor<?x3x224x224x!quant.uniform<i8:f32, 1.0774157047271729:-13>>
func.func @test_cast(%arg0: tensor<?x3x224x224xf32>) -> tensor<?x3x224x224x!quant.uniform<i8:f32, 1.0774157047271729:-13>> {
  %0 = "quant.qcast"(%arg0) {name = ""} : (tensor<?x3x224x224xf32>) -> tensor<?x3x224x224x!quant.uniform<i8:f32, 1.0774157047271729:-13>>
return %0 : tensor<?x3x224x224x!quant.uniform<i8:f32, 1.0774157047271729:-13>>
}

// -----

// CHECK-LABEL: @test_conv
// CHECK-NEXT: %[[VAR0:.*]] = "top.Weight"() {name = ""} : () -> tensor<2x3x7x7xi8>
// CHECK-NEXT: %[[VAR1:.*]] = "top.Weight"() {name = ""} : () -> tensor<2xi32>
// CHECK-NEXT: %[[VAR2:.*]] = "tpu.Conv"(%arg0, %[[VAR0]], %[[VAR1]])
// CHECK-SAME: multiplier = [1195430722, 1449933006]
// CHECK-SAME: rshift = [-10, -13]
func.func @test_conv(%arg0: tensor<?x3x224x224x!quant.uniform<i8:f32, 1.0774157047271729:-13>>) -> tensor<?x2x112x112x!quant.uniform<i8:f32, 0.13142697513103485:-128>> {
    %1 = "top.Weight"() {name = ""} : () -> tensor<2x3x7x7x!quant.uniform<i8<-127:127>:f32:0, {6.6312561102677137E-5,1.0053779078589287E-5}>>
    %2 = "top.Weight"() {name = ""} : () -> tensor<2x!quant.uniform<i32:f32:0, {7.1446193032898009E-5,1.0832099178514909E-5}>>
    %3 = "top.Conv"(%arg0, %1, %2) {dilations = [1, 1], do_relu = true, kernel_shape = [7, 7], name = "", pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<?x3x224x224x!quant.uniform<i8:f32, 1.0774157047271729:-13>>, tensor<2x3x7x7x!quant.uniform<i8<-127:127>:f32:0, {6.6312561102677137E-5,1.0053779078589287E-5}>>, tensor<2x!quant.uniform<i32:f32:0, {7.1446193032898009E-5,1.0832099178514909E-5}>>) -> tensor<?x2x112x112x!quant.uniform<i8:f32, 0.13142697513103485:-128>>
	return %3 : tensor<?x2x112x112x!quant.uniform<i8:f32, 0.13142697513103485:-128>>
}

// -----

// CHECK-LABEL: @test_add
// CHECK-NEXT: %[[VAR0:.*]] = "tpu.Add"(%arg0, %arg1)
// CHECK-SAME: multipliers = [1073741824, 1334010492, 1250638940]
// CHECK-SAME: rshifts = [20, 19, -17]
func.func @test_add(%arg0 : tensor<?x256x56x56x!quant.uniform<i8:f32, 0.17478941380977631:28>>, %arg1 : tensor<?x256x56x56x!quant.uniform<i8:f32, 0.10857866704463959:11>>) -> tensor<?x256x56x56x!quant.uniform<i8:f32, 0.075033128261566162:-128>> {
    %0 = "top.Add"(%arg0, %arg1) {name = ""} : (tensor<?x256x56x56x!quant.uniform<i8:f32, 0.17478941380977631:28>>, tensor<?x256x56x56x!quant.uniform<i8:f32, 0.10857866704463959:11>>) -> tensor<?x256x56x56x!quant.uniform<i8:f32, 0.075033128261566162:-128>>
	return %0 : tensor<?x256x56x56x!quant.uniform<i8:f32, 0.075033128261566162:-128>>
}

// -----

// CHECK-LABEL: @test_matmul
// CHECK-NEXT: %[[VAR0:.*]] = "tpu.MatMul"(%arg0, %arg1, %arg2)
// CHECK-SAME: multiplier = 1122827235
// CHECK-SAME: rshift = -7
func.func @test_matmul(%arg0 : tensor<?x2048x!quant.uniform<i8:f32, 0.10753946006298065:-128>>, %arg1 : tensor<2048x1000x!quant.uniform<i8<-127:127>:f32, 0.0057965274900197983>>, %arg2 : tensor<1000x!quant.uniform<i32:f32, 6.2335544498637319E-4>>) -> tensor<?x1000x!quant.uniform<i8:f32, 0.15260285139083862:-68>> {
    %0 = "top.MatMul"(%arg0, %arg1, %arg2) {do_relu = false, name = ""} : (tensor<?x2048x!quant.uniform<i8:f32, 0.10753946006298065:-128>>, tensor<2048x1000x!quant.uniform<i8<-127:127>:f32, 0.0057965274900197983>>, tensor<1000x!quant.uniform<i32:f32, 6.2335544498637319E-4>>) -> tensor<?x1000x!quant.uniform<i8:f32, 0.15260285139083862:-68>>
	return %0 : tensor<?x1000x!quant.uniform<i8:f32, 0.15260285139083862:-68>>
}

// -----

// CHECK-LABEL: @test_maxpool
// CHECK: %[[VAR0:.*]] = "tpu.MaxPool"(%arg0)
func.func @test_maxpool(%arg0 : tensor<?x64x112x112x!quant.uniform<i8:f32, 0.13142697513103485:-128>>) -> tensor<?x64x56x56x!quant.uniform<i8:f32, 0.13142697513103485:-128>> {
    %0 = "top.MaxPool"(%arg0) {kernel_shape = [3, 3], name = "", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<?x64x112x112x!quant.uniform<i8:f32, 0.13142697513103485:-128>>) -> tensor<?x64x56x56x!quant.uniform<i8:f32, 0.13142697513103485:-128>>
	return %0 : tensor<?x64x56x56x!quant.uniform<i8:f32, 0.13142697513103485:-128>>
}

// -----

// CHECK-LABEL: @test_avgpool
// CHECK: %[[VAR0:.*]] = "tpu.AvgPool"(%arg0)
func.func @test_avgpool(%arg0 : tensor<?x2048x7x7x!quant.uniform<i8:f32, 0.34225726127624512:-128>>) -> tensor<?x2048x!quant.uniform<i8:f32, 0.10753946006298065:-128>> {
    %0 = "top.AvgPool"(%arg0) {kernel_shape = [1, 2], name = "", pads = [0, 0, 0, 0], strides = [0, 0]} : (tensor<?x2048x7x7x!quant.uniform<i8:f32, 0.34225726127624512:-128>>) -> tensor<?x2048x!quant.uniform<i8:f32, 0.10753946006298065:-128>>
	return %0 : tensor<?x2048x!quant.uniform<i8:f32, 0.10753946006298065:-128>>
}

// -----

// CHECK-LABEL: @test_softmax
// CHECK: %[[VAR0:.*]] = "tpu.Softmax"(%arg0)
func.func @test_softmax(%arg0 : tensor<?x1000x!quant.uniform<i8:f32, 0.15260285139083862:-68>>) -> tensor<?x1000x!quant.uniform<i8:f32, 3.906250e-03:-128>> {
    %0 = "top.Softmax"(%arg0) {axis = 1 : i64, name = ""} : (tensor<?x1000x!quant.uniform<i8:f32, 0.15260285139083862:-68>>) -> tensor<?x1000x!quant.uniform<i8:f32, 3.906250e-03:-128>>
	return %0 : tensor<?x1000x!quant.uniform<i8:f32, 3.906250e-03:-128>>
}
