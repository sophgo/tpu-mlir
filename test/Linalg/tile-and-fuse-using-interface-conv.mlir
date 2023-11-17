// RUN: tpuc-test -test-tiling-interface=tile-size=1,8,32,16 -cse %s

func.func @conv_tensors_static(%input: tensor<1x225x225x3xf32>, %filter: tensor<3x3x3x32xf32>, %filter_0: tensor<3x3x32x32xf32>) -> tensor<1x110x110x32xf32> {


  %init = tensor.empty() : tensor<1x112x112x32xf32>

  %conv = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%input, %filter : tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>)
    outs(%init : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>

  %init_0 = tensor.empty() : tensor<1x110x110x32xf32>

  %conv_0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>,
    __internal_linalg_transform__ = "root"}
    ins(%conv, %filter_0 : tensor<1x112x112x32xf32>, tensor<3x3x32x32xf32>)
    outs(%init_0 : tensor<1x110x110x32xf32>) -> tensor<1x110x110x32xf32>

    return %conv_0: tensor<1x110x110x32xf32>
}
