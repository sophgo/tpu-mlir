// RUN: tpuc-test %s --codegen-tile-and-fuse-greedily --mlir-print-debuginfo --cse -o %t

#loc = loc(unknown)
#loc1 = loc("/workspace/torch-mlir/examples/attention.py":57:0)
#loc2 = loc("/workspace/torch-mlir/examples/attention.py":56:0)
#loc3 = loc("/workspace/torch-mlir/examples/attention.py":46:0)
#loc4 = loc("/workspace/torch-mlir/examples/attention.py":38:0)
#loc5 = loc("/workspace/torch-mlir/examples/attention.py":49:0)
#loc6 = loc("/workspace/torch-mlir/examples/attention.py":39:0)
#loc7 = loc("/workspace/torch-mlir/examples/attention.py":40:0)
#loc8 = loc("/workspace/torch-mlir/examples/attention.py":44:0)
#loc9 = loc("/workspace/torch-mlir/examples/attention.py":43:0)
#loc10 = loc("/workspace/torch-mlir/examples/attention.py":45:0)
#loc11 = loc("/workspace/torch-mlir/examples/attention.py":53:0)
#loc12 = loc("/workspace/torch-mlir/examples/attention.py":54:0)
#loc13 = loc("/workspace/torch-mlir/examples/attention.py":55:0)
#map = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (0, 0, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> ()>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map10 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>
#map11 = affine_map<(d0, d1, d2) -> ()>
#loc14 = loc("aten::matmul"(#loc4))
#loc15 = loc("aten::softmax"(#loc5))
#loc16 = loc("aten::add"(#loc4))
#loc17 = loc("aten::reshape"(#loc6))
#loc18 = loc("aten::transpose"(#loc7))
#loc19 = loc("aten::transpose"(#loc8))
#loc20 = loc("aten::transpose"(#loc9))
#loc21 = loc("aten::matmul"(#loc10))
#loc22 = loc("aten::div"(#loc3))
#loc23 = loc("aten::matmul"(#loc11))
#loc24 = loc("aten::transpose"(#loc12))
#loc25 = loc("aten::reshape"(#loc13))
#loc26 = loc("aten::matmul"(#loc2))
#loc27 = loc("aten::add"(#loc2))
#loc28 = loc("aten::add"(#loc1))
module attributes {torch.debug_module_name = "Model"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64> loc(#loc)
  func.func @forward(%arg0: tensor<1x384x768xf32> loc(unknown)) -> tensor<1x384x768xf32> {
    %cst = arith.constant dense<1> : tensor<i64> loc(#loc1)
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1x768xf32> loc(#loc2)
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<768x768xf32> loc(#loc2)
    %cst_2 = arith.constant dense<8.000000e+00> : tensor<f64> loc(#loc3)
    %cst_3 = arith.constant 0.000000e+00 : f32 loc(#loc14)
    %cst_4 = arith.constant 0xFF800000 : f32 loc(#loc15)
    %0 = tensor.empty() : tensor<1x384x768xf32> loc(#loc14)
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x384x768xf32>) outs(%0 : tensor<1x384x768xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc4)), %out: f32 loc("aten::matmul"(#loc4))):
      linalg.yield %in : f32 loc(#loc14)
    } -> tensor<1x384x768xf32> loc(#loc14)
    %2 = tensor.empty() : tensor<1x768x768xf32> loc(#loc14)
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_1 : tensor<768x768xf32>) outs(%2 : tensor<1x768x768xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc4)), %out: f32 loc("aten::matmul"(#loc4))):
      linalg.yield %in : f32 loc(#loc14)
    } -> tensor<1x768x768xf32> loc(#loc14)
    %4 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<1x384x768xf32>) -> tensor<1x384x768xf32> loc(#loc14)
    %5 = linalg.batch_matmul ins(%1, %3 : tensor<1x384x768xf32>, tensor<1x768x768xf32>) outs(%4 : tensor<1x384x768xf32>) -> tensor<1x384x768xf32> loc(#loc14)
    %6 = linalg.generic {__linalg_tiling__ = array<i64: 0, 64, 32>, indexing_maps = [#map, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %cst_0 : tensor<1x384x768xf32>, tensor<1x1x768xf32>) outs(%0 : tensor<1x384x768xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc4)), %in_10: f32 loc("aten::add"(#loc4)), %out: f32 loc("aten::add"(#loc4))):
      %38 = arith.addf %in, %in_10 : f32 loc(#loc16)
      linalg.yield %38 : f32 loc(#loc16)
    } -> tensor<1x384x768xf32> loc(#loc16)
    %expanded = tensor.expand_shape %6 [[0], [1], [2, 3]] : tensor<1x384x768xf32> into tensor<1x384x12x64xf32> loc(#loc17)
    %7 = tensor.empty() : tensor<1x12x384x64xf32> loc(#loc18)
    %8 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x384x12x64xf32>) outs(%7 : tensor<1x12x384x64xf32>) {
    ^bb0(%in: f32 loc("aten::reshape"(#loc6)), %out: f32 loc("aten::transpose"(#loc7))):
      linalg.yield %in : f32 loc(#loc18)
    } -> tensor<1x12x384x64xf32> loc(#loc18)
    %9 = tensor.empty() : tensor<1x12x64x384xf32> loc(#loc19)
    %10 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8 : tensor<1x12x384x64xf32>) outs(%9 : tensor<1x12x64x384xf32>) {
    ^bb0(%in: f32 loc("aten::transpose"(#loc9)), %out: f32 loc("aten::transpose"(#loc8))):
      linalg.yield %in : f32 loc(#loc19)
    } -> tensor<1x12x64x384xf32> loc(#loc19)
    %11 = linalg.generic {indexing_maps = [#map7, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8 : tensor<1x12x384x64xf32>) outs(%7 : tensor<1x12x384x64xf32>) {
    ^bb0(%in: f32 loc("aten::transpose"(#loc7)), %out: f32 loc("aten::matmul"(#loc10))):
      linalg.yield %in : f32 loc(#loc21)
    } -> tensor<1x12x384x64xf32> loc(#loc21)
    %12 = linalg.generic {__linalg_tiling__ = array<i64: 0, 4, 64, 32>, indexing_maps = [#map7, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<1x12x64x384xf32>) outs(%9 : tensor<1x12x64x384xf32>) {
    ^bb0(%in: f32 loc("aten::transpose"(#loc8)), %out: f32 loc("aten::matmul"(#loc10))):
      linalg.yield %in : f32 loc(#loc21)
    } -> tensor<1x12x64x384xf32> loc(#loc21)
    %collapsed = tensor.collapse_shape %11 [[0, 1], [2], [3]] : tensor<1x12x384x64xf32> into tensor<12x384x64xf32> loc(#loc21)
    %collapsed_5 = tensor.collapse_shape %12 [[0, 1], [2], [3]] : tensor<1x12x64x384xf32> into tensor<12x64x384xf32> loc(#loc21)
    %13 = tensor.empty() : tensor<12x384x384xf32> loc(#loc21)
    %14 = linalg.fill ins(%cst_3 : f32) outs(%13 : tensor<12x384x384xf32>) -> tensor<12x384x384xf32> loc(#loc21)
    %15 = linalg.batch_matmul {__linalg_tiling__ = array<i64: 4, 64, 32, 32>} ins(%collapsed, %collapsed_5 : tensor<12x384x64xf32>, tensor<12x64x384xf32>) outs(%14 : tensor<12x384x384xf32>) -> tensor<12x384x384xf32> loc(#loc21)
    %expanded_6 = tensor.expand_shape %15 [[0, 1], [2], [3]] : tensor<12x384x384xf32> into tensor<1x12x384x384xf32> loc(#loc21)
    %16 = tensor.empty() : tensor<1x12x384x384xf32> loc(#loc22)
    %17 = linalg.generic {indexing_maps = [#map8, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_2 : tensor<f64>) outs(%16 : tensor<1x12x384x384xf32>) {
    ^bb0(%in: f64 loc("/workspace/torch-mlir/examples/attention.py":46:0), %out: f32 loc("aten::div"(#loc3))):
      %38 = arith.truncf %in : f64 to f32 loc(#loc22)
      linalg.yield %38 : f32 loc(#loc22)
    } -> tensor<1x12x384x384xf32> loc(#loc22)
    %18 = linalg.generic {__linalg_tiling__ = array<i64: 0, 4, 64, 32>, indexing_maps = [#map7, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_6, %17 : tensor<1x12x384x384xf32>, tensor<1x12x384x384xf32>) outs(%16 : tensor<1x12x384x384xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc10)), %in_10: f32 loc("aten::div"(#loc3)), %out: f32 loc("aten::div"(#loc3))):
      %38 = arith.divf %in, %in_10 : f32 loc(#loc22)
      linalg.yield %38 : f32 loc(#loc22)
    } -> tensor<1x12x384x384xf32> loc(#loc22)
    %19 = tensor.empty() : tensor<1x12x384x1xf32> loc(#loc15)
    %20 = linalg.fill ins(%cst_4 : f32) outs(%19 : tensor<1x12x384x1xf32>) -> tensor<1x12x384x1xf32> loc(#loc15)
    %21 = linalg.generic {indexing_maps = [#map4, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%18 : tensor<1x12x384x384xf32>) outs(%20 : tensor<1x12x384x1xf32>) {
    ^bb0(%in: f32 loc("aten::div"(#loc3)), %out: f32 loc("aten::softmax"(#loc5))):
      %38 = arith.maxf %in, %out : f32 loc(#loc15)
      linalg.yield %38 : f32 loc(#loc15)
    } -> tensor<1x12x384x1xf32> loc(#loc15)
    %22 = linalg.generic {indexing_maps = [#map7, #map10, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18, %21 : tensor<1x12x384x384xf32>, tensor<1x12x384x1xf32>) outs(%16 : tensor<1x12x384x384xf32>) {
    ^bb0(%in: f32 loc("aten::div"(#loc3)), %in_10: f32 loc("aten::softmax"(#loc5)), %out: f32 loc("aten::softmax"(#loc5))):
      %38 = arith.subf %in, %in_10 : f32 loc(#loc15)
      linalg.yield %38 : f32 loc(#loc15)
    } -> tensor<1x12x384x384xf32> loc(#loc15)
    %23 = linalg.generic {indexing_maps = [#map7, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22 : tensor<1x12x384x384xf32>) outs(%16 : tensor<1x12x384x384xf32>) {
    ^bb0(%in: f32 loc("aten::softmax"(#loc5)), %out: f32 loc("aten::softmax"(#loc5))):
      %38 = math.exp %in : f32 loc(#loc15)
      linalg.yield %38 : f32 loc(#loc15)
    } -> tensor<1x12x384x384xf32> loc(#loc15)
    %24 = linalg.fill ins(%cst_3 : f32) outs(%19 : tensor<1x12x384x1xf32>) -> tensor<1x12x384x1xf32> loc(#loc15)
    %25 = linalg.generic {indexing_maps = [#map4, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%23 : tensor<1x12x384x384xf32>) outs(%24 : tensor<1x12x384x1xf32>) {
    ^bb0(%in: f32 loc("aten::softmax"(#loc5)), %out: f32 loc("aten::softmax"(#loc5))):
      %38 = arith.addf %in, %out : f32 loc(#loc15)
      linalg.yield %38 : f32 loc(#loc15)
    } -> tensor<1x12x384x1xf32> loc(#loc15)
    %26 = linalg.generic {indexing_maps = [#map7, #map10, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23, %25 : tensor<1x12x384x384xf32>, tensor<1x12x384x1xf32>) outs(%16 : tensor<1x12x384x384xf32>) {
    ^bb0(%in: f32 loc("aten::softmax"(#loc5)), %in_10: f32 loc("aten::softmax"(#loc5)), %out: f32 loc("aten::softmax"(#loc5))):
      %38 = arith.divf %in, %in_10 : f32 loc(#loc15)
      linalg.yield %38 : f32 loc(#loc15)
    } -> tensor<1x12x384x384xf32> loc(#loc15)
    %27 = linalg.generic {indexing_maps = [#map7, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26 : tensor<1x12x384x384xf32>) outs(%16 : tensor<1x12x384x384xf32>) {
    ^bb0(%in: f32 loc("aten::softmax"(#loc5)), %out: f32 loc("aten::matmul"(#loc11))):
      linalg.yield %in : f32 loc(#loc23)
    } -> tensor<1x12x384x384xf32> loc(#loc23)
    %collapsed_7 = tensor.collapse_shape %27 [[0, 1], [2], [3]] : tensor<1x12x384x384xf32> into tensor<12x384x384xf32> loc(#loc23)
    %28 = tensor.empty() : tensor<12x384x64xf32> loc(#loc23)
    %29 = linalg.fill ins(%cst_3 : f32) outs(%28 : tensor<12x384x64xf32>) -> tensor<12x384x64xf32> loc(#loc23)
    %30 = linalg.batch_matmul {__linalg_tiling__ = array<i64: 4, 64, 32>} ins(%collapsed_7, %collapsed : tensor<12x384x384xf32>, tensor<12x384x64xf32>) outs(%29 : tensor<12x384x64xf32>) -> tensor<12x384x64xf32> loc(#loc23)
    %expanded_8 = tensor.expand_shape %30 [[0, 1], [2], [3]] : tensor<12x384x64xf32> into tensor<1x12x384x64xf32> loc(#loc23)
    %31 = tensor.empty() : tensor<1x384x12x64xf32> loc(#loc24)
    %32 = linalg.generic {__linalg_tiling__ = array<i64: 0, 4, 64, 32>, indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_8 : tensor<1x12x384x64xf32>) outs(%31 : tensor<1x384x12x64xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc11)), %out: f32 loc("aten::transpose"(#loc12))):
      linalg.yield %in : f32 loc(#loc24)
    } -> tensor<1x384x12x64xf32> loc(#loc24)
    %collapsed_9 = tensor.collapse_shape %32 [[0], [1], [2, 3]] : tensor<1x384x12x64xf32> into tensor<1x384x768xf32> loc(#loc25)
    %33 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_9 : tensor<1x384x768xf32>) outs(%0 : tensor<1x384x768xf32>) {
    ^bb0(%in: f32 loc("aten::reshape"(#loc13)), %out: f32 loc("aten::matmul"(#loc2))):
      linalg.yield %in : f32 loc(#loc26)
    } -> tensor<1x384x768xf32> loc(#loc26)
    %34 = linalg.batch_matmul ins(%33, %3 : tensor<1x384x768xf32>, tensor<1x768x768xf32>) outs(%4 : tensor<1x384x768xf32>) -> tensor<1x384x768xf32> loc(#loc26)
    %35 = linalg.generic {indexing_maps = [#map, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %cst_0 : tensor<1x384x768xf32>, tensor<1x1x768xf32>) outs(%0 : tensor<1x384x768xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc2)), %in_10: f32 loc("aten::add"(#loc2)), %out: f32 loc("aten::add"(#loc2))):
      %38 = arith.addf %in, %in_10 : f32 loc(#loc27)
      linalg.yield %38 : f32 loc(#loc27)
    } -> tensor<1x384x768xf32> loc(#loc27)
    %36 = linalg.generic {indexing_maps = [#map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : tensor<i64>) outs(%0 : tensor<1x384x768xf32>) {
    ^bb0(%in: i64 loc("/workspace/torch-mlir/examples/attention.py":57:0), %out: f32 loc("aten::add"(#loc1))):
      %38 = arith.sitofp %in : i64 to f32 loc(#loc28)
      linalg.yield %38 : f32 loc(#loc28)
    } -> tensor<1x384x768xf32> loc(#loc28)
    %37 = linalg.generic {__linalg_tiling__ = array<i64: 0, 64, 32>, indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %36 : tensor<1x384x768xf32>, tensor<1x384x768xf32>) outs(%0 : tensor<1x384x768xf32>) {
    ^bb0(%in: f32 loc("aten::add"(#loc2)), %in_10: f32 loc("aten::add"(#loc1)), %out: f32 loc("aten::matmul"(#loc4))):
      %38 = arith.addf %in, %in_10 : f32 loc(#loc28)
      linalg.yield %38 : f32 loc(#loc28)
    } -> tensor<1x384x768xf32> loc(#loc28)
    return %37 : tensor<1x384x768xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
