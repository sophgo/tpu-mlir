// RUN: tpuc-test %s  --codegen-decompose-linalg-generic --mlir-elide-elementsattrs-if-larger=20 --mlir-print-debuginfo --cse -o %t

#loc = loc(unknown)
#loc1 = loc("/workspace/torch-mlir/examples/attention.py":57:0)
#loc2 = loc("/workspace/torch-mlir/examples/attention.py":56:0)
#loc3 = loc("/workspace/torch-mlir/examples/attention.py":50:0)
#loc4 = loc("/workspace/torch-mlir/examples/attention.py":46:0)
#loc5 = loc("/workspace/torch-mlir/examples/attention.py":41:0)
#loc6 = loc("/workspace/torch-mlir/examples/attention.py":38:0)
#loc7 = loc("/workspace/torch-mlir/examples/attention.py":49:0)
#loc8 = loc("/workspace/torch-mlir/examples/attention.py":39:0)
#loc9 = loc("/workspace/torch-mlir/examples/attention.py":40:0)
#loc10 = loc("/workspace/torch-mlir/examples/attention.py":42:0)
#loc11 = loc("/workspace/torch-mlir/examples/attention.py":43:0)
#loc12 = loc("/workspace/torch-mlir/examples/attention.py":44:0)
#loc13 = loc("/workspace/torch-mlir/examples/attention.py":45:0)
#loc14 = loc("/workspace/torch-mlir/examples/attention.py":51:0)
#loc15 = loc("/workspace/torch-mlir/examples/attention.py":52:0)
#loc16 = loc("/workspace/torch-mlir/examples/attention.py":53:0)
#loc17 = loc("/workspace/torch-mlir/examples/attention.py":54:0)
#loc18 = loc("/workspace/torch-mlir/examples/attention.py":55:0)
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
#loc19 = loc("aten::matmul"(#loc6))
#loc20 = loc("aten::softmax"(#loc7))
#loc21 = loc("aten::add"(#loc6))
#loc22 = loc("aten::reshape"(#loc8))
#loc23 = loc("aten::transpose"(#loc9))
#loc24 = loc("aten::matmul"(#loc5))
#loc25 = loc("aten::add"(#loc5))
#loc26 = loc("aten::reshape"(#loc10))
#loc27 = loc("aten::transpose"(#loc11))
#loc28 = loc("aten::transpose"(#loc12))
#loc29 = loc("aten::matmul"(#loc13))
#loc30 = loc("aten::div"(#loc4))
#loc31 = loc("aten::matmul"(#loc3))
#loc32 = loc("aten::add"(#loc3))
#loc33 = loc("aten::reshape"(#loc14))
#loc34 = loc("aten::transpose"(#loc15))
#loc35 = loc("aten::matmul"(#loc16))
#loc36 = loc("aten::transpose"(#loc17))
#loc37 = loc("aten::reshape"(#loc18))
#loc38 = loc("aten::matmul"(#loc2))
#loc39 = loc("aten::add"(#loc2))
#loc40 = loc("aten::add"(#loc1))
module attributes {torch.debug_module_name = "Model"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64> loc(#loc)
  func.func @forward(%arg0: tensor<1x384x768xf32> loc(unknown)) -> tensor<1x384x768xf32> {
    %cst = arith.constant dense<1> : tensor<i64> loc(#loc1)
    %cst_0 = arith.constant dense_resource<__elided__> : tensor<1x1x768xf32> loc(#loc2)
    %cst_1 = arith.constant dense_resource<__elided__> : tensor<768x768xf32> loc(#loc2)
    %cst_2 = arith.constant dense_resource<__elided__> : tensor<1x1x768xf32> loc(#loc3)
    %cst_3 = arith.constant dense_resource<__elided__> : tensor<768x768xf32> loc(#loc3)
    %cst_4 = arith.constant dense<8.000000e+00> : tensor<f64> loc(#loc4)
    %cst_5 = arith.constant dense_resource<__elided__> : tensor<1x1x768xf32> loc(#loc5)
    %cst_6 = arith.constant dense_resource<__elided__> : tensor<768x768xf32> loc(#loc5)
    %cst_7 = arith.constant dense_resource<__elided__> : tensor<1x1x768xf32> loc(#loc6)
    %cst_8 = arith.constant dense_resource<__elided__> : tensor<768x768xf32> loc(#loc6)
    %c0_i64 = arith.constant 0 : i64 loc(#loc19)
    %cst_9 = arith.constant 0.000000e+00 : f32 loc(#loc19)
    %cst_10 = arith.constant 0xFF800000 : f32 loc(#loc20)
    %0 = tensor.empty() : tensor<1x384x768xf32> loc(#loc19)
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x384x768xf32>) outs(%0 : tensor<1x384x768xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc6)), %out: f32 loc("aten::matmul"(#loc6))):
      linalg.yield %in : f32 loc(#loc19)
    } -> tensor<1x384x768xf32> loc(#loc19)
    %2 = tensor.empty() : tensor<1x768x768xf32> loc(#loc19)
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_8 : tensor<768x768xf32>) outs(%2 : tensor<1x768x768xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc6)), %out: f32 loc("aten::matmul"(#loc6))):
      linalg.yield %in : f32 loc(#loc19)
    } -> tensor<1x768x768xf32> loc(#loc19)
    %4 = linalg.fill ins(%cst_9 : f32) outs(%0 : tensor<1x384x768xf32>) -> tensor<1x384x768xf32> loc(#loc19)
    %5 = linalg.batch_matmul ins(%1, %3 : tensor<1x384x768xf32>, tensor<1x768x768xf32>) outs(%4 : tensor<1x384x768xf32>) -> tensor<1x384x768xf32> loc(#loc19)
    %6 = linalg.generic {indexing_maps = [#map, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %cst_7 : tensor<1x384x768xf32>, tensor<1x1x768xf32>) outs(%0 : tensor<1x384x768xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc6)), %in_19: f32 loc("aten::add"(#loc6)), %out: f32 loc("aten::add"(#loc6))):
      %48 = arith.addf %in, %in_19 : f32 loc(#loc21)
      linalg.yield %48 : f32 loc(#loc21)
    } -> tensor<1x384x768xf32> loc(#loc21)
    %expanded = tensor.expand_shape %6 [[0], [1], [2, 3]] : tensor<1x384x768xf32> into tensor<1x384x12x64xf32> loc(#loc22)
    %7 = tensor.empty() : tensor<1x12x384x64xf32> loc(#loc23)
    %8 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x384x12x64xf32>) outs(%7 : tensor<1x12x384x64xf32>) {
    ^bb0(%in: f32 loc("aten::reshape"(#loc8)), %out: f32 loc("aten::transpose"(#loc9))):
      linalg.yield %in : f32 loc(#loc23)
    } -> tensor<1x12x384x64xf32> loc(#loc23)
    %9 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_6 : tensor<768x768xf32>) outs(%2 : tensor<1x768x768xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc5)), %out: f32 loc("aten::matmul"(#loc5))):
      linalg.yield %in : f32 loc(#loc24)
    } -> tensor<1x768x768xf32> loc(#loc24)
    %10 = linalg.batch_matmul ins(%1, %9 : tensor<1x384x768xf32>, tensor<1x768x768xf32>) outs(%4 : tensor<1x384x768xf32>) -> tensor<1x384x768xf32> loc(#loc24)
    %11 = linalg.generic {indexing_maps = [#map, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %cst_5 : tensor<1x384x768xf32>, tensor<1x1x768xf32>) outs(%0 : tensor<1x384x768xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc5)), %in_19: f32 loc("aten::add"(#loc5)), %out: f32 loc("aten::add"(#loc5))):
      %48 = arith.addf %in, %in_19 : f32 loc(#loc25)
      linalg.yield %48 : f32 loc(#loc25)
    } -> tensor<1x384x768xf32> loc(#loc25)
    %expanded_11 = tensor.expand_shape %11 [[0], [1], [2, 3]] : tensor<1x384x768xf32> into tensor<1x384x12x64xf32> loc(#loc26)
    %12 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_11 : tensor<1x384x12x64xf32>) outs(%7 : tensor<1x12x384x64xf32>) {
    ^bb0(%in: f32 loc("aten::reshape"(#loc10)), %out: f32 loc("aten::transpose"(#loc11))):
      linalg.yield %in : f32 loc(#loc27)
    } -> tensor<1x12x384x64xf32> loc(#loc27)
    %13 = tensor.empty() : tensor<1x12x64x384xf32> loc(#loc28)
    %14 = linalg.generic {indexing_maps = [#map4, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : tensor<1x12x384x64xf32>) outs(%13 : tensor<1x12x64x384xf32>) {
    ^bb0(%in: f32 loc("aten::transpose"(#loc11)), %out: f32 loc("aten::transpose"(#loc12))):
      linalg.yield %in : f32 loc(#loc28)
    } -> tensor<1x12x64x384xf32> loc(#loc28)
    %15 = linalg.generic {indexing_maps = [#map7, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8 : tensor<1x12x384x64xf32>) outs(%7 : tensor<1x12x384x64xf32>) {
    ^bb0(%in: f32 loc("aten::transpose"(#loc9)), %out: f32 loc("aten::matmul"(#loc13))):
      linalg.yield %in : f32 loc(#loc29)
    } -> tensor<1x12x384x64xf32> loc(#loc29)
    %16 = linalg.generic {indexing_maps = [#map7, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<1x12x64x384xf32>) outs(%13 : tensor<1x12x64x384xf32>) {
    ^bb0(%in: f32 loc("aten::transpose"(#loc12)), %out: f32 loc("aten::matmul"(#loc13))):
      linalg.yield %in : f32 loc(#loc29)
    } -> tensor<1x12x64x384xf32> loc(#loc29)
    %collapsed = tensor.collapse_shape %15 [[0, 1], [2], [3]] : tensor<1x12x384x64xf32> into tensor<12x384x64xf32> loc(#loc29)
    %collapsed_12 = tensor.collapse_shape %16 [[0, 1], [2], [3]] : tensor<1x12x64x384xf32> into tensor<12x64x384xf32> loc(#loc29)
    %17 = tensor.empty() : tensor<12x384x384xf32> loc(#loc29)
    %18 = linalg.fill ins(%cst_9 : f32) outs(%17 : tensor<12x384x384xf32>) -> tensor<12x384x384xf32> loc(#loc29)
    %19 = linalg.batch_matmul ins(%collapsed, %collapsed_12 : tensor<12x384x64xf32>, tensor<12x64x384xf32>) outs(%18 : tensor<12x384x384xf32>) -> tensor<12x384x384xf32> loc(#loc29)
    %expanded_13 = tensor.expand_shape %19 [[0, 1], [2], [3]] : tensor<12x384x384xf32> into tensor<1x12x384x384xf32> loc(#loc29)
    %20 = tensor.empty() : tensor<1x12x384x384xf32> loc(#loc30)
    %21 = linalg.generic {indexing_maps = [#map7, #map8, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_13, %cst_4 : tensor<1x12x384x384xf32>, tensor<f64>) outs(%20 : tensor<1x12x384x384xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc13)), %in_19: f64 loc("aten::div"(#loc4)), %out: f32 loc("aten::div"(#loc4))):
      %48 = arith.truncf %in_19 : f64 to f32 loc(#loc30)
      %49 = arith.divf %in, %48 : f32 loc(#loc30)
      linalg.yield %49 : f32 loc(#loc30)
    } -> tensor<1x12x384x384xf32> loc(#loc30)
    %22 = tensor.empty() : tensor<1x12x384x1xi64> loc(#loc20)
    %23 = linalg.fill ins(%c0_i64 : i64) outs(%22 : tensor<1x12x384x1xi64>) -> tensor<1x12x384x1xi64> loc(#loc20)
    %24 = tensor.empty() : tensor<1x12x384x1xf32> loc(#loc20)
    %25 = linalg.fill ins(%cst_10 : f32) outs(%24 : tensor<1x12x384x1xf32>) -> tensor<1x12x384x1xf32> loc(#loc20)
    %26:2 = linalg.generic {indexing_maps = [#map4, #map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%21 : tensor<1x12x384x384xf32>) outs(%25, %23 : tensor<1x12x384x1xf32>, tensor<1x12x384x1xi64>) {
    ^bb0(%in: f32 loc("aten::div"(#loc4)), %out: f32 loc("aten::softmax"(#loc7)), %out_19: i64 loc("aten::softmax"(#loc7))):
      %48 = linalg.index 3 : index loc(#loc20)
      %49 = arith.index_cast %48 : index to i64 loc(#loc20)
      %50 = arith.maxf %in, %out : f32 loc(#loc20)
      %51 = arith.cmpf ogt, %in, %out : f32 loc(#loc20)
      %52 = arith.select %51, %49, %out_19 : i64 loc(#loc20)
      linalg.yield %50, %52 : f32, i64 loc(#loc20)
    } -> (tensor<1x12x384x1xf32>, tensor<1x12x384x1xi64>) loc(#loc20)
    %27 = linalg.generic {indexing_maps = [#map7, #map10, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21, %26#0 : tensor<1x12x384x384xf32>, tensor<1x12x384x1xf32>) outs(%20 : tensor<1x12x384x384xf32>) {
    ^bb0(%in: f32 loc("aten::div"(#loc4)), %in_19: f32 loc("aten::softmax"(#loc7)), %out: f32 loc("aten::softmax"(#loc7))):
      %48 = arith.subf %in, %in_19 : f32 loc(#loc20)
      linalg.yield %48 : f32 loc(#loc20)
    } -> tensor<1x12x384x384xf32> loc(#loc20)
    %28 = linalg.generic {indexing_maps = [#map7, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27 : tensor<1x12x384x384xf32>) outs(%20 : tensor<1x12x384x384xf32>) {
    ^bb0(%in: f32 loc("aten::softmax"(#loc7)), %out: f32 loc("aten::softmax"(#loc7))):
      %48 = math.exp %in : f32 loc(#loc20)
      linalg.yield %48 : f32 loc(#loc20)
    } -> tensor<1x12x384x384xf32> loc(#loc20)
    %29 = linalg.fill ins(%cst_9 : f32) outs(%24 : tensor<1x12x384x1xf32>) -> tensor<1x12x384x1xf32> loc(#loc20)
    %30 = linalg.generic {indexing_maps = [#map4, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%28 : tensor<1x12x384x384xf32>) outs(%29 : tensor<1x12x384x1xf32>) {
    ^bb0(%in: f32 loc("aten::softmax"(#loc7)), %out: f32 loc("aten::softmax"(#loc7))):
      %48 = arith.addf %in, %out : f32 loc(#loc20)
      linalg.yield %48 : f32 loc(#loc20)
    } -> tensor<1x12x384x1xf32> loc(#loc20)
    %31 = linalg.generic {indexing_maps = [#map7, #map10, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %30 : tensor<1x12x384x384xf32>, tensor<1x12x384x1xf32>) outs(%20 : tensor<1x12x384x384xf32>) {
    ^bb0(%in: f32 loc("aten::softmax"(#loc7)), %in_19: f32 loc("aten::softmax"(#loc7)), %out: f32 loc("aten::softmax"(#loc7))):
      %48 = arith.divf %in, %in_19 : f32 loc(#loc20)
      linalg.yield %48 : f32 loc(#loc20)
    } -> tensor<1x12x384x384xf32> loc(#loc20)
    %32 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_3 : tensor<768x768xf32>) outs(%2 : tensor<1x768x768xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc3)), %out: f32 loc("aten::matmul"(#loc3))):
      linalg.yield %in : f32 loc(#loc31)
    } -> tensor<1x768x768xf32> loc(#loc31)
    %33 = linalg.batch_matmul ins(%1, %32 : tensor<1x384x768xf32>, tensor<1x768x768xf32>) outs(%4 : tensor<1x384x768xf32>) -> tensor<1x384x768xf32> loc(#loc31)
    %34 = linalg.generic {indexing_maps = [#map, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %cst_2 : tensor<1x384x768xf32>, tensor<1x1x768xf32>) outs(%0 : tensor<1x384x768xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc3)), %in_19: f32 loc("aten::add"(#loc3)), %out: f32 loc("aten::add"(#loc3))):
      %48 = arith.addf %in, %in_19 : f32 loc(#loc32)
      linalg.yield %48 : f32 loc(#loc32)
    } -> tensor<1x384x768xf32> loc(#loc32)
    %expanded_14 = tensor.expand_shape %34 [[0], [1], [2, 3]] : tensor<1x384x768xf32> into tensor<1x384x12x64xf32> loc(#loc33)
    %35 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_14 : tensor<1x384x12x64xf32>) outs(%7 : tensor<1x12x384x64xf32>) {
    ^bb0(%in: f32 loc("aten::reshape"(#loc14)), %out: f32 loc("aten::transpose"(#loc15))):
      linalg.yield %in : f32 loc(#loc34)
    } -> tensor<1x12x384x64xf32> loc(#loc34)
    %36 = linalg.generic {indexing_maps = [#map7, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31 : tensor<1x12x384x384xf32>) outs(%20 : tensor<1x12x384x384xf32>) {
    ^bb0(%in: f32 loc("aten::softmax"(#loc7)), %out: f32 loc("aten::matmul"(#loc16))):
      linalg.yield %in : f32 loc(#loc35)
    } -> tensor<1x12x384x384xf32> loc(#loc35)
    %37 = linalg.generic {indexing_maps = [#map7, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35 : tensor<1x12x384x64xf32>) outs(%7 : tensor<1x12x384x64xf32>) {
    ^bb0(%in: f32 loc("aten::transpose"(#loc15)), %out: f32 loc("aten::matmul"(#loc16))):
      linalg.yield %in : f32 loc(#loc35)
    } -> tensor<1x12x384x64xf32> loc(#loc35)
    %collapsed_15 = tensor.collapse_shape %36 [[0, 1], [2], [3]] : tensor<1x12x384x384xf32> into tensor<12x384x384xf32> loc(#loc35)
    %collapsed_16 = tensor.collapse_shape %37 [[0, 1], [2], [3]] : tensor<1x12x384x64xf32> into tensor<12x384x64xf32> loc(#loc35)
    %38 = tensor.empty() : tensor<12x384x64xf32> loc(#loc35)
    %39 = linalg.fill ins(%cst_9 : f32) outs(%38 : tensor<12x384x64xf32>) -> tensor<12x384x64xf32> loc(#loc35)
    %40 = linalg.batch_matmul ins(%collapsed_15, %collapsed_16 : tensor<12x384x384xf32>, tensor<12x384x64xf32>) outs(%39 : tensor<12x384x64xf32>) -> tensor<12x384x64xf32> loc(#loc35)
    %expanded_17 = tensor.expand_shape %40 [[0, 1], [2], [3]] : tensor<12x384x64xf32> into tensor<1x12x384x64xf32> loc(#loc35)
    %41 = tensor.empty() : tensor<1x384x12x64xf32> loc(#loc36)
    %42 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_17 : tensor<1x12x384x64xf32>) outs(%41 : tensor<1x384x12x64xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc16)), %out: f32 loc("aten::transpose"(#loc17))):
      linalg.yield %in : f32 loc(#loc36)
    } -> tensor<1x384x12x64xf32> loc(#loc36)
    %collapsed_18 = tensor.collapse_shape %42 [[0], [1], [2, 3]] : tensor<1x384x12x64xf32> into tensor<1x384x768xf32> loc(#loc37)
    %43 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_18 : tensor<1x384x768xf32>) outs(%0 : tensor<1x384x768xf32>) {
    ^bb0(%in: f32 loc("aten::reshape"(#loc18)), %out: f32 loc("aten::matmul"(#loc2))):
      linalg.yield %in : f32 loc(#loc38)
    } -> tensor<1x384x768xf32> loc(#loc38)
    %44 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_1 : tensor<768x768xf32>) outs(%2 : tensor<1x768x768xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc2)), %out: f32 loc("aten::matmul"(#loc2))):
      linalg.yield %in : f32 loc(#loc38)
    } -> tensor<1x768x768xf32> loc(#loc38)
    %45 = linalg.batch_matmul ins(%43, %44 : tensor<1x384x768xf32>, tensor<1x768x768xf32>) outs(%4 : tensor<1x384x768xf32>) -> tensor<1x384x768xf32> loc(#loc38)
    %46 = linalg.generic {indexing_maps = [#map, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %cst_0 : tensor<1x384x768xf32>, tensor<1x1x768xf32>) outs(%0 : tensor<1x384x768xf32>) {
    ^bb0(%in: f32 loc("aten::matmul"(#loc2)), %in_19: f32 loc("aten::add"(#loc2)), %out: f32 loc("aten::add"(#loc2))):
      %48 = arith.addf %in, %in_19 : f32 loc(#loc39)
      linalg.yield %48 : f32 loc(#loc39)
    } -> tensor<1x384x768xf32> loc(#loc39)
    %47 = linalg.generic {__linalg_tiling__ = array<i64: 0, 64, 32>, indexing_maps = [#map, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %cst : tensor<1x384x768xf32>, tensor<i64>) outs(%0 : tensor<1x384x768xf32>) {
    ^bb0(%in: f32 loc("aten::add"(#loc2)), %in_19: i64 loc("aten::add"(#loc1)), %out: f32 loc("aten::add"(#loc1))):
      %48 = arith.sitofp %in_19 : i64 to f32 loc(#loc40)
      %49 = arith.addf %in, %48 : f32 loc(#loc40)
      linalg.yield %49 : f32 loc(#loc40)
    } -> tensor<1x384x768xf32> loc(#loc40)
    return %47 : tensor<1x384x768xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
