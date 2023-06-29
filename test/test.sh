model_transform.py --model_name test --model_def ./test_model.prototxt --model_data ./test_model.caffemodel --input_shapes [[1,3,14,14]] --test_input ./test_model_in_f32.npz --test_result ./test.npz --mlir test.mlir


model_deploy.py --mlir test.mlir --quantize F32 --chip bm1684x --test_input test_in_f32.npz --test_reference test_in_f32.npz --model test.bmodel


# task2
 model_transform.py --model_name task2 --model_def ./test_model2.onnx --input_shapes [[1,16,28,28]] --test_input ./test_model2_in_f32.npz --test_result ./task2.npz --mlir task2.mlir
