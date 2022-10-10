![](./assets/sophgo_chip.png)



# Purpose

As shown in [README.md](../README.md), `model_transform.py` and `model_deploy.py` are wrapper scripts for getting started quickly.

For more detailed instructions, this guide is showing step-by-step work flows of porting a model.

you can refer to script: [step_by_step](../regression/basic/run_step_by_step.sh)

# Build

Refer to [README.md](../README.md).

```bash
docker pull sophgo/tpuc_dev:latest

cd sophgo-mlir

# w/o dataset
docker run --name mymlir \
    -v $PWD:/workspace \
    -it sophgo/tpuc_dev:latest

# w/ dataset mount as /dataset, for model eval
docker run --name mymlir \
    -v $PWD:/workspace \
    -v /data/dataset:/dataset \
    -it sophgo/tpuc_dev:latest

# to build
source ./envsetup.sh
./build.sh

# to cleanup
rm ./build -rf
docker stop mymlir
docker rm mymlir
```


# Tutorial 1. Port an ONNX model with F32 mode

## step 0. Prepare

Create a clean work dir.

```bash
mkdir work_dir
cd work_dir
```

### 1. Copy `resnet18.onnx` to the current dir

```bash
cp ../regression/resnet18.onnx ./
```

### 2. Copy an image for testing

```bash
cp ../regression/image/cat.jpg ./
# feh cat.jpg
```

### 3. Generate the input data for testing (in npz format) from the image

`model_transform.py` will generate this file (resnet18_in_f32.npz) if --test_input argument is provided with a jpg file.

<mark>TODO: to add a specific tool for input data convesion, and support preprocessing arguments.</mark>

```bash
# use the pre-generated resnet18_in_f32.npz file for time being
cp ../regression/resnet18_in_f32.npz ./
# to examine the data
npz_tool.py dump resnet18_in_f32.npz
npz_tool.py dump resnet18_in_f32.npz input
# shape (1, 3, 224, 224)
# dtype float32
# max [2.245657 2.4276   2.635578]
# min [-2.063543 -1.9474   -1.801422]
# mean [ 0.125982  0.399004 -0.288606]
# abs mean fp32 [0.897471 0.641968 0.865129]
# std fp32 [1.06295  0.71965  0.944557]
```

### 4. Prepare dataset for calibration

<mark>TODO: to add a specific tool for generating `image_list.txt` from real dataset, randomly select files according to `--input_num`.</mark>

Right now use the regression/image dir, which has one 2 image files, and construct image_list.txt with find.

```bash
mkdir ./dataset_cali
cp ../regression/image/*.jpg ./dataset_cali/
find ./dataset_cali -type f -name "*.jpg" > ./dataset_cali/image_list.txt
```

### 5. Prepare dataset for eval

Download `ILSVRC2012_img_val.tar` from `http://image-net.org/download-images`.

Check integrity and precess with [valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

Assuming, `ILSVRC2012_img_val.tar` is in /dataset/ dir as we have mounted.

```bash
cd /dataset

# check integrity
md5sum -c <(echo "29b22e2961454d5413ddabcf34fc5622 ILSVRC2012_img_val.tar")

# get valprep.sh
# wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
cp /work/regression/valprep.sh ./

# unzip and build folder tree
mkdir -p imagenet/val
tar -xvf ILSVRC2012_img_val.tar -C ./imagenet/val
pushd imagenet/val;bash ../../valprep.sh; popd
```

## step 1. Import the model into MLIR TOP Dialect, and run test

As a wrapper, `model_transform.py` can do importing, testing and result comparing in one command.

```bash
# no need to run this, if run detailed steps separately
model_transform.py \
    --model_name resnet18 \
    --model_def ./resnet18.onnx \
    --input_shapes [[1,3,224,224]] \
    --resize_dims 256,256 \
    --mean 123.675,116.28,103.53 \
    --scale 0.0171,0.0175,0.0174 \
    --pixel_format rgb \
    --test_input ./cat.jpg \
    --test_result resnet18_f32_outputs.npz \
    --tolerance 0.99,0.99 \
    --mlir resnet18.mlir
```

Break into detailed steps.

### 1. Run `model_transform.py` w/o `--test_input` argument, do importing only

Note: preprocess arguments are provided, and will be stored in the IR. Usage includes calibration, accruacy regression, runtime info retrieval, and potential TPU preprocessing support.

```bash
model_transform.py \
    --model_name resnet18 \
    --model_def ./resnet18.onnx \
    --input_shapes [[1,3,224,224]] \
    --resize_dims 256,256 \
    --mean 123.675,116.28,103.53 \
    --scale 0.0171,0.0175,0.0174 \
    --pixel_format rgb \
    --mlir resnet18.mlir
```

Result in `resnet18.mlir`, which is a human readable file

```bash
more resnet18.mlir
```

And weights are stored in a npz file `resnet18_top_f32_all_weight.npz`, which can be examined by `npz_tool.py`

```bash
npz_tool.py dump resnet18_top_f32_all_weight.npz                  # to show the list
npz_tool.py dump resnet18_top_f32_all_weight.npz fc.weight_fix    # to show tensor fc.weight_fix
```

### 2. Test both the original ONNX model and the imported MLIR model, then do comparing tensor-by-tensor

Run inference with the ONNX model

```bash
# before transform, opt has been applied to the onnx model
# therefore we run the onnx inference with the optimized model ./resnet18_opt.onnx
# the opt is done by onnxsim (pip install onnx-simplifier)
model_runner.py \
    --model ./resnet18_opt.onnx \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_all_tensors_onnx.npz

npz_tool.py dump resnet18_all_tensors_onnx.npz output_Gemm
```

Run inference with the imported MLIR model

```bash
model_runner.py \
    --model ./resnet18.mlir \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_all_tensors_mlir_f32.npz

npz_tool.py dump resnet18_all_tensors_mlir_f32.npz output_Gemm
```

Compare the results

```bash
npz_tool.py compare \
    resnet18_all_tensors_mlir_f32.npz \
    resnet18_all_tensors_onnx.npz \
    --tolerance 0.999,0.999
```

Check Top-K for correctness

```bash
npz_tool.py dump resnet18_all_tensors_mlir_f32.npz output_Gemm 5
# shape (1, 1000)
# dtype float32
# Show Top-K 5
# (281, 10.266102)
# (282, 9.682159)
# (285, 8.585421)
# (287, 7.7521954)
# (463, 7.3861647)
```

According to `https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/`, 281 is a lovely `tabby cat`, which is correct, and prob of 282 (`tiger cat`) is very close to 281, which is reasonable.

## step 2. Generate bmodel with F32, and run test

### 1. Generate bmodel with backend optimzations

```bash
tpuc-opt resnet18.mlir \
    --convert-top-to-tpu="mode=F32 chip=bm1684x" \
    --save-weight \
    -o resnet18_f32_1684x.mlir

tpuc-opt resnet18_f32_1684x.mlir \
    --weight-reorder \
    --subnet-divide \
    --layer-group \
    --address-assign \
    --save-weight \
    --codegen="model_file=resnet18_f32_1684x.bmodel" \
    -o resnet18_f32_1684x_final.mlir
```


<mark>TODO: if give mode=FP32, or any invalid mode, will core dump and gives no hints, refine the error messages</mark>

### 2. Run test with the bmodel simulator

This is calling the simulation

```bash
model_runner.py \
    --model resnet18_f32_1684x.bmodel \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_all_tensors_f32_1684x_bmodel.npz
```

<mark>TODO: to dump profiling info when running simulation</mark>

Compare the results against the mlir interpreter results

```bash
npz_tool.py compare \
    resnet18_all_tensors_f32_1684x_bmodel.npz \
    resnet18_all_tensors_mlir_f32.npz \
    --tolerance 0.999,0.999 \
    -v
```

### 3. Run bmodel with the TPU hardware

<mark>TODO: todo</mark>

<mark>TODO: to dump PMU profiling info as well</mark>


# Tutorial 2. Port an ONNX model with INT8 Symmetirc mode

## step 0. Prepare

Same as [Tutorial 1. step 0. Prepare](#step-0-prepare).

## step 1. Import the model into MLIR TOP Dialect, and run test

Same as [Tutorial 1. step 1. Import Model](#step-1-import-the-model-into-mlir-top-dialect-and-run-test).

## step 2. Calibration for quantization

This step is to generate a table by running statistics upon inference results of the dataset images. For each tensor, 3 stats are collected
- threshold
- min
- max

This table is used by both symmetric and asymmetric quantization.

Use the dataset_cali dir, and the `image_list.txt` file describes the image file list.

```bash
run_calibration.py resnet18.mlir \
    --data_list ./dataset_cali/image_list.txt \
    --input_num 2 \
    -o resnet18_cali_table

# resnet18_cali_table is a txt file
cat resnet18_cali_table
```

Another way is to use `--dataset` argument, pointing to a dataset dir, e.g. the imagenet dataset. The calibration program will randomly select `--input_num` number of images from the dir.

```bash
run_calibration.py resnet18.mlir \
    --dataset /dataset/imagenet/val \
    --input_num 100 \
    -o resnet18_cali_table_100

# resnet18_cali_table is a txt file
cat resnet18_cali_table_100
```

## step 3. Quantization, and run test and eval

### 1. Import cali table and do quantization

```bash
tpuc-opt resnet18.mlir \
    --import-calibration-table='file=resnet18_cali_table asymmetric=false' \
    --convert-top-to-tpu="mode=INT8 asymmetric=false chip=bm1684x" \
    --save-weight \
    -o resnet18_int8_sym_1684x.mlir
```

<mark>TODO: resnet18_tpu_lowered_bm1684x_weight.npz is a hardcoded file name, could be overwritten if run multiple lowering simutaneously.</mark>

### 2. Run test with the quantized mlir model

<mark>TODO: this is convenient but a little strange, as it takes fp32 file as input. The reason we can do this is because we stored the preprecess info in the IR. Better to make it more precise with 2 separate steps, one for input data quantization, and the model take int8 input directly (except for the case that we are doing TPU preprocessing, and take fp32 input directly).</mark>

```bash
model_runner.py \
    --model resnet18_int8_sym_1684x.mlir \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_all_tensors_int8_sym_1684x.npz
```

Compare the results against the f32 results.

```bash
npz_tool.py compare \
    resnet18_all_tensors_int8_sym_1684x.npz \
    resnet18_all_tensors_mlir_f32.npz \
    --tolerance 0.95,0.70 \
    -v
```

Check Top-K

```bash
npz_tool.py dump resnet18_all_tensors_int8_sym_1684x.npz output_Gemm 5
```

### 3. Run eval with the quantized mlir model

The preprocess info is stored in the MLIR IR.

```bash
# eval with 1000 images
model_eval_imagenet.py \
    --model resnet18_int8_sym_1684x.mlir \
    --dataset /dataset/imagenet/val \
    --count 1000

# eval with the full dataset, which is 50000
model_eval_imagenet.py \
    --model resnet18_int8_sym_1684x.mlir \
    --dataset /dataset/imagenet/val
```

This tutorial model will get accuracy around `Acc@1 100.00 ( 60.93)	Acc@5 100.00 ( 87.34)`, because we only used 2 images for calibration. Later we will show how to improve the accuracy.

To compare with F32 model, we can run the eval with the f32 mlir model (assuming we have done Tutorial 1)

```bash
model_eval_imagenet.py \
    --model resnet18.mlir \
    --dataset /dataset/imagenet/val
```

The F32 model is around `Acc@1 100.00 ( 67.76)	Acc@5 100.00 ( 88.22)`.

<mark>TODO: model_eval_imagenet.py support mlir only for now, need to support onnx, tflite, and bmodel</mark>

<mark>TODO: bmodel needs to contain preprocess info</mark>

## step 4. Generate bmodel with INT8 symmetric quantization, and run test

### 1. Generate bmodel with backend optimzations

```bash
tpuc-opt resnet18_int8_sym_1684x.mlir \
    --weight-reorder \
    --subnet-divide \
    --layer-group \
    --address-assign \
    --save-weight \
    --codegen="model_file=resnet18_int8_sym_1684x.bmodel" \
    -o resnet18_int8_sym_1684x_final.mlir
```

### 2. Run test with the bmodel simulator

```bash
model_runner.py \
    --model resnet18_int8_sym_1684x.bmodel \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_all_tensors_int8_sym_1684x_bmodel.npz
```

<mark>TODO: to dump profiling info when running simulation</mark>

Compare the results against the mlir interpreter results

```bash
npz_tool.py compare \
    resnet18_all_tensors_int8_sym_1684x_bmodel.npz \
    resnet18_all_tensors_int8_sym_1684x.npz \
    --tolerance 0.97,0.85 \
    -v
```

Check Top-K

```bash
npz_tool.py dump resnet18_all_tensors_int8_sym_1684x_bmodel.npz output_Gemm_f32 5
```

<mark>TODO: output are still in f32, need to expose `express_type` to command line.</mark>

<mark>TODO: Even with `express_type` been set properly, the mlir interpreter is not bit-accurate with the bmodel simulator (the HW dehavior) for now, which means it is not capable of verifying the accuracy of the model yet. This is an important issue, and should be solved ASAP. </mark>

### 3. Run bmodel with the TPU hardware

<mark>TODO: todo</mark>


# Tutorial 3. Port an ONNX model with INT8 Asymmetirc mode

## step 0. Prepare

Same as [Tutorial 1. step 0. Prepare](#step-0-prepare).

## step 1. Import the model into MLIR TOP Dialect, and run test

Same as [Tutorial 1. step 1. Import Model](#step-1-import-the-model-into-mlir-top-dialect-and-run-test).

## step 2. Calibration for quantization

Same as [Tutorial 2. step 2. Calibration](#step-2-calibration-for-quantization)

## step 3. Quantization, and run test and eval

### 1. Import cali table and do quantization

```bash
tpuc-opt resnet18.mlir \
    --import-calibration-table='file=resnet18_cali_table asymmetric=true' \
    --convert-top-to-tpu="mode=INT8 asymmetric=true chip=bm1684x" \
    --save-weight \
    -o resnet18_int8_asym_1684x.mlir
```

### 2. Run test with the quantized mlir model

```bash
model_runner.py \
    --model resnet18_int8_asym_1684x.mlir \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_all_tensors_int8_asym_1684x.npz
```

Compare the results against the f32 results (or onnx results)

```bash
npz_tool.py compare \
    resnet18_all_tensors_int8_asym_1684x.npz \
    resnet18_all_tensors_mlir_f32.npz \
    --tolerance 0.97,0.75 \
    -v
```

Check Top-K

```bash
npz_tool.py dump resnet18_all_tensors_int8_asym_1684x.npz output_Gemm 5
```

### 3. Run eval with the quantized mlir model

The preprocess info is stored in the MLIR IR.

```bash
# eval with 1000 images
model_eval_imagenet.py \
    --model resnet18_int8_asym_1684x.mlir \
    --dataset /dataset/imagenet/val \
    --count 1000

# eval with the full dataset, which is 50000
model_eval_imagenet.py \
    --model resnet18_int8_asym_1684x.mlir \
    --dataset /dataset/imagenet/val
```

## step 4. Generate bmodel with INT8 Asymmetric quantization, and run test

### 1. Generate bmodel with backend optimzations

```bash
tpuc-opt resnet18_int8_asym_1684x.mlir \
    --weight-reorder \
    --subnet-divide \
    --layer-group \
    --address-assign \
    --save-weight \
    --codegen="model_file=resnet18_int8_asym_1684x.bmodel" \
    -o resnet18_int8_asym_1684x_final.mlir
```

### 2. Run test with the bmodel simulator

```bash
model_runner.py \
    --model resnet18_int8_asym_1684x.bmodel \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_all_tensors_int8_asym_1684x_bmodel.npz
```

Compare the results against the mlir interpreter results

```bash
npz_tool.py compare \
    resnet18_all_tensors_int8_asym_1684x_bmodel.npz \
    resnet18_all_tensors_int8_asym_1684x.npz \
    --tolerance 0.97,0.80 \
    -v
```

Check Top-K

```bash
npz_tool.py dump resnet18_all_tensors_int8_asym_1684x_bmodel.npz output_Gemm_f32 5
```

### 3. Run bmodel with the TPU hardware

<mark>TODO: todo</mark>


# Tutorial 4. Port an ONNX model with F16 mode

## step 0. Prepare

Same as [Tutorial 1. step 0. Prepare](#step-0-prepare).

## step 1. Import the model into MLIR TOP Dialect, and run test

Same as [Tutorial 1. step 1. Import Model](#step-1-import-the-model-into-mlir-top-dialect-and-run-test).

## step 2. Quantization with F16, and run test and eval

This does not need the calibration table

### 1. Do quantization

```bash
tpuc-opt resnet18.mlir \
    --convert-top-to-tpu="mode=F16 chip=bm1684x" \
    --save-weight \
    -o resnet18_f16_1684x.mlir
```

### 2. Run test with the quantized mlir model

```bash
model_runner.py \
    --model resnet18_f16_1684x.mlir \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_all_tensors_f16_1684x.npz
```

Compare the results against the f32 results (or onnx results)

```bash
npz_tool.py compare \
    resnet18_all_tensors_f16_1684x.npz \
    resnet18_all_tensors_mlir_f32.npz \
    --tolerance 0.99,0.93 \
    -v
```

Check Top-K

```bash
npz_tool.py dump resnet18_all_tensors_f16_1684x.npz output_Gemm 5
```

### 3. Run eval with the quantized mlir model

The preprocess info is stored in the MLIR IR.

```bash
# eval with 1000 images
model_eval_imagenet.py \
    --model resnet18_f16_1684x.mlir \
    --dataset /dataset/imagenet/val \
    --count 1000

# eval with the full dataset, which is 50000
model_eval_imagenet.py \
    --model resnet18_f16_1684x.mlir \
    --dataset /dataset/imagenet/val
```

## step 3. Generate bmodel with F16, and run test

<mark>TODO: todo</mark>


# Tutorial 5. Port an ONNX model with BF16 mode

## step 0. Prepare

Same as [Tutorial 1. step 0. Prepare](#step-0-prepare).

## step 1. Import the model into MLIR TOP Dialect, and run test

Same as [Tutorial 1. step 1. Import Model](#step-1-import-the-model-into-mlir-top-dialect-and-run-test).

## step 2. Quantization with BF16, and run test and eval

This does not need the calibration table

### 1. Do quantization

```bash
tpuc-opt resnet18.mlir \
    --convert-top-to-tpu="mode=BF16 chip=bm1684x" \
    --save-weight \
    -o resnet18_bf16_1684x.mlir
```

### 2. Run test with the quantized mlir model

```bash
model_runner.py \
    --model resnet18_bf16_1684x.mlir \
    --input resnet18_in_f32.npz \
    --dump_all_tensors \
    --output resnet18_all_tensors_bf16_1684x.npz
```

Compare the results against the f32 results (or onnx results)

```bash
npz_tool.py compare \
    resnet18_all_tensors_bf16_1684x.npz \
    resnet18_all_tensors_mlir_f32.npz \
    --tolerance 0.99,0.87 \
    -v
```

Check Top-K

```bash
npz_tool.py dump resnet18_all_tensors_bf16_1684x.npz output_Gemm 5
```

### 3. Run eval with the quantized mlir model

The preprocess info is stored in the MLIR IR.

```bash
# eval with 1000 images
model_eval_imagenet.py \
    --model resnet18_bf16_1684x.mlir \
    --dataset /dataset/imagenet/val \
    --count 1000

# eval with the full dataset, which is 50000
model_eval_imagenet.py \
    --model resnet18_bf16_1684x.mlir \
    --dataset /dataset/imagenet/val
```

## step 3. Generate bmodel with Bf16, and run test

<mark>TODO: todo</mark>


# Tutorial 6. Optimize quantization accuracy by refining threshold

This Tutorial shows basic steps of optimizing quantization accuracy, including enlarging calibration dataset size, and threshold finetuning.

## step 1. enlarge calibration dataset

Use the imagenet val dataset, and randomly select `--input_num` number of images as calibration dataset.

```bash
# to use 1000 images
run_calibration.py resnet18.mlir \
    --dataset /dataset/imagenet/val \
    --input_num 1000 \
    -o resnet18_cali_table_1000
```

<mark>TODO: add progress bar for inferencing</mark>

<mark>TODO: Optimize for memory usage, killed if dataset too large, may need to consider calibrate layer by layer, or batch of layers (run multiple times inferencing for each image though)</mark>

For reference, accuracy of different mode is collected here.

| Mode             |  Top-1   |  Top-5  |
|------------------|----------|---------|
| F32              |   67.76  |   88.22 |
| F16              |   67.68  |   88.21 |
| BF16             |   67.35  |   87.88 |
| INT8_Sym(1000)   |   67.28  |   87.79 |
| INT8_Asym(1000)  |   65.88  |   86.95 |

## step 2. threshold finetuning

<mark>TODO: todo</mark>


# Tutorial 7. Optimize quantization accuracy by mix-precision

This Tutorial shows steps of optimizing quantization accuracy by using mix-precision quantization.

<mark>TODO: todo</mark>


# Tutorial 8. Port a TFLite model and do quantization

<mark>TODO: todo</mark>


# Tutorial 9. Port a pre-quantized INT8 Asymmetric TFLite model

<mark>TODO: todo</mark>


# Tutorial 10. Port a Caffe model and do quantization

<mark>TODO: todo</mark>
