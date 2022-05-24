## sophgo-mlir

This project provides compiler to transform NN model into sophgo graph with sophgo runtime support.

Get more info from [sophgo](http://sophgo.com).

## start docker

* pull docker image from dockerhub

    ``` shell
    docker pull sophgo/sophgo_dev:1.1-ubuntu-18.04
    ```

* create a container, and run

    ``` shell
    # myname1234 just a example, you can set your own name
    docker run --privileged --name myname1234 -v $PWD:/work -it sophgo/sophgo_dev:1.1-ubuntu-18.04
    ```

## Build

``` shell
source ./envsetup.sh
./build.sh
```

## Test

``` shell
pushd regression
./run.sh
popd
```

## How to use toolchain

you can refer to regression/run_deploy.sh

### transform model to mlir (top)

``` shell
model_transform.py \
    --model_type onnx \
    --model_name resnet18 \
    --input_shapes [[1,3,224,224]] \
    --model_def  resnet18.onnx \
    --test_input resnet18_in.npz \
    --test_result resnet18_top_outputs.npz \
    --mlir resnet18.mlir
```

### deploy mlir to bmodel

##### To fp32 model

``` shell
model_deploy.py \
  --mlir resnet18.mlir \
  --quantize F32 \
  --chip bm1686 \
  --test_input resnet18_in.npz \
  --test_reference resnet18_top_outputs.npz \
  --tolerance 0.99,0.99 \
  --model resnet18_1686_f32.bmodel
```

##### To int8 model

prepare inputs in forder "dataset", and run calibration:

``` shell
run_calibration.py resnet18.mlir \
    --dataset dataset \
    --input_num 1 \
    -o resnet18_cali_table
```
###### to symmetric

``` shell
model_deploy.py \
  --mlir resnet18.mlir \
  --quantize INT8 \
  --calibration_table resnet18_cali_table \
  --chip bm1686 \
  --test_input resnet18_in.npz \
  --test_reference resnet18_top_outputs.npz \
  --tolerance 0.97,0.75 \
  --model resnet18_1686_int8_sym.bmodel
```

###### to asymmetric

``` shell
model_deploy.py \
  --mlir resnet18.mlir \
  --quantize INT8 \
  --asymmetric \
  --calibration_table resnet18_cali_table \
  --chip bm1686 \
  --test_input resnet18_in.npz \
  --test_reference resnet18_top_outputs.npz \
  --tolerance 0.97,0.75 \
  --model resnet18_1686_int8_asym.bmodel
```
