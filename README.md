## sophgo-mlir

This project provides compiler to transform NN model into sophgo graph with sophgo runtime support.

Get more info from [sophgo](http://sophgo.com).

## start docker

```
# myname1234 just a example, you can set your own name
docker run --name myname1234 -v $PWD:/work -it sophgo/sophgo_dev:1.0-ubuntu-20.04
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

#### transform model to mlir (tops)

``` shell
model_transform.py \
    --model_type onnx \
    --model_name resnet18 \
    --model_def  ../resnet18.onnx \
    --input ../resnet18_in_f32.npz \
    --mlir resnet18.mlir
```

#### do calibration

prepare inputs in forder "dataset", and run:

``` shell
run_calibration.py resnet18.mlir \
    --dataset dataset \
    --input_num 1 \
    -o resnet18_cali_table
```

#### deploy mlir to bmodel
