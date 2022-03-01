# 设计随笔

## 要求

* 用vscode编辑代码，提交前都格式化一下

* 所有文档用markdown编写

* td文件的注视要完整，以便自动生成文档

* 所有对外的工具都用python，（runtime工具除外）

* 尽可能减少git submodule;第三方优先直接在docker里面安装(如pybind11)，不能安装的则使用编译后的库（如llvm,oneDNN)

## 设计

* 用mlir自带的文档生成工具生成td文档

* 保证calibration高效：

  * operation的资源申请与forward过程分离；
  * interpet能用oneDNN的，用oneDNN；不能用oneDNN的，尽量用OMP

* 用'F32'，不要用'FP32'，与mlir保持一致

* mlir默认整型是INT64，默认浮点型是F64. attribute的类型尽量保持一致.好处如下：

``` mlir
# 用INT32
%4 = "tops.Conv"(%1, %2, %3) {kernel_shape = [3 : i32, 3 : i32], name = "output_Conv"}
# 用INT64
%4 = "tops.Conv"(%1, %2, %3) {kernel_shape = [3, 3], name = "output_Conv"}
```

## 代码编译方法
source ./envsetup.sh
./build.sh

## 工具使用方式

``` shell
model_transform.py \
    --model_type onnx \
    --model_name resnet18 \
    --model_def  ../resnet18.onnx \
    --tolerance 0.99,0.99,0.96 \
    --mlir resnet18.mlir

# 后续集成到model_transform.py里面
tpuc-opt resnet18.mlir \
    --canonicalize \
    -o resnet18_opt.mlir
```


## 待解决问题

* llvm库精简 (目前从2GB精简到200MB，希望能减到100MB以内)

* op注视补充完整

* docker需要配置成用普通用户权限

* dnnl pool与conv参考matmul优化

* 数据全部按fp32存储，后续支持其他类型

* 目前所有op都认为是一个输出，后续支持多输出op ?

* tpuc-opt能否需要实现python版本？

* tops的pass只有一个：Fuse Relu到conv和add，其他pass需要补充完整

* 目前shape还不支持动态shape和shape推导，后续支持

* vscode的pybind11路径配置没有配好，Python.hu也没有定位好

## 一些思考

#### 是否第一层用TOSA Dialect ?
不用：
如果由客户转，增加客户工作量；
tosa定义的op还是太少，没有lrn、layernorm等等op；
自己定义tops dialect，可以完全掌控

#### caffe需要前端转吗？
需要：
目前caffe转onnx没有组织维护，支持的算子特别少

#### 开源后的亮点

1 tops层和quant层采用oneDNN和omp推导结果，做验证
2 工具全部用python，方便使用和维护
3 可以随时更新到llvm最新版本

#### 是否要支持预处理？以及将预处理合并到tpu layer?

预处理事情比较多，先不做预处理的考虑，假定用户的输入是预处理后的数据；
后续时间充足再考虑预处理的事情，边端芯片还是需要的，用户从vpss出来后就需要直接导入到模型，
预处理只能有tpu完成。
