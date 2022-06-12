# Top转TPU



## 基本过程

![](./assets/lowering.png)



* Top算子可以分f32和int8两种，前者是大多数网络的情况；后者是如tflite等量化过的网络的情况
* f32算子可以直接转换成f32/f16/bf16的tpu层算子，如果要转int8，则需要类型是calibrated_type
* int8算子只能直接转换成tpu层int8算子



## 混合精度

![](./assets/mix_prec.png)

当OP之间的类型不一致时，则插入CastOp。

需要注意的有以下几点：

* 当同一个OP被cast多次时，重复的cast需要合并