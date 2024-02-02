[bmprofile] arch=4
[bmprofile] net_name=resnet50_v1
[bmprofile] tpu_freq=0
[bmprofile] is_mlir=1
...Start Profile Log...
[bmprofile] start to run subnet_id=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=16 layer_type=Load layer_name=
[bmprofile] tensor_id=-1 is_in=1 shape=[1x3x224x224] dtype=2 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=16 is_in=0 shape=[1x3x224x224] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=224 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=17 layer_type=Load layer_name=
[bmprofile] tensor_id=12 is_in=1 shape=[1x32x1x3168] dtype=2 is_const=1 gaddr=618475290624 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=17 is_in=0 shape=[1x32x1x3168] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=18 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=16 is_in=1 shape=[1x3x224x224] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=224 l2addr=0
[bmprofile] tensor_id=17 is_in=1 shape=[1x32x1x3168] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=18 is_in=0 shape=[1x64x112x112] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=112 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=2 bd_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=20 layer_type=Pool2D layer_name=
[bmprofile] tensor_id=18 is_in=1 shape=[1x64x112x112] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=112 l2addr=0
[bmprofile] tensor_id=20 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=50176 nslice=1 hslice=56 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=2 bd_func=1 core=0
[bmprofile] local_layer: layer_id=19 layer_type=Load layer_name=
[bmprofile] tensor_id=13 is_in=1 shape=[1x32x1x160] dtype=2 is_const=1 gaddr=618475393024 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=19 is_in=0 shape=[1x32x1x160] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=2 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=22 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=20 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=50176 nslice=1 hslice=56 l2addr=0
[bmprofile] tensor_id=19 is_in=1 shape=[1x32x1x160] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=22 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=56 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=3 bd_func=0 core=0
[bmprofile] local_layer: layer_id=21 layer_type=Load layer_name=
[bmprofile] tensor_id=14 is_in=1 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=618475401216 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=21 is_in=0 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=3 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=24 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=22 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=56 l2addr=0
[bmprofile] tensor_id=21 is_in=1 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=24 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=90112 nslice=1 hslice=56 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=4 bd_func=0 core=0
[bmprofile] local_layer: layer_id=23 layer_type=Store layer_name=
[bmprofile] tensor_id=20 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=50176 nslice=1 hslice=56 l2addr=0
[bmprofile] tensor_id=23 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=50176 nslice=1 hslice=56 l2addr=0
[bmprofile] gdma cmd_id bd_id=5 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=25 layer_type=Store layer_name=
[bmprofile] tensor_id=24 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=90112 nslice=1 hslice=56 l2addr=0
[bmprofile] tensor_id=25 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=90112 nslice=1 hslice=56 l2addr=0
[bmprofile] gdma cmd_id bd_id=7 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=6 bd_func=15 core=0
[bmprofile] gdma cmd_id bd_id=8 gdma_id=0 gdma_dir=0 gdma_func=6 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=15 core=0
[bmprofile] gdma cmd_id bd_id=9 gdma_id=0 gdma_dir=0 gdma_func=6 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=8 bd_func=3 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=32 layer_type=Load layer_name=
[bmprofile] tensor_id=25 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=687195119616 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=32 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=10 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=33 layer_type=Load layer_name=
[bmprofile] tensor_id=28 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=618475442176 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=33 is_in=0 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=10 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=36 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=32 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=33 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=36 is_in=0 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=10 bd_func=0 core=0
[bmprofile] local_layer: layer_id=34 layer_type=Load layer_name=
[bmprofile] tensor_id=23 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=687194918912 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=34 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=10 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=35 layer_type=Load layer_name=
[bmprofile] tensor_id=29 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=618475462656 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=35 is_in=0 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=45312 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=10 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=37 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=34 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=35 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=45312 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=37 is_in=0 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=12 bd_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=39 layer_type=Add layer_name=
[bmprofile] tensor_id=36 is_in=1 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=37 is_in=1 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=39 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=12 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=12 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=12 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=12 bd_func=3 core=0
[bmprofile] local_layer: layer_id=38 layer_type=Load layer_name=
[bmprofile] tensor_id=30 is_in=1 shape=[1x32x1x544] dtype=2 is_const=1 gaddr=618475483136 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=38 is_in=0 shape=[1x32x1x544] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=52288 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=14 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=41 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=39 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=38 is_in=1 shape=[1x32x1x544] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=52288 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=41 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=13 bd_func=0 core=0
[bmprofile] local_layer: layer_id=40 layer_type=Store layer_name=
[bmprofile] tensor_id=39 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=40 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=18 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=42 layer_type=Store layer_name=
[bmprofile] tensor_id=41 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=42 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=20 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=15 bd_func=15 core=0
[bmprofile] gdma cmd_id bd_id=21 gdma_id=0 gdma_dir=0 gdma_func=6 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=16 bd_func=15 core=0
[bmprofile] gdma cmd_id bd_id=22 gdma_id=0 gdma_dir=0 gdma_func=6 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=17 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=0 bd_func=15 core=1
[bmprofile] gdma cmd_id bd_id=1 gdma_id=0 gdma_dir=0 gdma_func=6 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1 bd_func=15 core=1
[bmprofile] gdma cmd_id bd_id=2 gdma_id=0 gdma_dir=0 gdma_func=6 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=2 bd_func=3 core=1
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=32 layer_type=Load layer_name=
[bmprofile] tensor_id=25 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=687195119616 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=32 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=3 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] local_layer: layer_id=33 layer_type=Load layer_name=
[bmprofile] tensor_id=28 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=618475442176 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=33 is_in=0 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=3 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=36 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=32 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=33 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=36 is_in=0 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=4 bd_func=0 core=1
[bmprofile] local_layer: layer_id=34 layer_type=Load layer_name=
[bmprofile] tensor_id=23 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=687194918912 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=34 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=3 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] local_layer: layer_id=35 layer_type=Load layer_name=
[bmprofile] tensor_id=29 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=618475462656 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=35 is_in=0 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=45312 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=3 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=37 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=34 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=35 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=45312 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=37 is_in=0 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=6 bd_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=39 layer_type=Add layer_name=
[bmprofile] tensor_id=36 is_in=1 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=37 is_in=1 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=39 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=6 bd_func=3 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=6 bd_func=3 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=6 bd_func=3 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=6 bd_func=3 core=1
[bmprofile] local_layer: layer_id=38 layer_type=Load layer_name=
[bmprofile] tensor_id=30 is_in=1 shape=[1x32x1x544] dtype=2 is_const=1 gaddr=618475483136 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=38 is_in=0 shape=[1x32x1x544] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=52288 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=7 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=41 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=39 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=38 is_in=1 shape=[1x32x1x544] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=52288 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=41 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=0 core=1
[bmprofile] local_layer: layer_id=40 layer_type=Store layer_name=
[bmprofile] tensor_id=39 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=40 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=11 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=42 layer_type=Store layer_name=
[bmprofile] tensor_id=41 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=42 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=13 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=9 bd_func=15 core=1
[bmprofile] gdma cmd_id bd_id=14 gdma_id=0 gdma_dir=0 gdma_func=6 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=10 bd_func=15 core=1
[bmprofile] gdma cmd_id bd_id=15 gdma_id=0 gdma_dir=0 gdma_func=6 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=11 bd_func=3 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=17 bd_func=15 core=0
[bmprofile] gdma cmd_id bd_id=24 gdma_id=0 gdma_dir=0 gdma_func=6 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=18 bd_func=15 core=0
[bmprofile] gdma cmd_id bd_id=25 gdma_id=0 gdma_dir=0 gdma_func=6 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=19 bd_func=3 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=49 layer_type=Load layer_name=
[bmprofile] tensor_id=42 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=687196524544 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=49 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=29 l2addr=0
[bmprofile] gdma cmd_id bd_id=26 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=50 layer_type=Load layer_name=
[bmprofile] tensor_id=45 is_in=1 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=618475503616 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=50 is_in=0 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=26 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=53 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=49 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=29 l2addr=0
[bmprofile] tensor_id=50 is_in=1 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=53 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=21 bd_func=0 core=0
[bmprofile] local_layer: layer_id=51 layer_type=Load layer_name=
[bmprofile] tensor_id=46 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=618475544576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=51 is_in=0 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=45312 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=26 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=52 layer_type=Load layer_name=
[bmprofile] tensor_id=40 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=687195721728 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=52 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=26 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=54 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=53 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=51 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=45312 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=54 is_in=0 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=23 bd_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=56 layer_type=Add layer_name=
[bmprofile] tensor_id=54 is_in=1 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=52 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=56 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=23 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=23 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=23 bd_func=3 core=0
[bmprofile] local_layer: layer_id=55 layer_type=Load layer_name=
[bmprofile] tensor_id=47 is_in=1 shape=[1x32x1x544] dtype=2 is_const=1 gaddr=618475565056 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=55 is_in=0 shape=[1x32x1x544] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=52416 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=30 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=58 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=56 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=55 is_in=1 shape=[1x32x1x544] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=52416 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=58 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=24 bd_func=0 core=0
[bmprofile] local_layer: layer_id=57 layer_type=Store layer_name=
[bmprofile] tensor_id=56 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=57 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=33 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=59 layer_type=Store layer_name=
[bmprofile] tensor_id=58 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=59 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=35 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=26 bd_func=15 core=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=0 gdma_dir=0 gdma_func=6 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=27 bd_func=15 core=0
[bmprofile] gdma cmd_id bd_id=37 gdma_id=0 gdma_dir=0 gdma_func=6 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=28 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=11 bd_func=15 core=1
[bmprofile] gdma cmd_id bd_id=17 gdma_id=0 gdma_dir=0 gdma_func=6 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=12 bd_func=15 core=1
[bmprofile] gdma cmd_id bd_id=18 gdma_id=0 gdma_dir=0 gdma_func=6 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=13 bd_func=3 core=1
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=49 layer_type=Load layer_name=
[bmprofile] tensor_id=42 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=687196524544 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=49 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=29 l2addr=0
[bmprofile] gdma cmd_id bd_id=19 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] local_layer: layer_id=50 layer_type=Load layer_name=
[bmprofile] tensor_id=45 is_in=1 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=618475503616 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=50 is_in=0 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=19 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=53 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=49 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=29 l2addr=0
[bmprofile] tensor_id=50 is_in=1 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=53 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=15 bd_func=0 core=1
[bmprofile] local_layer: layer_id=51 layer_type=Load layer_name=
[bmprofile] tensor_id=46 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=618475544576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=51 is_in=0 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=45312 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=19 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] local_layer: layer_id=52 layer_type=Load layer_name=
[bmprofile] tensor_id=40 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=687195721728 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=52 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=19 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=54 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=53 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=51 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=45312 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=54 is_in=0 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=17 bd_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=56 layer_type=Add layer_name=
[bmprofile] tensor_id=54 is_in=1 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=52 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=56 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=17 bd_func=3 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=17 bd_func=3 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=17 bd_func=3 core=1
[bmprofile] local_layer: layer_id=55 layer_type=Load layer_name=
[bmprofile] tensor_id=47 is_in=1 shape=[1x32x1x544] dtype=2 is_const=1 gaddr=618475565056 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=55 is_in=0 shape=[1x32x1x544] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=52416 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=58 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=56 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=55 is_in=1 shape=[1x32x1x544] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=52416 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=58 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=18 bd_func=0 core=1
[bmprofile] local_layer: layer_id=57 layer_type=Store layer_name=
[bmprofile] tensor_id=56 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=57 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=26 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=59 layer_type=Store layer_name=
[bmprofile] tensor_id=58 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=59 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=28 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=20 bd_func=15 core=1
[bmprofile] gdma cmd_id bd_id=29 gdma_id=0 gdma_dir=0 gdma_func=6 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=21 bd_func=15 core=1
[bmprofile] gdma cmd_id bd_id=30 gdma_id=0 gdma_dir=0 gdma_func=6 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=22 bd_func=3 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=28 bd_func=15 core=0
[bmprofile] gdma cmd_id bd_id=39 gdma_id=0 gdma_dir=0 gdma_func=6 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=29 bd_func=15 core=0
[bmprofile] gdma cmd_id bd_id=40 gdma_id=0 gdma_dir=0 gdma_func=6 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=30 bd_func=3 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=66 layer_type=Load layer_name=
[bmprofile] tensor_id=59 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=687196725248 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=66 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=29 l2addr=0
[bmprofile] gdma cmd_id bd_id=41 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=67 layer_type=Load layer_name=
[bmprofile] tensor_id=62 is_in=1 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=618475585536 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=67 is_in=0 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=41 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=70 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=66 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=29 l2addr=0
[bmprofile] tensor_id=67 is_in=1 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=70 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=32 bd_func=0 core=0
[bmprofile] local_layer: layer_id=68 layer_type=Load layer_name=
[bmprofile] tensor_id=63 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=618475626496 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=68 is_in=0 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=45312 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=41 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=69 layer_type=Load layer_name=
[bmprofile] tensor_id=57 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=687194918912 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=69 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=41 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=71 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=70 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=68 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=45312 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=71 is_in=0 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=34 bd_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=73 layer_type=Add layer_name=
[bmprofile] tensor_id=71 is_in=1 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=69 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=73 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=34 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=34 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=34 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=34 bd_func=3 core=0
[bmprofile] local_layer: layer_id=72 layer_type=Load layer_name=
[bmprofile] tensor_id=64 is_in=1 shape=[1x32x1x1072] dtype=2 is_const=1 gaddr=618475646976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=72 is_in=0 shape=[1x32x1x1072] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=52416 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=45 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=75 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=73 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=72 is_in=1 shape=[1x32x1x1072] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=52416 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=75 is_in=0 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=35 bd_func=0 core=0
[bmprofile] local_layer: layer_id=74 layer_type=Store layer_name=
[bmprofile] tensor_id=73 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=74 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=49 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=76 layer_type=Store layer_name=
[bmprofile] tensor_id=75 is_in=1 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=76 is_in=0 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=51 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=37 bd_func=15 core=0
[bmprofile] gdma cmd_id bd_id=52 gdma_id=0 gdma_dir=0 gdma_func=6 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=38 bd_func=15 core=0
[bmprofile] gdma cmd_id bd_id=53 gdma_id=0 gdma_dir=0 gdma_func=6 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=39 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=22 bd_func=15 core=1
[bmprofile] gdma cmd_id bd_id=32 gdma_id=0 gdma_dir=0 gdma_func=6 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=23 bd_func=15 core=1
[bmprofile] gdma cmd_id bd_id=33 gdma_id=0 gdma_dir=0 gdma_func=6 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=24 bd_func=3 core=1
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=66 layer_type=Load layer_name=
[bmprofile] tensor_id=59 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=687196725248 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=66 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=29 l2addr=0
[bmprofile] gdma cmd_id bd_id=34 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] local_layer: layer_id=67 layer_type=Load layer_name=
[bmprofile] tensor_id=62 is_in=1 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=618475585536 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=67 is_in=0 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=34 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=70 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=66 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=29 l2addr=0
[bmprofile] tensor_id=67 is_in=1 shape=[1x32x1x1184] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=70 is_in=0 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=26 bd_func=0 core=1
[bmprofile] local_layer: layer_id=68 layer_type=Load layer_name=
[bmprofile] tensor_id=63 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=618475626496 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=68 is_in=0 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=45312 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=34 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] local_layer: layer_id=69 layer_type=Load layer_name=
[bmprofile] tensor_id=57 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=687194918912 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=69 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=34 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=71 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=70 is_in=1 shape=[1x64x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=68 is_in=1 shape=[1x32x1x608] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=45312 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=71 is_in=0 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=28 bd_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=73 layer_type=Add layer_name=
[bmprofile] tensor_id=71 is_in=1 shape=[1x256x56x56] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=69 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=73 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=28 bd_func=3 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=28 bd_func=3 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=28 bd_func=3 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=28 bd_func=3 core=1
[bmprofile] local_layer: layer_id=72 layer_type=Load layer_name=
[bmprofile] tensor_id=64 is_in=1 shape=[1x32x1x1072] dtype=2 is_const=1 gaddr=618475646976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=72 is_in=0 shape=[1x32x1x1072] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=52416 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=38 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=75 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=73 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=72 is_in=1 shape=[1x32x1x1072] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=52416 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=75 is_in=0 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=29 bd_func=0 core=1
[bmprofile] local_layer: layer_id=74 layer_type=Store layer_name=
[bmprofile] tensor_id=73 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=74 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=42 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=76 layer_type=Store layer_name=
[bmprofile] tensor_id=75 is_in=1 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=76 is_in=0 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=44 gdma_id=0 gdma_dir=0 gdma_func=0 core=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=31 bd_func=15 core=1
[bmprofile] gdma cmd_id bd_id=45 gdma_id=0 gdma_dir=0 gdma_func=6 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=32 bd_func=15 core=1
[bmprofile] gdma cmd_id bd_id=46 gdma_id=0 gdma_dir=0 gdma_func=6 core=1
[bmprofile] gdma cmd_id bd_id=0 gdma_id=33 bd_func=3 core=1
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=112 layer_type=Load layer_name=
[bmprofile] tensor_id=79 is_in=1 shape=[1x32x1x4656] dtype=2 is_const=1 gaddr=618475683840 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=112 is_in=0 shape=[1x32x1x4656] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=55808 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=54 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=113 layer_type=Load layer_name=
[bmprofile] tensor_id=76 is_in=1 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=687196524544 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=113 is_in=0 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=77040 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=54 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=116 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=113 is_in=1 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=77040 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=112 is_in=1 shape=[1x32x1x4656] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=55808 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=116 is_in=0 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=41 bd_func=0 core=0
[bmprofile] local_layer: layer_id=114 layer_type=Load layer_name=
[bmprofile] tensor_id=80 is_in=1 shape=[1x32x1x2240] dtype=2 is_const=1 gaddr=618475835392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=114 is_in=0 shape=[1x32x1x2240] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=25088 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=54 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=115 layer_type=Load layer_name=
[bmprofile] tensor_id=74 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=687195721728 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=115 is_in=0 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=56 l2addr=0
[bmprofile] gdma cmd_id bd_id=54 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=118 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=116 is_in=1 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=114 is_in=1 shape=[1x32x1x2240] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=25088 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=118 is_in=0 shape=[1x512x28x28] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=73728 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=43 bd_func=0 core=0
[bmprofile] local_layer: layer_id=117 layer_type=Load layer_name=
[bmprofile] tensor_id=81 is_in=1 shape=[1x32x1x4288] dtype=2 is_const=1 gaddr=618475909120 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=117 is_in=0 shape=[1x32x1x4288] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=56 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=119 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=115 is_in=1 shape=[1x256x56x56] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=56 l2addr=0
[bmprofile] tensor_id=117 is_in=1 shape=[1x32x1x4288] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=119 is_in=0 shape=[1x512x28x28] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=90112 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=44 bd_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=121 layer_type=Add layer_name=
[bmprofile] tensor_id=118 is_in=1 shape=[1x512x28x28] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=73728 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=119 is_in=1 shape=[1x512x28x28] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=90112 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=121 is_in=0 shape=[1x512x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=44 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=44 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=44 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=44 bd_func=3 core=0
[bmprofile] local_layer: layer_id=120 layer_type=Load layer_name=
[bmprofile] tensor_id=82 is_in=1 shape=[1x32x1x2096] dtype=2 is_const=1 gaddr=618476048384 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=120 is_in=0 shape=[1x32x1x2096] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=106496 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=60 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=123 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=121 is_in=1 shape=[1x512x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=120 is_in=1 shape=[1x32x1x2096] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=106496 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=123 is_in=0 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=45 bd_func=0 core=0
[bmprofile] local_layer: layer_id=122 layer_type=Load layer_name=
[bmprofile] tensor_id=83 is_in=1 shape=[1x32x1x4656] dtype=2 is_const=1 gaddr=618476118016 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=122 is_in=0 shape=[1x32x1x4656] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=64 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=125 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=123 is_in=1 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=122 is_in=1 shape=[1x32x1x4656] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=125 is_in=0 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=46 bd_func=0 core=0
[bmprofile] local_layer: layer_id=124 layer_type=Load layer_name=
[bmprofile] tensor_id=84 is_in=1 shape=[1x32x1x2240] dtype=2 is_const=1 gaddr=618476269568 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=124 is_in=0 shape=[1x32x1x2240] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=126 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=125 is_in=1 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=12544 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=124 is_in=1 shape=[1x32x1x2240] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=126 is_in=0 shape=[1x512x28x28] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=47 bd_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=128 layer_type=Add layer_name=
[bmprofile] tensor_id=126 is_in=1 shape=[1x512x28x28] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=121 is_in=1 shape=[1x512x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=128 is_in=0 shape=[1x512x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=47 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=47 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=47 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=47 bd_func=3 core=0
[bmprofile] local_layer: layer_id=127 layer_type=Load layer_name=
[bmprofile] tensor_id=85 is_in=1 shape=[1x32x1x2096] dtype=2 is_const=1 gaddr=618476343296 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=127 is_in=0 shape=[1x32x1x2096] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=106496 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=70 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=130 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=128 is_in=1 shape=[1x512x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=127 is_in=1 shape=[1x32x1x2096] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=106496 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=130 is_in=0 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=48 bd_func=0 core=0
[bmprofile] local_layer: layer_id=129 layer_type=Load layer_name=
[bmprofile] tensor_id=86 is_in=1 shape=[1x32x1x4656] dtype=2 is_const=1 gaddr=618476412928 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=129 is_in=0 shape=[1x32x1x4656] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=74 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=132 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=130 is_in=1 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=129 is_in=1 shape=[1x32x1x4656] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=132 is_in=0 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=28928 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=49 bd_func=0 core=0
[bmprofile] local_layer: layer_id=131 layer_type=Load layer_name=
[bmprofile] tensor_id=87 is_in=1 shape=[1x32x1x2240] dtype=2 is_const=1 gaddr=618476564480 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=131 is_in=0 shape=[1x32x1x2240] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=76 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=133 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=132 is_in=1 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=28928 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=131 is_in=1 shape=[1x32x1x2240] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=133 is_in=0 shape=[1x512x28x28] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=50 bd_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=135 layer_type=Add layer_name=
[bmprofile] tensor_id=133 is_in=1 shape=[1x512x28x28] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=128 is_in=1 shape=[1x512x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=135 is_in=0 shape=[1x512x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=90112 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=50 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=50 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=50 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=50 bd_func=3 core=0
[bmprofile] local_layer: layer_id=134 layer_type=Load layer_name=
[bmprofile] tensor_id=88 is_in=1 shape=[1x32x1x2096] dtype=2 is_const=1 gaddr=618476638208 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=134 is_in=0 shape=[1x32x1x2096] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=106496 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=80 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=137 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=135 is_in=1 shape=[1x512x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=90112 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=134 is_in=1 shape=[1x32x1x2096] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=106496 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=137 is_in=0 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=51 bd_func=0 core=0
[bmprofile] local_layer: layer_id=136 layer_type=Load layer_name=
[bmprofile] tensor_id=89 is_in=1 shape=[1x32x1x4656] dtype=2 is_const=1 gaddr=618476707840 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=136 is_in=0 shape=[1x32x1x4656] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=84 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=140 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=137 is_in=1 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=136 is_in=1 shape=[1x32x1x4656] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=140 is_in=0 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=52 bd_func=0 core=0
[bmprofile] local_layer: layer_id=138 layer_type=Load layer_name=
[bmprofile] tensor_id=90 is_in=1 shape=[1x32x1x2240] dtype=2 is_const=1 gaddr=618476859392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=138 is_in=0 shape=[1x32x1x2240] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=18528 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=139 layer_type=Load layer_name=
[bmprofile] tensor_id=92 is_in=1 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=618477068288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=139 is_in=0 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=142 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=140 is_in=1 shape=[1x128x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=138 is_in=1 shape=[1x32x1x2240] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=18528 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=142 is_in=0 shape=[1x512x28x28] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=115072 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=54 bd_func=0 core=0
[bmprofile] local_layer: layer_id=141 layer_type=Load layer_name=
[bmprofile] tensor_id=91 is_in=1 shape=[1x32x1x4192] dtype=2 is_const=1 gaddr=618476933120 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=141 is_in=0 shape=[1x32x1x4192] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=31072 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=88 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=144 layer_type=Add layer_name=
[bmprofile] tensor_id=142 is_in=1 shape=[1x512x28x28] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=115072 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=135 is_in=1 shape=[1x512x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=90112 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=144 is_in=0 shape=[1x512x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=18528 nslice=1 hslice=28 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=55 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=55 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=55 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=55 bd_func=3 core=0
[bmprofile] local_layer: layer_id=145 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=144 is_in=1 shape=[1x512x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=18528 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=141 is_in=1 shape=[1x32x1x4192] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=31072 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=145 is_in=0 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=102656 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=55 bd_func=0 core=0
[bmprofile] local_layer: layer_id=143 layer_type=Load layer_name=
[bmprofile] tensor_id=93 is_in=1 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=618477662208 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=143 is_in=0 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=106496 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=90 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=147 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=145 is_in=1 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=102656 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=139 is_in=1 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=147 is_in=0 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=31072 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=56 bd_func=0 core=0
[bmprofile] local_layer: layer_id=146 layer_type=Load layer_name=
[bmprofile] tensor_id=94 is_in=1 shape=[1x32x1x16768] dtype=2 is_const=1 gaddr=618477936640 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=146 is_in=0 shape=[1x32x1x16768] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=96 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=150 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=147 is_in=1 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=31072 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=143 is_in=1 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=106496 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=150 is_in=0 shape=[1x1024x14x14] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=122880 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=57 bd_func=0 core=0
[bmprofile] local_layer: layer_id=151 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=144 is_in=1 shape=[1x512x28x28] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=18528 nslice=1 hslice=28 l2addr=0
[bmprofile] tensor_id=146 is_in=1 shape=[1x32x1x16768] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=151 is_in=0 shape=[1x1024x14x14] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=57 bd_func=0 core=0
[bmprofile] local_layer: layer_id=148 layer_type=Load layer_name=
[bmprofile] tensor_id=95 is_in=1 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=618478473216 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=148 is_in=0 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=51296 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=98 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=149 layer_type=Load layer_name=
[bmprofile] tensor_id=96 is_in=1 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=618478739456 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=149 is_in=0 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=98 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=154 layer_type=Add layer_name=
[bmprofile] tensor_id=150 is_in=1 shape=[1x1024x14x14] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=122880 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=151 is_in=1 shape=[1x1024x14x14] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=154 is_in=0 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=90112 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=59 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=59 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=59 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=59 bd_func=3 core=0
[bmprofile] local_layer: layer_id=155 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=154 is_in=1 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=90112 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=148 is_in=1 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=51296 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=155 is_in=0 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=87872 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=59 bd_func=0 core=0
[bmprofile] local_layer: layer_id=156 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=155 is_in=1 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=87872 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=149 is_in=1 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=156 is_in=0 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=59584 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=59 bd_func=0 core=0
[bmprofile] local_layer: layer_id=152 layer_type=Load layer_name=
[bmprofile] tensor_id=97 is_in=1 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=618479333376 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=152 is_in=0 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=18528 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=102 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=153 layer_type=Load layer_name=
[bmprofile] tensor_id=99 is_in=1 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=618479874048 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=153 is_in=0 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=102 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=158 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=156 is_in=1 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=59584 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=152 is_in=1 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=18528 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=158 is_in=0 shape=[1x1024x14x14] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=61 bd_func=0 core=0
[bmprofile] local_layer: layer_id=157 layer_type=Load layer_name=
[bmprofile] tensor_id=98 is_in=1 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=618479607808 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=157 is_in=0 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=46816 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=110 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=161 layer_type=Add layer_name=
[bmprofile] tensor_id=158 is_in=1 shape=[1x1024x14x14] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=154 is_in=1 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=90112 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=161 is_in=0 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=106496 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=62 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=62 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=62 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=62 bd_func=3 core=0
[bmprofile] local_layer: layer_id=162 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=161 is_in=1 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=106496 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=157 is_in=1 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=46816 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=162 is_in=0 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=18528 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=62 bd_func=0 core=0
[bmprofile] local_layer: layer_id=163 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=162 is_in=1 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=18528 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=153 is_in=1 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=163 is_in=0 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=55104 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=62 bd_func=0 core=0
[bmprofile] local_layer: layer_id=159 layer_type=Load layer_name=
[bmprofile] tensor_id=100 is_in=1 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=618480467968 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=159 is_in=0 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=75872 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=112 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=160 layer_type=Load layer_name=
[bmprofile] tensor_id=102 is_in=1 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=618481008640 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=160 is_in=0 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=112 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=165 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=163 is_in=1 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=55104 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=159 is_in=1 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=75872 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=165 is_in=0 shape=[1x1024x14x14] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=122880 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=64 bd_func=0 core=0
[bmprofile] local_layer: layer_id=164 layer_type=Load layer_name=
[bmprofile] tensor_id=101 is_in=1 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=618480742400 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=164 is_in=0 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=120 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=168 layer_type=Add layer_name=
[bmprofile] tensor_id=165 is_in=1 shape=[1x1024x14x14] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=122880 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=161 is_in=1 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=106496 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=168 is_in=0 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=65 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=65 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=65 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=65 bd_func=3 core=0
[bmprofile] local_layer: layer_id=169 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=168 is_in=1 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=164 is_in=1 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=169 is_in=0 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=52960 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=65 bd_func=0 core=0
[bmprofile] local_layer: layer_id=170 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=169 is_in=1 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=52960 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=160 is_in=1 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=170 is_in=0 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=24672 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=65 bd_func=0 core=0
[bmprofile] local_layer: layer_id=166 layer_type=Load layer_name=
[bmprofile] tensor_id=103 is_in=1 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=618481602560 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=166 is_in=0 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=122 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=167 layer_type=Load layer_name=
[bmprofile] tensor_id=105 is_in=1 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=618482143232 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=167 is_in=0 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=122 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=172 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=170 is_in=1 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=24672 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=166 is_in=1 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=172 is_in=0 shape=[1x1024x14x14] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=100448 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=67 bd_func=0 core=0
[bmprofile] local_layer: layer_id=171 layer_type=Load layer_name=
[bmprofile] tensor_id=104 is_in=1 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=618481876992 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=171 is_in=0 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=130 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=175 layer_type=Add layer_name=
[bmprofile] tensor_id=172 is_in=1 shape=[1x1024x14x14] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=100448 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=168 is_in=1 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=175 is_in=0 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=73728 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=68 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=68 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=68 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=68 bd_func=3 core=0
[bmprofile] local_layer: layer_id=176 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=175 is_in=1 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=73728 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=171 is_in=1 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=176 is_in=0 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=69344 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=68 bd_func=0 core=0
[bmprofile] local_layer: layer_id=177 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=176 is_in=1 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=69344 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=167 is_in=1 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=177 is_in=0 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=41056 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=68 bd_func=0 core=0
[bmprofile] local_layer: layer_id=173 layer_type=Load layer_name=
[bmprofile] tensor_id=106 is_in=1 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=618482737152 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=173 is_in=0 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=18528 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=132 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=174 layer_type=Load layer_name=
[bmprofile] tensor_id=108 is_in=1 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=618483277824 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=174 is_in=0 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=132 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=179 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=177 is_in=1 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=41056 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=173 is_in=1 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=18528 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=179 is_in=0 shape=[1x1024x14x14] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=70 bd_func=0 core=0
[bmprofile] local_layer: layer_id=178 layer_type=Load layer_name=
[bmprofile] tensor_id=107 is_in=1 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=618483011584 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=178 is_in=0 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=140 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=180 layer_type=Add layer_name=
[bmprofile] tensor_id=179 is_in=1 shape=[1x1024x14x14] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=175 is_in=1 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=73728 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=180 is_in=0 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=71 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=71 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=71 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=71 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=182 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=180 is_in=1 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=178 is_in=1 shape=[1x32x1x8288] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=182 is_in=0 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=71 bd_func=0 core=0
[bmprofile] local_layer: layer_id=181 layer_type=Load layer_name=
[bmprofile] tensor_id=109 is_in=1 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=618483871744 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=181 is_in=0 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=18528 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=146 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=184 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=182 is_in=1 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=174 is_in=1 shape=[1x32x1x18528] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=184 is_in=0 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=72 bd_func=0 core=0
[bmprofile] local_layer: layer_id=183 layer_type=Load layer_name=
[bmprofile] tensor_id=110 is_in=1 shape=[1x32x1x16576] dtype=2 is_const=1 gaddr=618484146176 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=183 is_in=0 shape=[1x32x1x16576] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=60464 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=148 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=185 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=184 is_in=1 shape=[1x256x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=181 is_in=1 shape=[1x32x1x8576] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=18528 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=185 is_in=0 shape=[1x1024x14x14] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=73 bd_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=186 layer_type=Add layer_name=
[bmprofile] tensor_id=185 is_in=1 shape=[1x1024x14x14] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=180 is_in=1 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=186 is_in=0 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=73 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=73 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=73 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=73 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=188 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=186 is_in=1 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=183 is_in=1 shape=[1x32x1x16576] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=60464 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=188 is_in=0 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=37056 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=73 bd_func=0 core=0
[bmprofile] local_layer: layer_id=187 layer_type=Store layer_name=
[bmprofile] tensor_id=186 is_in=1 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=187 is_in=0 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=156 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=189 layer_type=Store layer_name=
[bmprofile] tensor_id=188 is_in=1 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=37056 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=189 is_in=0 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=37056 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=158 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=197 layer_type=Load layer_name=
[bmprofile] tensor_id=189 is_in=1 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=687195119616 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=197 is_in=0 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=78016 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=158 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=198 layer_type=Load layer_name=
[bmprofile] tensor_id=192 is_in=1 shape=[1x32x32x2310] dtype=2 is_const=1 gaddr=618484678656 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=198 is_in=0 shape=[1x32x32x2310] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=158 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=200 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=197 is_in=1 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=78016 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=198 is_in=1 shape=[1x32x32x2310] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=200 is_in=0 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=115456 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=77 bd_func=0 core=0
[bmprofile] local_layer: layer_id=199 layer_type=Load layer_name=
[bmprofile] tensor_id=193 is_in=1 shape=[1x32x1x33536] dtype=2 is_const=1 gaddr=618487046144 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=199 is_in=0 shape=[1x32x1x33536] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=158 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=203 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=200 is_in=1 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=115456 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=199 is_in=1 shape=[1x32x1x33536] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=203 is_in=0 shape=[1x2048x7x7] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=122880 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=78 bd_func=0 core=0
[bmprofile] local_layer: layer_id=201 layer_type=Load layer_name=
[bmprofile] tensor_id=187 is_in=1 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=687194918912 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=201 is_in=0 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=73728 nslice=1 hslice=14 l2addr=0
[bmprofile] gdma cmd_id bd_id=160 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=202 layer_type=Load layer_name=
[bmprofile] tensor_id=194 is_in=1 shape=[1x32x32x2072] dtype=2 is_const=1 gaddr=618488119296 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=202 is_in=0 shape=[1x32x32x2072] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=160 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=205 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=201 is_in=1 shape=[1x1024x14x14] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=73728 nslice=1 hslice=14 l2addr=0
[bmprofile] tensor_id=202 is_in=1 shape=[1x32x32x2072] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=205 is_in=0 shape=[1x2048x7x7] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=66304 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=80 bd_func=0 core=0
[bmprofile] local_layer: layer_id=204 layer_type=Load layer_name=
[bmprofile] tensor_id=195 is_in=1 shape=[1x32x1x32960] dtype=2 is_const=1 gaddr=618490241024 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=204 is_in=0 shape=[1x32x1x32960] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=162 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=206 layer_type=Add layer_name=
[bmprofile] tensor_id=203 is_in=1 shape=[1x2048x7x7] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=122880 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=205 is_in=1 shape=[1x2048x7x7] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=66304 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=206 is_in=0 shape=[1x2048x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=73920 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=81 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=81 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=81 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=81 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=208 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=206 is_in=1 shape=[1x2048x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=73920 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=204 is_in=1 shape=[1x32x1x32960] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=208 is_in=0 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=80384 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=81 bd_func=0 core=0
[bmprofile] local_layer: layer_id=207 layer_type=Store layer_name=
[bmprofile] tensor_id=206 is_in=1 shape=[1x2048x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=73920 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=207 is_in=0 shape=[1x2048x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=73920 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=168 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=209 layer_type=Store layer_name=
[bmprofile] tensor_id=208 is_in=1 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=80384 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=209 is_in=0 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=80384 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=170 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=218 layer_type=Load layer_name=
[bmprofile] tensor_id=212 is_in=1 shape=[1x32x32x2310] dtype=2 is_const=1 gaddr=618491297792 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=218 is_in=0 shape=[1x32x32x2310] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=170 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=219 layer_type=Load layer_name=
[bmprofile] tensor_id=209 is_in=1 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=687195250688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=219 is_in=0 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=79040 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=170 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=221 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=219 is_in=1 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=79040 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=218 is_in=1 shape=[1x32x32x2310] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=221 is_in=0 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=122880 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=85 bd_func=0 core=0
[bmprofile] local_layer: layer_id=220 layer_type=Load layer_name=
[bmprofile] tensor_id=213 is_in=1 shape=[1x32x1x33536] dtype=2 is_const=1 gaddr=618493665280 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=220 is_in=0 shape=[1x32x1x33536] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=170 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=223 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=221 is_in=1 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=122880 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=220 is_in=1 shape=[1x32x1x33536] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=223 is_in=0 shape=[1x2048x7x7] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=86 bd_func=0 core=0
[bmprofile] local_layer: layer_id=222 layer_type=Load layer_name=
[bmprofile] tensor_id=207 is_in=1 shape=[1x2048x7x7] dtype=3 is_const=0 gaddr=687195148288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=222 is_in=0 shape=[1x2048x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=172 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=225 layer_type=Add layer_name=
[bmprofile] tensor_id=223 is_in=1 shape=[1x2048x7x7] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=222 is_in=1 shape=[1x2048x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=225 is_in=0 shape=[1x2048x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=122880 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=87 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=87 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=87 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=87 bd_func=3 core=0
[bmprofile] local_layer: layer_id=224 layer_type=Load layer_name=
[bmprofile] tensor_id=214 is_in=1 shape=[1x32x1x32960] dtype=2 is_const=1 gaddr=618494738432 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=224 is_in=0 shape=[1x32x1x32960] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=174 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=227 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=225 is_in=1 shape=[1x2048x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=122880 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=224 is_in=1 shape=[1x32x1x32960] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=227 is_in=0 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=73920 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=88 bd_func=0 core=0
[bmprofile] local_layer: layer_id=226 layer_type=Load layer_name=
[bmprofile] tensor_id=215 is_in=1 shape=[1x32x32x2310] dtype=2 is_const=1 gaddr=618495795200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=226 is_in=0 shape=[1x32x32x2310] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=178 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=229 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=227 is_in=1 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=73920 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=226 is_in=1 shape=[1x32x32x2310] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=229 is_in=0 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=126976 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=89 bd_func=0 core=0
[bmprofile] local_layer: layer_id=228 layer_type=Load layer_name=
[bmprofile] tensor_id=216 is_in=1 shape=[1x32x1x33536] dtype=2 is_const=1 gaddr=618498162688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=228 is_in=0 shape=[1x32x1x33536] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=180 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=230 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=229 is_in=1 shape=[1x512x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=126976 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=228 is_in=1 shape=[1x32x1x33536] dtype=2 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=230 is_in=0 shape=[1x2048x7x7] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=73920 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=163840 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=90 bd_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=231 layer_type=Add layer_name=
[bmprofile] tensor_id=230 is_in=1 shape=[1x2048x7x7] dtype=2 is_const=0 gaddr=0 gsize=0 loffset=73920 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=225 is_in=1 shape=[1x2048x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=122880 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=231 is_in=0 shape=[1x2048x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=73920 nslice=1 hslice=7 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=90 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=90 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=90 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=90 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=232 layer_type=Pool2D layer_name=
[bmprofile] tensor_id=231 is_in=1 shape=[1x2048x7x7] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=73920 nslice=1 hslice=7 l2addr=0
[bmprofile] tensor_id=232 is_in=0 shape=[1x2048x1x1] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=196608 bd_func=12 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=90 bd_func=1 core=0
[bmprofile] local_layer: layer_id=233 layer_type=Reshape layer_name=
[bmprofile] tensor_id=232 is_in=1 shape=[1x2048x1x1] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=233 is_in=0 shape=[1x2048] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=78016 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=90 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=234 layer_type=Store layer_name=
[bmprofile] tensor_id=233 is_in=1 shape=[1x2048] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=78016 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=234 is_in=0 shape=[1x2048] dtype=3 is_const=0 gaddr=0 gsize=0 loffset=78016 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=191 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=239 layer_type=MatMul layer_name=
[bmprofile] tensor_id=234 is_in=1 shape=[1x2048] dtype=3 is_const=0 gaddr=687194918912 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=237 is_in=1 shape=[2048x1000] dtype=2 is_const=1 gaddr=618499235840 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=238 is_in=1 shape=[1x1000] dtype=6 is_const=1 gaddr=618501283840 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=239 is_in=0 shape=[1x1000] dtype=2 is_const=0 gaddr=687194923008 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=191 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=191 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=191 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=191 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=191 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=94 bd_func=2 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=192 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=192 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=96 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=96 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=194 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=194 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=98 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=98 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=196 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=196 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=100 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=100 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=198 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=198 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=102 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=102 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=200 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=200 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=104 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=104 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=202 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=202 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=106 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=106 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=204 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=204 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=108 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=108 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=206 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=206 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=110 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=110 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=208 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=208 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=112 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=112 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=210 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=210 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=114 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=114 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=212 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=212 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=116 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=116 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=214 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=214 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=118 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=118 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=216 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=216 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=120 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=120 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=218 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=218 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=122 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=122 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=220 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=220 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=124 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=124 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=222 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=222 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=126 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=126 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=224 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=224 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=128 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=128 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=226 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=226 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=130 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=130 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=228 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=228 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=132 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=132 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=230 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=230 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=134 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=134 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=232 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=232 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=136 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=136 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=234 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=234 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=138 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=138 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=236 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=236 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=140 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=140 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=238 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=238 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=142 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=142 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=240 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=240 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=144 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=144 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=242 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=242 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=146 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=146 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=244 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=244 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=148 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=148 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=246 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=246 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=150 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=150 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=248 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=248 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=152 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=152 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=250 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=250 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=154 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=154 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=156 bd_func=2 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=156 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=156 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=156 bd_func=4 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=256 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] end parallel.
[bmprofile] end to run subnet_id=0
[bmprofile] core_list=0,1,
[bmprofile] mtype=1 addr=4924944384 size=2158592 alloc=1705374960595187 free=1705374962286652 desc=neuron_mem0
[bmprofile] mtype=1 addr=4898947072 size=25997312 alloc=1705374960595225 free=0 desc=coeff
[bmprofile] mtype=1 addr=4927102976 size=12800 alloc=1705374960595423 free=1705374962286665 desc=bd_cmd_mem
[bmprofile] mtype=1 addr=4927119360 size=14208 alloc=1705374960601050 free=1705374962286677 desc=gdma_cmd_mem
[bmprofile] mtype=1 addr=4927135744 size=2048 alloc=1705374960605100 free=1705374962286688 desc=bd_cmd_mem
[bmprofile] mtype=1 addr=4927139840 size=2304 alloc=1705374960606182 free=1705374962286703 desc=gdma_cmd_mem
[bmprofile] mtype=1 addr=4927143936 size=1000 alloc=1705374960607805 free=0 desc=tensor
[bmprofile] mtype=1 addr=4927148032 size=150528 alloc=1705374960607869 free=0 desc=tensor
[bmprofile] mtype=1 addr=4927299584 size=16777216 alloc=1705374960611855 free=1705374962191740 desc=dyn_profile
[bmprofile] mtype=1 addr=4944076800 size=33554432 alloc=1705374960616514 free=1705374962184214 desc=bdc_perf_monitor
[bmprofile] mtype=1 addr=4977631232 size=268435456 alloc=1705374960652540 free=1705374962191717 desc=gdma_perf_monitor
[bmprofile] mtype=1 addr=5246066688 size=16777216 alloc=1705374960654828 free=1705374962284866 desc=dyn_profile
[bmprofile] mtype=1 addr=5262843904 size=33554432 alloc=1705374960659366 free=1705374962203000 desc=bdc_perf_monitor
[bmprofile] mtype=1 addr=5296398336 size=268435456 alloc=1705374960695373 free=1705374962275999 desc=gdma_perf_monitor
