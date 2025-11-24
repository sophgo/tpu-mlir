[bmprofile] arch=4
[bmprofile] net_name=blazeface
[bmprofile] tpu_freq=0
[bmprofile] is_mlir=1
...Start Profile Log...
[bmprofile] start to run subnet_id=0

[bmprofile] global_layer: layer_id=12 layer_type=Cast layer_name=
[bmprofile] tensor_id=-1 is_in=1 shape=[1x3x128x128] dtype=0 is_const=0 gaddr=687195422720 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=12 is_in=0 shape=[1x3x128x128] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=1 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=2 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=2 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=20 layer_type=Load layer_name=
[bmprofile] tensor_id=12 is_in=1 shape=[1x3x128x128] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=20 is_in=0 shape=[1x3x128x128] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=128 l2addr=0
[bmprofile] gdma cmd_id bd_id=2 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=21 layer_type=Load layer_name=
[bmprofile] tensor_id=14 is_in=1 shape=[1x24x1x80] dtype=1 is_const=1 gaddr=618475294720 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=21 is_in=0 shape=[1x24x1x80] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=73728 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=2 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=22 layer_type=Load layer_name=
[bmprofile] tensor_id=13 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=618475290624 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=22 is_in=0 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=2 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=25 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=20 is_in=1 shape=[1x3x128x128] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=128 l2addr=0
[bmprofile] tensor_id=21 is_in=1 shape=[1x24x1x80] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=73728 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=22 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=25 is_in=0 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=5 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=5 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=5 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=5 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=6 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=5 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=5 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=1 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 bd_func=3 core=0
[bmprofile] local_layer: layer_id=23 layer_type=Load layer_name=
[bmprofile] tensor_id=15 is_in=1 shape=[1x24x3x3] dtype=1 is_const=1 gaddr=618475298816 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=23 is_in=0 shape=[1x24x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=2 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=24 layer_type=Load layer_name=
[bmprofile] tensor_id=16 is_in=1 shape=[1x24x1x1] dtype=1 is_const=1 gaddr=618475302912 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=24 is_in=0 shape=[1x24x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=106496 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=2 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=28 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=25 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=23 is_in=1 shape=[1x24x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=24 is_in=1 shape=[1x24x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=106496 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=28 is_in=0 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=9 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=9 bd_func=1 core=0
[bmprofile] local_layer: layer_id=26 layer_type=Load layer_name=
[bmprofile] tensor_id=18 is_in=1 shape=[1x24x1x32] dtype=1 is_const=1 gaddr=618475311104 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=26 is_in=0 shape=[1x24x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=17 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=27 layer_type=Load layer_name=
[bmprofile] tensor_id=17 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=618475307008 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=27 is_in=0 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=90112 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=17 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=29 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=28 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=26 is_in=1 shape=[1x24x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=27 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=90112 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=29 is_in=0 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=11 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=11 bd_func=0 core=0
[bmprofile] local_layer: layer_id=30 layer_type=Add layer_name=
[bmprofile] tensor_id=29 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=25 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=30 is_in=0 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=11 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=11 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=31 layer_type=Store layer_name=
[bmprofile] tensor_id=30 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=31 is_in=0 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=34 layer_type=Pad layer_name=
[bmprofile] tensor_id=31 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=687195226112 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=34 is_in=0 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=687194996736 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=40 layer_type=Load layer_name=
[bmprofile] tensor_id=31 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=687195226112 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=40 is_in=0 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=41 layer_type=Load layer_name=
[bmprofile] tensor_id=35 is_in=1 shape=[1x24x3x3] dtype=1 is_const=1 gaddr=618475315200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=41 is_in=0 shape=[1x24x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=42 layer_type=Load layer_name=
[bmprofile] tensor_id=36 is_in=1 shape=[1x24x1x1] dtype=1 is_const=1 gaddr=618475319296 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=42 is_in=0 shape=[1x24x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=45 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=40 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=41 is_in=1 shape=[1x24x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=42 is_in=1 shape=[1x24x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=45 is_in=0 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=17 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=17 bd_func=1 core=0
[bmprofile] local_layer: layer_id=43 layer_type=Load layer_name=
[bmprofile] tensor_id=38 is_in=1 shape=[1x28x1x32] dtype=1 is_const=1 gaddr=618475327488 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=43 is_in=0 shape=[1x28x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=44 layer_type=Load layer_name=
[bmprofile] tensor_id=37 is_in=1 shape=[1x28x1x1] dtype=0 is_const=1 gaddr=618475323392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=44 is_in=0 shape=[1x28x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=47 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=45 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=43 is_in=1 shape=[1x28x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=44 is_in=1 shape=[1x28x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=47 is_in=0 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=19 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=19 bd_func=0 core=0
[bmprofile] local_layer: layer_id=46 layer_type=Load layer_name=
[bmprofile] tensor_id=34 is_in=1 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=687194996736 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=46 is_in=0 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=25 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=48 layer_type=Add layer_name=
[bmprofile] tensor_id=47 is_in=1 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=46 is_in=1 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=48 is_in=0 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=20 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=20 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=50 layer_type=Pool2D layer_name=
[bmprofile] tensor_id=48 is_in=1 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=50 is_in=0 shape=[1x28x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=20 bd_func=1 core=0
[bmprofile] local_layer: layer_id=49 layer_type=Store layer_name=
[bmprofile] tensor_id=48 is_in=1 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=49 is_in=0 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=29 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=51 layer_type=Store layer_name=
[bmprofile] tensor_id=50 is_in=1 shape=[1x28x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=51 is_in=0 shape=[1x28x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=30 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=54 layer_type=Pad layer_name=
[bmprofile] tensor_id=51 is_in=1 shape=[1x28x32x32] dtype=1 is_const=0 gaddr=687195619328 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=54 is_in=0 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=687195078656 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=30 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=30 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=60 layer_type=Load layer_name=
[bmprofile] tensor_id=49 is_in=1 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=60 is_in=0 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=30 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=61 layer_type=Load layer_name=
[bmprofile] tensor_id=55 is_in=1 shape=[1x28x3x3] dtype=1 is_const=1 gaddr=618475331584 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=61 is_in=0 shape=[1x28x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=30 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=62 layer_type=Load layer_name=
[bmprofile] tensor_id=56 is_in=1 shape=[1x28x1x1] dtype=1 is_const=1 gaddr=618475335680 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=62 is_in=0 shape=[1x28x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=30 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=65 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=60 is_in=1 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=61 is_in=1 shape=[1x28x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=62 is_in=1 shape=[1x28x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=65 is_in=0 shape=[1x28x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=27 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=27 bd_func=1 core=0
[bmprofile] local_layer: layer_id=63 layer_type=Load layer_name=
[bmprofile] tensor_id=58 is_in=1 shape=[1x32x1x32] dtype=1 is_const=1 gaddr=618475343872 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=63 is_in=0 shape=[1x32x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=34816 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=30 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=64 layer_type=Load layer_name=
[bmprofile] tensor_id=57 is_in=1 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=618475339776 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=64 is_in=0 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=30 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=67 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=65 is_in=1 shape=[1x28x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=63 is_in=1 shape=[1x32x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=34816 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=64 is_in=1 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=67 is_in=0 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=29 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=29 bd_func=0 core=0
[bmprofile] local_layer: layer_id=66 layer_type=Load layer_name=
[bmprofile] tensor_id=54 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=687195078656 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=66 is_in=0 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=32 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=68 layer_type=Add layer_name=
[bmprofile] tensor_id=67 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=66 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=68 is_in=0 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=30 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=30 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=69 layer_type=Store layer_name=
[bmprofile] tensor_id=68 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=69 is_in=0 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=72 layer_type=Pad layer_name=
[bmprofile] tensor_id=69 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=687195013120 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=72 is_in=0 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=78 layer_type=Load layer_name=
[bmprofile] tensor_id=69 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=687195013120 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=78 is_in=0 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=79 layer_type=Load layer_name=
[bmprofile] tensor_id=73 is_in=1 shape=[1x32x3x3] dtype=1 is_const=1 gaddr=618475347968 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=79 is_in=0 shape=[1x32x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=80 layer_type=Load layer_name=
[bmprofile] tensor_id=74 is_in=1 shape=[1x32x1x1] dtype=1 is_const=1 gaddr=618475352064 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=80 is_in=0 shape=[1x32x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=83 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=78 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=79 is_in=1 shape=[1x32x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=80 is_in=1 shape=[1x32x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=83 is_in=0 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=36 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=36 bd_func=1 core=0
[bmprofile] local_layer: layer_id=81 layer_type=Load layer_name=
[bmprofile] tensor_id=76 is_in=1 shape=[1x36x1x32] dtype=1 is_const=1 gaddr=618475360256 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=81 is_in=0 shape=[1x36x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=82 layer_type=Load layer_name=
[bmprofile] tensor_id=75 is_in=1 shape=[1x36x1x1] dtype=0 is_const=1 gaddr=618475356160 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=82 is_in=0 shape=[1x36x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=85 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=83 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=81 is_in=1 shape=[1x36x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=82 is_in=1 shape=[1x36x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=85 is_in=0 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=38 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=38 bd_func=0 core=0
[bmprofile] local_layer: layer_id=84 layer_type=Load layer_name=
[bmprofile] tensor_id=72 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=84 is_in=0 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=38 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=86 layer_type=Add layer_name=
[bmprofile] tensor_id=85 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=84 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=86 is_in=0 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=39 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=39 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=87 layer_type=Store layer_name=
[bmprofile] tensor_id=86 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=87 is_in=0 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=42 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=90 layer_type=Pad layer_name=
[bmprofile] tensor_id=87 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=687194939392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=90 is_in=0 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=687194853376 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=42 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=42 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=96 layer_type=Load layer_name=
[bmprofile] tensor_id=87 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=687194939392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=96 is_in=0 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=42 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=97 layer_type=Load layer_name=
[bmprofile] tensor_id=91 is_in=1 shape=[1x36x3x3] dtype=1 is_const=1 gaddr=618475364352 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=97 is_in=0 shape=[1x36x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=42 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=98 layer_type=Load layer_name=
[bmprofile] tensor_id=92 is_in=1 shape=[1x36x1x1] dtype=1 is_const=1 gaddr=618475368448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=98 is_in=0 shape=[1x36x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=42 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=101 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=96 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=97 is_in=1 shape=[1x36x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=98 is_in=1 shape=[1x36x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=101 is_in=0 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=45 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=45 bd_func=1 core=0
[bmprofile] local_layer: layer_id=99 layer_type=Load layer_name=
[bmprofile] tensor_id=94 is_in=1 shape=[1x42x1x48] dtype=1 is_const=1 gaddr=618475376640 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=99 is_in=0 shape=[1x42x1x48] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=42 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=100 layer_type=Load layer_name=
[bmprofile] tensor_id=93 is_in=1 shape=[1x42x1x1] dtype=0 is_const=1 gaddr=618475372544 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=100 is_in=0 shape=[1x42x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=42 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=103 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=101 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=99 is_in=1 shape=[1x42x1x48] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=100 is_in=1 shape=[1x42x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=103 is_in=0 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=47 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=47 bd_func=0 core=0
[bmprofile] local_layer: layer_id=102 layer_type=Load layer_name=
[bmprofile] tensor_id=90 is_in=1 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=687194853376 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=102 is_in=0 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=44 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=104 layer_type=Add layer_name=
[bmprofile] tensor_id=103 is_in=1 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=102 is_in=1 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=104 is_in=0 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=48 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=48 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=106 layer_type=Pool2D layer_name=
[bmprofile] tensor_id=104 is_in=1 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=106 is_in=0 shape=[1x42x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=48 bd_func=1 core=0
[bmprofile] local_layer: layer_id=105 layer_type=Store layer_name=
[bmprofile] tensor_id=104 is_in=1 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=105 is_in=0 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=48 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=107 layer_type=Store layer_name=
[bmprofile] tensor_id=106 is_in=1 shape=[1x42x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=107 is_in=0 shape=[1x42x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=49 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=110 layer_type=Pad layer_name=
[bmprofile] tensor_id=107 is_in=1 shape=[1x42x16x16] dtype=1 is_const=0 gaddr=687195013120 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=110 is_in=0 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=687194877952 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=49 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=49 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=116 layer_type=Load layer_name=
[bmprofile] tensor_id=105 is_in=1 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=116 is_in=0 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=49 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=117 layer_type=Load layer_name=
[bmprofile] tensor_id=111 is_in=1 shape=[1x42x3x3] dtype=1 is_const=1 gaddr=618475380736 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=117 is_in=0 shape=[1x42x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=49 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=118 layer_type=Load layer_name=
[bmprofile] tensor_id=112 is_in=1 shape=[1x42x1x1] dtype=1 is_const=1 gaddr=618475384832 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=118 is_in=0 shape=[1x42x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=49 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=121 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=116 is_in=1 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=117 is_in=1 shape=[1x42x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=118 is_in=1 shape=[1x42x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=121 is_in=0 shape=[1x42x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=55 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=55 bd_func=1 core=0
[bmprofile] local_layer: layer_id=119 layer_type=Load layer_name=
[bmprofile] tensor_id=114 is_in=1 shape=[1x48x1x48] dtype=1 is_const=1 gaddr=618475393024 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=119 is_in=0 shape=[1x48x1x48] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=33792 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=49 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=120 layer_type=Load layer_name=
[bmprofile] tensor_id=113 is_in=1 shape=[1x48x1x1] dtype=0 is_const=1 gaddr=618475388928 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=120 is_in=0 shape=[1x48x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=49 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=123 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=121 is_in=1 shape=[1x42x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=119 is_in=1 shape=[1x48x1x48] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=33792 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=120 is_in=1 shape=[1x48x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=123 is_in=0 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=57 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=57 bd_func=0 core=0
[bmprofile] local_layer: layer_id=122 layer_type=Load layer_name=
[bmprofile] tensor_id=110 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=687194877952 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=122 is_in=0 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=51 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=124 layer_type=Add layer_name=
[bmprofile] tensor_id=123 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=122 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=124 is_in=0 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=58 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=58 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=125 layer_type=Store layer_name=
[bmprofile] tensor_id=124 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=125 is_in=0 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=55 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=128 layer_type=Pad layer_name=
[bmprofile] tensor_id=125 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=687194853376 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=128 is_in=0 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=55 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=55 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=134 layer_type=Load layer_name=
[bmprofile] tensor_id=125 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=687194853376 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=134 is_in=0 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=55 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=135 layer_type=Load layer_name=
[bmprofile] tensor_id=129 is_in=1 shape=[1x48x3x3] dtype=1 is_const=1 gaddr=618475401216 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=135 is_in=0 shape=[1x48x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=55 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=136 layer_type=Load layer_name=
[bmprofile] tensor_id=130 is_in=1 shape=[1x48x1x1] dtype=1 is_const=1 gaddr=618475405312 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=136 is_in=0 shape=[1x48x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=55 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=139 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=134 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=135 is_in=1 shape=[1x48x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=136 is_in=1 shape=[1x48x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=139 is_in=0 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=64 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=64 bd_func=1 core=0
[bmprofile] local_layer: layer_id=137 layer_type=Load layer_name=
[bmprofile] tensor_id=132 is_in=1 shape=[1x56x1x48] dtype=1 is_const=1 gaddr=618475413504 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=137 is_in=0 shape=[1x56x1x48] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=33792 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=55 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=138 layer_type=Load layer_name=
[bmprofile] tensor_id=131 is_in=1 shape=[1x56x1x1] dtype=0 is_const=1 gaddr=618475409408 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=138 is_in=0 shape=[1x56x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=55 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=141 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=139 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=137 is_in=1 shape=[1x56x1x48] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=33792 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=138 is_in=1 shape=[1x56x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=141 is_in=0 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=66 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=66 bd_func=0 core=0
[bmprofile] local_layer: layer_id=140 layer_type=Load layer_name=
[bmprofile] tensor_id=128 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=140 is_in=0 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=57 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=142 layer_type=Add layer_name=
[bmprofile] tensor_id=141 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=140 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=142 is_in=0 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=67 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=67 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=143 layer_type=Store layer_name=
[bmprofile] tensor_id=142 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=143 is_in=0 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=146 layer_type=Pad layer_name=
[bmprofile] tensor_id=143 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=687194800128 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=146 is_in=0 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=152 layer_type=Load layer_name=
[bmprofile] tensor_id=143 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=687194800128 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=152 is_in=0 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=153 layer_type=Load layer_name=
[bmprofile] tensor_id=147 is_in=1 shape=[1x56x3x3] dtype=1 is_const=1 gaddr=618475421696 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=153 is_in=0 shape=[1x56x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=154 layer_type=Load layer_name=
[bmprofile] tensor_id=148 is_in=1 shape=[1x56x1x1] dtype=1 is_const=1 gaddr=618475425792 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=154 is_in=0 shape=[1x56x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=157 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=152 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=153 is_in=1 shape=[1x56x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=154 is_in=1 shape=[1x56x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=157 is_in=0 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=73 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=73 bd_func=1 core=0
[bmprofile] local_layer: layer_id=155 layer_type=Load layer_name=
[bmprofile] tensor_id=150 is_in=1 shape=[1x64x1x64] dtype=1 is_const=1 gaddr=618475433984 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=155 is_in=0 shape=[1x64x1x64] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=33792 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=156 layer_type=Load layer_name=
[bmprofile] tensor_id=149 is_in=1 shape=[1x64x1x1] dtype=0 is_const=1 gaddr=618475429888 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=156 is_in=0 shape=[1x64x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=159 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=157 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=155 is_in=1 shape=[1x64x1x64] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=33792 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=156 is_in=1 shape=[1x64x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=159 is_in=0 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=75 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=75 bd_func=0 core=0
[bmprofile] local_layer: layer_id=158 layer_type=Load layer_name=
[bmprofile] tensor_id=146 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=158 is_in=0 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=63 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=160 layer_type=Add layer_name=
[bmprofile] tensor_id=159 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=158 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=160 is_in=0 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=76 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=76 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=161 layer_type=Store layer_name=
[bmprofile] tensor_id=160 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=161 is_in=0 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=67 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=164 layer_type=Pad layer_name=
[bmprofile] tensor_id=161 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=687194845184 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=164 is_in=0 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=67 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=67 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=170 layer_type=Load layer_name=
[bmprofile] tensor_id=161 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=687194845184 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=170 is_in=0 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=67 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=171 layer_type=Load layer_name=
[bmprofile] tensor_id=165 is_in=1 shape=[1x64x3x3] dtype=1 is_const=1 gaddr=618475442176 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=171 is_in=0 shape=[1x64x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=67 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=172 layer_type=Load layer_name=
[bmprofile] tensor_id=166 is_in=1 shape=[1x64x1x1] dtype=1 is_const=1 gaddr=618475446272 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=172 is_in=0 shape=[1x64x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=67 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=175 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=170 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=171 is_in=1 shape=[1x64x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=172 is_in=1 shape=[1x64x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=175 is_in=0 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=82 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=82 bd_func=1 core=0
[bmprofile] local_layer: layer_id=173 layer_type=Load layer_name=
[bmprofile] tensor_id=168 is_in=1 shape=[1x72x1x64] dtype=1 is_const=1 gaddr=618475454464 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=173 is_in=0 shape=[1x72x1x64] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=17920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=67 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=174 layer_type=Load layer_name=
[bmprofile] tensor_id=167 is_in=1 shape=[1x72x1x1] dtype=0 is_const=1 gaddr=618475450368 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=174 is_in=0 shape=[1x72x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=67 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=177 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=175 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=173 is_in=1 shape=[1x72x1x64] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=17920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=174 is_in=1 shape=[1x72x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=177 is_in=0 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=84 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=84 bd_func=0 core=0
[bmprofile] local_layer: layer_id=176 layer_type=Load layer_name=
[bmprofile] tensor_id=164 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=176 is_in=0 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=69 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=178 layer_type=Add layer_name=
[bmprofile] tensor_id=177 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=176 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=178 is_in=0 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=85 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=85 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=179 layer_type=Store layer_name=
[bmprofile] tensor_id=178 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=179 is_in=0 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=73 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=182 layer_type=Pad layer_name=
[bmprofile] tensor_id=179 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=687194808320 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=182 is_in=0 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=73 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=73 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=188 layer_type=Load layer_name=
[bmprofile] tensor_id=179 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=687194808320 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=188 is_in=0 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=73 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=189 layer_type=Load layer_name=
[bmprofile] tensor_id=183 is_in=1 shape=[1x72x3x3] dtype=1 is_const=1 gaddr=618475466752 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=189 is_in=0 shape=[1x72x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=73 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=190 layer_type=Load layer_name=
[bmprofile] tensor_id=184 is_in=1 shape=[1x72x1x1] dtype=1 is_const=1 gaddr=618475470848 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=190 is_in=0 shape=[1x72x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=73 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=193 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=188 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=189 is_in=1 shape=[1x72x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=190 is_in=1 shape=[1x72x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=193 is_in=0 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=91 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=91 bd_func=1 core=0
[bmprofile] local_layer: layer_id=191 layer_type=Load layer_name=
[bmprofile] tensor_id=186 is_in=1 shape=[1x80x1x80] dtype=1 is_const=1 gaddr=618475479040 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=191 is_in=0 shape=[1x80x1x80] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=73 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=192 layer_type=Load layer_name=
[bmprofile] tensor_id=185 is_in=1 shape=[1x80x1x1] dtype=0 is_const=1 gaddr=618475474944 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=192 is_in=0 shape=[1x80x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=73 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=195 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=193 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=191 is_in=1 shape=[1x80x1x80] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=192 is_in=1 shape=[1x80x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=195 is_in=0 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=93 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=93 bd_func=0 core=0
[bmprofile] local_layer: layer_id=194 layer_type=Load layer_name=
[bmprofile] tensor_id=182 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=194 is_in=0 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=75 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=196 layer_type=Add layer_name=
[bmprofile] tensor_id=195 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=194 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=196 is_in=0 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=94 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=94 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=197 layer_type=Store layer_name=
[bmprofile] tensor_id=196 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=197 is_in=0 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=79 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=200 layer_type=Pad layer_name=
[bmprofile] tensor_id=197 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=687194857472 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=200 is_in=0 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=687194812416 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=79 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=79 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=206 layer_type=Load layer_name=
[bmprofile] tensor_id=197 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=687194857472 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=206 is_in=0 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=79 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=207 layer_type=Load layer_name=
[bmprofile] tensor_id=201 is_in=1 shape=[1x80x3x3] dtype=1 is_const=1 gaddr=618475495424 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=207 is_in=0 shape=[1x80x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=79 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=208 layer_type=Load layer_name=
[bmprofile] tensor_id=202 is_in=1 shape=[1x80x1x1] dtype=1 is_const=1 gaddr=618475499520 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=208 is_in=0 shape=[1x80x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=79 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=211 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=206 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=207 is_in=1 shape=[1x80x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=208 is_in=1 shape=[1x80x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=211 is_in=0 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=100 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=100 bd_func=1 core=0
[bmprofile] local_layer: layer_id=209 layer_type=Load layer_name=
[bmprofile] tensor_id=204 is_in=1 shape=[1x88x1x80] dtype=1 is_const=1 gaddr=618475507712 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=209 is_in=0 shape=[1x88x1x80] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=24960 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=79 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=210 layer_type=Load layer_name=
[bmprofile] tensor_id=203 is_in=1 shape=[1x88x1x1] dtype=0 is_const=1 gaddr=618475503616 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=210 is_in=0 shape=[1x88x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=79 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=213 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=211 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=209 is_in=1 shape=[1x88x1x80] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=24960 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=210 is_in=1 shape=[1x88x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=213 is_in=0 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=102 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=102 bd_func=0 core=0
[bmprofile] local_layer: layer_id=212 layer_type=Load layer_name=
[bmprofile] tensor_id=200 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=687194812416 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=212 is_in=0 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=81 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=214 layer_type=Add layer_name=
[bmprofile] tensor_id=213 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=212 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=214 is_in=0 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=103 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=103 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=216 layer_type=Pool2D layer_name=
[bmprofile] tensor_id=214 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=216 is_in=0 shape=[1x88x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=103 bd_func=1 core=0
[bmprofile] local_layer: layer_id=215 layer_type=Store layer_name=
[bmprofile] tensor_id=214 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=215 is_in=0 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=85 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=217 layer_type=Store layer_name=
[bmprofile] tensor_id=216 is_in=1 shape=[1x88x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=217 is_in=0 shape=[1x88x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=220 layer_type=Pad layer_name=
[bmprofile] tensor_id=217 is_in=1 shape=[1x88x8x8] dtype=1 is_const=0 gaddr=687194898432 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=220 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=687194824704 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
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
[bmprofile] local_layer: layer_id=244 layer_type=Load layer_name=
[bmprofile] tensor_id=222 is_in=1 shape=[1x88x1x1] dtype=1 is_const=1 gaddr=618475528192 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=244 is_in=0 shape=[1x88x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32960 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=245 layer_type=Load layer_name=
[bmprofile] tensor_id=221 is_in=1 shape=[1x88x3x3] dtype=1 is_const=1 gaddr=618475524096 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=245 is_in=0 shape=[1x88x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=246 layer_type=Load layer_name=
[bmprofile] tensor_id=215 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=246 is_in=0 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=249 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=246 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=245 is_in=1 shape=[1x88x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=244 is_in=1 shape=[1x88x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32960 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=249 is_in=0 shape=[1x88x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=110 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=110 bd_func=1 core=0
[bmprofile] local_layer: layer_id=247 layer_type=Load layer_name=
[bmprofile] tensor_id=224 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=618475536384 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=247 is_in=0 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=8704 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=248 layer_type=Load layer_name=
[bmprofile] tensor_id=223 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=618475532288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=248 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=251 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=249 is_in=1 shape=[1x88x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=247 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=8704 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=248 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=251 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=112 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=112 bd_func=0 core=0
[bmprofile] local_layer: layer_id=250 layer_type=Load layer_name=
[bmprofile] tensor_id=220 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=687194824704 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=250 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=88 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=254 layer_type=Add layer_name=
[bmprofile] tensor_id=251 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=250 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=254 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=1536 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=113 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=113 bd_func=3 core=0
[bmprofile] local_layer: layer_id=252 layer_type=Load layer_name=
[bmprofile] tensor_id=225 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=618475556864 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=252 is_in=0 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=8704 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=90 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=253 layer_type=Load layer_name=
[bmprofile] tensor_id=226 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=618475560960 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=253 is_in=0 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=90 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=257 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=254 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=1536 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=252 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=8704 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=253 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=257 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=115 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=115 bd_func=1 core=0
[bmprofile] local_layer: layer_id=255 layer_type=Load layer_name=
[bmprofile] tensor_id=228 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=618475569152 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=255 is_in=0 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=92 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=256 layer_type=Load layer_name=
[bmprofile] tensor_id=227 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=618475565056 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=256 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=92 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=260 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=257 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=255 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=256 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=260 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=1920 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=117 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=117 bd_func=0 core=0
[bmprofile] local_layer: layer_id=261 layer_type=Add layer_name=
[bmprofile] tensor_id=260 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=1920 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=254 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=1536 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=261 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=117 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=117 bd_func=3 core=0
[bmprofile] local_layer: layer_id=258 layer_type=Load layer_name=
[bmprofile] tensor_id=230 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=618475593728 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=258 is_in=0 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=94 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=259 layer_type=Load layer_name=
[bmprofile] tensor_id=229 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=618475589632 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=259 is_in=0 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=94 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=264 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=261 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=259 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=258 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=264 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=119 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=119 bd_func=1 core=0
[bmprofile] local_layer: layer_id=262 layer_type=Load layer_name=
[bmprofile] tensor_id=232 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=618475601920 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=262 is_in=0 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=1536 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=98 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=263 layer_type=Load layer_name=
[bmprofile] tensor_id=231 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=618475597824 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=263 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=98 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=267 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=264 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=262 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=1536 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=263 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=267 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8576 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=121 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=121 bd_func=0 core=0
[bmprofile] local_layer: layer_id=268 layer_type=Add layer_name=
[bmprofile] tensor_id=267 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8576 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=261 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=268 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=121 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=121 bd_func=3 core=0
[bmprofile] local_layer: layer_id=265 layer_type=Load layer_name=
[bmprofile] tensor_id=234 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=618475626496 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=265 is_in=0 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=100 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=266 layer_type=Load layer_name=
[bmprofile] tensor_id=233 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=618475622400 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=266 is_in=0 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=100 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=271 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=268 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=266 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=265 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=271 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=123 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=123 bd_func=1 core=0
[bmprofile] local_layer: layer_id=269 layer_type=Load layer_name=
[bmprofile] tensor_id=236 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=618475634688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=269 is_in=0 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=1536 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=104 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=270 layer_type=Load layer_name=
[bmprofile] tensor_id=235 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=618475630592 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=270 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=104 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=274 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=271 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=269 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=1536 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=270 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=274 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8576 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=125 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=125 bd_func=0 core=0
[bmprofile] local_layer: layer_id=275 layer_type=Add layer_name=
[bmprofile] tensor_id=274 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8576 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=268 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=275 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=125 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=125 bd_func=3 core=0
[bmprofile] local_layer: layer_id=272 layer_type=Load layer_name=
[bmprofile] tensor_id=238 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=618475659264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=272 is_in=0 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=106 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=273 layer_type=Load layer_name=
[bmprofile] tensor_id=237 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=618475655168 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=273 is_in=0 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=106 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=278 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=275 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=273 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=272 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=278 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=127 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=127 bd_func=1 core=0
[bmprofile] local_layer: layer_id=276 layer_type=Load layer_name=
[bmprofile] tensor_id=240 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=618475667456 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=276 is_in=0 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=1536 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=110 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=277 layer_type=Load layer_name=
[bmprofile] tensor_id=239 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=618475663360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=277 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=110 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=281 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=278 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=276 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=1536 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=277 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=281 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16768 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=129 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=129 bd_func=0 core=0
[bmprofile] local_layer: layer_id=282 layer_type=Add layer_name=
[bmprofile] tensor_id=281 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16768 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=275 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=282 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=129 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=129 bd_func=3 core=0
[bmprofile] local_layer: layer_id=279 layer_type=Load layer_name=
[bmprofile] tensor_id=242 is_in=1 shape=[1x2x1x96] dtype=1 is_const=1 gaddr=618475692032 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=279 is_in=0 shape=[1x2x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=112 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=280 layer_type=Load layer_name=
[bmprofile] tensor_id=241 is_in=1 shape=[1x2x1x1] dtype=0 is_const=1 gaddr=618475687936 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=280 is_in=0 shape=[1x2x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=112 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=284 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=246 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=279 is_in=1 shape=[1x2x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=280 is_in=1 shape=[1x2x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=284 is_in=0 shape=[1x2x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=131 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=131 bd_func=0 core=0
[bmprofile] local_layer: layer_id=283 layer_type=Store layer_name=
[bmprofile] tensor_id=282 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=283 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=116 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
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
[bmprofile] local_layer: layer_id=285 layer_type=Store layer_name=
[bmprofile] tensor_id=284 is_in=1 shape=[1x2x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=285 is_in=0 shape=[1x2x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=118 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
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

[bmprofile] global_layer: layer_id=288 layer_type=Permute layer_name=
[bmprofile] tensor_id=285 is_in=1 shape=[1x2x16x16] dtype=1 is_const=0 gaddr=687194836992 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=288 is_in=0 shape=[1x16x16x2] dtype=1 is_const=0 gaddr=687194832896 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=118 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=134 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=119 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=289 layer_type=Reshape layer_name=
[bmprofile] tensor_id=288 is_in=1 shape=[1x16x16x2] dtype=1 is_const=0 gaddr=687194832896 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=289 is_in=0 shape=[1x512x1] dtype=1 is_const=0 gaddr=687194832896 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=292 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=283 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=687194812416 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=291 is_in=1 shape=[1x6x1x96] dtype=1 is_const=1 gaddr=618475700224 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=290 is_in=1 shape=[1x6x1x1] dtype=0 is_const=1 gaddr=618475696128 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=292 is_in=0 shape=[1x6x8x8] dtype=1 is_const=0 gaddr=687194824704 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=119 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=119 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=119 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=138 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=138 bd_func=0 core=0
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=121 gdma_id=0 gdma_dir=0 gdma_func=0 core=0

[bmprofile] global_layer: layer_id=293 layer_type=Permute layer_name=
[bmprofile] tensor_id=292 is_in=1 shape=[1x6x8x8] dtype=1 is_const=0 gaddr=687194824704 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=293 is_in=0 shape=[1x8x8x6] dtype=1 is_const=0 gaddr=687194828800 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=121 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=140 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=122 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=294 layer_type=Reshape layer_name=
[bmprofile] tensor_id=293 is_in=1 shape=[1x8x8x6] dtype=1 is_const=0 gaddr=687194828800 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=294 is_in=0 shape=[1x384x1] dtype=1 is_const=0 gaddr=687194828800 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=295 layer_type=Concat layer_name=
[bmprofile] tensor_id=289 is_in=1 shape=[1x512x1] dtype=1 is_const=0 gaddr=687194832896 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=294 is_in=1 shape=[1x384x1] dtype=1 is_const=0 gaddr=687194828800 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=295 is_in=0 shape=[1x896x1] dtype=1 is_const=0 gaddr=687194824704 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=122 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=122 gdma_id=0 gdma_dir=0 gdma_func=0 core=0

[bmprofile] global_layer: layer_id=296 layer_type=Cast layer_name=
[bmprofile] tensor_id=295 is_in=1 shape=[1x896x1] dtype=1 is_const=0 gaddr=687194824704 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=296 is_in=0 shape=[1x896x1] dtype=0 is_const=0 gaddr=687194857472 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=122 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=144 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=123 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=299 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=215 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=298 is_in=1 shape=[1x32x1x96] dtype=1 is_const=1 gaddr=618475708416 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=297 is_in=1 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=618475704320 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=299 is_in=0 shape=[1x32x16x16] dtype=1 is_const=0 gaddr=687194841088 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=123 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=123 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=123 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=148 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=148 bd_func=0 core=0
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=125 gdma_id=0 gdma_dir=0 gdma_func=0 core=0

[bmprofile] global_layer: layer_id=300 layer_type=Permute layer_name=
[bmprofile] tensor_id=299 is_in=1 shape=[1x32x16x16] dtype=1 is_const=0 gaddr=687194841088 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=300 is_in=0 shape=[1x16x16x32] dtype=1 is_const=0 gaddr=687194824704 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=125 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=150 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=126 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=301 layer_type=Reshape layer_name=
[bmprofile] tensor_id=300 is_in=1 shape=[1x16x16x32] dtype=1 is_const=0 gaddr=687194824704 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=301 is_in=0 shape=[1x512x16] dtype=1 is_const=0 gaddr=687194824704 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=304 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=283 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=687194812416 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=303 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=618475720704 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=302 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=618475716608 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=304 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=126 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=126 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=126 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=154 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=154 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=126 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=126 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=156 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=156 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=128 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=128 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=128 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=159 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=159 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=130 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=132 gdma_id=0 gdma_dir=0 gdma_func=0 core=0

[bmprofile] global_layer: layer_id=305 layer_type=Permute layer_name=
[bmprofile] tensor_id=304 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=687194767360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=305 is_in=0 shape=[1x8x8x96] dtype=1 is_const=0 gaddr=687194841088 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=132 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=162 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=133 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=306 layer_type=Reshape layer_name=
[bmprofile] tensor_id=305 is_in=1 shape=[1x8x8x96] dtype=1 is_const=0 gaddr=687194841088 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=306 is_in=0 shape=[1x384x16] dtype=1 is_const=0 gaddr=687194841088 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=308 layer_type=Load layer_name=
[bmprofile] tensor_id=301 is_in=1 shape=[1x512x16] dtype=1 is_const=0 gaddr=687194824704 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=308 is_in=0 shape=[1x512x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=133 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=309 layer_type=Load layer_name=
[bmprofile] tensor_id=306 is_in=1 shape=[1x384x16] dtype=1 is_const=0 gaddr=687194841088 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=309 is_in=0 shape=[1x384x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=133 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=310 layer_type=Concat layer_name=
[bmprofile] tensor_id=308 is_in=1 shape=[1x512x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=309 is_in=1 shape=[1x384x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=310 is_in=0 shape=[1x896x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=165 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=165 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=311 layer_type=Cast layer_name=
[bmprofile] tensor_id=310 is_in=1 shape=[1x896x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=311 is_in=0 shape=[1x896x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=165 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=312 layer_type=Store layer_name=
[bmprofile] tensor_id=311 is_in=1 shape=[1x896x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=312 is_in=0 shape=[1x896x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=136 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] end to run subnet_id=0
[bmprofile] core_list=0,
[bmprofile] mtype=1 addr=4898947072 size=450560 alloc=1764052996410254 free=0 desc=coeff
[bmprofile] mtype=1 addr=4899397632 size=8576 alloc=1764052996410394 free=1764052997435287 desc=bd_cmd_mem
[bmprofile] mtype=1 addr=4899409920 size=16000 alloc=1764052996411132 free=1764052997435298 desc=gdma_cmd_mem
[bmprofile] mtype=1 addr=4899426304 size=57344 alloc=1764052996413493 free=0 desc=io_mem
[bmprofile] mtype=1 addr=4899483648 size=3584 alloc=1764052996413510 free=0 desc=io_mem
[bmprofile] mtype=1 addr=4899487744 size=196608 alloc=1764052996413589 free=0 desc=io_mem
[bmprofile] mtype=1 addr=4899684352 size=909312 alloc=1764052996414224 free=1764052997435341 desc=neuron_mem_1
[bmprofile] mtype=1 addr=4900593664 size=16777216 alloc=1764052996418556 free=1764052997434542 desc=dyn_profile
[bmprofile] mtype=1 addr=4917370880 size=33554432 alloc=1764052996423117 free=1764052997332667 desc=bdc_perf_monitor
[bmprofile] mtype=1 addr=4950925312 size=268435456 alloc=1764052996459178 free=1764052997429305 desc=gdma_perf_monitor
