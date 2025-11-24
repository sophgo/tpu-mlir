[bmprofile] arch=8
[bmprofile] net_name=blazeface
[bmprofile] tpu_freq=1000
[bmprofile] is_mlir=1
...Start Profile Log...
[bmprofile] start to run subnet_id=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=15 layer_type=Load layer_name=
[bmprofile] tensor_id=-1 is_in=1 shape=[1x3x128x128] dtype=0 is_const=0 gaddr=2201171394560 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=15 is_in=0 shape=[1x3x128x128] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=46 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=18 layer_type=Cast layer_name=
[bmprofile] tensor_id=15 is_in=1 shape=[1x3x128x128] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=46 l2addr=0
[bmprofile] tensor_id=18 is_in=0 shape=[1x3x128x128] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=46 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048577 bd_func=3 core=0
[bmprofile] local_layer: layer_id=16 layer_type=Load layer_name=
[bmprofile] tensor_id=13 is_in=1 shape=[1x24x1x200] dtype=8 is_const=1 gaddr=1101659115520 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=16 is_in=0 shape=[1x24x1x200] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=17 layer_type=Load layer_name=
[bmprofile] tensor_id=12 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=1101659111424 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=17 is_in=0 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=19 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=18 is_in=1 shape=[1x3x128x128] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=46 l2addr=0
[bmprofile] tensor_id=16 is_in=1 shape=[1x24x1x200] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=17 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=19 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=22 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048579 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048579 bd_func=3 core=0
[bmprofile] local_layer: layer_id=15 layer_type=Load layer_name=
[bmprofile] tensor_id=-2 is_in=1 shape=[1x3x128x128] dtype=0 is_const=0 gaddr=2201171394560 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=15 is_in=0 shape=[1x3x128x128] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=45 l2addr=0
[bmprofile] gdma cmd_id bd_id=1 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=18 layer_type=Cast layer_name=
[bmprofile] tensor_id=15 is_in=1 shape=[1x3x128x128] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=45 l2addr=0
[bmprofile] tensor_id=18 is_in=0 shape=[1x3x128x128] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=45 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048580 bd_func=3 core=0
[bmprofile] local_layer: layer_id=20 layer_type=Store layer_name=
[bmprofile] tensor_id=19 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=22 l2addr=0
[bmprofile] tensor_id=20 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=22 l2addr=0
[bmprofile] gdma cmd_id bd_id=3 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=19 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=18 is_in=1 shape=[1x3x128x128] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=45 l2addr=0
[bmprofile] tensor_id=16 is_in=1 shape=[1x24x1x200] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=17 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=19 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=21 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048581 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048581 bd_func=3 core=0
[bmprofile] local_layer: layer_id=15 layer_type=Load layer_name=
[bmprofile] tensor_id=-3 is_in=1 shape=[1x3x128x128] dtype=0 is_const=0 gaddr=2201171394560 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=15 is_in=0 shape=[1x3x128x128] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=43 l2addr=0
[bmprofile] gdma cmd_id bd_id=4 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=18 layer_type=Cast layer_name=
[bmprofile] tensor_id=15 is_in=1 shape=[1x3x128x128] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=43 l2addr=0
[bmprofile] tensor_id=18 is_in=0 shape=[1x3x128x128] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=43 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048582 bd_func=3 core=0
[bmprofile] local_layer: layer_id=20 layer_type=Store layer_name=
[bmprofile] tensor_id=19 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=21 l2addr=0
[bmprofile] tensor_id=20 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=21 l2addr=0
[bmprofile] gdma cmd_id bd_id=6 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=19 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=18 is_in=1 shape=[1x3x128x128] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=43 l2addr=0
[bmprofile] tensor_id=16 is_in=1 shape=[1x24x1x200] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=17 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=19 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=21 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048583 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048583 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=20 layer_type=Store layer_name=
[bmprofile] tensor_id=19 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=21 l2addr=0
[bmprofile] tensor_id=20 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=21 l2addr=0
[bmprofile] gdma cmd_id bd_id=9 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=25 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=20 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=2201170935808 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=23 is_in=1 shape=[1x24x3x3] dtype=8 is_const=1 gaddr=1101659127808 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=24 is_in=1 shape=[1x24x1x1] dtype=8 is_const=1 gaddr=1101659131904 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=25 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=9 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=9 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=9 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048587 bd_func=1 core=0
[bmprofile] gdma cmd_id bd_id=9 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=10 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048588 bd_func=1 core=0
[bmprofile] gdma cmd_id bd_id=10 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=11 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048590 bd_func=1 core=0
[bmprofile] gdma cmd_id bd_id=11 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=12 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048592 bd_func=1 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=13 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=29 layer_type=Load layer_name=
[bmprofile] tensor_id=25 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=29 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=13 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=30 layer_type=Load layer_name=
[bmprofile] tensor_id=27 is_in=1 shape=[1x24x1x24] dtype=8 is_const=1 gaddr=1101659140096 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=30 is_in=0 shape=[1x24x1x24] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=13 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=31 layer_type=Load layer_name=
[bmprofile] tensor_id=26 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=1101659136000 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=31 is_in=0 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=13 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=33 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=29 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=30 is_in=1 shape=[1x24x1x24] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=31 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=33 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048597 bd_func=0 core=0
[bmprofile] local_layer: layer_id=32 layer_type=Load layer_name=
[bmprofile] tensor_id=20 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=2201170935808 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=32 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=13 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=34 layer_type=Add layer_name=
[bmprofile] tensor_id=33 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=32 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=34 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048598 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048598 bd_func=3 core=0
[bmprofile] local_layer: layer_id=29 layer_type=Load layer_name=
[bmprofile] tensor_id=25 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=29 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=14 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=33 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=29 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=30 is_in=1 shape=[1x24x1x24] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=31 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=33 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048599 bd_func=0 core=0
[bmprofile] local_layer: layer_id=32 layer_type=Load layer_name=
[bmprofile] tensor_id=20 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=2201170935808 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=32 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=16 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=35 layer_type=Store layer_name=
[bmprofile] tensor_id=34 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=35 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=16 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=34 layer_type=Add layer_name=
[bmprofile] tensor_id=33 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=32 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=34 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048601 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048601 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=35 layer_type=Store layer_name=
[bmprofile] tensor_id=34 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=35 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=19 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=38 layer_type=Pad layer_name=
[bmprofile] tensor_id=35 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=2201171197952 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=38 is_in=0 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=2201170968576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=19 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=19 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=44 layer_type=Load layer_name=
[bmprofile] tensor_id=35 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=2201171197952 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=44 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=33 l2addr=0
[bmprofile] gdma cmd_id bd_id=19 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=45 layer_type=Load layer_name=
[bmprofile] tensor_id=39 is_in=1 shape=[1x24x3x3] dtype=8 is_const=1 gaddr=1101659144192 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=45 is_in=0 shape=[1x24x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=12864 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=19 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=46 layer_type=Load layer_name=
[bmprofile] tensor_id=40 is_in=1 shape=[1x24x1x1] dtype=8 is_const=1 gaddr=1101659148288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=46 is_in=0 shape=[1x24x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=12944 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=19 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=49 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=44 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=33 l2addr=0
[bmprofile] tensor_id=45 is_in=1 shape=[1x24x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=12864 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=46 is_in=1 shape=[1x24x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=12944 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=49 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048607 bd_func=1 core=0
[bmprofile] local_layer: layer_id=47 layer_type=Load layer_name=
[bmprofile] tensor_id=42 is_in=1 shape=[1x28x1x24] dtype=8 is_const=1 gaddr=1101659156480 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=47 is_in=0 shape=[1x28x1x24] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=12672 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=19 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=48 layer_type=Load layer_name=
[bmprofile] tensor_id=41 is_in=1 shape=[1x28x1x1] dtype=0 is_const=1 gaddr=1101659152384 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=48 is_in=0 shape=[1x28x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=12928 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=19 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=51 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=49 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=47 is_in=1 shape=[1x28x1x24] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=12672 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=48 is_in=1 shape=[1x28x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=12928 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=51 is_in=0 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048609 bd_func=0 core=0
[bmprofile] local_layer: layer_id=50 layer_type=Load layer_name=
[bmprofile] tensor_id=38 is_in=1 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=2201170968576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=50 is_in=0 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=20 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=52 layer_type=Add layer_name=
[bmprofile] tensor_id=51 is_in=1 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=50 is_in=1 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=52 is_in=0 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048610 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048610 bd_func=3 core=0
[bmprofile] local_layer: layer_id=44 layer_type=Load layer_name=
[bmprofile] tensor_id=35 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=2201171197952 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=44 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=33 l2addr=0
[bmprofile] gdma cmd_id bd_id=21 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=54 layer_type=Pool2D layer_name=
[bmprofile] tensor_id=52 is_in=1 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=54 is_in=0 shape=[1x28x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048611 bd_func=1 core=0
[bmprofile] local_layer: layer_id=53 layer_type=Store layer_name=
[bmprofile] tensor_id=52 is_in=1 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=53 is_in=0 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=49 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=44 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=33 l2addr=0
[bmprofile] tensor_id=45 is_in=1 shape=[1x24x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=12864 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=46 is_in=1 shape=[1x24x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=12944 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=49 is_in=0 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048612 bd_func=1 core=0
[bmprofile] local_layer: layer_id=55 layer_type=Store layer_name=
[bmprofile] tensor_id=54 is_in=1 shape=[1x28x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=55 is_in=0 shape=[1x28x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=24 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=51 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=49 is_in=1 shape=[1x24x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=47 is_in=1 shape=[1x28x1x24] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=12672 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=48 is_in=1 shape=[1x28x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=12928 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=51 is_in=0 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048613 bd_func=0 core=0
[bmprofile] local_layer: layer_id=50 layer_type=Load layer_name=
[bmprofile] tensor_id=38 is_in=1 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=2201170968576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=50 is_in=0 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=25 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=52 layer_type=Add layer_name=
[bmprofile] tensor_id=51 is_in=1 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=50 is_in=1 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=52 is_in=0 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048614 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048614 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=54 layer_type=Pool2D layer_name=
[bmprofile] tensor_id=52 is_in=1 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=54 is_in=0 shape=[1x28x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048614 bd_func=1 core=0
[bmprofile] local_layer: layer_id=53 layer_type=Store layer_name=
[bmprofile] tensor_id=52 is_in=1 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=53 is_in=0 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=28 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=55 layer_type=Store layer_name=
[bmprofile] tensor_id=54 is_in=1 shape=[1x28x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=55 is_in=0 shape=[1x28x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=29 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=58 layer_type=Pad layer_name=
[bmprofile] tensor_id=55 is_in=1 shape=[1x28x32x32] dtype=8 is_const=0 gaddr=2201171591168 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=58 is_in=0 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=2201171050496 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=29 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=29 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=64 layer_type=Load layer_name=
[bmprofile] tensor_id=53 is_in=1 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=64 is_in=0 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=33 l2addr=0
[bmprofile] gdma cmd_id bd_id=29 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=65 layer_type=Load layer_name=
[bmprofile] tensor_id=59 is_in=1 shape=[1x28x3x3] dtype=8 is_const=1 gaddr=1101659160576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=65 is_in=0 shape=[1x28x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=29 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=66 layer_type=Load layer_name=
[bmprofile] tensor_id=60 is_in=1 shape=[1x28x1x1] dtype=8 is_const=1 gaddr=1101659164672 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=66 is_in=0 shape=[1x28x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=29 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=69 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=64 is_in=1 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=33 l2addr=0
[bmprofile] tensor_id=65 is_in=1 shape=[1x28x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=66 is_in=1 shape=[1x28x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=69 is_in=0 shape=[1x28x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048621 bd_func=1 core=0
[bmprofile] local_layer: layer_id=67 layer_type=Load layer_name=
[bmprofile] tensor_id=62 is_in=1 shape=[1x32x1x32] dtype=8 is_const=1 gaddr=1101659172864 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=67 is_in=0 shape=[1x32x1x32] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=45056 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=29 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=68 layer_type=Load layer_name=
[bmprofile] tensor_id=61 is_in=1 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=1101659168768 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=68 is_in=0 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=29 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=71 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=69 is_in=1 shape=[1x28x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=67 is_in=1 shape=[1x32x1x32] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=45056 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=68 is_in=1 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=71 is_in=0 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048623 bd_func=0 core=0
[bmprofile] local_layer: layer_id=70 layer_type=Load layer_name=
[bmprofile] tensor_id=58 is_in=1 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=2201171050496 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=70 is_in=0 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=30 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=72 layer_type=Add layer_name=
[bmprofile] tensor_id=71 is_in=1 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=70 is_in=1 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=72 is_in=0 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048624 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048624 bd_func=3 core=0
[bmprofile] local_layer: layer_id=64 layer_type=Load layer_name=
[bmprofile] tensor_id=53 is_in=1 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=64 is_in=0 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=31 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=69 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=64 is_in=1 shape=[1x28x64x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=65 is_in=1 shape=[1x28x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=66 is_in=1 shape=[1x28x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=69 is_in=0 shape=[1x28x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048625 bd_func=1 core=0
[bmprofile] local_layer: layer_id=73 layer_type=Store layer_name=
[bmprofile] tensor_id=72 is_in=1 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=73 is_in=0 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=33 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=71 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=69 is_in=1 shape=[1x28x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=67 is_in=1 shape=[1x32x1x32] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=45056 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=68 is_in=1 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=71 is_in=0 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048626 bd_func=0 core=0
[bmprofile] local_layer: layer_id=70 layer_type=Load layer_name=
[bmprofile] tensor_id=58 is_in=1 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=2201171050496 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=70 is_in=0 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=34 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=72 layer_type=Add layer_name=
[bmprofile] tensor_id=71 is_in=1 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=70 is_in=1 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=72 is_in=0 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048627 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048627 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=73 layer_type=Store layer_name=
[bmprofile] tensor_id=72 is_in=1 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=73 is_in=0 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=37 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=76 layer_type=Pad layer_name=
[bmprofile] tensor_id=73 is_in=1 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=2201170984960 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=76 is_in=0 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=37 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=37 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=82 layer_type=Load layer_name=
[bmprofile] tensor_id=73 is_in=1 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=2201170984960 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=82 is_in=0 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=45056 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=37 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=83 layer_type=Load layer_name=
[bmprofile] tensor_id=77 is_in=1 shape=[1x32x3x3] dtype=8 is_const=1 gaddr=1101659176960 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=83 is_in=0 shape=[1x32x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=37 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=84 layer_type=Load layer_name=
[bmprofile] tensor_id=78 is_in=1 shape=[1x32x1x1] dtype=8 is_const=1 gaddr=1101659181056 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=84 is_in=0 shape=[1x32x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=61440 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=37 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=87 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=82 is_in=1 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=45056 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=83 is_in=1 shape=[1x32x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=84 is_in=1 shape=[1x32x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=61440 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=87 is_in=0 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048633 bd_func=1 core=0
[bmprofile] local_layer: layer_id=85 layer_type=Load layer_name=
[bmprofile] tensor_id=80 is_in=1 shape=[1x36x1x32] dtype=8 is_const=1 gaddr=1101659189248 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=85 is_in=0 shape=[1x36x1x32] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=34816 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=37 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=86 layer_type=Load layer_name=
[bmprofile] tensor_id=79 is_in=1 shape=[1x36x1x1] dtype=0 is_const=1 gaddr=1101659185152 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=86 is_in=0 shape=[1x36x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=37 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=89 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=87 is_in=1 shape=[1x32x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=85 is_in=1 shape=[1x36x1x32] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=34816 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=86 is_in=1 shape=[1x36x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=89 is_in=0 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048635 bd_func=0 core=0
[bmprofile] local_layer: layer_id=88 layer_type=Load layer_name=
[bmprofile] tensor_id=76 is_in=1 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=88 is_in=0 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=38 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=90 layer_type=Add layer_name=
[bmprofile] tensor_id=89 is_in=1 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=88 is_in=1 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=90 is_in=0 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048636 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048636 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=91 layer_type=Store layer_name=
[bmprofile] tensor_id=90 is_in=1 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=91 is_in=0 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=41 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=94 layer_type=Pad layer_name=
[bmprofile] tensor_id=91 is_in=1 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=2201170911232 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=94 is_in=0 shape=[1x42x32x32] dtype=8 is_const=0 gaddr=2201170825216 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=41 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=41 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=100 layer_type=Load layer_name=
[bmprofile] tensor_id=96 is_in=1 shape=[1x36x1x1] dtype=8 is_const=1 gaddr=1101659197440 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=100 is_in=0 shape=[1x36x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=10240 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=41 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=101 layer_type=Load layer_name=
[bmprofile] tensor_id=91 is_in=1 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=2201170911232 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=101 is_in=0 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=41 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=102 layer_type=Load layer_name=
[bmprofile] tensor_id=95 is_in=1 shape=[1x36x3x3] dtype=8 is_const=1 gaddr=1101659193344 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=102 is_in=0 shape=[1x36x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=41 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=105 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=101 is_in=1 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=102 is_in=1 shape=[1x36x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=100 is_in=1 shape=[1x36x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=10240 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=105 is_in=0 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=45056 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048642 bd_func=1 core=0
[bmprofile] local_layer: layer_id=103 layer_type=Load layer_name=
[bmprofile] tensor_id=98 is_in=1 shape=[1x42x1x40] dtype=8 is_const=1 gaddr=1101659205632 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=103 is_in=0 shape=[1x42x1x40] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=41 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=104 layer_type=Load layer_name=
[bmprofile] tensor_id=97 is_in=1 shape=[1x42x1x1] dtype=0 is_const=1 gaddr=1101659201536 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=104 is_in=0 shape=[1x42x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=61440 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=41 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=107 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=105 is_in=1 shape=[1x36x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=45056 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=103 is_in=1 shape=[1x42x1x40] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=57344 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=104 is_in=1 shape=[1x42x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=61440 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=107 is_in=0 shape=[1x42x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048644 bd_func=0 core=0
[bmprofile] local_layer: layer_id=106 layer_type=Load layer_name=
[bmprofile] tensor_id=94 is_in=1 shape=[1x42x32x32] dtype=8 is_const=0 gaddr=2201170825216 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=106 is_in=0 shape=[1x42x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=42 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=108 layer_type=Add layer_name=
[bmprofile] tensor_id=107 is_in=1 shape=[1x42x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=106 is_in=1 shape=[1x42x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=108 is_in=0 shape=[1x42x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048645 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048645 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=110 layer_type=Pool2D layer_name=
[bmprofile] tensor_id=108 is_in=1 shape=[1x42x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=110 is_in=0 shape=[1x42x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048645 bd_func=1 core=0
[bmprofile] local_layer: layer_id=109 layer_type=Store layer_name=
[bmprofile] tensor_id=108 is_in=1 shape=[1x42x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=109 is_in=0 shape=[1x42x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=45 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=111 layer_type=Store layer_name=
[bmprofile] tensor_id=110 is_in=1 shape=[1x42x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=111 is_in=0 shape=[1x42x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=46 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=114 layer_type=Pad layer_name=
[bmprofile] tensor_id=111 is_in=1 shape=[1x42x16x16] dtype=8 is_const=0 gaddr=2201170984960 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=114 is_in=0 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=2201170849792 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=46 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=46 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=120 layer_type=Load layer_name=
[bmprofile] tensor_id=116 is_in=1 shape=[1x42x1x1] dtype=8 is_const=1 gaddr=1101659213824 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=120 is_in=0 shape=[1x42x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=46 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=121 layer_type=Load layer_name=
[bmprofile] tensor_id=109 is_in=1 shape=[1x42x32x32] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=121 is_in=0 shape=[1x42x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=46 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=122 layer_type=Load layer_name=
[bmprofile] tensor_id=115 is_in=1 shape=[1x42x3x3] dtype=8 is_const=1 gaddr=1101659209728 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=122 is_in=0 shape=[1x42x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=46 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=125 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=121 is_in=1 shape=[1x42x32x32] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=122 is_in=1 shape=[1x42x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=120 is_in=1 shape=[1x42x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=125 is_in=0 shape=[1x42x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048652 bd_func=1 core=0
[bmprofile] local_layer: layer_id=123 layer_type=Load layer_name=
[bmprofile] tensor_id=118 is_in=1 shape=[1x48x1x48] dtype=8 is_const=1 gaddr=1101659222016 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=123 is_in=0 shape=[1x48x1x48] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=31744 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=46 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=124 layer_type=Load layer_name=
[bmprofile] tensor_id=117 is_in=1 shape=[1x48x1x1] dtype=0 is_const=1 gaddr=1101659217920 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=124 is_in=0 shape=[1x48x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=46 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=127 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=125 is_in=1 shape=[1x42x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=123 is_in=1 shape=[1x48x1x48] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=31744 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=124 is_in=1 shape=[1x48x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=127 is_in=0 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048654 bd_func=0 core=0
[bmprofile] local_layer: layer_id=126 layer_type=Load layer_name=
[bmprofile] tensor_id=114 is_in=1 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=2201170849792 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=126 is_in=0 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=47 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=128 layer_type=Add layer_name=
[bmprofile] tensor_id=127 is_in=1 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=126 is_in=1 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=128 is_in=0 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048655 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048655 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=129 layer_type=Store layer_name=
[bmprofile] tensor_id=128 is_in=1 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=129 is_in=0 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=50 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=132 layer_type=Pad layer_name=
[bmprofile] tensor_id=129 is_in=1 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=2201170825216 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=132 is_in=0 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=50 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=50 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=138 layer_type=Load layer_name=
[bmprofile] tensor_id=134 is_in=1 shape=[1x48x1x1] dtype=8 is_const=1 gaddr=1101659234304 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=138 is_in=0 shape=[1x48x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=50 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=139 layer_type=Load layer_name=
[bmprofile] tensor_id=129 is_in=1 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=2201170825216 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=139 is_in=0 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=50 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=140 layer_type=Load layer_name=
[bmprofile] tensor_id=133 is_in=1 shape=[1x48x3x3] dtype=8 is_const=1 gaddr=1101659230208 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=140 is_in=0 shape=[1x48x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=50 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=143 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=139 is_in=1 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=140 is_in=1 shape=[1x48x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=138 is_in=1 shape=[1x48x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=143 is_in=0 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048661 bd_func=1 core=0
[bmprofile] local_layer: layer_id=141 layer_type=Load layer_name=
[bmprofile] tensor_id=136 is_in=1 shape=[1x56x1x48] dtype=8 is_const=1 gaddr=1101659242496 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=141 is_in=0 shape=[1x56x1x48] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=11776 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=50 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=142 layer_type=Load layer_name=
[bmprofile] tensor_id=135 is_in=1 shape=[1x56x1x1] dtype=0 is_const=1 gaddr=1101659238400 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=142 is_in=0 shape=[1x56x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=50 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=145 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=143 is_in=1 shape=[1x48x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=141 is_in=1 shape=[1x56x1x48] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=11776 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=142 is_in=1 shape=[1x56x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=145 is_in=0 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048663 bd_func=0 core=0
[bmprofile] local_layer: layer_id=144 layer_type=Load layer_name=
[bmprofile] tensor_id=132 is_in=1 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=144 is_in=0 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=4096 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=51 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=146 layer_type=Add layer_name=
[bmprofile] tensor_id=145 is_in=1 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=144 is_in=1 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=4096 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=146 is_in=0 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048664 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048664 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=147 layer_type=Store layer_name=
[bmprofile] tensor_id=146 is_in=1 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=147 is_in=0 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=54 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=150 layer_type=Pad layer_name=
[bmprofile] tensor_id=147 is_in=1 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=2201170771968 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=150 is_in=0 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=54 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=54 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=156 layer_type=Load layer_name=
[bmprofile] tensor_id=152 is_in=1 shape=[1x56x1x1] dtype=8 is_const=1 gaddr=1101659254784 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=156 is_in=0 shape=[1x56x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=54 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=157 layer_type=Load layer_name=
[bmprofile] tensor_id=151 is_in=1 shape=[1x56x3x3] dtype=8 is_const=1 gaddr=1101659250688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=157 is_in=0 shape=[1x56x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=54 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=158 layer_type=Load layer_name=
[bmprofile] tensor_id=147 is_in=1 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=2201170771968 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=158 is_in=0 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=54 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=161 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=158 is_in=1 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=157 is_in=1 shape=[1x56x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=156 is_in=1 shape=[1x56x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=161 is_in=0 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048670 bd_func=1 core=0
[bmprofile] local_layer: layer_id=159 layer_type=Load layer_name=
[bmprofile] tensor_id=154 is_in=1 shape=[1x64x1x56] dtype=8 is_const=1 gaddr=1101659262976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=159 is_in=0 shape=[1x64x1x56] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=54 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=160 layer_type=Load layer_name=
[bmprofile] tensor_id=153 is_in=1 shape=[1x64x1x1] dtype=0 is_const=1 gaddr=1101659258880 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=160 is_in=0 shape=[1x64x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=54 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=163 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=161 is_in=1 shape=[1x56x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=159 is_in=1 shape=[1x64x1x56] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=160 is_in=1 shape=[1x64x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=163 is_in=0 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048672 bd_func=0 core=0
[bmprofile] local_layer: layer_id=162 layer_type=Load layer_name=
[bmprofile] tensor_id=150 is_in=1 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=162 is_in=0 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=4096 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=55 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=164 layer_type=Add layer_name=
[bmprofile] tensor_id=163 is_in=1 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=162 is_in=1 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=4096 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=164 is_in=0 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048673 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048673 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=165 layer_type=Store layer_name=
[bmprofile] tensor_id=164 is_in=1 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=165 is_in=0 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=58 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=168 layer_type=Pad layer_name=
[bmprofile] tensor_id=165 is_in=1 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=2201170817024 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=168 is_in=0 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=58 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=58 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=174 layer_type=Load layer_name=
[bmprofile] tensor_id=165 is_in=1 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=2201170817024 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=174 is_in=0 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=58 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=175 layer_type=Load layer_name=
[bmprofile] tensor_id=169 is_in=1 shape=[1x64x3x3] dtype=8 is_const=1 gaddr=1101659271168 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=175 is_in=0 shape=[1x64x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=58 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=176 layer_type=Load layer_name=
[bmprofile] tensor_id=170 is_in=1 shape=[1x64x1x1] dtype=8 is_const=1 gaddr=1101659275264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=176 is_in=0 shape=[1x64x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=45056 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=58 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=179 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=174 is_in=1 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=175 is_in=1 shape=[1x64x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=176 is_in=1 shape=[1x64x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=45056 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=179 is_in=0 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048679 bd_func=1 core=0
[bmprofile] local_layer: layer_id=177 layer_type=Load layer_name=
[bmprofile] tensor_id=172 is_in=1 shape=[1x72x1x64] dtype=8 is_const=1 gaddr=1101659283456 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=177 is_in=0 shape=[1x72x1x64] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=58 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=178 layer_type=Load layer_name=
[bmprofile] tensor_id=171 is_in=1 shape=[1x72x1x1] dtype=0 is_const=1 gaddr=1101659279360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=178 is_in=0 shape=[1x72x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=58 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=181 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=179 is_in=1 shape=[1x64x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=177 is_in=1 shape=[1x72x1x64] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=178 is_in=1 shape=[1x72x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=181 is_in=0 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048681 bd_func=0 core=0
[bmprofile] local_layer: layer_id=180 layer_type=Load layer_name=
[bmprofile] tensor_id=168 is_in=1 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=180 is_in=0 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=59 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=182 layer_type=Add layer_name=
[bmprofile] tensor_id=181 is_in=1 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=180 is_in=1 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=182 is_in=0 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048682 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048682 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=183 layer_type=Store layer_name=
[bmprofile] tensor_id=182 is_in=1 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=183 is_in=0 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=62 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=186 layer_type=Pad layer_name=
[bmprofile] tensor_id=183 is_in=1 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=2201170780160 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=186 is_in=0 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=62 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=62 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=192 layer_type=Load layer_name=
[bmprofile] tensor_id=183 is_in=1 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=2201170780160 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=192 is_in=0 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=62 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=193 layer_type=Load layer_name=
[bmprofile] tensor_id=187 is_in=1 shape=[1x72x3x3] dtype=8 is_const=1 gaddr=1101659295744 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=193 is_in=0 shape=[1x72x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=45056 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=62 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=194 layer_type=Load layer_name=
[bmprofile] tensor_id=188 is_in=1 shape=[1x72x1x1] dtype=8 is_const=1 gaddr=1101659299840 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=194 is_in=0 shape=[1x72x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=62 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=197 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=192 is_in=1 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=193 is_in=1 shape=[1x72x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=45056 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=194 is_in=1 shape=[1x72x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=53248 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=197 is_in=0 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048688 bd_func=1 core=0
[bmprofile] local_layer: layer_id=195 layer_type=Load layer_name=
[bmprofile] tensor_id=190 is_in=1 shape=[1x80x1x72] dtype=8 is_const=1 gaddr=1101659308032 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=195 is_in=0 shape=[1x80x1x72] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=62 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=196 layer_type=Load layer_name=
[bmprofile] tensor_id=189 is_in=1 shape=[1x80x1x1] dtype=0 is_const=1 gaddr=1101659303936 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=196 is_in=0 shape=[1x80x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=62 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=199 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=197 is_in=1 shape=[1x72x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=195 is_in=1 shape=[1x80x1x72] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=196 is_in=1 shape=[1x80x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=199 is_in=0 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048690 bd_func=0 core=0
[bmprofile] local_layer: layer_id=198 layer_type=Load layer_name=
[bmprofile] tensor_id=186 is_in=1 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=198 is_in=0 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=63 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=200 layer_type=Add layer_name=
[bmprofile] tensor_id=199 is_in=1 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=198 is_in=1 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=200 is_in=0 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048691 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048691 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=201 layer_type=Store layer_name=
[bmprofile] tensor_id=200 is_in=1 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=201 is_in=0 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=204 layer_type=Pad layer_name=
[bmprofile] tensor_id=201 is_in=1 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=2201170829312 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=204 is_in=0 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=2201170784256 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=210 layer_type=Load layer_name=
[bmprofile] tensor_id=201 is_in=1 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=2201170829312 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=210 is_in=0 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=211 layer_type=Load layer_name=
[bmprofile] tensor_id=206 is_in=1 shape=[1x80x1x1] dtype=8 is_const=1 gaddr=1101659324416 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=211 is_in=0 shape=[1x80x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=45056 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=212 layer_type=Load layer_name=
[bmprofile] tensor_id=205 is_in=1 shape=[1x80x3x3] dtype=8 is_const=1 gaddr=1101659320320 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=212 is_in=0 shape=[1x80x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=215 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=210 is_in=1 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=212 is_in=1 shape=[1x80x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=36864 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=211 is_in=1 shape=[1x80x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=45056 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=215 is_in=0 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048697 bd_func=1 core=0
[bmprofile] local_layer: layer_id=213 layer_type=Load layer_name=
[bmprofile] tensor_id=208 is_in=1 shape=[1x88x1x80] dtype=8 is_const=1 gaddr=1101659332608 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=213 is_in=0 shape=[1x88x1x80] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=214 layer_type=Load layer_name=
[bmprofile] tensor_id=207 is_in=1 shape=[1x88x1x1] dtype=0 is_const=1 gaddr=1101659328512 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=214 is_in=0 shape=[1x88x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=217 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=215 is_in=1 shape=[1x80x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=213 is_in=1 shape=[1x88x1x80] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=214 is_in=1 shape=[1x88x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=217 is_in=0 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048699 bd_func=0 core=0
[bmprofile] local_layer: layer_id=216 layer_type=Load layer_name=
[bmprofile] tensor_id=204 is_in=1 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=2201170784256 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=216 is_in=0 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=5120 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=67 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=218 layer_type=Add layer_name=
[bmprofile] tensor_id=217 is_in=1 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=216 is_in=1 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=5120 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=218 is_in=0 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048700 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048700 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=220 layer_type=Pool2D layer_name=
[bmprofile] tensor_id=218 is_in=1 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=220 is_in=0 shape=[1x88x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048700 bd_func=1 core=0
[bmprofile] local_layer: layer_id=219 layer_type=Store layer_name=
[bmprofile] tensor_id=218 is_in=1 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=219 is_in=0 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=70 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=221 layer_type=Store layer_name=
[bmprofile] tensor_id=220 is_in=1 shape=[1x88x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=221 is_in=0 shape=[1x88x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=71 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=224 layer_type=Pad layer_name=
[bmprofile] tensor_id=221 is_in=1 shape=[1x88x8x8] dtype=8 is_const=0 gaddr=2201170870272 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=224 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=2201170784256 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=71 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=71 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
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
[bmprofile] local_layer: layer_id=248 layer_type=Load layer_name=
[bmprofile] tensor_id=226 is_in=1 shape=[1x88x1x1] dtype=8 is_const=1 gaddr=1101659353088 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=248 is_in=0 shape=[1x88x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=71 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=249 layer_type=Load layer_name=
[bmprofile] tensor_id=225 is_in=1 shape=[1x88x3x3] dtype=8 is_const=1 gaddr=1101659348992 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=249 is_in=0 shape=[1x88x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=71 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=250 layer_type=Load layer_name=
[bmprofile] tensor_id=219 is_in=1 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=250 is_in=0 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=71 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=253 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=250 is_in=1 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=249 is_in=1 shape=[1x88x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=248 is_in=1 shape=[1x88x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=253 is_in=0 shape=[1x88x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048707 bd_func=1 core=0
[bmprofile] local_layer: layer_id=251 layer_type=Load layer_name=
[bmprofile] tensor_id=228 is_in=1 shape=[1x96x1x88] dtype=8 is_const=1 gaddr=1101659361280 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=251 is_in=0 shape=[1x96x1x88] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=71 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=252 layer_type=Load layer_name=
[bmprofile] tensor_id=227 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=1101659357184 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=252 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=20992 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=71 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=257 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=253 is_in=1 shape=[1x88x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=251 is_in=1 shape=[1x96x1x88] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=252 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=20992 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=257 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=5632 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048709 bd_func=0 core=0
[bmprofile] local_layer: layer_id=254 layer_type=Load layer_name=
[bmprofile] tensor_id=224 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=2201170784256 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=254 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=72 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=255 layer_type=Load layer_name=
[bmprofile] tensor_id=230 is_in=1 shape=[1x96x1x1] dtype=8 is_const=1 gaddr=1101659385856 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=255 is_in=0 shape=[1x96x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=28720 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=72 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=256 layer_type=Load layer_name=
[bmprofile] tensor_id=231 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=1101659389952 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=256 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=72 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=259 layer_type=Add layer_name=
[bmprofile] tensor_id=257 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=5632 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=254 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=259 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048712 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048712 bd_func=3 core=0
[bmprofile] local_layer: layer_id=258 layer_type=Load layer_name=
[bmprofile] tensor_id=229 is_in=1 shape=[1x96x3x3] dtype=8 is_const=1 gaddr=1101659381760 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=258 is_in=0 shape=[1x96x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=20992 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=73 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=261 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=259 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=258 is_in=1 shape=[1x96x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=20992 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=255 is_in=1 shape=[1x96x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=28720 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=261 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048713 bd_func=1 core=0
[bmprofile] local_layer: layer_id=260 layer_type=Load layer_name=
[bmprofile] tensor_id=232 is_in=1 shape=[1x96x1x96] dtype=8 is_const=1 gaddr=1101659394048 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=260 is_in=0 shape=[1x96x1x96] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=75 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=265 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=261 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=260 is_in=1 shape=[1x96x1x96] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=256 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=265 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=9728 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048714 bd_func=0 core=0
[bmprofile] local_layer: layer_id=266 layer_type=Add layer_name=
[bmprofile] tensor_id=265 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=9728 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=259 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=266 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048714 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048714 bd_func=3 core=0
[bmprofile] local_layer: layer_id=262 layer_type=Load layer_name=
[bmprofile] tensor_id=234 is_in=1 shape=[1x96x1x1] dtype=8 is_const=1 gaddr=1101659418624 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=262 is_in=0 shape=[1x96x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=76 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=263 layer_type=Load layer_name=
[bmprofile] tensor_id=233 is_in=1 shape=[1x96x3x3] dtype=8 is_const=1 gaddr=1101659414528 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=263 is_in=0 shape=[1x96x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=7936 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=76 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=264 layer_type=Load layer_name=
[bmprofile] tensor_id=236 is_in=1 shape=[1x96x1x96] dtype=8 is_const=1 gaddr=1101659426816 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=264 is_in=0 shape=[1x96x1x96] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=5632 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=76 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=268 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=266 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=263 is_in=1 shape=[1x96x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=7936 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=262 is_in=1 shape=[1x96x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=268 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=17920 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048717 bd_func=1 core=0
[bmprofile] local_layer: layer_id=267 layer_type=Load layer_name=
[bmprofile] tensor_id=235 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=1101659422720 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=267 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=79 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=272 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=268 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=17920 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=264 is_in=1 shape=[1x96x1x96] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=5632 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=267 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=272 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=9728 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048718 bd_func=0 core=0
[bmprofile] local_layer: layer_id=273 layer_type=Add layer_name=
[bmprofile] tensor_id=272 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=9728 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=266 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=273 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048718 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048718 bd_func=3 core=0
[bmprofile] local_layer: layer_id=269 layer_type=Load layer_name=
[bmprofile] tensor_id=238 is_in=1 shape=[1x96x1x1] dtype=8 is_const=1 gaddr=1101659451392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=269 is_in=0 shape=[1x96x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=80 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=270 layer_type=Load layer_name=
[bmprofile] tensor_id=237 is_in=1 shape=[1x96x3x3] dtype=8 is_const=1 gaddr=1101659447296 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=270 is_in=0 shape=[1x96x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=14592 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=80 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=271 layer_type=Load layer_name=
[bmprofile] tensor_id=240 is_in=1 shape=[1x96x1x96] dtype=8 is_const=1 gaddr=1101659459584 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=271 is_in=0 shape=[1x96x1x96] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=80 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=275 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=273 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=270 is_in=1 shape=[1x96x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=14592 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=269 is_in=1 shape=[1x96x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=275 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048721 bd_func=1 core=0
[bmprofile] local_layer: layer_id=274 layer_type=Load layer_name=
[bmprofile] tensor_id=239 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=1101659455488 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=274 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=83 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=279 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=275 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=271 is_in=1 shape=[1x96x1x96] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=274 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=279 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=9728 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048722 bd_func=0 core=0
[bmprofile] local_layer: layer_id=280 layer_type=Add layer_name=
[bmprofile] tensor_id=279 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=9728 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=273 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=280 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048722 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048722 bd_func=3 core=0
[bmprofile] local_layer: layer_id=276 layer_type=Load layer_name=
[bmprofile] tensor_id=242 is_in=1 shape=[1x96x1x1] dtype=8 is_const=1 gaddr=1101659484160 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=276 is_in=0 shape=[1x96x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=84 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=277 layer_type=Load layer_name=
[bmprofile] tensor_id=241 is_in=1 shape=[1x96x3x3] dtype=8 is_const=1 gaddr=1101659480064 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=277 is_in=0 shape=[1x96x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=7936 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=84 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=278 layer_type=Load layer_name=
[bmprofile] tensor_id=244 is_in=1 shape=[1x96x1x96] dtype=8 is_const=1 gaddr=1101659492352 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=278 is_in=0 shape=[1x96x1x96] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=5632 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=84 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=282 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=280 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=277 is_in=1 shape=[1x96x3x3] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=7936 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=276 is_in=1 shape=[1x96x1x1] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=28672 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=282 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048725 bd_func=1 core=0
[bmprofile] local_layer: layer_id=281 layer_type=Load layer_name=
[bmprofile] tensor_id=243 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=1101659488256 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=281 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=87 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=283 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=282 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=278 is_in=1 shape=[1x96x1x96] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=5632 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=281 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=283 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048726 bd_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=286 layer_type=Add layer_name=
[bmprofile] tensor_id=283 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=280 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=8192 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=286 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048726 bd_func=3 core=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048726 bd_func=3 core=0
[bmprofile] local_layer: layer_id=284 layer_type=Load layer_name=
[bmprofile] tensor_id=246 is_in=1 shape=[1x2x1x88] dtype=8 is_const=1 gaddr=1101659516928 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=284 is_in=0 shape=[1x2x1x88] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=24784 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=89 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] local_layer: layer_id=285 layer_type=Load layer_name=
[bmprofile] tensor_id=245 is_in=1 shape=[1x2x1x1] dtype=0 is_const=1 gaddr=1101659512832 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=285 is_in=0 shape=[1x2x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=28704 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=89 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=288 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=250 is_in=1 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=284 is_in=1 shape=[1x2x1x88] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=24784 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=285 is_in=1 shape=[1x2x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=28704 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=288 is_in=0 shape=[1x2x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048728 bd_func=0 core=0
[bmprofile] local_layer: layer_id=287 layer_type=Store layer_name=
[bmprofile] tensor_id=286 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=287 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=12288 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=91 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
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
[bmprofile] local_layer: layer_id=289 layer_type=Store layer_name=
[bmprofile] tensor_id=288 is_in=1 shape=[1x2x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=289 is_in=0 shape=[1x2x16x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=20480 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=92 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
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

[bmprofile] global_layer: layer_id=292 layer_type=Permute layer_name=
[bmprofile] tensor_id=289 is_in=1 shape=[1x2x16x16] dtype=8 is_const=0 gaddr=2201170796544 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=292 is_in=0 shape=[1x16x16x2] dtype=8 is_const=0 gaddr=2201170792448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=92 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048731 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=93 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=293 layer_type=Reshape layer_name=
[bmprofile] tensor_id=292 is_in=1 shape=[1x16x16x2] dtype=8 is_const=0 gaddr=2201170792448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=293 is_in=0 shape=[1x512x1] dtype=8 is_const=0 gaddr=2201170792448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=296 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=287 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=2201170800640 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=295 is_in=1 shape=[1x6x1x96] dtype=8 is_const=1 gaddr=1101659525120 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=294 is_in=1 shape=[1x6x1x1] dtype=0 is_const=1 gaddr=1101659521024 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=296 is_in=0 shape=[1x6x8x8] dtype=8 is_const=0 gaddr=2201170788352 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=93 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=93 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=93 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048735 bd_func=0 core=0
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=94 gdma_id=0 gdma_dir=0 gdma_func=0 core=0

[bmprofile] global_layer: layer_id=297 layer_type=Permute layer_name=
[bmprofile] tensor_id=296 is_in=1 shape=[1x6x8x8] dtype=8 is_const=0 gaddr=2201170788352 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=297 is_in=0 shape=[1x8x8x6] dtype=8 is_const=0 gaddr=2201170784256 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=94 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048737 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=95 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=298 layer_type=Reshape layer_name=
[bmprofile] tensor_id=297 is_in=1 shape=[1x8x8x6] dtype=8 is_const=0 gaddr=2201170784256 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=298 is_in=0 shape=[1x384x1] dtype=8 is_const=0 gaddr=2201170784256 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=299 layer_type=Concat layer_name=
[bmprofile] tensor_id=293 is_in=1 shape=[1x512x1] dtype=8 is_const=0 gaddr=2201170792448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=298 is_in=1 shape=[1x384x1] dtype=8 is_const=0 gaddr=2201170784256 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=299 is_in=0 shape=[1x896x1] dtype=8 is_const=0 gaddr=2201170812928 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=95 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=95 gdma_id=0 gdma_dir=0 gdma_func=0 core=0

[bmprofile] global_layer: layer_id=300 layer_type=Cast layer_name=
[bmprofile] tensor_id=299 is_in=1 shape=[1x896x1] dtype=8 is_const=0 gaddr=2201170812928 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=300 is_in=0 shape=[1x896x1] dtype=0 is_const=0 gaddr=2201170825216 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=95 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048741 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=96 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=303 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=219 is_in=1 shape=[1x88x16x16] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=302 is_in=1 shape=[1x32x1x88] dtype=8 is_const=1 gaddr=1101659533312 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=301 is_in=1 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=1101659529216 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=303 is_in=0 shape=[1x32x16x16] dtype=8 is_const=0 gaddr=2201170784256 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=96 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=96 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=96 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048745 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=96 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=96 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048747 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=97 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=97 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=97 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048750 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=98 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=98 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=98 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048753 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=99 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=100 gdma_id=0 gdma_dir=0 gdma_func=0 core=0

[bmprofile] global_layer: layer_id=304 layer_type=Permute layer_name=
[bmprofile] tensor_id=303 is_in=1 shape=[1x32x16x16] dtype=8 is_const=0 gaddr=2201170784256 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=304 is_in=0 shape=[1x16x16x32] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=100 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048756 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=101 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=305 layer_type=Reshape layer_name=
[bmprofile] tensor_id=304 is_in=1 shape=[1x16x16x32] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=305 is_in=0 shape=[1x512x16] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=308 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=287 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=2201170800640 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=307 is_in=1 shape=[1x96x1x96] dtype=8 is_const=1 gaddr=1101659545600 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=306 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=1101659541504 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=308 is_in=0 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=2201170767872 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=101 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=101 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=101 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048760 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=101 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=101 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048762 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=102 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=102 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=102 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048765 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=103 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=103 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=103 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048768 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=104 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=104 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=104 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048771 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=105 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=105 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=105 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048774 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=106 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=106 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=106 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048777 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=107 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=107 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=107 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048780 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=108 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=108 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=108 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048783 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=109 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=109 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=109 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048786 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=110 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=110 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=110 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048789 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=111 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=111 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=111 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048792 bd_func=0 core=0
[bmprofile] gdma cmd_id bd_id=112 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=113 gdma_id=0 gdma_dir=0 gdma_func=0 core=0

[bmprofile] global_layer: layer_id=309 layer_type=Permute layer_name=
[bmprofile] tensor_id=308 is_in=1 shape=[1x96x8x8] dtype=8 is_const=0 gaddr=2201170767872 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=309 is_in=0 shape=[1x8x8x96] dtype=8 is_const=0 gaddr=2201170755584 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=113 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048795 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=114 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=310 layer_type=Reshape layer_name=
[bmprofile] tensor_id=309 is_in=1 shape=[1x8x8x96] dtype=8 is_const=0 gaddr=2201170755584 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=310 is_in=0 shape=[1x384x16] dtype=8 is_const=0 gaddr=2201170755584 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=311 layer_type=Concat layer_name=
[bmprofile] tensor_id=305 is_in=1 shape=[1x512x16] dtype=8 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=310 is_in=1 shape=[1x384x16] dtype=8 is_const=0 gaddr=2201170755584 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=311 is_in=0 shape=[1x896x16] dtype=8 is_const=0 gaddr=2201170796544 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=114 gdma_id=0 gdma_dir=0 gdma_func=0 core=0
[bmprofile] gdma cmd_id bd_id=114 gdma_id=0 gdma_dir=0 gdma_func=0 core=0

[bmprofile] global_layer: layer_id=312 layer_type=Cast layer_name=
[bmprofile] tensor_id=311 is_in=1 shape=[1x896x16] dtype=8 is_const=0 gaddr=2201170796544 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=312 is_in=0 shape=[1x896x16] dtype=0 is_const=0 gaddr=2201170739200 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=114 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1048799 bd_func=3 core=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=115 gdma_id=0 gdma_dir=0 gdma_func=1 core=0
[bmprofile] end parallel.
[bmprofile] end to run subnet_id=0
[bmprofile] core_list=0,
[bmprofile] mtype=1 addr=2237067264 size=909312 alloc=1764054188708048 free=1764054188733942 desc=neuron_mem
[bmprofile] mtype=1 addr=2236612608 size=454656 alloc=1764054188708065 free=0 desc=coeff
[bmprofile] mtype=1 addr=2237976576 size=7424 alloc=1764054188708180 free=1764054188733914 desc=bd_cmd_mem
[bmprofile] mtype=1 addr=2237984768 size=21632 alloc=1764054188709648 free=1764054188733928 desc=gdma_cmd_mem
[bmprofile] mtype=1 addr=2238009344 size=57344 alloc=1764054188712295 free=0 desc=io_mem
[bmprofile] mtype=1 addr=2238066688 size=3584 alloc=1764054188712354 free=0 desc=io_mem
[bmprofile] mtype=1 addr=2238070784 size=196608 alloc=1764054188712551 free=0 desc=io_mem
[bmprofile] mtype=1 addr=2238267392 size=262144 alloc=1764054188715570 free=1764054188732066 desc=bdc_perf_monitor
[bmprofile] mtype=1 addr=2238529536 size=1048576 alloc=1764054188715640 free=1764054188732371 desc=gdma_perf_monitor
