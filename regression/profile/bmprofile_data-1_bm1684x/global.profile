[bmprofile] arch=3
[bmprofile] net_name=blazeface
[bmprofile] tpu_freq=1000
[bmprofile] is_mlir=1
...Start Profile Log...
[bmprofile] start to run subnet_id=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=20 layer_type=Load layer_name=
[bmprofile] tensor_id=-1 is_in=1 shape=[1x3x128x128] dtype=0 is_const=0 gaddr=4296138752 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=20 is_in=0 shape=[1x3x128x128] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=128 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=23 layer_type=Cast layer_name=
[bmprofile] tensor_id=20 is_in=1 shape=[1x3x128x128] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=128 l2addr=0
[bmprofile] tensor_id=23 is_in=0 shape=[1x3x128x128] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=128 l2addr=0
[bmprofile] bd cmd_id bd_id=1 gdma_id=1 bd_func=3
[bmprofile] local_layer: layer_id=21 layer_type=Load layer_name=
[bmprofile] tensor_id=13 is_in=1 shape=[1x24x1x160] dtype=1 is_const=1 gaddr=4294971392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=21 is_in=0 shape=[1x24x1x160] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=147456 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=2 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=22 layer_type=Load layer_name=
[bmprofile] tensor_id=12 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=4294967296 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=22 is_in=0 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=163840 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=3 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=26 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=23 is_in=1 shape=[1x3x128x128] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=128 l2addr=0
[bmprofile] tensor_id=21 is_in=1 shape=[1x24x1x160] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=147456 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=22 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=163840 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=26 is_in=0 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=64 l2addr=0
[bmprofile] bd cmd_id bd_id=2 gdma_id=3 bd_func=5
[bmprofile] bd cmd_id bd_id=3 gdma_id=3 bd_func=5
[bmprofile] bd cmd_id bd_id=4 gdma_id=3 bd_func=5
[bmprofile] bd cmd_id bd_id=5 gdma_id=3 bd_func=5
[bmprofile] bd cmd_id bd_id=6 gdma_id=3 bd_func=3
[bmprofile] bd cmd_id bd_id=7 gdma_id=3 bd_func=3
[bmprofile] bd cmd_id bd_id=8 gdma_id=3 bd_func=3
[bmprofile] bd cmd_id bd_id=9 gdma_id=3 bd_func=6
[bmprofile] bd cmd_id bd_id=10 gdma_id=3 bd_func=5
[bmprofile] bd cmd_id bd_id=11 gdma_id=3 bd_func=5
[bmprofile] bd cmd_id bd_id=12 gdma_id=3 bd_func=1
[bmprofile] bd cmd_id bd_id=13 gdma_id=3 bd_func=0
[bmprofile] bd cmd_id bd_id=14 gdma_id=3 bd_func=3
[bmprofile] local_layer: layer_id=24 layer_type=Load layer_name=
[bmprofile] tensor_id=16 is_in=1 shape=[1x24x1x288] dtype=1 is_const=1 gaddr=4294987776 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=24 is_in=0 shape=[1x24x1x288] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=131072 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=1 gdma_id=4 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=25 layer_type=Load layer_name=
[bmprofile] tensor_id=14 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=4294979584 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=25 is_in=0 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=180224 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=1 gdma_id=5 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=29 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=26 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=24 is_in=1 shape=[1x24x1x288] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=131072 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=25 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=180224 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=29 is_in=0 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=64 l2addr=0
[bmprofile] bd cmd_id bd_id=15 gdma_id=5 bd_func=0
[bmprofile] local_layer: layer_id=27 layer_type=Load layer_name=
[bmprofile] tensor_id=18 is_in=1 shape=[1x24x1x32] dtype=1 is_const=1 gaddr=4295008256 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=27 is_in=0 shape=[1x24x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=14 gdma_id=6 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=28 layer_type=Load layer_name=
[bmprofile] tensor_id=17 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=4295004160 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=28 is_in=0 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=147456 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=14 gdma_id=7 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=30 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=29 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=27 is_in=1 shape=[1x24x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=28 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=147456 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=30 is_in=0 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=64 l2addr=0
[bmprofile] bd cmd_id bd_id=16 gdma_id=7 bd_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=31 layer_type=Add layer_name=
[bmprofile] tensor_id=30 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=26 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=31 is_in=0 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=64 l2addr=0
[bmprofile] bd cmd_id bd_id=17 gdma_id=7 bd_func=3
[bmprofile] bd cmd_id bd_id=18 gdma_id=7 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=32 layer_type=Store layer_name=
[bmprofile] tensor_id=31 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=32 is_in=0 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=18 gdma_id=8 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=35 layer_type=Pad layer_name=
[bmprofile] tensor_id=32 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=4295942144 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=35 is_in=0 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=4295712768 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=18 gdma_id=9 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=18 gdma_id=10 gdma_dir=0 gdma_func=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=40 layer_type=Load layer_name=
[bmprofile] tensor_id=32 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=4295942144 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=40 is_in=0 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=18 gdma_id=11 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=41 layer_type=Load layer_name=
[bmprofile] tensor_id=15 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=4294983680 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=41 is_in=0 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=18 gdma_id=12 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=42 layer_type=Load layer_name=
[bmprofile] tensor_id=36 is_in=1 shape=[1x24x1x288] dtype=1 is_const=1 gaddr=4295012352 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=42 is_in=0 shape=[1x24x1x288] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=18 gdma_id=13 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=46 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=40 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=42 is_in=1 shape=[1x24x1x288] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=41 is_in=1 shape=[1x24x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=46 is_in=0 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=64 l2addr=0
[bmprofile] bd cmd_id bd_id=19 gdma_id=13 bd_func=0
[bmprofile] local_layer: layer_id=43 layer_type=Load layer_name=
[bmprofile] tensor_id=38 is_in=1 shape=[1x28x1x32] dtype=1 is_const=1 gaddr=4295032832 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=43 is_in=0 shape=[1x28x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=18 gdma_id=14 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=44 layer_type=Load layer_name=
[bmprofile] tensor_id=37 is_in=1 shape=[1x28x1x1] dtype=0 is_const=1 gaddr=4295028736 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=44 is_in=0 shape=[1x28x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=18 gdma_id=15 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=45 layer_type=Load layer_name=
[bmprofile] tensor_id=35 is_in=1 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=4295712768 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=45 is_in=0 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=18 gdma_id=16 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=47 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=46 is_in=1 shape=[1x24x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=43 is_in=1 shape=[1x28x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=44 is_in=1 shape=[1x28x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=47 is_in=0 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=64 l2addr=0
[bmprofile] bd cmd_id bd_id=20 gdma_id=16 bd_func=0
[bmprofile] local_layer: layer_id=48 layer_type=Add layer_name=
[bmprofile] tensor_id=47 is_in=1 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=45 is_in=1 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=48 is_in=0 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=64 l2addr=0
[bmprofile] bd cmd_id bd_id=21 gdma_id=16 bd_func=3
[bmprofile] bd cmd_id bd_id=22 gdma_id=16 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=50 layer_type=Pool2D layer_name=
[bmprofile] tensor_id=48 is_in=1 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=50 is_in=0 shape=[1x28x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=32 l2addr=0
[bmprofile] bd cmd_id bd_id=23 gdma_id=16 bd_func=1
[bmprofile] local_layer: layer_id=49 layer_type=Store layer_name=
[bmprofile] tensor_id=48 is_in=1 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=49 is_in=0 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=24576 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=22 gdma_id=17 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=51 layer_type=Store layer_name=
[bmprofile] tensor_id=50 is_in=1 shape=[1x28x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=51 is_in=0 shape=[1x28x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=40960 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=18 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=54 layer_type=Pad layer_name=
[bmprofile] tensor_id=51 is_in=1 shape=[1x28x32x32] dtype=1 is_const=0 gaddr=4296335360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=54 is_in=0 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=4295794688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=19 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=20 gdma_dir=0 gdma_func=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=60 layer_type=Load layer_name=
[bmprofile] tensor_id=49 is_in=1 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=60 is_in=0 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=64 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=21 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=61 layer_type=Load layer_name=
[bmprofile] tensor_id=56 is_in=1 shape=[1x28x1x288] dtype=1 is_const=1 gaddr=4295041024 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=61 is_in=0 shape=[1x28x1x288] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=22 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=62 layer_type=Load layer_name=
[bmprofile] tensor_id=55 is_in=1 shape=[1x28x1x1] dtype=0 is_const=1 gaddr=4295036928 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=62 is_in=0 shape=[1x28x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=23 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=65 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=60 is_in=1 shape=[1x28x64x64] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=64 l2addr=0
[bmprofile] tensor_id=61 is_in=1 shape=[1x28x1x288] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=62 is_in=1 shape=[1x28x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=65 is_in=0 shape=[1x28x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] bd cmd_id bd_id=24 gdma_id=23 bd_func=0
[bmprofile] local_layer: layer_id=63 layer_type=Load layer_name=
[bmprofile] tensor_id=58 is_in=1 shape=[1x32x1x32] dtype=1 is_const=1 gaddr=4295061504 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=63 is_in=0 shape=[1x32x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=67584 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=24 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=64 layer_type=Load layer_name=
[bmprofile] tensor_id=57 is_in=1 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=4295057408 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=64 is_in=0 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=25 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=67 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=65 is_in=1 shape=[1x28x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=63 is_in=1 shape=[1x32x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=67584 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=64 is_in=1 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=67 is_in=0 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] bd cmd_id bd_id=25 gdma_id=25 bd_func=0
[bmprofile] local_layer: layer_id=66 layer_type=Load layer_name=
[bmprofile] tensor_id=54 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=4295794688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=66 is_in=0 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=24 gdma_id=26 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=68 layer_type=Add layer_name=
[bmprofile] tensor_id=67 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=66 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=68 is_in=0 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=32 l2addr=0
[bmprofile] bd cmd_id bd_id=26 gdma_id=26 bd_func=3
[bmprofile] bd cmd_id bd_id=27 gdma_id=26 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=69 layer_type=Store layer_name=
[bmprofile] tensor_id=68 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=69 is_in=0 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=27 gdma_id=27 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=72 layer_type=Pad layer_name=
[bmprofile] tensor_id=69 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=4295729152 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=72 is_in=0 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=27 gdma_id=28 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=27 gdma_id=29 gdma_dir=0 gdma_func=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=78 layer_type=Load layer_name=
[bmprofile] tensor_id=69 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=4295729152 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=78 is_in=0 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=27 gdma_id=30 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=79 layer_type=Load layer_name=
[bmprofile] tensor_id=74 is_in=1 shape=[1x32x1x288] dtype=1 is_const=1 gaddr=4295069696 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=79 is_in=0 shape=[1x32x1x288] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=27 gdma_id=31 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=80 layer_type=Load layer_name=
[bmprofile] tensor_id=73 is_in=1 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=4295065600 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=80 is_in=0 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=27 gdma_id=32 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=83 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=78 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=79 is_in=1 shape=[1x32x1x288] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=80 is_in=1 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=83 is_in=0 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] bd cmd_id bd_id=28 gdma_id=32 bd_func=0
[bmprofile] local_layer: layer_id=81 layer_type=Load layer_name=
[bmprofile] tensor_id=76 is_in=1 shape=[1x36x1x32] dtype=1 is_const=1 gaddr=4295094272 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=81 is_in=0 shape=[1x36x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=67584 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=27 gdma_id=33 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=82 layer_type=Load layer_name=
[bmprofile] tensor_id=75 is_in=1 shape=[1x36x1x1] dtype=0 is_const=1 gaddr=4295090176 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=82 is_in=0 shape=[1x36x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=27 gdma_id=34 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=85 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=83 is_in=1 shape=[1x32x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=81 is_in=1 shape=[1x36x1x32] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=67584 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=82 is_in=1 shape=[1x36x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=85 is_in=0 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] bd cmd_id bd_id=29 gdma_id=34 bd_func=0
[bmprofile] local_layer: layer_id=84 layer_type=Load layer_name=
[bmprofile] tensor_id=72 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=84 is_in=0 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=28 gdma_id=35 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=86 layer_type=Add layer_name=
[bmprofile] tensor_id=85 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=84 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=86 is_in=0 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=32 l2addr=0
[bmprofile] bd cmd_id bd_id=30 gdma_id=35 bd_func=3
[bmprofile] bd cmd_id bd_id=31 gdma_id=35 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=87 layer_type=Store layer_name=
[bmprofile] tensor_id=86 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=87 is_in=0 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=31 gdma_id=36 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=90 layer_type=Pad layer_name=
[bmprofile] tensor_id=87 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=4295655424 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=90 is_in=0 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=4295569408 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=31 gdma_id=37 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=31 gdma_id=38 gdma_dir=0 gdma_func=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=96 layer_type=Load layer_name=
[bmprofile] tensor_id=87 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=4295655424 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=96 is_in=0 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=31 gdma_id=39 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=97 layer_type=Load layer_name=
[bmprofile] tensor_id=91 is_in=1 shape=[1x36x3x3] dtype=1 is_const=1 gaddr=4295098368 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=97 is_in=0 shape=[1x36x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=31 gdma_id=40 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=98 layer_type=Load layer_name=
[bmprofile] tensor_id=92 is_in=1 shape=[1x36x1x1] dtype=1 is_const=1 gaddr=4295102464 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=98 is_in=0 shape=[1x36x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=31 gdma_id=41 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=101 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=96 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=97 is_in=1 shape=[1x36x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=98 is_in=1 shape=[1x36x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=101 is_in=0 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] bd cmd_id bd_id=32 gdma_id=41 bd_func=1
[bmprofile] local_layer: layer_id=99 layer_type=Load layer_name=
[bmprofile] tensor_id=94 is_in=1 shape=[1x42x1x64] dtype=1 is_const=1 gaddr=4295110656 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=99 is_in=0 shape=[1x42x1x64] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=49664 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=31 gdma_id=42 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=100 layer_type=Load layer_name=
[bmprofile] tensor_id=93 is_in=1 shape=[1x42x1x1] dtype=0 is_const=1 gaddr=4295106560 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=100 is_in=0 shape=[1x42x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=31 gdma_id=43 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=103 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=101 is_in=1 shape=[1x36x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=99 is_in=1 shape=[1x42x1x64] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=49664 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=100 is_in=1 shape=[1x42x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=103 is_in=0 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] bd cmd_id bd_id=33 gdma_id=43 bd_func=0
[bmprofile] local_layer: layer_id=102 layer_type=Load layer_name=
[bmprofile] tensor_id=90 is_in=1 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=4295569408 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=102 is_in=0 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=32 gdma_id=44 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=104 layer_type=Add layer_name=
[bmprofile] tensor_id=103 is_in=1 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=102 is_in=1 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=104 is_in=0 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] bd cmd_id bd_id=34 gdma_id=44 bd_func=3
[bmprofile] bd cmd_id bd_id=35 gdma_id=44 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=106 layer_type=Pool2D layer_name=
[bmprofile] tensor_id=104 is_in=1 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=106 is_in=0 shape=[1x42x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=36 gdma_id=44 bd_func=1
[bmprofile] local_layer: layer_id=105 layer_type=Store layer_name=
[bmprofile] tensor_id=104 is_in=1 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=105 is_in=0 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=35 gdma_id=45 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=107 layer_type=Store layer_name=
[bmprofile] tensor_id=106 is_in=1 shape=[1x42x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=107 is_in=0 shape=[1x42x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=46 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=110 layer_type=Pad layer_name=
[bmprofile] tensor_id=107 is_in=1 shape=[1x42x16x16] dtype=1 is_const=0 gaddr=4295729152 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=110 is_in=0 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=4295593984 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=47 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=48 gdma_dir=0 gdma_func=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=116 layer_type=Load layer_name=
[bmprofile] tensor_id=105 is_in=1 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=116 is_in=0 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=49 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=117 layer_type=Load layer_name=
[bmprofile] tensor_id=111 is_in=1 shape=[1x42x3x3] dtype=1 is_const=1 gaddr=4295118848 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=117 is_in=0 shape=[1x42x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=50 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=118 layer_type=Load layer_name=
[bmprofile] tensor_id=112 is_in=1 shape=[1x42x1x1] dtype=1 is_const=1 gaddr=4295122944 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=118 is_in=0 shape=[1x42x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=51 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=121 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=116 is_in=1 shape=[1x42x32x32] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=32 l2addr=0
[bmprofile] tensor_id=117 is_in=1 shape=[1x42x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=118 is_in=1 shape=[1x42x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=121 is_in=0 shape=[1x42x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=37 gdma_id=51 bd_func=1
[bmprofile] local_layer: layer_id=119 layer_type=Load layer_name=
[bmprofile] tensor_id=114 is_in=1 shape=[1x48x1x64] dtype=1 is_const=1 gaddr=4295131136 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=119 is_in=0 shape=[1x48x1x64] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=66048 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=52 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=120 layer_type=Load layer_name=
[bmprofile] tensor_id=113 is_in=1 shape=[1x48x1x1] dtype=0 is_const=1 gaddr=4295127040 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=120 is_in=0 shape=[1x48x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=53 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=123 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=121 is_in=1 shape=[1x42x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=119 is_in=1 shape=[1x48x1x64] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=66048 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=120 is_in=1 shape=[1x48x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=123 is_in=0 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=38 gdma_id=53 bd_func=0
[bmprofile] local_layer: layer_id=122 layer_type=Load layer_name=
[bmprofile] tensor_id=110 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=4295593984 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=122 is_in=0 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=37 gdma_id=54 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=124 layer_type=Add layer_name=
[bmprofile] tensor_id=123 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=122 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=124 is_in=0 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=39 gdma_id=54 bd_func=3
[bmprofile] bd cmd_id bd_id=40 gdma_id=54 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=125 layer_type=Store layer_name=
[bmprofile] tensor_id=124 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=125 is_in=0 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=40 gdma_id=55 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=128 layer_type=Pad layer_name=
[bmprofile] tensor_id=125 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=4295569408 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=128 is_in=0 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=40 gdma_id=56 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=40 gdma_id=57 gdma_dir=0 gdma_func=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=134 layer_type=Load layer_name=
[bmprofile] tensor_id=125 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=4295569408 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=134 is_in=0 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=40 gdma_id=58 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=135 layer_type=Load layer_name=
[bmprofile] tensor_id=129 is_in=1 shape=[1x48x3x3] dtype=1 is_const=1 gaddr=4295139328 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=135 is_in=0 shape=[1x48x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=40 gdma_id=59 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=136 layer_type=Load layer_name=
[bmprofile] tensor_id=130 is_in=1 shape=[1x48x1x1] dtype=1 is_const=1 gaddr=4295143424 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=136 is_in=0 shape=[1x48x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=40 gdma_id=60 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=139 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=134 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=135 is_in=1 shape=[1x48x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=136 is_in=1 shape=[1x48x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=139 is_in=0 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=41 gdma_id=60 bd_func=1
[bmprofile] local_layer: layer_id=137 layer_type=Load layer_name=
[bmprofile] tensor_id=132 is_in=1 shape=[1x56x1x64] dtype=1 is_const=1 gaddr=4295151616 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=137 is_in=0 shape=[1x56x1x64] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=66048 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=40 gdma_id=61 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=138 layer_type=Load layer_name=
[bmprofile] tensor_id=131 is_in=1 shape=[1x56x1x1] dtype=0 is_const=1 gaddr=4295147520 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=138 is_in=0 shape=[1x56x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=40 gdma_id=62 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=141 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=139 is_in=1 shape=[1x48x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=137 is_in=1 shape=[1x56x1x64] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=66048 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=138 is_in=1 shape=[1x56x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=141 is_in=0 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=42 gdma_id=62 bd_func=0
[bmprofile] local_layer: layer_id=140 layer_type=Load layer_name=
[bmprofile] tensor_id=128 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=140 is_in=0 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=41 gdma_id=63 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=142 layer_type=Add layer_name=
[bmprofile] tensor_id=141 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=140 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=142 is_in=0 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=43 gdma_id=63 bd_func=3
[bmprofile] bd cmd_id bd_id=44 gdma_id=63 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=143 layer_type=Store layer_name=
[bmprofile] tensor_id=142 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=143 is_in=0 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=44 gdma_id=64 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=146 layer_type=Pad layer_name=
[bmprofile] tensor_id=143 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=4295516160 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=146 is_in=0 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=44 gdma_id=65 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=44 gdma_id=66 gdma_dir=0 gdma_func=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=152 layer_type=Load layer_name=
[bmprofile] tensor_id=143 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=4295516160 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=152 is_in=0 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=44 gdma_id=67 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=153 layer_type=Load layer_name=
[bmprofile] tensor_id=147 is_in=1 shape=[1x56x3x3] dtype=1 is_const=1 gaddr=4295159808 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=153 is_in=0 shape=[1x56x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=44 gdma_id=68 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=154 layer_type=Load layer_name=
[bmprofile] tensor_id=148 is_in=1 shape=[1x56x1x1] dtype=1 is_const=1 gaddr=4295163904 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=154 is_in=0 shape=[1x56x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=44 gdma_id=69 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=157 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=152 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=153 is_in=1 shape=[1x56x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=154 is_in=1 shape=[1x56x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=157 is_in=0 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=45 gdma_id=69 bd_func=1
[bmprofile] local_layer: layer_id=155 layer_type=Load layer_name=
[bmprofile] tensor_id=150 is_in=1 shape=[1x64x1x64] dtype=1 is_const=1 gaddr=4295172096 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=155 is_in=0 shape=[1x64x1x64] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=66048 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=44 gdma_id=70 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=156 layer_type=Load layer_name=
[bmprofile] tensor_id=149 is_in=1 shape=[1x64x1x1] dtype=0 is_const=1 gaddr=4295168000 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=156 is_in=0 shape=[1x64x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=44 gdma_id=71 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=159 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=157 is_in=1 shape=[1x56x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=155 is_in=1 shape=[1x64x1x64] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=66048 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=156 is_in=1 shape=[1x64x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=159 is_in=0 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=46 gdma_id=71 bd_func=0
[bmprofile] local_layer: layer_id=158 layer_type=Load layer_name=
[bmprofile] tensor_id=146 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=158 is_in=0 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=45 gdma_id=72 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=160 layer_type=Add layer_name=
[bmprofile] tensor_id=159 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=158 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=160 is_in=0 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=47 gdma_id=72 bd_func=3
[bmprofile] bd cmd_id bd_id=48 gdma_id=72 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=161 layer_type=Store layer_name=
[bmprofile] tensor_id=160 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=161 is_in=0 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=48 gdma_id=73 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=164 layer_type=Pad layer_name=
[bmprofile] tensor_id=161 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=4295561216 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=164 is_in=0 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=48 gdma_id=74 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=48 gdma_id=75 gdma_dir=0 gdma_func=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=170 layer_type=Load layer_name=
[bmprofile] tensor_id=161 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=4295561216 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=170 is_in=0 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=48 gdma_id=76 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=171 layer_type=Load layer_name=
[bmprofile] tensor_id=165 is_in=1 shape=[1x64x3x3] dtype=1 is_const=1 gaddr=4295180288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=171 is_in=0 shape=[1x64x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=48 gdma_id=77 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=172 layer_type=Load layer_name=
[bmprofile] tensor_id=166 is_in=1 shape=[1x64x1x1] dtype=1 is_const=1 gaddr=4295184384 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=172 is_in=0 shape=[1x64x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=48 gdma_id=78 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=175 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=170 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=171 is_in=1 shape=[1x64x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=172 is_in=1 shape=[1x64x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=175 is_in=0 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=49 gdma_id=78 bd_func=1
[bmprofile] local_layer: layer_id=173 layer_type=Load layer_name=
[bmprofile] tensor_id=168 is_in=1 shape=[1x72x1x64] dtype=1 is_const=1 gaddr=4295192576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=173 is_in=0 shape=[1x72x1x64] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=33792 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=48 gdma_id=79 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=174 layer_type=Load layer_name=
[bmprofile] tensor_id=167 is_in=1 shape=[1x72x1x1] dtype=0 is_const=1 gaddr=4295188480 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=174 is_in=0 shape=[1x72x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=48 gdma_id=80 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=177 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=175 is_in=1 shape=[1x64x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=173 is_in=1 shape=[1x72x1x64] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=33792 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=174 is_in=1 shape=[1x72x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=177 is_in=0 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=50 gdma_id=80 bd_func=0
[bmprofile] local_layer: layer_id=176 layer_type=Load layer_name=
[bmprofile] tensor_id=164 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=176 is_in=0 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=49 gdma_id=81 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=178 layer_type=Add layer_name=
[bmprofile] tensor_id=177 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=176 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=178 is_in=0 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=51 gdma_id=81 bd_func=3
[bmprofile] bd cmd_id bd_id=52 gdma_id=81 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=179 layer_type=Store layer_name=
[bmprofile] tensor_id=178 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=179 is_in=0 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=52 gdma_id=82 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=182 layer_type=Pad layer_name=
[bmprofile] tensor_id=179 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=4295524352 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=182 is_in=0 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=52 gdma_id=83 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=52 gdma_id=84 gdma_dir=0 gdma_func=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=188 layer_type=Load layer_name=
[bmprofile] tensor_id=179 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=4295524352 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=188 is_in=0 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=52 gdma_id=85 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=189 layer_type=Load layer_name=
[bmprofile] tensor_id=183 is_in=1 shape=[1x72x3x3] dtype=1 is_const=1 gaddr=4295204864 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=189 is_in=0 shape=[1x72x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=52 gdma_id=86 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=190 layer_type=Load layer_name=
[bmprofile] tensor_id=184 is_in=1 shape=[1x72x1x1] dtype=1 is_const=1 gaddr=4295208960 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=190 is_in=0 shape=[1x72x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=52 gdma_id=87 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=193 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=188 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=189 is_in=1 shape=[1x72x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=190 is_in=1 shape=[1x72x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=193 is_in=0 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=53 gdma_id=87 bd_func=1
[bmprofile] local_layer: layer_id=191 layer_type=Load layer_name=
[bmprofile] tensor_id=186 is_in=1 shape=[1x80x1x96] dtype=1 is_const=1 gaddr=4295217152 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=191 is_in=0 shape=[1x80x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=66560 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=52 gdma_id=88 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=192 layer_type=Load layer_name=
[bmprofile] tensor_id=185 is_in=1 shape=[1x80x1x1] dtype=0 is_const=1 gaddr=4295213056 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=192 is_in=0 shape=[1x80x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=52 gdma_id=89 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=195 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=193 is_in=1 shape=[1x72x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=191 is_in=1 shape=[1x80x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=66560 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=192 is_in=1 shape=[1x80x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=195 is_in=0 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=54 gdma_id=89 bd_func=0
[bmprofile] local_layer: layer_id=194 layer_type=Load layer_name=
[bmprofile] tensor_id=182 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=194 is_in=0 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=53 gdma_id=90 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=196 layer_type=Add layer_name=
[bmprofile] tensor_id=195 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=194 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=196 is_in=0 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=55 gdma_id=90 bd_func=3
[bmprofile] bd cmd_id bd_id=56 gdma_id=90 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=197 layer_type=Store layer_name=
[bmprofile] tensor_id=196 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=197 is_in=0 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=56 gdma_id=91 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=200 layer_type=Pad layer_name=
[bmprofile] tensor_id=197 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=4295573504 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=200 is_in=0 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=4295528448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=56 gdma_id=92 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=56 gdma_id=93 gdma_dir=0 gdma_func=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=206 layer_type=Load layer_name=
[bmprofile] tensor_id=197 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=4295573504 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=206 is_in=0 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=56 gdma_id=94 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=207 layer_type=Load layer_name=
[bmprofile] tensor_id=201 is_in=1 shape=[1x80x3x3] dtype=1 is_const=1 gaddr=4295233536 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=207 is_in=0 shape=[1x80x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=56 gdma_id=95 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=208 layer_type=Load layer_name=
[bmprofile] tensor_id=202 is_in=1 shape=[1x80x1x1] dtype=1 is_const=1 gaddr=4295237632 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=208 is_in=0 shape=[1x80x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=56 gdma_id=96 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=211 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=206 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=207 is_in=1 shape=[1x80x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=208 is_in=1 shape=[1x80x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=211 is_in=0 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=57 gdma_id=96 bd_func=1
[bmprofile] local_layer: layer_id=209 layer_type=Load layer_name=
[bmprofile] tensor_id=204 is_in=1 shape=[1x88x1x96] dtype=1 is_const=1 gaddr=4295245824 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=209 is_in=0 shape=[1x88x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=49408 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=56 gdma_id=97 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=210 layer_type=Load layer_name=
[bmprofile] tensor_id=203 is_in=1 shape=[1x88x1x1] dtype=0 is_const=1 gaddr=4295241728 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=210 is_in=0 shape=[1x88x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=56 gdma_id=98 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=213 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=211 is_in=1 shape=[1x80x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=209 is_in=1 shape=[1x88x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=49408 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=210 is_in=1 shape=[1x88x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=213 is_in=0 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=58 gdma_id=98 bd_func=0
[bmprofile] local_layer: layer_id=212 layer_type=Load layer_name=
[bmprofile] tensor_id=200 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=4295528448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=212 is_in=0 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=57 gdma_id=99 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=214 layer_type=Add layer_name=
[bmprofile] tensor_id=213 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=212 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=214 is_in=0 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=59 gdma_id=99 bd_func=3
[bmprofile] bd cmd_id bd_id=60 gdma_id=99 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=216 layer_type=Pool2D layer_name=
[bmprofile] tensor_id=214 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=216 is_in=0 shape=[1x88x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=61 gdma_id=99 bd_func=1
[bmprofile] local_layer: layer_id=215 layer_type=Store layer_name=
[bmprofile] tensor_id=214 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=215 is_in=0 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=60 gdma_id=100 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=217 layer_type=Store layer_name=
[bmprofile] tensor_id=216 is_in=1 shape=[1x88x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=217 is_in=0 shape=[1x88x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=101 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=220 layer_type=Pad layer_name=
[bmprofile] tensor_id=217 is_in=1 shape=[1x88x8x8] dtype=1 is_const=0 gaddr=4295614464 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=220 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=4295528448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=102 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=103 gdma_dir=0 gdma_func=0
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
[bmprofile] tensor_id=222 is_in=1 shape=[1x88x1x1] dtype=1 is_const=1 gaddr=4295270400 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=244 is_in=0 shape=[1x88x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=104 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=245 layer_type=Load layer_name=
[bmprofile] tensor_id=221 is_in=1 shape=[1x88x3x3] dtype=1 is_const=1 gaddr=4295266304 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=245 is_in=0 shape=[1x88x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=65728 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=105 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=246 layer_type=Load layer_name=
[bmprofile] tensor_id=215 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=246 is_in=0 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=106 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=249 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=246 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=245 is_in=1 shape=[1x88x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=65728 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=244 is_in=1 shape=[1x88x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=249 is_in=0 shape=[1x88x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=62 gdma_id=106 bd_func=1
[bmprofile] local_layer: layer_id=247 layer_type=Load layer_name=
[bmprofile] tensor_id=224 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=4295278592 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=247 is_in=0 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=16896 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=107 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=248 layer_type=Load layer_name=
[bmprofile] tensor_id=223 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=4295274496 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=248 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=108 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=251 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=249 is_in=1 shape=[1x88x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=247 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=16896 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=248 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=251 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=63 gdma_id=108 bd_func=0
[bmprofile] local_layer: layer_id=250 layer_type=Load layer_name=
[bmprofile] tensor_id=220 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=4295528448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=250 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=62 gdma_id=109 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=254 layer_type=Add layer_name=
[bmprofile] tensor_id=251 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=250 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=254 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=1024 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=64 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=65 gdma_id=109 bd_func=3
[bmprofile] local_layer: layer_id=252 layer_type=Load layer_name=
[bmprofile] tensor_id=225 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=4295299072 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=252 is_in=0 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=16896 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=63 gdma_id=110 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=253 layer_type=Load layer_name=
[bmprofile] tensor_id=226 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=4295303168 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=253 is_in=0 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=63 gdma_id=111 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=257 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=254 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=1024 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=252 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=16896 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=253 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=257 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=66 gdma_id=111 bd_func=1
[bmprofile] local_layer: layer_id=255 layer_type=Load layer_name=
[bmprofile] tensor_id=228 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=4295311360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=255 is_in=0 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=65 gdma_id=112 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=256 layer_type=Load layer_name=
[bmprofile] tensor_id=227 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=4295307264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=256 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=65 gdma_id=113 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=260 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=257 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=255 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=256 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=260 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=1280 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=67 gdma_id=113 bd_func=0
[bmprofile] local_layer: layer_id=261 layer_type=Add layer_name=
[bmprofile] tensor_id=260 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=1280 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=254 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=1024 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=261 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=68 gdma_id=113 bd_func=3
[bmprofile] bd cmd_id bd_id=69 gdma_id=113 bd_func=3
[bmprofile] local_layer: layer_id=258 layer_type=Load layer_name=
[bmprofile] tensor_id=229 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=4295331840 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=258 is_in=0 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=114 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=259 layer_type=Load layer_name=
[bmprofile] tensor_id=230 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=4295335936 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=259 is_in=0 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=115 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=264 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=261 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=258 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=259 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=264 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=70 gdma_id=115 bd_func=1
[bmprofile] local_layer: layer_id=262 layer_type=Load layer_name=
[bmprofile] tensor_id=232 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=4295344128 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=262 is_in=0 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=1024 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=69 gdma_id=116 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=263 layer_type=Load layer_name=
[bmprofile] tensor_id=231 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=4295340032 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=263 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=69 gdma_id=117 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=267 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=264 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=262 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=1024 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=263 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=267 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16640 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=71 gdma_id=117 bd_func=0
[bmprofile] local_layer: layer_id=268 layer_type=Add layer_name=
[bmprofile] tensor_id=267 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16640 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=261 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=268 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=72 gdma_id=117 bd_func=3
[bmprofile] bd cmd_id bd_id=73 gdma_id=117 bd_func=3
[bmprofile] local_layer: layer_id=265 layer_type=Load layer_name=
[bmprofile] tensor_id=234 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=4295368704 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=265 is_in=0 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=70 gdma_id=118 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=266 layer_type=Load layer_name=
[bmprofile] tensor_id=233 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=4295364608 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=266 is_in=0 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=70 gdma_id=119 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=271 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=268 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=266 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=265 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=114688 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=271 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=74 gdma_id=119 bd_func=1
[bmprofile] local_layer: layer_id=269 layer_type=Load layer_name=
[bmprofile] tensor_id=236 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=4295376896 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=269 is_in=0 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=1024 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=73 gdma_id=120 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=270 layer_type=Load layer_name=
[bmprofile] tensor_id=235 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=4295372800 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=270 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=73 gdma_id=121 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=274 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=271 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=269 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=1024 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=270 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=274 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16640 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=75 gdma_id=121 bd_func=0
[bmprofile] local_layer: layer_id=275 layer_type=Add layer_name=
[bmprofile] tensor_id=274 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16640 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=268 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=275 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=76 gdma_id=121 bd_func=3
[bmprofile] bd cmd_id bd_id=77 gdma_id=121 bd_func=3
[bmprofile] local_layer: layer_id=272 layer_type=Load layer_name=
[bmprofile] tensor_id=238 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=4295401472 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=272 is_in=0 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=74 gdma_id=122 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=273 layer_type=Load layer_name=
[bmprofile] tensor_id=237 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=4295397376 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=273 is_in=0 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=3 l2addr=0
[bmprofile] gdma cmd_id bd_id=74 gdma_id=123 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=278 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=275 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=273 is_in=1 shape=[1x96x3x3] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=3 l2addr=0
[bmprofile] tensor_id=272 is_in=1 shape=[1x96x1x1] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=278 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=78 gdma_id=123 bd_func=1
[bmprofile] local_layer: layer_id=276 layer_type=Load layer_name=
[bmprofile] tensor_id=240 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=4295409664 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=276 is_in=0 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=1024 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=77 gdma_id=124 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=277 layer_type=Load layer_name=
[bmprofile] tensor_id=239 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=4295405568 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=277 is_in=0 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=77 gdma_id=125 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=281 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=278 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=276 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=1024 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=277 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=281 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=33024 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=79 gdma_id=125 bd_func=0
[bmprofile] local_layer: layer_id=282 layer_type=Add layer_name=
[bmprofile] tensor_id=281 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=33024 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=275 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=282 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=80 gdma_id=125 bd_func=3
[bmprofile] bd cmd_id bd_id=81 gdma_id=125 bd_func=3
[bmprofile] local_layer: layer_id=279 layer_type=Load layer_name=
[bmprofile] tensor_id=241 is_in=1 shape=[1x2x1x1] dtype=0 is_const=1 gaddr=4295430144 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=279 is_in=0 shape=[1x2x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98368 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=78 gdma_id=126 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=280 layer_type=Load layer_name=
[bmprofile] tensor_id=242 is_in=1 shape=[1x2x1x96] dtype=1 is_const=1 gaddr=4295434240 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=280 is_in=0 shape=[1x2x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=78 gdma_id=127 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=284 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=246 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=280 is_in=1 shape=[1x2x1x96] dtype=1 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=279 is_in=1 shape=[1x2x1x1] dtype=0 is_const=1 gaddr=0 gsize=0 loffset=98368 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=284 is_in=0 shape=[1x2x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=82 gdma_id=127 bd_func=0
[bmprofile] local_layer: layer_id=283 layer_type=Store layer_name=
[bmprofile] tensor_id=282 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=283 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=81 gdma_id=128 gdma_dir=0 gdma_func=0
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
[bmprofile] tensor_id=284 is_in=1 shape=[1x2x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=285 is_in=0 shape=[1x2x16x16] dtype=1 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=82 gdma_id=129 gdma_dir=0 gdma_func=0
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
[bmprofile] tensor_id=285 is_in=1 shape=[1x2x16x16] dtype=1 is_const=0 gaddr=4295540736 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=288 is_in=0 shape=[1x16x16x2] dtype=1 is_const=0 gaddr=4295536640 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=82 gdma_id=130 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=83 gdma_id=130 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=83 gdma_id=131 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=289 layer_type=Reshape layer_name=
[bmprofile] tensor_id=288 is_in=1 shape=[1x16x16x2] dtype=1 is_const=0 gaddr=4295536640 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=289 is_in=0 shape=[1x512x1] dtype=1 is_const=0 gaddr=4295536640 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=292 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=283 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=4295544832 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=291 is_in=1 shape=[1x6x1x96] dtype=1 is_const=1 gaddr=4295442432 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=290 is_in=1 shape=[1x6x1x1] dtype=0 is_const=1 gaddr=4295438336 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=292 is_in=0 shape=[1x6x8x8] dtype=1 is_const=0 gaddr=4295532544 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=83 gdma_id=132 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=83 gdma_id=133 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=83 gdma_id=134 gdma_dir=0 gdma_func=0
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=84 gdma_id=134 bd_func=0
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=84 gdma_id=135 gdma_dir=0 gdma_func=0

[bmprofile] global_layer: layer_id=293 layer_type=Permute layer_name=
[bmprofile] tensor_id=292 is_in=1 shape=[1x6x8x8] dtype=1 is_const=0 gaddr=4295532544 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=293 is_in=0 shape=[1x8x8x6] dtype=1 is_const=0 gaddr=4295528448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=84 gdma_id=136 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=85 gdma_id=136 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=85 gdma_id=137 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=294 layer_type=Reshape layer_name=
[bmprofile] tensor_id=293 is_in=1 shape=[1x8x8x6] dtype=1 is_const=0 gaddr=4295528448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=294 is_in=0 shape=[1x384x1] dtype=1 is_const=0 gaddr=4295528448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=295 layer_type=Concat layer_name=
[bmprofile] tensor_id=289 is_in=1 shape=[1x512x1] dtype=1 is_const=0 gaddr=4295536640 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=294 is_in=1 shape=[1x384x1] dtype=1 is_const=0 gaddr=4295528448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=295 is_in=0 shape=[1x896x1] dtype=1 is_const=0 gaddr=4295557120 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=85 gdma_id=138 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=85 gdma_id=139 gdma_dir=0 gdma_func=0

[bmprofile] global_layer: layer_id=296 layer_type=Cast layer_name=
[bmprofile] tensor_id=295 is_in=1 shape=[1x896x1] dtype=1 is_const=0 gaddr=4295557120 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=296 is_in=0 shape=[1x896x1] dtype=0 is_const=0 gaddr=4295569408 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=85 gdma_id=140 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=86 gdma_id=140 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=86 gdma_id=141 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=299 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=215 is_in=1 shape=[1x88x16x16] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=298 is_in=1 shape=[1x32x1x96] dtype=1 is_const=1 gaddr=4295450624 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=297 is_in=1 shape=[1x32x1x1] dtype=0 is_const=1 gaddr=4295446528 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=299 is_in=0 shape=[1x32x16x16] dtype=1 is_const=0 gaddr=4295528448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=142 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=143 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=144 gdma_dir=0 gdma_func=0
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=87 gdma_id=144 bd_func=0
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=87 gdma_id=145 gdma_dir=0 gdma_func=0

[bmprofile] global_layer: layer_id=300 layer_type=Permute layer_name=
[bmprofile] tensor_id=299 is_in=1 shape=[1x32x16x16] dtype=1 is_const=0 gaddr=4295528448 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=300 is_in=0 shape=[1x16x16x32] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=87 gdma_id=146 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=88 gdma_id=146 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=88 gdma_id=147 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=301 layer_type=Reshape layer_name=
[bmprofile] tensor_id=300 is_in=1 shape=[1x16x16x32] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=301 is_in=0 shape=[1x512x16] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=304 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=283 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=4295544832 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=303 is_in=1 shape=[1x96x1x96] dtype=1 is_const=1 gaddr=4295462912 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=302 is_in=1 shape=[1x96x1x1] dtype=0 is_const=1 gaddr=4295458816 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=304 is_in=0 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=4295512064 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=88 gdma_id=148 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=88 gdma_id=149 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=88 gdma_id=150 gdma_dir=0 gdma_func=0
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=89 gdma_id=150 bd_func=0
[bmprofile] gdma cmd_id bd_id=88 gdma_id=151 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=88 gdma_id=152 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=90 gdma_id=152 bd_func=0
[bmprofile] gdma cmd_id bd_id=89 gdma_id=153 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] gdma cmd_id bd_id=90 gdma_id=154 gdma_dir=0 gdma_func=0

[bmprofile] global_layer: layer_id=305 layer_type=Permute layer_name=
[bmprofile] tensor_id=304 is_in=1 shape=[1x96x8x8] dtype=1 is_const=0 gaddr=4295512064 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=305 is_in=0 shape=[1x8x8x96] dtype=1 is_const=0 gaddr=4295499776 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=90 gdma_id=155 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=91 gdma_id=155 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=91 gdma_id=156 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=306 layer_type=Reshape layer_name=
[bmprofile] tensor_id=305 is_in=1 shape=[1x8x8x96] dtype=1 is_const=0 gaddr=4295499776 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=306 is_in=0 shape=[1x384x16] dtype=1 is_const=0 gaddr=4295499776 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=307 layer_type=Concat layer_name=
[bmprofile] tensor_id=301 is_in=1 shape=[1x512x16] dtype=1 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=306 is_in=1 shape=[1x384x16] dtype=1 is_const=0 gaddr=4295499776 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=307 is_in=0 shape=[1x896x16] dtype=1 is_const=0 gaddr=4295540736 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=91 gdma_id=157 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=91 gdma_id=158 gdma_dir=0 gdma_func=0

[bmprofile] global_layer: layer_id=308 layer_type=Cast layer_name=
[bmprofile] tensor_id=307 is_in=1 shape=[1x896x16] dtype=1 is_const=0 gaddr=4295540736 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=308 is_in=0 shape=[1x896x16] dtype=0 is_const=0 gaddr=4295483392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=91 gdma_id=159 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=92 gdma_id=159 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=92 gdma_id=160 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] insert bd end (cmd_id bd_id=93)
[bmprofile] bd cmd_id bd_id=93 gdma_id=0 bd_func=15
[bmprofile] insert gdma end (cmd_id gdma_id=161)
[bmprofile] gdma cmd_id bd_id=0 gdma_id=161 gdma_dir=0 gdma_func=6
[bmprofile] end to run subnet_id=0
[bmprofile] core_list=0,
[bmprofile] mtype=1 addr=4900511744 size=909312 alloc=1764051392743691 free=1764051393005501 desc=neuron_mem
[bmprofile] mtype=1 addr=4899995648 size=516096 alloc=1764051392743694 free=0 desc=coeff
[bmprofile] mtype=1 addr=4901421056 size=5760 alloc=1764051392743737 free=1764051393005494 desc=bd_cmd_mem
[bmprofile] mtype=1 addr=4901429248 size=15488 alloc=1764051392743892 free=1764051393005496 desc=gdma_cmd_mem
[bmprofile] mtype=1 addr=4901445632 size=57344 alloc=1764051392744120 free=0 desc=io_mem
[bmprofile] mtype=1 addr=4901502976 size=3584 alloc=1764051392744123 free=0 desc=io_mem
[bmprofile] mtype=1 addr=4901507072 size=196608 alloc=1764051392744149 free=0 desc=io_mem
[bmprofile] mtype=1 addr=4901703680 size=16777216 alloc=1764051392744475 free=1764051393005287 desc=dyn_profile
[bmprofile] mtype=1 addr=4918480896 size=33554432 alloc=1764051392744486 free=1764051392991112 desc=bdc_perf_monitor
[bmprofile] mtype=1 addr=4952035328 size=268435456 alloc=1764051392744508 free=1764051393005283 desc=gdma_perf_monitor
