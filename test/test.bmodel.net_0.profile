[bmprofile] is_mlir=1
...Start Profile Log...
[bmprofile] start to run subnet_id=0

[bmprofile] global_layer: layer_id=12 layer_type=Conv2D layer_name=
[bmprofile] tensor_id=-1 is_in=1 shape=[1x3x14x14] dtype=0 is_const=0 gaddr=4294975488 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=10 is_in=1 shape=[1x3x1x1] dtype=0 is_const=1 gaddr=4294967296 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=11 is_in=1 shape=[1x3x1x1] dtype=0 is_const=1 gaddr=4294971392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=12 is_in=0 shape=[1x3x14x14] dtype=0 is_const=0 gaddr=4294979584 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=2 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=3 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=1 gdma_id=3 bd_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=1 gdma_id=4 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=13 layer_type=AbsAdd layer_name=
[bmprofile] tensor_id=12 is_in=1 shape=[1x3x14x14] dtype=0 is_const=0 gaddr=4294979584 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=13 is_in=0 shape=[1x3x14x14] dtype=0 is_const=0 gaddr=4294983680 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=1 gdma_id=5 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=2 gdma_id=5 bd_func=3
[bmprofile] bd cmd_id bd_id=3 gdma_id=5 bd_func=3
[bmprofile] gdma cmd_id bd_id=3 gdma_id=6 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=0 gdma_id=0 bd_func=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=7 gdma_dir=0 gdma_func=6
[bmprofile] end to run subnet_id=0
