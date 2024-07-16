# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from include.layer import TotalLayerInfo


def generate_layer(global_info, writer, out_file, tiu_instance_map, gdma_instance_map, chip_arch):
    layer_list = []
    for sub_net in global_info.subnet_list:
        if sub_net is not None:
            layer_list.extend(sub_net.layer_list)
    layer_infos = TotalLayerInfo(writer, layer_list)
    layer_infos.add_kpi_field(tiu_instance_map, gdma_instance_map, chip_arch)
    lay_info_map = layer_infos.pop_data()
    layer_infos.write(chip_arch)
    return lay_info_map
