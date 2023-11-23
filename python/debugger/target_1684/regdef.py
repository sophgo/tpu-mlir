# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""
regdef for 1684 is special, should be maintained manully.
"""
import ctypes
from typing import Dict, Type
from ..target_common import atomic_reg
from itertools import accumulate


class dma_cmd_reg(atomic_reg):
    _fields_ = [
        ("pio_gdma_enable", ctypes.c_uint64, 1),
        ("des_type", ctypes.c_uint64, 1),
        ("chain_end", ctypes.c_uint64, 1),
        ("intr_en", ctypes.c_uint64, 1),
        ("barrier_enable", ctypes.c_uint64, 1),
        ("stride_enable", ctypes.c_uint64, 1),
        ("direction", ctypes.c_uint64, 2),
        ("acc_write_enable", ctypes.c_uint64, 1),
        ("common_mode", ctypes.c_uint64, 1),
        ("prefetch_disable", ctypes.c_uint64, 1),
        ("hold_des_value", ctypes.c_uint64, 1),
        ("reserved", ctypes.c_uint64, 4),
        ("cmd_id", ctypes.c_uint64, 16),
        ("special_func", ctypes.c_uint64, 3),
        ("dst_data_format", ctypes.c_uint64, 3),
        ("chw_copy", ctypes.c_uint64, 1),
        ("sys_mem_type", ctypes.c_uint64, 1),
        ("src_data_format", ctypes.c_uint64, 3),
        ("lrn_shift_num", ctypes.c_uint64, 4),
        ("lrn_shift_dir", ctypes.c_uint64, 1),
        ("eng0_sync_id", ctypes.c_uint64, 16),
        ("eng1_sync_id", ctypes.c_uint64, 16),
        ("eng3_sync_id", ctypes.c_uint64, 16),
        ("constant_value", ctypes.c_uint64, 32),
        ("src_nstride", ctypes.c_uint64, 32),
        ("src_cstride", ctypes.c_uint64, 32),
        ("src_hstride", ctypes.c_uint64, 32),
        ("src_wstride", ctypes.c_uint64, 32),
        ("dst_nstride", ctypes.c_uint64, 32),
        ("dst_cstride", ctypes.c_uint64, 32),
        ("dst_hstride", ctypes.c_uint64, 32),
        ("dst_wstride", ctypes.c_uint64, 32),
        ("src_nsize", ctypes.c_uint64, 16),
        ("src_csize", ctypes.c_uint64, 16),
        ("src_hsize", ctypes.c_uint64, 16),
        ("src_wsize", ctypes.c_uint64, 16),
        ("dst_nsize", ctypes.c_uint64, 16),
        ("dst_csize", ctypes.c_uint64, 16),
        ("dst_hsize", ctypes.c_uint64, 16),
        ("dst_wsize", ctypes.c_uint64, 16),
        ("src_start_addr_l32", ctypes.c_uint64, 32),
        ("dst_start_addr_l32", ctypes.c_uint64, 32),
        ("src_start_addr_h8", ctypes.c_uint64, 8),
        ("dst_start_addr_h8", ctypes.c_uint64, 8),
        ("src_hshift", ctypes.c_uint64, 24),
        ("src_wshift", ctypes.c_uint64, 8),
        ("dst_hshift", ctypes.c_uint64, 8),
        ("dst_wshift", ctypes.c_uint64, 8),
        ("localmem_mask_l32", ctypes.c_uint64, 32),
        ("localmem_mask_h32", ctypes.c_uint64, 32),
        ("single_step", ctypes.c_uint64, 1),
        ("debug_mode", ctypes.c_uint64, 1),
    ]

    pio_gdma_enable: int
    des_type: int
    chain_end: int
    intr_en: int
    barrier_enable: int
    stride_enable: int
    direction: int
    acc_write_enable: int
    common_mode: int
    prefetch_disable: int
    hold_des_value: int
    reserved: int
    cmd_id: int
    special_func: int
    dst_data_format: int
    chw_copy: int
    sys_mem_type: int
    src_data_format: int
    lrn_shift_num: int
    lrn_shift_dir: int
    eng0_sync_id: int
    eng1_sync_id: int
    eng3_sync_id: int
    constant_value: int
    src_nstride: int
    src_cstride: int
    src_hstride: int
    src_wstride: int
    dst_nstride: int
    dst_cstride: int
    dst_hstride: int
    dst_wstride: int
    src_nsize: int
    src_csize: int
    src_hsize: int
    src_wsize: int
    dst_nsize: int
    dst_csize: int
    dst_hsize: int
    dst_wsize: int
    src_start_addr_l32: int
    dst_start_addr_l32: int
    src_start_addr_h8: int
    dst_start_addr_h8: int
    src_hshift: int
    src_wshift: int
    dst_hshift: int
    dst_wshift: int
    localmem_mask_l32: int
    localmem_mask_h32: int
    single_step: int
    debug_mode: int

    length = 1024
    core_id = 0

    @property
    def cmd_id_dep(self):
        return self.eng0_sync_id


class tiu_cmd_reg(atomic_reg):
    _fields_ = [
        ("cmd_en", ctypes.c_uint64, 1),
        ("cmd_end", ctypes.c_uint64, 1),
        ("cmd_id_en", ctypes.c_uint64, 1),
        ("cmd_id_tpu", ctypes.c_uint64, 16),
        ("cmd_id_gdma", ctypes.c_uint64, 16),
        ("cmd_keep", ctypes.c_uint64, 1),
        ("cmd_intr_en", ctypes.c_uint64, 1),
        ("tsk_typ", ctypes.c_uint64, 4),
        ("tsk_eu_typ", ctypes.c_uint64, 5),
        ("tsk_opd_num", ctypes.c_uint64, 2),
        ("opt_right_shift", ctypes.c_uint64, 5),
        ("opt_left_shift", ctypes.c_uint64, 5),
        ("opt_shift_typ", ctypes.c_uint64, 1),
        ("opt_res_add", ctypes.c_uint64, 1),
        ("opt_relu", ctypes.c_uint64, 1),
        ("opt_left_tran", ctypes.c_uint64, 1),
        ("opt_winograd", ctypes.c_uint64, 1),
        ("opt_kernel_rotate", ctypes.c_uint64, 1),
        ("opt_opd0_sign", ctypes.c_uint64, 1),
        ("opt_opd1_sign", ctypes.c_uint64, 1),
        ("opt_opd2_sign", ctypes.c_uint64, 1),
        ("opt_res0_prec", ctypes.c_uint64, 3),
        ("opt_opd0_prec", ctypes.c_uint64, 3),
        ("opt_opd1_prec", ctypes.c_uint64, 3),
        ("opt_opd2_prec", ctypes.c_uint64, 3),
        ("opt_opd0_const", ctypes.c_uint64, 1),
        ("opt_opd1_const", ctypes.c_uint64, 1),
        ("opt_opd2_const", ctypes.c_uint64, 1),
        ("short_res0_str", ctypes.c_uint64, 3),
        ("short_opd0_str", ctypes.c_uint64, 3),
        ("short_opd1_str", ctypes.c_uint64, 3),
        ("short_opd2_str", ctypes.c_uint64, 3),
        # start not aligned
        ("opd0_x_ins0", ctypes.c_uint64, 4),
        ("opd0_y_ins0", ctypes.c_uint64, 4),
        ("opd1_x_ins0", ctypes.c_uint64, 4),
        ("opd1_y_ins0", ctypes.c_uint64, 4),
        ("opd0_up_pad", ctypes.c_uint64, 4),
        ("opd0_dn_pad", ctypes.c_uint64, 4),
        ("opd0_lf_pad", ctypes.c_uint64, 4),
        ("opd0_rt_pad", ctypes.c_uint64, 4),
        ("res_op_x_str", ctypes.c_uint64, 4),
        ("res_op_y_str", ctypes.c_uint64, 4),
        ("tsk_lane_num", ctypes.c_uint64, 64),
        ("res0_n", ctypes.c_uint64, 16),
        ("res0_c", ctypes.c_uint64, 12),
        ("res0_h", ctypes.c_uint64, 16),
        ("res0_w", ctypes.c_uint64, 16),
        ("opd0_n", ctypes.c_uint64, 16),
        ("opd0_c", ctypes.c_uint64, 12),
        ("opd0_h", ctypes.c_uint64, 16),
        ("opd0_w", ctypes.c_uint64, 16),
        ("opd1_n", ctypes.c_uint64, 12),
        ("opd1_c", ctypes.c_uint64, 12),
        ("opd1_h", ctypes.c_uint64, 16),
        ("opd1_w", ctypes.c_uint64, 16),
        ("res0_h_shift", ctypes.c_uint64, 4),
        ("res0_w_shift", ctypes.c_uint64, 4),
        ("opd0_h_shift", ctypes.c_uint64, 4),
        ("opd0_w_shift", ctypes.c_uint64, 4),
        ("opd1_h_shift", ctypes.c_uint64, 4),
        ("opd1_w_shift", ctypes.c_uint64, 4),
        ("res0_n_str", ctypes.c_uint64, 19),
        ("res0_c_str", ctypes.c_uint64, 19),
        ("opd0_n_str", ctypes.c_uint64, 19),
        ("opd0_c_str", ctypes.c_uint64, 19),
        ("opd1_n_str", ctypes.c_uint64, 19),
        ("opd1_c_str", ctypes.c_uint64, 19),
        ("opd2_n_str", ctypes.c_uint64, 19),
        ("opd2_c_str", ctypes.c_uint64, 19),
        ("opt_res_add_sign", ctypes.c_uint64, 1),
        ("opd0_neq1", ctypes.c_uint64, 1),
        ("opd1_neq1", ctypes.c_uint64, 1),
        ("opt_opd3_const", ctypes.c_uint64, 1),
        ("rsvd0", ctypes.c_uint64, 22),
        ("res0_addr", ctypes.c_uint64, 32),
        ("opd0_addr", ctypes.c_uint64, 32),
        ("opd1_addr", ctypes.c_uint64, 32),
        ("opd2_addr", ctypes.c_uint64, 32),
        ("res0_h_str", ctypes.c_uint64, 32),
        ("res0_w_str", ctypes.c_uint64, 32),
        ("opd0_h_str", ctypes.c_uint64, 32),
        ("opd0_w_str", ctypes.c_uint64, 32),
        ("opd1_h_str", ctypes.c_uint64, 32),
        ("opd1_w_str", ctypes.c_uint64, 32),
        ("opd2_h_str", ctypes.c_uint64, 32),
        ("opd2_w_str", ctypes.c_uint64, 32),
        ("res1_addr", ctypes.c_uint64, 32),
        ("opd3_addr", ctypes.c_uint64, 32),
    ]

    cmd_en: int
    cmd_end: int
    cmd_id_en: int
    cmd_id_tpu: int
    cmd_id_gdma: int
    cmd_keep: int
    cmd_intr_en: int
    tsk_typ: int
    tsk_eu_typ: int
    tsk_opd_num: int
    opt_right_shift: int
    opt_left_shift: int
    opt_shift_typ: int
    opt_res_add: int
    opt_relu: int
    opt_left_tran: int
    opt_winograd: int
    opt_kernel_rotate: int
    opt_opd0_sign: int
    opt_opd1_sign: int
    opt_opd2_sign: int
    opt_res0_prec: int
    opt_opd0_prec: int
    opt_opd1_prec: int
    opt_opd2_prec: int
    opt_opd0_const: int
    opt_opd1_const: int
    opt_opd2_const: int
    short_res0_str: int
    short_opd0_str: int
    short_opd1_str: int
    short_opd2_str: int
    opd0_x_ins0: int
    opd0_y_ins0: int
    opd1_x_ins0: int
    opd1_y_ins0: int
    opd0_up_pad: int
    opd0_dn_pad: int
    opd0_lf_pad: int
    opd0_rt_pad: int
    res_op_x_str: int
    res_op_y_str: int
    tsk_lane_num: int
    res0_n: int
    res0_c: int
    res0_h: int
    res0_w: int
    opd0_n: int
    opd0_c: int
    opd0_h: int
    opd0_w: int
    opd1_n: int
    opd1_c: int
    opd1_h: int
    opd1_w: int
    res0_h_shift: int
    res0_w_shift: int
    opd0_h_shift: int
    opd0_w_shift: int
    opd1_h_shift: int
    opd1_w_shift: int
    res0_n_str: int
    res0_c_str: int
    opd0_n_str: int
    opd0_c_str: int
    opd1_n_str: int
    opd1_c_str: int
    opd2_n_str: int
    opd2_c_str: int
    opt_res_add_sign: int
    opd0_neq1: int
    opd1_neq1: int
    opt_opd3_const: int
    rsvd0: int
    res0_addr: int
    opd0_addr: int
    opd1_addr: int
    opd2_addr: int
    res0_h_str: int
    res0_w_str: int
    opd0_h_str: int
    opd0_w_str: int
    opd1_h_str: int
    opd1_w_str: int
    opd2_h_str: int
    opd2_w_str: int
    res1_addr: int
    opd3_addr: int

    core_id = 0
    length = 1024

    @property
    def cmd_id(self):
        return self.cmd_id_tpu

    @property
    def cmd_id_dep(self):
        return self.cmd_id_gdma


class conv_op_reg(tiu_cmd_reg):
    OP_NAME = "conv_op"


class pord_op_reg(tiu_cmd_reg):
    OP_NAME = "pord_op"


class mm_op_reg(tiu_cmd_reg):
    OP_NAME = "mm_op"


class ar_op_reg(tiu_cmd_reg):
    OP_NAME = "ar_op"


class mm2_op_reg(tiu_cmd_reg):
    OP_NAME = "mm2_op"


class cc_op_reg(tiu_cmd_reg):
    OP_NAME = "cc_op"


class lut_op_reg(tiu_cmd_reg):
    OP_NAME = "lut_op"


class md_sum_op_reg(tiu_cmd_reg):
    OP_NAME = "md_sum_op"


class md_scalar_op_reg(tiu_cmd_reg):
    OP_NAME = "md_scalar_op"


class md_sfu_op_reg(tiu_cmd_reg):
    OP_NAME = "md_sfu_op"


class md_linear_op_reg(tiu_cmd_reg):
    OP_NAME = "md_linear_op"


class lma_op_reg(tiu_cmd_reg):
    OP_NAME = "lma_op"


class decompress_op_reg(tiu_cmd_reg):
    OP_NAME = "decompress_op"


class md_cmp_op_reg(tiu_cmd_reg):
    OP_NAME = "md_cmp_op"


class vc_op_reg(tiu_cmd_reg):
    OP_NAME = "vc_op"


class dma_tensor_reg(dma_cmd_reg):
    OP_NAME = "dma_tensor"


op_class_dic: Dict[str, Type[atomic_reg]] = {
    "conv_op": conv_op_reg,
    "pord_op": pord_op_reg,
    "mm_op": mm_op_reg,
    "ar_op": ar_op_reg,
    "mm2_op": mm2_op_reg,
    "cc_op": cc_op_reg,
    "lut_op": lut_op_reg,
    "md_sum_op": md_sum_op_reg,
    "md_scalar_op": md_scalar_op_reg,
    "md_sfu_op": md_sfu_op_reg,
    "md_linear_op": md_linear_op_reg,
    "lma_op": lma_op_reg,
    "decompress_op": decompress_op_reg,
    "md_cmp_op": md_cmp_op_reg,
    "vc_op": vc_op_reg,
    "dma_tensor": dma_tensor_reg,
}


_, _, dma_bitwidths = zip(*dma_cmd_reg._fields_)
_, _, tiu_bitwidths = zip(*tiu_cmd_reg._fields_)

tiu_high_bits = list(accumulate(tiu_bitwidths))[:-1]
dma_high_bits = list(accumulate(dma_bitwidths))[:-1]
