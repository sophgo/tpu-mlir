import os

chip = 'bm1690'

#print_ori_fx_graph/dump_fx_graph/skip_tpu_compile/dump_bmodel_input
debug_cmd = os.environ.get("TORCH_TPU_DEBUG_CMD", "disable_dot,const_name")

only_compile_graph_id = int(os.environ.get("TORCH_TPU_ONLY_COMPILE_GRAPH_ID", -1))

num_core = int(os.environ.get("TORCH_TPU_CORE_NUM", 1))


def get_num_core(chip_type):
    if chip_type == 'bm1690':
        num_core = 8
    if chip_type == 'bm1688':
        num_core = 2
    if chip_type == 'bm1684x':
        num_core = 1
    return num_core


compile_opt = int(os.environ.get("TORCH_TPU_MLIR_COMPILE_OPT", 2))

#fp_mode
#test_input
cmp = False

unit_test = False

run_on_cmodel = True


def print_config_info():
    print('config_info:')
    print('  chip:', chip)
    print('  debug_cmd:', debug_cmd)
    print('  only_compile_graph_id:', only_compile_graph_id)
    print('  num_core:', num_core)
    print('  compile_opt:', compile_opt)
    print('  cmp:', cmp)
    print('  unit_test:', unit_test)
    print('  run_on_cmodel:', run_on_cmodel)
