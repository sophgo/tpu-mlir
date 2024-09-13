# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .MLIRImporter import MLIRImporter, Platform
from .TorchConverter import TorchConverter
from .TorchHelper import *
from mlir.ir import *
from tools.tool_maskrcnn import MaskRCNN_Tester_Basic
from utils.misc import str2list
import mlir.dialects.top as top
import numpy as np
import torchvision
import logging
import copy
import os
import re
import yaml
import ast
logger = logging.getLogger("root")




# parser.add_argument("--maskrcnn_io_map", type=dict_type, default=dict(),
#                         help="io map, -1:whole model's inputs, -2:whole model's outputs, 0,1,2...:id of maskrcnn sections\
#                         like:{(0,0):(-1,0),(1,0):(0,0),(-2,0):(1,0)}, which means \
#                         section[0]'s input[0] comes from input[0] of the whole model, \
#                         section[1]'s input[0] comes from section[0]'s output[0], \
#                         output[0] of the whole model comes from section[1]'s output[0]")
#     parser.add_argument("--maskrcnn_output_num", type=int, default=1, help='maskrcnn output num')
#     parser.add_argument("--maskrcnn_structure", type=str2list, default=list(),
#                         help='1 means pt model, 0 means ppl op, like:1,0,1\
#                         which means 1st launch a pt model, 2nd launch a ppl op, 3rd launch a pt model')
#     parser.add_argument("--maskrcnn_ppl_op", type=str2list, default=list(),
#                         help='lists of PPLOp names, ex:ppl_rpn,ppl_roi_align')
#     parser.add_argument("--numPPLOp_InWithoutWeight_MaskRCNN", type=str2list, default=list(),
#                         help='lists of input nums of each PPLOp, like:[2,3]')

class MaskRCNNHelper(MaskRCNN_Tester_Basic):

    TOP_INPUT  = -1
    TOP_OUTPUT = -2

    IO_PATTERN_STRING = 0
    IO_PATTERN_INT    = 1

    Symbol_PPLSubBlock = 0
    Symbol_TorchSubBlock =1

    SubTorchBlock   = "SubTorchBlock"
    Prefix_TOPInput = ""#"TOPINPUT_"

    name_Utest_Pt = "Model_Fake_Utest.pt"

    def __init__(self, debug_flag):
        self.dict_content_yaml = None
        MaskRCNN_Tester_Basic.__init__(self, debug_flag)
        assert self.Prefix_TOPInput == "", "[Error] Prefix_TOPInput must be empty for model_runner"

    def get_Input_prefix(self):
        return self.Prefix_TOPInput

    def get_ppl_name(self, idx: int) -> str:
        return "PPL_{}_".format(idx)

    def get_ppl_weight_name(self, idx: int) -> str:
        return "PPL_{}_Weight_".format(idx)

    def read_YAML_config(self, path_yaml: str):
        assert isinstance(path_yaml, str), "[MaskRCNN-Error]Only support path_yaml as pure string now!"
        self.path_yaml = path_yaml
        with open(self.path_yaml, 'r') as file:
                self.dict_content_yaml = yaml.unsafe_load(file)
                self.print_debug("-----------Print YAML To Check!---------------------")
                self.print_debug( self.dict_content_yaml)
                self.print_debug("-----------Print MaskRCNN Parameters!---------------------")
                self.print_debug( list(self.dict_content_yaml.keys()))
                self.print_debug("-----------YAML END!---------------------")

    def parser_MASKRCNN_CONFIG_YAML(self):
        def YAML_parser_helper():
             context_parser = ""
             for idx, name in enumerate(self.dict_content_yaml.keys()):
                value = self.dict_content_yaml[name]
                if (not isinstance(value, dict)) or  (not isinstance(value, list)):
                 context_parser += "self.{} = self.dict_content_yaml[\"{}\"]#value:{} \n".format(name,name,value)
             self.print_debug(context_parser)
             assert 0

        # YAML_parser_helper()
        self.CHANNEL_ROI = self.dict_content_yaml["CHANNEL_ROI"]#value:256
        self.CHANNEL_RPN_BBOXES = self.dict_content_yaml["CHANNEL_RPN_BBOXES"]#value:12
        self.CHANNEL_RPN_SCORES = self.dict_content_yaml["CHANNEL_RPN_SCORES"]#value:3
        self.CONF_THRESHOLD_1st = self.dict_content_yaml["CONF_THRESHOLD_1st"]#value:0.0
        self.CONF_THRESHOLD_2nd = self.dict_content_yaml["CONF_THRESHOLD_2nd"]#value:0.0
        self.DELTA2BBOX_1st_MAX_SHAPE_H = self.dict_content_yaml["DELTA2BBOX_1st_MAX_SHAPE_H"]#value:800.0
        self.DELTA2BBOX_1st_MAX_SHAPE_W = self.dict_content_yaml["DELTA2BBOX_1st_MAX_SHAPE_W"]#value:1216.0
        self.DELTA2BBOX_1st_MEAN = self.dict_content_yaml["DELTA2BBOX_1st_MEAN"]#value:0
        self.DELTA2BBOX_1st_STD_0 = self.dict_content_yaml["DELTA2BBOX_1st_STD_0"]#value:1
        self.DELTA2BBOX_1st_STD_1 = self.dict_content_yaml["DELTA2BBOX_1st_STD_1"]#value:1
        self.DELTA2BBOX_2nd_MAX_SHAPE_H = self.dict_content_yaml["DELTA2BBOX_2nd_MAX_SHAPE_H"]#value:800.0
        self.DELTA2BBOX_2nd_MAX_SHAPE_W = self.dict_content_yaml["DELTA2BBOX_2nd_MAX_SHAPE_W"]#value:1216.0
        self.DELTA2BBOX_2nd_MEAN = self.dict_content_yaml["DELTA2BBOX_2nd_MEAN"]#value:0
        self.DELTA2BBOX_2nd_STD_0 = self.dict_content_yaml["DELTA2BBOX_2nd_STD_0"]#value:0.1
        self.DELTA2BBOX_2nd_STD_1 = self.dict_content_yaml["DELTA2BBOX_2nd_STD_1"]#value:0.2
        self.FAKE_DYNAMIC_LENGTH = self.dict_content_yaml["FAKE_DQ_DYNAMIC_LENGTH"]#value:4741
        self.GLOBAL_BATCH_SIZE = self.dict_content_yaml["GLOBAL_BATCH_SIZE"]#value:1
        self.GetBboxB_SCORE_EQ = self.dict_content_yaml["GetBboxB_SCORE_EQ"]#value:0.05
        self.Global_strides = self.dict_content_yaml["Global_strides_DQ"]#value:[(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]
        self.HARDWARE_FACTOR_TOPK = self.dict_content_yaml["HARDWARE_FACTOR_TOPK"]#value:10
        self.H_RPN_DYN_MAX = self.dict_content_yaml["H_RPN_DYN_MAX"]#value:200
        self.MASK_POOLER_FAKE_DYNAMIC_LENGTH = self.dict_content_yaml["MASK_POOLER_FAKE_DYNAMIC_LENGTH"]#value:100
        self.MAX_LENGTH_STATIC_STRECHED = self.dict_content_yaml["MAX_LENGTH_STATIC_STRECHED"]#value:4741
        self.MAX_PER_IMG = self.dict_content_yaml["MAX_PER_IMG"]#value:1000
        self.MAX_PER_IMG_GetBboxB = self.dict_content_yaml["MAX_PER_IMG_GetBboxB"]#value:100
        self.MAX_SCALAR_C = self.dict_content_yaml["MAX_SCALAR_C"]#value:4.1351665567
        self.NMS_MAX_LENGTH_1st = self.dict_content_yaml["NMS_MAX_LENGTH_1st"]#value:4741
        self.NMS_MAX_LENGTH_2nd = self.dict_content_yaml["NMS_MAX_LENGTH_2nd"]#value:80000
        self.NMS_PRE = self.dict_content_yaml["NMS_PRE"]#value:1000
        self.NMS_THRE_1st = self.dict_content_yaml["NMS_THRE_1st"]#value:0.7
        self.NMS_THRE_2nd = self.dict_content_yaml["NMS_THRE_2nd"]#value:0.5
        self.NUM_CLASSES = self.dict_content_yaml["NUM_CLASSES"]#value:1
        self.NUM_CLASSES_GetBboxB = self.dict_content_yaml["NUM_CLASSES_GetBboxB"]#value:80
        self.NUM_INDEXES = self.dict_content_yaml["NUM_INDEXES"]#value:4
        self.NUM_LEVELS = self.dict_content_yaml["NUM_LEVELS"]#value:5
        self.NUM_LEVELS_ROI = self.dict_content_yaml["NUM_LEVELS_ROI"]#value:4
        self.ROI_H = self.dict_content_yaml["ROI_H"]#value:200
        self.ROI_LEN = self.dict_content_yaml["ROI_LEN"]#value:5
        self.ROI_PH_BBOX_POOLER = self.dict_content_yaml["ROI_PH_BBOX_POOLER"]#value:7
        self.ROI_PH_MASK_POOLER = self.dict_content_yaml["ROI_PH_MASK_POOLER"]#value:14
        self.ROI_PW_BBOX_POOLER = self.dict_content_yaml["ROI_PW_BBOX_POOLER"]#value:7
        self.ROI_PW_MASK_POOLER = self.dict_content_yaml["ROI_PW_MASK_POOLER"]#value:14
        self.ROI_SLICE_BBOX_POOLER = self.dict_content_yaml["ROI_SLICE_BBOX_POOLER"]#value:1000
        self.ROI_SLICE_MASK_POOLER = self.dict_content_yaml["ROI_SLICE_MASK_POOLER"]#value:100
        self.ROI_W = self.dict_content_yaml["ROI_W"]#value:304
        self.TOPK_ONNX_NMS_1st = self.dict_content_yaml["TOPK_ONNX_NMS_1st"]#value:250
        self.TOPK_ONNX_NMS_2nd = self.dict_content_yaml["TOPK_ONNX_NMS_2nd"]#value:250
        self.TPU1686_TOPK_FACTOR = self.dict_content_yaml["TPU1686_TOPK_FACTOR"]#value:3
        self.W_RPN_DYN_MAX = self.dict_content_yaml["W_RPN_DYN_MAX"]#value:304

    def trans_str2list(self, input_A):
        if input_A is None: return []
        if isinstance(input_A, list):
            return input_A
        if isinstance(input_A, str):
            return str2list(input_A)
        assert 0, "[Error] either be string or list, but got {}!".format(type(input_A))

    def trans_str2dict(self, input_A):
        if input_A is None: return dict()
        if isinstance(input_A, dict):
            return input_A
        if isinstance(input_A, str):
            return  ast.literal_eval(input_A)
        assert 0, "[Error] either be dict or list, but got {}!".format(type(input_A))

class MaskRCNNConverter(TorchConverter, MaskRCNNHelper):

    def __init__(self,
                 model_name: str,
                 model_def:  str,
                 model_external:        list = [],
                 input_shapes_AllBlock: list = [],
                 input_types:           list = [],
                 output_names:          list = [],
                 preprocess_args:       dict = {},
                 path_yaml:             list = [],
                 debug:                 bool = True):
        '''
        input_shapes_AllBlock: input shapes of complete maskrcnn model
        input_types:  input types of complete maskrcnn model
        output_names: output names of complete maskrcnn model
        io_map:       dict like {(i,j): (m,n)}, defined as {(dst_id,operand_id):(src_id,operand_id)}
                      which means the j-th input of the i-th submodel is the n-th output of the m-th submodel
        maskrcnn_num_output: necessary as last sub-block is PPLOp/TorchPt is never certain
        list_encoded_block:   sequential order of all sub-blocks; 0:PPLOp, 1: TorchPt;    Folded WeightOp is created in corresponding PPLOp.
        list_name_PPLop:      sequential order of the names for every PPLops
        nums_PPLOp_Inputs:    abundant for MaskRCNN, sequential order of numInputs for every PPLops
        '''
        MaskRCNNHelper.__init__(self, debug)
        self.read_YAML_config(path_yaml)
        assert self.params_model_transform_MaskRCNN in  self.dict_content_yaml.keys(),"[MaskRCNN-Error] your config yaml: {} must contain necessary external maskrcnn parameters:{}".format(path_yaml)
        io_map = dict()
        if 'maskrcnn_io_map' in self.dict_content_yaml[self.params_model_transform_MaskRCNN].keys():
            io_map = self.dict_content_yaml[self.params_model_transform_MaskRCNN]['maskrcnn_io_map']
            io_map = self.trans_str2dict(io_map)

        maskrcnn_num_output  = int(self.dict_content_yaml[self.params_model_transform_MaskRCNN]['maskrcnn_num_output'])
        list_encoded_block   = self.dict_content_yaml[self.params_model_transform_MaskRCNN]['list_encoded_block']
        list_name_PPLop      = self.dict_content_yaml[self.params_model_transform_MaskRCNN]['list_name_PPLop']
        nums_PPLOp_Inputs    = self.dict_content_yaml[self.params_model_transform_MaskRCNN]['nums_PPLOp_Inputs']

        list_encoded_block   = self.trans_str2list(list_encoded_block)
        list_name_PPLop      = self.trans_str2list(list_name_PPLop)
        nums_PPLOp_Inputs    = self.trans_str2list(nums_PPLOp_Inputs)

        multi_torch_files = [model_def] + self.trans_str2list(model_external)
        if not model_external and len(list_encoded_block)==1:
            assert not io_map, "[MaskRCNN-Error] io_map should be None if only Utest for PPLOp!"
        list_encoded_block,multi_torch_files,nums_PPLOp_Inputs ,input_shapes_AllBlock,list_name_PPLop,io_map, maskrcnn_num_output,num_InputOperand_BlockTorch_1st =self.Gen_Utest_Torch_pt(list_encoded_block,multi_torch_files,nums_PPLOp_Inputs ,input_shapes_AllBlock,list_name_PPLop,io_map, maskrcnn_num_output)
        if isinstance(multi_torch_files, str):
            multi_torch_files = [multi_torch_files]
        TorchConverter.__init__(self,
                                model_name,
                                multi_torch_files[0],
                                input_shapes_AllBlock[0: num_InputOperand_BlockTorch_1st],
                                input_types,output_names,
                                preprocess_args)
        #[Note]Complexor Representation Compeleteness is ensured by 3 params: {multi_torch_files, list_name_PPLop} + list_encoded_block
        #Structure Info
        self.list_encoded_block = [int(s) for s in list_encoded_block]
        assert int(self.list_encoded_block[0])>=1, "[Inherit-Error]1st sub_block must be Torch!"
        self.num_sub_blocks_without_TOPINOUT = len(self.list_encoded_block)
        assert len(list_name_PPLop) + len(multi_torch_files) == self.num_sub_blocks_without_TOPINOUT,(len(list_name_PPLop) , len(multi_torch_files) , self.num_sub_blocks_without_TOPINOUT)
        assert len(list_name_PPLop)==len(nums_PPLOp_Inputs)

        #Individual Op info
        self.multi_torch_files = multi_torch_files
        self.list_name_PPLop = list_name_PPLop
        self.nums_PPLOp_Inputs = [int(n) for n in nums_PPLOp_Inputs]
        self.list_block_discontinuous_id_OrderbyStructure_IndexbyBlockType = []
        self.list_all_block_str_names_order = []
        self.reorganize_sub_block_info()

        self.model_name = model_name
        self.weight_file = "{}_top_origin_weight.npz".format(model_name)
        self.ppl_factory = {
            "ppl::RPN_get_bboxes": lambda input_list, op_type, id_PPLOp: self.convert_RPN_get_bboxes_op(input_list, op_type, id_PPLOp),
            "ppl::Bbox_Pooler": lambda input_list, op_type, id_PPLOp: self.convert_BBox_Pooler_op(input_list, op_type, id_PPLOp),
            "ppl::get_bboxes_B": lambda input_list, op_type, id_PPLOp: self.convert_get_bboxes_B_op(input_list, op_type, id_PPLOp),
            "ppl::Mask_Pooler": lambda input_list, op_type, id_PPLOp: self.convert_Mask_Pooler_op(input_list, op_type, id_PPLOp)
        }
        self.input_shapes_AllBlock   = input_shapes_AllBlock
        self.maskrcnn_num_output = maskrcnn_num_output
        self.io_map = self.MaskRCNN_IOMAP_Transformer(io_map)
        if self.DEBUG_MASKRCNN:
            self.draw_io_as_matplot(self.io_map)
        self.detect_idle_io_map(self.io_map)

        self.mlir = None
        self.torch_graphs = []
        self.torch_models = []
        self.state_dicts  = []
        self.input_shapes = []
        self.input_types  = []
        self.all_input_names  = []
        self.all_output_names = []

        #[Note]Initialize TOP_INPUT/TOP_OUTPUT Info
        self.num_InputOperand_BlockTorch_1st = num_InputOperand_BlockTorch_1st
        self.num_InputOperand_AllBlock   = len(self.input_shapes_AllBlock)
        self.input_names = []
        for i in range(self.num_InputOperand_BlockTorch_1st):
            self.input_names.append("{}".format(i))

        self.output_names = []
        for i in range(self.maskrcnn_num_output):
            self.output_names.append("{}".format(i))
        self.output_shapes = [[]] * self.maskrcnn_num_output

        self.process_model_info(multi_torch_files,
                              input_types, output_names)
        self.init_MLIRImporter()
        self.unranked_type = self.mlir.get_tensor_type([])
        self.preprocess_args = {}
        self.deal_with_preprocess(preprocess_args)
        self.converted_nodes = list()
        self.const_val = dict()

        # yapf: enable
        self.check_op_types_Superior_MaskRCNN()
        self.parser_MASKRCNN_CONFIG_YAML()

    '''
        1 pplop => Model_Fake_Utest.pt + pplop; then later TorchConverter can be applied
    '''
    def Gen_Utest_Torch_pt(self, maskrcnn_structure: list[int],
                                 model_def: list[str],
                                 numPPLOp_InWithoutWeight_MaskRCNN: list[int],
                                 input_shapes: list,
                                 maskrcnn_ppl_op: list[int],
                                 maskrcnn_io_map: dict,
                                 maskrcnn_output_num: int):

        num_InputOperand_BlockTorch_1st = 1
        is_FirstOp_PPLOp    = int(maskrcnn_structure[0]) == 0
        is_OnlyOneOp        = len(maskrcnn_structure)    == 1
        if isinstance(model_def, str):
            model_def = [model_def]
        if is_FirstOp_PPLOp:
            maskrcnn_structure = ["1"] + maskrcnn_structure
            assert self.name_Utest_Pt != "TOP_IN.pt","[Error] TOP_IN is a proprietary model name, don't use it freely!"
            path_modified_fake_torch = os.path.abspath(os.getcwd()) +"/{}".format(self.name_Utest_Pt)
            if model_def is not None and (not is_OnlyOneOp):
                for idx_model, each_exist_model in enumerate(model_def):
                    if (idx_model > 0):
                        assert self.name_Utest_Pt not in each_exist_model, "[Error]{} will be added, but it's conflict with your existed model_def:{}!".format(each_exist_model)
                    else:
                        assert self.name_Utest_Pt in each_exist_model, "[Error]now {} must be the first name in your model_def!".format(each_exist_model)
                model_def = [path_modified_fake_torch] + model_def[1:]
            elif model_def is not None and is_OnlyOneOp:
                assert  len(model_def) == 1, "[Error] MaskRCNN-Utest Mode Enable, model_def contains only 1 .pt file, but got: {}, which len is {}!".format(model_def,  len(model_def))
                assert  model_def[0] == self.name_Utest_Pt, "[Error] MaskRCNN-Utest Mode Enable, model_def must be {}, as there's only 1 Op in maskrcnn_structure and must be 0==PPLOp".format(self.name_Utest_Pt)
                model_def = [path_modified_fake_torch]
            else:
                model_def = [path_modified_fake_torch]
            num_InputOperands_block_1st = int(numPPLOp_InWithoutWeight_MaskRCNN[0])
            assert num_InputOperands_block_1st <= len(input_shapes),"[Error][Case:ppl-op first]: nums of Top_Input {} must be compatible with nums of input_shapes {}".format(int(numPPLOp_InWithoutWeight_MaskRCNN[0]), len(input_shapes))

            class Modified_Fake_Model(torch.nn.Module):
                def __init__(self):
                    super(Modified_Fake_Model, self).__init__()
                def forward(self,*x):
                    list1 = []
                    for i in range(len(x)):
                        list1 +=[x[i]*1]
                    list1 = tuple(list1)
                    return list1

            model_fake = Modified_Fake_Model()
            example_forward_input =tuple([torch.rand(input_shapes[i]) for i in range(num_InputOperands_block_1st)])
            traced_model_fake = torch.jit.trace(model_fake, example_forward_input)
            torch.jit.save(traced_model_fake, path_modified_fake_torch)
            traced_model_fake=torch.jit.load(path_modified_fake_torch, map_location=torch.device('cpu'))
            simulated_output = traced_model_fake(*example_forward_input)
            for i in range(num_InputOperands_block_1st):
                assert torch.abs(torch.sum(example_forward_input[i]-simulated_output[i]))<1e-6

            '''
             revise io_map for:1 pplop => Model_Fake_Utest.pt + pplop;
            '''
            def gen_attched_fake_io_map():
                dst = maskrcnn_ppl_op[0]
                src = self.name_Utest_Pt.split(".pt")[0]
                fake_io_map = dict()
                #[Note] Force inputs of Utest aligned with 1st PPLop
                for i in range(num_InputOperands_block_1st):
                    fake_io_map[(dst,i)]=(src,i)
                #[Case-0]: Top_In2FirstTorch
                dst = self.name_Utest_Pt.split(".pt")[0]
                offset_TOP = 0

                for dst_key in  maskrcnn_io_map.keys():
                    src_per = maskrcnn_io_map[dst_key]
                    if not isinstance(dst_key[0], int): break
                    if dst_key[0]==0 and (src_per[0]== -1 or src_per[0]== "TOP_IN"):
                        offset_TOP = src_per[1] if offset_TOP==0 else min(offset_TOP,src_per[1])

                for i in range(num_InputOperands_block_1st):
                    fake_io_map[(dst,i)]=("TOP_IN",i + offset_TOP)

                #[Case-1]:torch/ppl2TOP_OUT
                src = maskrcnn_ppl_op[-1] if int(maskrcnn_structure[-1])==0 else model_def[-1]
                for i in range(int(maskrcnn_output_num)):
                    fake_io_map[("TOP_OUT",i)]=(src,i)
                return fake_io_map

            list_id_dst_block = [i[0] for i in list(maskrcnn_io_map.keys())]
            assert  np.sum([isinstance(j,int) for j in list_id_dst_block])==len(list_id_dst_block),"[Utest-PPL] only allow (str:int)  for (dst/src:id)] now!"

            #[Note] Only update Utest.pt connections, regraphize other connections later
            maskrcnn_io_map.update(gen_attched_fake_io_map())

            #Model_Fake_Utest is same as pplOp
            num_InputOperand_BlockTorch_1st = int(numPPLOp_InWithoutWeight_MaskRCNN[0])

        return maskrcnn_structure, model_def, numPPLOp_InWithoutWeight_MaskRCNN,input_shapes,maskrcnn_ppl_op,maskrcnn_io_map, maskrcnn_output_num, num_InputOperand_BlockTorch_1st

    '''
        id_PPLOp
        idx_TorchBlock
        =>
        list_all_block_str_names_order #names in order

        list_block_discontinuous_id_OrderbyStructure_IndexbyBlockType #concat id in order but not reindex them
        [0,1,2],[0,1,2] => [0,1,0,1,2,2] not [10,1,2,3,4,5]
    '''
    def reorganize_sub_block_info(self):
        idx_TorchBlock = 0
        id_PPLOp   = 0
        for _, block_id in enumerate(self.list_encoded_block):
            if   block_id == self.Symbol_PPLSubBlock:
                self.list_all_block_str_names_order +=[self.list_name_PPLop[id_PPLOp]]
                self.list_block_discontinuous_id_OrderbyStructure_IndexbyBlockType.append(id_PPLOp)
                id_PPLOp += 1
            elif block_id == self.Symbol_TorchSubBlock:
                self.list_all_block_str_names_order +=[self.multi_torch_files[idx_TorchBlock]]
                self.list_block_discontinuous_id_OrderbyStructure_IndexbyBlockType.append(idx_TorchBlock)
                idx_TorchBlock  += 1
            else: assert 0,(block_id, self.Symbol_TorchSubBlock,self.Symbol_PPLSubBlock)

    '''
        preprocess_args [TODO]
    '''
    def deal_with_preprocess(self, preprocess_args):
        if 'preprocess_list' in preprocess_args:
            if preprocess_args['preprocess_list'] is not None:
                for input_index in preprocess_args['preprocess_list']:
                    assert (0 < input_index <= self.num_InputOperand_BlockTorch_1st
                            and "Please check --preprocess_list is right input")
            else:
                preprocess_args['preprocess_list'] = [
                    i + 1 for i in range(self.num_InputOperand_BlockTorch_1st)]
        if 'channel_format' in preprocess_args:
            if preprocess_args['channel_format'] != "none":
                self.preprocess_args = preprocess_args
        self.preprocess_args = []
        for i in range(self.num_InputOperand_BlockTorch_1st):
            self.preprocess_args += [self.preprocess_args]
        for i in range(self.num_InputOperand_AllBlock - self.num_InputOperand_BlockTorch_1st):
                self.preprocess_args +=[{None}]
        assert len(self.preprocess_args)==len(self.input_names),(len(self.preprocess_args),"vs", len(self.input_names))

    '''
        check if io_map illegal
    '''
    def detect_idle_io_map(self, init_io_map):
        #[Check] if all TOP_IN-id is used,  otherwise  redundant dynamic ir will be parsed in irgen.
        init_top_idx = dict()
        for dst_per in init_io_map.keys():
            src_per =init_io_map[dst_per]
            if src_per[0]==self.TOP_INPUT or src_per[0]=="TOP_IN":
                if src_per[1] not in list(init_top_idx.keys()):
                    init_top_idx[src_per[1]] = False
        # for dst_per in init_io_map.keys():
        #     if dst_per[1] in list(init_top_idx.keys()):
        #         init_top_idx[dst_per[1]] = True
        # list_idle_list = []
        # for  idx, temp in enumerate(init_top_idx.keys()):
        #    if init_top_idx[temp]==False:
        #        list_idle_list+=[temp]
        # assert 0,init_io_map
        if len(init_top_idx)!=len(self.input_shapes_AllBlock):
               assert 0,"[Error] Some Inputs-([Warning]is the TOP_INPUT name!) are  not used, which will lead to dynamic_ir parser overflow error! Check revised_io_map_{}.svg!".format(self.model_name)
        '''
            [Check] self.maskrcnn_num_output
            [Check] TOP_OUT can't exist in src
            [Check] TOP_IN can't exist in dst
        '''
        for _, dst_per in enumerate(init_io_map.keys()):
            src_per =  init_io_map[dst_per]
            assert src_per[0] !=self.TOP_OUTPUT ," src can't be TOP_OUT({})".format(self.TOP_OUTPUT)
            assert dst_per[0] !=self.TOP_INPUT ," src can't be TOP_IN({})".format(self.TOP_INPUT)
        #[Check] ensure given maskrcnn_num_output is same as io_map
        sum_simulated_num_output = 0
        for _, dst_per in enumerate(init_io_map.keys()):
            sum_simulated_num_output +=int(dst_per[0]==self.TOP_OUTPUT)
        assert  self.maskrcnn_num_output==sum_simulated_num_output,"[Error] io_map requires {}-outputs but {} is given from model-transform!".format(sum_simulated_num_output, self.maskrcnn_num_output)

        #[Check] if dst is  illegally reused
        dict_replicated_dst_per = {}
        for _, dst_per in enumerate(init_io_map.keys()):
            block_id = dst_per[0]
            operand_id = dst_per[1]
            if block_id in dict_replicated_dst_per.keys():
                if operand_id in dict_replicated_dst_per[block_id]:
                    name_sub_block = self.list_all_block_str_names_order[block_id]  if block_id!=-2 else "TOP_OUT"
                    assert 0, "{} Subblock-{}-operand_id_{} has illegal multi-incomes, it forced to be single!".format(name_sub_block, block_id,operand_id )
            else:
               dict_replicated_dst_per[block_id]= [operand_id]
        '''
            [Check]If nums_PPLOp_Inputs from io_map-dst is same as self.nums_PPLOp_Inputs
            [Note] adapt dst, because src is reused
        '''
        # assert 0,self.list_encoded_block
        assert self.list_encoded_block[0]==self.Symbol_TorchSubBlock, "[Error]first input sub-block must be torch.pt!"
        dict_static_iomap_num_input = {}
        for _, dst_per in enumerate(init_io_map.keys()):
            block_id = dst_per[0]
            operand_id = dst_per[1]
            if block_id not in dict_static_iomap_num_input.keys():
                dict_static_iomap_num_input[block_id] = 1
            else:
                dict_static_iomap_num_input[block_id] +=1

        ppl_op_count_from_iomap = 0
        for _, block_id_withUtest in enumerate(dict_static_iomap_num_input.keys()):
         if block_id_withUtest!=self.TOP_OUTPUT:
            if self.list_encoded_block[block_id_withUtest]==self.Symbol_PPLSubBlock:
                num_input_PPLOp_from_user = self.nums_PPLOp_Inputs[ppl_op_count_from_iomap]
                num_input_PPLOp_atDst_fromIomap = dict_static_iomap_num_input[block_id_withUtest]
                if num_input_PPLOp_from_user != num_input_PPLOp_atDst_fromIomap:
                    PPLOP_name = self.list_name_PPLop[ppl_op_count_from_iomap]
                    assert 0, "[Error] name-{} nums of input from_iomap is {},but user gives {}".format(PPLOP_name, num_input_PPLOp_atDst_fromIomap,num_input_PPLOp_from_user)
                ppl_op_count_from_iomap += 1
        assert ppl_op_count_from_iomap==len(self.list_name_PPLop), "[Error] nums of pplop in io_map is {}, but user gives {}!".format(ppl_op_count_from_iomap,len(self.list_name_PPLop) )

    '''
        Transformer io_map to pure digital representaiton
    '''
    def  MaskRCNN_IOMAP_Transformer(self,init_io_map):
        dst_keys = list(init_io_map.keys())
        io_style = self.IO_PATTERN_INT
        flag_Model_Fake_Utest = False
        for per_key in  dst_keys:
            assert isinstance(per_key, tuple),(per_key)
            if  isinstance(per_key[0],str):
                io_style = self.IO_PATTERN_STRING
            if "Model_Fake_Utest"==per_key[0]:
                io_style = self.IO_PATTERN_STRING
                flag_Model_Fake_Utest= True

        #deal with "Model_Fake_Utest": int2string
        def search_string2int(target):
            if isinstance(target,str):
                return target
            if target==self.TOP_INPUT or  target==self.TOP_OUTPUT or target=='TOP_OUT' or target=='TOP_IN':
                return target
            assert isinstance(target, int)
            #[Note]] "0" will not be conflict with origin 1st PPLOP or Utset
            real_target  = target + int(target>=0)
            return self.list_all_block_str_names_order[real_target]

        io_map = dict()
        if(flag_Model_Fake_Utest):
            for dst_per in  dst_keys:
                src_per = init_io_map[dst_per]
                dst_set = (search_string2int(dst_per[0]),dst_per[1])
                src_set = (search_string2int(src_per[0]),src_per[1])
                io_map[dst_set] = src_set
        else:   io_map = init_io_map
        self.print_debug("[init_io_map]",init_io_map)
        self.print_debug("--------------------------")
        #TOP_IN=>-1
        #TOP_OUT=>-2
        new_io_map = dict()
        dst_keys = list(io_map.keys())
        for block_dst in dst_keys:
            block_src = io_map[block_dst]
            if io_style == self.IO_PATTERN_STRING:
                block_dst_0 = 'TOP_OUT' if block_dst[0]==self.TOP_OUTPUT else block_dst[0]
                block_src_0 = 'TOP_IN'  if block_src[0]==self.TOP_INPUT else block_src[0]
            elif io_style == self.IO_PATTERN_INT:
                block_dst_0 = self.TOP_OUTPUT if block_dst[0]=='TOP_OUT' else block_dst[0]
                block_src_0 = self.TOP_INPUT  if block_src[0]=='TOP_IN' else block_src[0]

            new_block_dst = (block_dst_0,block_dst[1])
            new_block_src = (block_src_0,block_src[1])
            self.print_debug("error",new_block_dst,new_block_src)
            if new_block_dst not in new_io_map.keys():
                new_io_map[new_block_dst] =new_block_src
            else:
                exist_set = new_io_map[new_block_dst]
                conflict_set = new_block_src
                self.print_debug("[Warning] dst {} is conflict, exist/conflict: {}-----{} , I choose exist user_given io_map {} as first prioity".format(new_block_dst,exist_set, conflict_set, exist_set))
        self.print_debug("[new_io_map]",new_io_map)
        #[Pure-Check] type-check
        dst_keys = list(new_io_map.keys())
        for block_dst in dst_keys:
            block_src = new_io_map[block_dst]
            assert isinstance(block_dst,  tuple),(block_dst)
            assert isinstance(block_dst[1], int),(block_dst)
            assert isinstance(block_src,  tuple),(block_src)
            assert isinstance(block_src[1], int),(block_src)
            if io_style == self.IO_PATTERN_STRING:
                 assert isinstance(block_dst[0], str),(block_dst)
                 assert isinstance(block_src[0], str),(block_src)
            elif io_style == self.IO_PATTERN_INT:
                 assert isinstance(block_dst[0], int),(block_dst)
                 assert isinstance(block_src[0], int),(block_src)
        #[Pure-Check]name-check
        #0)ppl_factory is verfied earlier
        for ppl_op_type in self.list_name_PPLop:
                    if ppl_op_type not in self.ppl_factory:
                        raise RuntimeError("PPL op not support:{}".format(ppl_op_type))
        #1)STRING_MODEL check if every name from io_map is in model_def;
        if (io_style == self.IO_PATTERN_STRING):
            def name_checker(name_dst,name_src,source_list,hist_flag_torch_dst,hist_flag_torch_src):
                flag_dst=0
                flag_src=0
                for name_per in source_list:
                    flag_dst = len(re.findall(name_dst,name_per))>0
                    if(flag_dst): break
                for name_per in source_list:
                    flag_src = len(re.findall(name_src,name_per))>0
                    if(flag_src): break
                flag_dst = 1 if name_dst=="TOP_OUT" else flag_dst
                flag_src = 1 if name_src=="TOP_IN"  else flag_src
                flag_dst = 1 if hist_flag_torch_dst else flag_dst
                flag_src = 1 if hist_flag_torch_src else flag_src
                return flag_dst,flag_src
            for set_block in dst_keys:
                name_dst = set_block[0]
                name_src = new_io_map[set_block][0]
                hist_flag_torch_dst,hist_flag_torch_src = name_checker(name_dst,name_src, self.multi_torch_files,0,0)
                flag_dst,flag_src = name_checker(name_dst,name_src, self.list_name_PPLop,hist_flag_torch_dst,hist_flag_torch_src)
                assert flag_dst,(name_dst, "not in model_def, correct this error!")
                assert flag_src,(name_src, "not in model_def, correct this error!")
        #convert string to idx
        if (io_style == self.IO_PATTERN_STRING):
            new_io_map_digital = dict()
            def match_block_str_name_to_idx(block):
                if block=="TOP_IN" or block==self.TOP_INPUT: return self.TOP_INPUT
                if block=="TOP_OUT"or block==self.TOP_OUTPUT: return self.TOP_OUTPUT
                self.print_debug("block:",block)
                for idx_record,name_digital in enumerate(self.list_all_block_str_names_order):
                    if len(re.findall(block, name_digital))>0:
                        return idx_record
                assert 0,"[Error] such {} not desired".format(block)
            for block_dst in dst_keys:
                block_src = new_io_map[block_dst]
                block_dst_0 = match_block_str_name_to_idx(block_dst[0])
                block_src_0 = match_block_str_name_to_idx(block_src[0])
                new_block_dst = (block_dst_0,block_dst[1])
                new_block_src = (block_src_0,block_src[1])
                new_io_map_digital[new_block_dst] =new_block_src
            new_io_map =new_io_map_digital

        return new_io_map

    '''
        draw tramsformed digital io_map to png
    '''
    def draw_io_as_matplot(self,io_map):
        self.print_debug("[Draw io_map]:",io_map)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15,20))
        dst_set = io_map.keys()
        width_from_dst = dict()
        width_from_src = dict()
        dict_id_input_dst = dict()
        dict_id_input_src = dict()
        def update_width(width_dict,dict_id, per):
            if  per[0] not in list(width_dict.keys()):
                width_dict[per[0]] = 1
                dict_id[per[0]] = [per[1]]
            else:
                if per[1] not in dict_id[per[0]]:
                    width_dict[per[0]] +=1
                    dict_id[per[0]] += [per[1]]

        for dst_per in dst_set:
            src_per = io_map[dst_per]
            update_width(width_from_dst,dict_id_input_dst, dst_per)
            assert src_per[0]!=self.TOP_OUTPUT
            update_width(width_from_src, dict_id_input_src, src_per)
        width_sub_block = dict()
        width_sub_block.update(width_from_src)
        width_sub_block.update(width_from_dst)
        for i in width_sub_block.keys():
            if i in width_from_src.keys() and  i in width_from_dst.keys():
                width_sub_block[i] = max(width_from_src[i], width_from_dst[i])

        block_start_dict = dict()
        block_start_dict[-1] = 0
        pointer_io = dict()
        pointer_io.update(io_map)
        for dst_per in dst_set:
            block_start_dict[dst_per[0]] = 65535

        design_id_series = [ i-1 for i in range(len(width_sub_block.keys()))]
        design_id_series[-1] = -2
        # assert 0,pointer_io
        for idx in range(len(width_sub_block.keys())):
            real_idx = design_id_series[idx]
            # witdh_per =  width_sub_block[real_idx]
            # self.print_debug("------------")
            for dst_per in pointer_io:
                src_per =  pointer_io[dst_per]
                if dst_per[0]==real_idx:
                    block_start_dict[real_idx]  = min(block_start_dict[real_idx], src_per[1])
                    # self.print_debug(block_start_dict[real_idx], dst_per,src_per)
            # if idx==4: assert 0,block_start_dict
            fresh_pointer_io= dict()
            for update_dst in pointer_io:
                update_src =  pointer_io[update_dst]
                if update_dst[0] == real_idx or update_src[0] ==  real_idx:
                        #input is unique, so dst first
                        offset_dst = block_start_dict[real_idx] if  update_dst[0] ==  real_idx else 0
                        offset_src = block_start_dict[real_idx] if  update_src[0] ==  real_idx else 0
                        new_dst = ( update_dst[0],  update_dst[1]+ offset_dst)
                        new_src = ( update_src[0],  update_src[1]+ offset_src)
                        fresh_pointer_io[new_dst]=new_src
                else:
                        fresh_pointer_io[update_dst] = update_src
            pointer_io = fresh_pointer_io
        #     if idx==3:
        #          assert 0,(block_start_dict,pointer_io)
        # assert 0,block_start_dict
        p_dst = []
        p_src = []
        max_dst_y =0
        min_src_y =0
        dst_set = pointer_io.keys()
        for _, dst_per in enumerate(dst_set):
            src_per =  pointer_io[dst_per]
            line_height = 4
            block_height = 8

            def gen_absolute_offset_y(block_idx_dst, block_idx_src):
                 base = self.TOP_INPUT
                 if block_idx_src==self.TOP_INPUT:
                    y_src =  base
                 else:
                    y_src = base + line_height + block_height +(block_idx_src)*(line_height+block_height)
                 if block_idx_dst==self.TOP_OUTPUT:
                    y_dst = base + line_height + (len(self.multi_torch_files)+len(self.nums_PPLOp_Inputs))*(line_height+block_height)
                 else:
                    y_dst = base + line_height + block_idx_dst*(line_height+block_height)
                 return y_dst, y_src
            y_dst,y_src = gen_absolute_offset_y(dst_per[0],src_per[0])
            x_dst = dst_per[1]
            x_src = src_per[1]
            #draw sub_block
            if dst_per[0]!=self.TOP_OUTPUT:

                block_width = width_sub_block[dst_per[0]]  #if not self.list_encoded_block[dst_per[0]] else width_sub_block[dst_per[0]]
                # assert block_width==width_sub_block[dst_per[0]]
                pointer_x_00 = block_start_dict[dst_per[0]] - 1
                pointer_x_01 = block_start_dict[dst_per[0]] - 1
                pointer_x_10 = block_start_dict[dst_per[0]] + block_width + 1
                pointer_x_11 = block_start_dict[dst_per[0]] + block_width + 1
                pointer_y_00 = y_dst
                pointer_y_01 = pointer_y_00 + block_height
                pointer_y_10 = pointer_y_00
                pointer_y_11 = pointer_y_10 + block_height
                plt.plot([pointer_x_00,pointer_x_01],[pointer_y_00,pointer_y_01],color="black")
                plt.plot([pointer_x_00,pointer_x_10],[pointer_y_00,pointer_y_10],color="black")
                plt.plot([pointer_x_10,pointer_x_11],[pointer_y_10,pointer_y_11],color="black")
                plt.plot([pointer_x_01,pointer_x_11],[pointer_y_01,pointer_y_11],color="black")
                #PPLOp
                if not self.list_encoded_block[dst_per[0]]:
                    idx_op = int(dst_per[0]-np.sum(np.array(self.list_encoded_block[0:dst_per[0]])))
                else:
                    idx_op =  int(np.sum(np.array(self.list_encoded_block[0:dst_per[0]])))
                tag = self.multi_torch_files[idx_op].split("/")[-1] if  self.list_encoded_block[dst_per[0]] else self.list_name_PPLop[idx_op].split("/")[-1]
                plt.text(x=pointer_x_00+block_width/2, y=pointer_y_00+block_height/2, s="[{}]-{}".format(dst_per[0],tag))

            p_dst  =[x_dst, y_dst]
            p_src  =[x_src, y_src]
            self.print_debug("[Connect-Info]",p_src,"to",p_dst, "[origin]:",  [src_per[1],src_per[0]],"to",[dst_per[1],dst_per[0]])
            point_x = [p_dst[0] ,p_src[0]]
            point_y = [p_dst[1] ,p_src[1]]
            max_dst_y = max(max_dst_y,p_dst[1])
            min_src_y = min(min_src_y,p_src[1])
            def hanging_line(point1, point2):
                import numpy as np
                a = (point2[1] - point1[1])/(np.cosh(point2[0]) - np.cosh(point1[0]))
                b = point1[1] - a*np.cosh(point1[0])
                x = np.linspace(point1[0], point2[0], 100)
                y = a*np.cosh(x) + b
                return (x,y)
            from scipy.optimize import curve_fit
            plt.scatter(point_x,point_y)
            (x,y) = hanging_line(p_dst, p_src)
            plt.plot(x,y)
            if np.sum(x)==0 or np.sum(y)==0.0 or np.isnan(x[0]) or np.isnan(y[0]):
                if point_x[0]==point_x[1]:
                    point_x = [point_x[0],point_x[0]- 0.01329*(point_y[1]-point_y[0])/2,point_x[1]]
                    point_y = [point_y[0],(point_y[0]+point_y[1])/2,point_y[1]]
                plt.plot(point_x,point_y,color="black")
            plt.text(x=p_dst[0]+ 0.1, y=p_dst[1]+ 0.5, s="{}".format(str(int(dst_per[1]-block_start_dict[dst_per[0]]))))
            plt.text(x=p_src[0]+ 0.1, y=p_src[1]+ 0.5, s="{}".format(str(int(src_per[1]-block_start_dict[src_per[0]]))))

        plt.axhline(y=max_dst_y,color='r',ls="--", label = "TOP_OUT")
        plt.axhline(y=min_src_y,color='r',ls="--",  label = "TOP_IN")
        path_io_plt = os.path.abspath(os.getcwd()) +"/revised_io_map_{}.svg".format(self.model_name)
        plt.savefig(path_io_plt,format="svg")

    '''
        check if every op type is illegal
        must be overridden in __init__()
    '''
    #load_multi_torch_model/check_op_types/get_all_op_types
    #must be overridden as these 3 in init()
    def check_op_types_Superior_MaskRCNN(self):
        op_types = self.get_all_op_types_Superior_MaskRCNN()
        known_ops = list(self.op_factory.keys())
        unknown_ops = []
        for op_type in op_types:
            logger.info(op_type)
            if op_type not in known_ops:
                if not (op_type.endswith("_") and op_type[:-1] in known_ops):
                    unknown_ops.append(op_type)
        if len(unknown_ops) != 0:
            raise RuntimeError(
                "The following operators are not implemented: {}".format(unknown_ops))

    '''
        collect all names in torch pts to self.nodes
        must be overridden in __init__()
    '''
    def get_all_op_types_Superior_MaskRCNN(self):
        """Return all operator names in the input graph"""
        self.nodes = []
        for graph in self.torch_graphs:
            self.nodes.extend(list(graph.nodes()))
            prim_blocks = ["prim::If", "prim::Loop"]
            for prim in prim_blocks:
                prim_nodes = graph.findAllNodes(prim, recurse=True)
                for prim_node in prim_nodes:
                    for block in prim_node.blocks():
                        self.nodes += block.nodes()
        return set(node.kind() for node in self.nodes)

    '''
        collect all names in torch pts to self.nodes
        must be overridden in __init__()
    '''
    def process_model_info(
            self,
            multi_torch_files: list[str],
            input_types: list,
            output_names: list):
        assert self.num_InputOperand_BlockTorch_1st<=self.num_InputOperand_AllBlock
        if self.num_InputOperand_BlockTorch_1st <self.num_InputOperand_AllBlock:
            self.print_debug("[Warning] Input Shapes exist other formats for later PPLOp: {}".format(self.input_shapes_AllBlock[self.num_InputOperand_BlockTorch_1st:]))
        self.load_multi_torch_model_Superior_MaskRCNN(multi_torch_files, self.input_shapes_AllBlock[:self.num_InputOperand_BlockTorch_1st],
                              input_types, output_names)
        #[Note]continue to process PPL input-shape-names
        for i in range(self.num_InputOperand_BlockTorch_1st, self.num_InputOperand_AllBlock ):
            self.input_names.append("{}".format(i))

        for idx, name in enumerate(self.input_names):
            if idx>=self.num_InputOperand_BlockTorch_1st:
                self.input_shapes +=[self.input_shapes_AllBlock[idx]]
                self.addShape(name, self.input_shapes_AllBlock[idx])

        assert len(self.input_shapes)==len(self.input_shapes_AllBlock)

    '''
        collect torch ir info
    '''
    def load_multi_torch_model_Superior_MaskRCNN(
            self,
            multi_torch_files: list[str],
            input_shapes: list,
            input_types: list,
            output_names: list):
        for idx, torch_file in enumerate(multi_torch_files):
            model = None
            model_name = self.get_submodel_prefix(idx)
            if isinstance(torch_file, str):
                model = torch.jit.load(
                    torch_file, map_location=torch.device('cpu'))
            else:
                model = torch_file
            model.eval()
            graph = model.inlined_graph
            self.state_dicts.append(model.state_dict())
            is_module = isinstance(model, torch.jit.ScriptModule)
            inputs = list(graph.inputs())
            inputs = inputs[1:] if is_module else inputs
            cur_input_names = []
            for inp in inputs:
                cur_input_names.append(inp.debugName())
            self.all_input_names.append(cur_input_names)

            cur_output_names = []
            for outp in graph.outputs():
                if outp.node().kind() == 'prim::TupleConstruct' or \
                outp.node().kind() == 'prim::ListConstruct':
                    ins = outp.node().inputs()
                    cur_output_names.extend([i.debugName() for i in ins])
                elif outp.node().kind() == 'prim::DictConstruct':
                    ins = outp.node().inputs()
                    ls_ins = list(ins)
                    in_num = len(ls_ins)
                    assert in_num % 2 == 0
                    cur_output_names.extend(
                        [ls_ins[i*2+1].debugName() for i in range(int(in_num/2))])
                else:
                    cur_output_names.append(outp.debugName())
            self.all_output_names.append(cur_output_names)

            self.weight_names = []
            self.torch_graphs.append(graph)
            self.torch_models.append(model)
        # assert 0,self.all_output_names
        assert len(self.torch_models) == len(self.torch_graphs)

        for idx, name in enumerate(self.input_names):
            self.addShape(name, input_shapes[idx])
        self.input_shapes = input_shapes
        for t in input_types:
            if t.lower() not in self.TypeMap:
                raise RuntimeError(f"Unknown type {t}")
            self.input_types.append(self.TypeMap[t.lower()])

    '''
        model_name => model_name.
    '''
    def get_submodel_prefix(self, idx: int) -> str:
        torch_file = self.multi_torch_files[idx]
        filename = os.path.basename(torch_file)
        return os.path.splitext(filename)[0] + "."

    '''
        reverse of reorganize_sub_block_info
    '''
    def get_structure_prefix(self, id_Block: int) -> str:
        idx = self.list_block_discontinuous_id_OrderbyStructure_IndexbyBlockType[id_Block]
        if self.list_encoded_block[id_Block]:
            return self.get_submodel_prefix(idx)
        return self.get_ppl_name(idx)

    '''
        list_inout_map stores names of top_in/top_out
    '''
    def generate_list_inout_map(self):
        self.list_inout_map = dict()
        for idx, _name in enumerate(self.output_names1):
            self.list_inout_map[self.input_names2[idx]] = _name

    '''
        switch to another torchnpt
    '''
    def Update_TorchConverter(self, idx: int):
        self.model = self.torch_models[idx]

    def generate_mlir(self, mlir_file: str, save_in_mem: bool = False):
        """convert all to mlir"""
        #TOP_INPUT
        for idx, _name in enumerate(self.input_names):
            unified_name = self.get_Input_prefix() + _name
            input_ = self.mlir.create_input_op(
                self.get_loc(unified_name), idx, self.preprocess_args[idx])
            self.addOperand(unified_name, input_)

        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))

        self.tensor_list = {}
        unsupported = set()
        id_graph  = 0
        id_PPLOp  = 0
        complete_onames = []
        for idx, flag_isNotPPL in enumerate(self.list_encoded_block):
            self.print_debug("[GenMlirStart]{}-sub_structure: {}".format(idx,".pt" if flag_isNotPPL else "PPL"))
            is_last_block = idx == len(self.list_encoded_block) - 1
            if flag_isNotPPL:
                self.Update_TorchConverter(id_graph)
                graph = self.torch_graphs[id_graph]
                cur_input_names = self.all_input_names[id_graph]
                for j, _name in enumerate(cur_input_names):
                    id_Block, output_id = self.io_map[(idx, j)]
                    output_name = self.input_names[output_id] if  id_Block==self.TOP_INPUT \
                                  else complete_onames[id_Block][output_id]
                    prefix_isolated = self.get_Input_prefix() if  id_Block==self.TOP_INPUT \
                                  else self.get_structure_prefix(id_Block)
                    output_name = output_name if len(re.findall(prefix_isolated,output_name))>=1 else prefix_isolated+output_name
                    self.print_debug("prefix_isolated/output_name: {}/{},{}".format(prefix_isolated,output_name,re.findall(prefix_isolated,output_name)))
                    op = self.getOperand(output_name)

                    prefix_unified = self.get_submodel_prefix(id_graph)
                    unified_name = _name if len(re.findall(prefix_unified,output_name))>=1 else prefix_unified+_name

                    self.addOperand(unified_name, op)
                    self.print_debug("prefix_unified/unified_name: {}/{}".format(prefix_unified,unified_name))

                self.print_debug("[Gen Unified_prefix]:",self.get_submodel_prefix(id_graph))
                #Ensure same-idx diff-value ConstantOp in different submodule won't override each other
                self.converted_nodes.clear()
                for node in graph.nodes():
                    self.converted_nodes.append(TorchNode(node))
                for n in self.converted_nodes:
                    prefix_unified = self.get_submodel_prefix(id_graph)
                    n.name    =  prefix_unified + n.name
                    n.inputs  = [prefix_unified+each_name for each_name in n.inputs] if len(n.inputs) >=1 else []
                    n.outputs = [prefix_unified+each_name for each_name in n.outputs]

                # checkout all type is supported
                for n in self.converted_nodes:
                    if n.op_type not in self.op_factory:
                        unsupported.add(n.op_type)
                if unsupported:
                    raise RuntimeError("Op not support:{}".format(unsupported))

                self.generate_list_map()
                for n in self.converted_nodes:
                    self.op_factory.get(n.op_type, lambda x: NoneAndRaise(x))(n)

                complete_onames.append(self.all_output_names[id_graph])
                id_graph += 1
            #PPLOp
            else:
                ppl_op_type = self.list_name_PPLop[id_PPLOp]
                if ppl_op_type not in self.ppl_factory:
                    raise RuntimeError("PPL op not support:{}".format(ppl_op_type))
                ppl_op_input_names = []
                nums_PPL_input_per = self.nums_PPLOp_Inputs[id_PPLOp]
                ppl_op_name = self.get_ppl_name(id_PPLOp)
                for j in range(nums_PPL_input_per):
                    id_Block, output_id = self.io_map[(idx, j)]
                    output_name = self.input_names[output_id] if  id_Block < 0 \
                                  else complete_onames[id_Block][output_id]
                    prefix_isolated = self.get_Input_prefix() if  id_Block < 0 \
                                  else self.get_structure_prefix(id_Block)
                    output_name = output_name if len(re.findall(prefix_isolated,output_name))>=1 else prefix_isolated+output_name
                    self.print_debug("[{}-prefix_isolated/output_name]: {}, {}".format(output_id,prefix_isolated,output_name))
                    op = self.getOperand(output_name)

                    _name = "{}_input_{}".format(ppl_op_type, j)
                    unified_name = ppl_op_name+_name
                    self.print_debug("[PPL-unified_name/ppl_op_name/name]: {}/{}/{}".format(unified_name, ppl_op_name,_name))
                    self.print_debug("-------------------------------")
                    self.addOperand(unified_name, op)
                    ppl_op_input_names.append(unified_name)

                cur_output_names = self.ppl_factory.get(ppl_op_type, lambda x: NoneAndRaise(x))(ppl_op_input_names, ppl_op_type,id_PPLOp)
                #[Check] num of outputs is same as user given self.num_outputs, if last sub-block is PPLOp
                if idx==len(self.list_encoded_block)-1:
                    assert len(self.output_names)==len(cur_output_names),"[Output-Shape Error] last sub_block is PPLOp,\
                       which num_outputs is {} != given num_outputs={}".format(len(cur_output_names),len(self.output_names))
                complete_onames.append(cur_output_names)
                id_PPLOp += 1
        # add return op
        return_op = list()
        # Set output
        for i in range(self.maskrcnn_num_output):
            id_Block, output_id = self.io_map[(self.TOP_OUTPUT, i)]
            assert id_Block != self.TOP_INPUT, "[Error]top-level in/out can't be inplaced, it's meaningless!"
            # self.print_debug("error here",id_Block,output_id)
            output_name = complete_onames[id_Block][output_id]
            unified_prefix = "" if self.list_encoded_block[id_Block]==self.Symbol_PPLSubBlock else  self.get_structure_prefix(id_Block)
            self.print_debug("[unified_prefix+output_name]:",unified_prefix+output_name)
            op = self.getOperand(unified_prefix+output_name)
            return_op.append(op)

        self.mlir.create_return_op(return_op)
        mlir_txt = self.mlir.print_module()
        if save_in_mem:
            mlir_txt = self.MlirModify(mlir_txt, self.weight_file)
            self.WeightToNpzInMem(self.weight_file)
        else:
            self.WeightToNpz(self.weight_file)
        with open(mlir_file, "w") as f:
            f.write(mlir_txt)
        logger.info("Save mlir file: {}".format(mlir_file))


    def convert_RPN_get_bboxes_op(self, input_list: list, op_type: str, id_PPLOp: int):
        Global_num_levels = self.NUM_LEVELS
        Global_strides = self.Global_strides
        def _gen_single_level_base_anchors(base_size, scales, ratios):
            w = base_size
            h = base_size
            center_offset = 0
            x_center = center_offset * w
            y_center = center_offset * h

            h_ratios = torch.sqrt(ratios)
            w_ratios = 1 / h_ratios
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
            base_anchors = [
                x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
                y_center + 0.5 * hs
            ]
            base_anchors = torch.stack(base_anchors, dim=-1)
            return base_anchors

        def _gen_base_anchors():
            multi_level_base_anchors = []
            base_sizes = [4, 8, 16, 32, 64]
            scales = torch.tensor([8.])
            ratios = torch.tensor([0.5000, 1.0000, 2.0000])
            for i, base_size in enumerate(base_sizes):
                multi_level_base_anchors.append(
                    _gen_single_level_base_anchors(
                        base_size,
                        scales=scales,
                        ratios=ratios))
            return multi_level_base_anchors

        def _grid_anchors(featmap_sizes):
            multi_level_anchors = []
            base_anchors = _gen_base_anchors()
            for i in range(Global_num_levels):
                anchors = _single_level_grid_anchors(
                    base_anchors[i],
                    featmap_sizes[i],
                    Global_strides[i])
                multi_level_anchors.append(anchors)
            return multi_level_anchors

        def _single_level_grid_anchors(base_anchors,featmap_size,stride=(16, 16)):
            feat_h, feat_w = featmap_size
            shift_x = torch.arange(0, feat_w) * stride[0]
            shift_y = torch.arange(0, feat_h) * stride[1]
            def _meshgrid( x, y, row_major=True):
                xx = x.repeat(y.shape[0])
                yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
                if row_major:
                    return xx, yy
                else:
                    return yy, xx
            shift_xx, shift_yy = _meshgrid(shift_x, shift_y)
            shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
            shifts = shifts.type_as(base_anchors)
            all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
            all_anchors = all_anchors.view(-1, 4)
            return all_anchors
        def div_up_internal(a,b):
            return int(a/b) + int(a%b!=0)
        dynamic_featmap_sizes = [[div_up_internal(self.H_RPN_DYN_MAX,2**i),div_up_internal(self.W_RPN_DYN_MAX,2**i)] for i in range(self.NUM_LEVELS)]
        mlvl_anchors = _grid_anchors(dynamic_featmap_sizes)
        mlvl_anchors = [per.reshape([1,1,-1,4]).numpy() for per in mlvl_anchors]

        name_anchor_0 = self.get_ppl_weight_name(id_PPLOp) + "anchor_0"
        name_anchor_1 = self.get_ppl_weight_name(id_PPLOp) + "anchor_1"
        name_anchor_2 = self.get_ppl_weight_name(id_PPLOp) + "anchor_2"
        name_anchor_3 = self.get_ppl_weight_name(id_PPLOp) + "anchor_3"
        name_anchor_4 = self.get_ppl_weight_name(id_PPLOp) + "anchor_4"

        self.addWeight(name_anchor_0, mlvl_anchors[0])
        self.addWeight(name_anchor_1, mlvl_anchors[1])
        self.addWeight(name_anchor_2, mlvl_anchors[2])
        self.addWeight(name_anchor_3, mlvl_anchors[3])
        self.addWeight(name_anchor_4, mlvl_anchors[4])
         #  self.getWeightOp(name_anchor_0),
        name_PPL_output_0 = self.get_ppl_name(id_PPLOp) + "{}_output_{}".format(op_type, 0)
        out = top.MaskRCNNRPNGetBboxesOp(self.unranked_type,
                                 self.getOp(input_list[0]),
                                 self.getOp(input_list[1]),
                                 self.getOp(input_list[2]),
                                 self.getOp(input_list[3]),
                                 self.getOp(input_list[4]),
                                 self.getOp(input_list[5]),
                                 self.getOp(input_list[6]),
                                 self.getOp(input_list[7]),
                                 self.getOp(input_list[8]),
                                 self.getOp(input_list[9]),
                                 self.getOp(input_list[10]),
                                 self.getWeightOp(name_anchor_0),
                                 self.getWeightOp(name_anchor_1),
                                 self.getWeightOp(name_anchor_2),
                                 self.getWeightOp(name_anchor_3),
                                 self.getWeightOp(name_anchor_4),
                                 self.DELTA2BBOX_1st_MEAN,
                                 self.DELTA2BBOX_1st_MEAN,
                                 self.DELTA2BBOX_1st_MEAN,
                                 self.DELTA2BBOX_1st_MEAN,
                                 1,1,1,1,
                                 self.MAX_SCALAR_C,
                                 self.NMS_THRE_1st,
                                 0.0,
                                 self.MAX_LENGTH_STATIC_STRECHED,
                                 self.NUM_INDEXES,
                                 self.NUM_CLASSES,
                                 self.CHANNEL_RPN_BBOXES,
                                 self.CHANNEL_RPN_SCORES,
                                 self.NMS_PRE,
                                 self.HARDWARE_FACTOR_TOPK,
                                 self.NMS_MAX_LENGTH_1st,
                                 self.TOPK_ONNX_NMS_1st,
                                 self.H_RPN_DYN_MAX,
                                 self.W_RPN_DYN_MAX,
                                 self.MAX_PER_IMG,
                                 loc=self.get_loc(name_PPL_output_0),
                                 ip=self.mlir.insert_point).result_list
        #[Note if params wrong check:]/workspace/tpu-mlir/install/python/mlir/dialects/_top_ops_gen.py
        self.addOperand(name_PPL_output_0 ,out)
        return [name_PPL_output_0]

    def convert_BBox_Pooler_op(self, input_list: list, op_type: str, id_PPLOp: int):
        #self.dict_content_yaml
        name_PPL_output_0 = self.get_ppl_name(id_PPLOp) + "{}_output_{}".format(op_type, 0)
        name_PPL_output_1 = self.get_ppl_name(id_PPLOp) + "{}_output_{}".format(op_type, 1)

        new_op = top.MaskRCNNBboxPoolerOp(self.unranked_type,
                                          self.unranked_type,
                                 self.getOp(input_list[0]),
                                 self.getOp(input_list[1]),
                                 self.getOp(input_list[2]),
                                 self.getOp(input_list[3]),
                                 self.getOp(input_list[4]),
                                 self.NUM_LEVELS_ROI,
                                 self.ROI_H,
                                 self.ROI_W,
                                 self.CHANNEL_ROI,
                                 self.ROI_SLICE_BBOX_POOLER,
                                 self.ROI_PH_BBOX_POOLER,
                                 self.ROI_PW_BBOX_POOLER,
                                 self.ROI_LEN,
                                 loc=self.get_loc([name_PPL_output_0,name_PPL_output_1]),
                                 ip=self.mlir.insert_point)
        #[Note if params wrong check:]/workspace/tpu-mlir/install/python/mlir/dialects/_top_ops_gen.py
        result_res = new_op.result_res
        result_rois = new_op.result_rois
        self.addOperand(name_PPL_output_0, result_res)
        self.addOperand(name_PPL_output_1, result_rois)
        return [name_PPL_output_0,name_PPL_output_1]


    def convert_get_bboxes_B_op(self, input_list: list, op_type: str, id_PPLOp: int):
        name_PPL_output_0 = self.get_ppl_name(id_PPLOp) + "{}_output_{}".format(op_type, 0)
        name_PPL_output_1 = self.get_ppl_name(id_PPLOp) + "{}_output_{}".format(op_type, 1)
        new_op = top.MaskRCNNGetBboxBOp(self.unranked_type,
                                        self.unranked_type,
                                 self.getOp(input_list[0]),
                                 self.getOp(input_list[1]),
                                 self.getOp(input_list[2]),
                                 self.getOp(input_list[3]),
                                 self.getOp(input_list[4]),
                                 self.GetBboxB_SCORE_EQ,
                                 self.MAX_SCALAR_C,
                                 self.NMS_THRE_2nd,
                                 self.DELTA2BBOX_2nd_MEAN,
                                 self.DELTA2BBOX_2nd_STD_0,
                                 self.DELTA2BBOX_2nd_STD_1,
                                 self.NUM_INDEXES,
                                 self.NUM_CLASSES,
                                 self.TOPK_ONNX_NMS_2nd,
                                 self.NUM_CLASSES_GetBboxB,
                                 self.NMS_MAX_LENGTH_2nd,
                                 self.MAX_PER_IMG,
                                 self.MAX_PER_IMG_GetBboxB,
                                 loc=self.get_loc([name_PPL_output_0,name_PPL_output_1]),
                                 ip=self.mlir.insert_point)
        #[Note if params wrong check:]/workspace/tpu-mlir/install/python/mlir/dialects/_top_ops_gen.py
        result_det_bboxes = new_op.result_det_bboxes
        result_det_labels = new_op.result_det_labels
        self.addOperand(name_PPL_output_0, result_det_bboxes)
        self.addOperand(name_PPL_output_1, result_det_labels)
        return [name_PPL_output_0,name_PPL_output_1]

    def convert_Mask_Pooler_op(self, input_list: list, op_type: str, id_PPLOp: int):
        name_PPL_output_0 = self.get_ppl_name(id_PPLOp) + "{}_output_{}".format(op_type, 0)

        out = top.MaskRCNNMaskPoolerOp(self.unranked_type,
                                 self.getOp(input_list[0]),
                                 self.getOp(input_list[1]),
                                 self.getOp(input_list[2]),
                                 self.getOp(input_list[3]),
                                 self.getOp(input_list[4]),
                                 self.getOp(input_list[5]),
                                 self.getOp(input_list[6]),
                                 self.NUM_LEVELS_ROI,
                                 self.ROI_H,
                                 self.ROI_W,
                                 self.CHANNEL_ROI,
                                 self.ROI_SLICE_MASK_POOLER,
                                 self.ROI_PH_MASK_POOLER,
                                 self.ROI_PW_MASK_POOLER,
                                 self.ROI_LEN,
                                 loc=self.get_loc(name_PPL_output_0),
                                 ip=self.mlir.insert_point).result_res
        #[Note if params wrong check:]/workspace/tpu-mlir/install/python/mlir/dialects/_top_ops_gen.py
        self.addOperand(name_PPL_output_0 ,out)
        return [name_PPL_output_0]
