import numpy as np
from typing import List, Union

import yaml
import argparse
import pathlib
import os

import torch

def str2listInt(v):
    files = v.split(',')
    files = [s.strip() for s in files]
    while files.count('') > 0:
        files.remove('')
    files = [int(s) for s in files]
    return files

def str2listFloat(v):
    files = v.split(',')
    files = [s.strip() for s in files]
    while files.count('') > 0:
        files.remove('')
    files = [float(s) for s in files]
    return files

class MaskRCNN_Tester_Basic(object):
    path_default_MaskRCNN_dataset   = os.path.dirname(os.path.abspath(__file__)) +  "/../../../nnmodels/maskrcnn_models/"
    params_model_transform_MaskRCNN = "maskrcnn_model_transform_extern_params"

    def __init__(self, debug: bool = True, path_custom_dataset: str = None):
        self.DEBUG_MASKRCNN = debug
        if path_custom_dataset is not None:
            self.print_debug("[Warning] path_default_MaskRCNN_dataset is changed from default path: {} to new path: {}".format(self.path_default_MaskRCNN_dataset, path_custom_dataset))
            self.path_default_MaskRCNN_dataset =  path_custom_dataset
        if not os.path.exists(self.path_default_MaskRCNN_dataset):
          if os.getenv('NNMODELS_PATH') is not None:
            self.path_default_MaskRCNN_dataset = os.getenv('NNMODELS_PATH') + "/maskrcnn_models/"
            if not os.path.exists(self.path_default_MaskRCNN_dataset):
               git_command = 'git --git-dir={}/.git log'.format(os.getenv('NNMODELS_PATH'))
               os.system(git_command)
               assert 0, "[MaskRCNN-Error] dataset path not found, expect {}".format(self.path_default_MaskRCNN_dataset)
        else:
          print("[MaskRCNN-Warning] NNMODELS_PATH is not exist, might used for model-zoo")
    def print_debug(self, *info):
        if self.DEBUG_MASKRCNN:
            print(*info)

class MaskRCNN_InputPreprocessor(MaskRCNN_Tester_Basic):
    Suffix_Universal = "SuperiorMaskRCNNInputPreprocessed"
    Mode_BackBone    = "BackBone"
    Mode_Complete    = "Complete"
    Mode_StartFromRPN2END        = "RPN2END"
    Mode_StartFromRPN2BBOXPOOLER = "RPN2BBOXPOOLER"

    def deal_path_save_preprocessed(self):
        if self.path_preprocessed_npz is not None:
            self.path_save_preprocessed               = self.path_preprocessed_npz
        else:
            if self.mode_input_generator == self.Mode_BackBone:
                self.SAVE_suffix = self.Mode_BackBone
            else:
                self.SAVE_suffix = self.Suffix_Universal
            path_save_preprocessed_lv0                = self.path_input_image.replace(".npy","").split(".npz")[0]+"_{}.npz".format(self.SAVE_suffix)
            name_npz_save                = path_save_preprocessed_lv0.split('/')[-1]
            self.path_current            = str(pathlib.Path().resolve())

            # [Save-Path-0] save to python/test/MaskRCNN_test_bm1684x file
            # This is changed as os.chdir, not where you start to run test_MaskRCNN.py
            self.path_save_preprocessed               = self.path_current  + "/" + name_npz_save

        ## [Save-Path-1] save to regression/dataset
        # self.path_save_preprocessed             = self.path_default_MaskRCNN_dataset  + "/" + name_npz_save

    def __init__(self,
                    path_yaml:                  str = None,
                    path_input_image:           str = None,
                    path_preprocessed_npz :     str = None,
                    basic_max_shape_inverse: list[int] =  None,
                    basic_scalar_factor:   list[float] =  None,
                    debug: bool = True,
                    mode_input_generator:  str  = "Complete"):

          MaskRCNN_Tester_Basic.__init__(self, debug)
          self.processed_input_data    = dict()
          self.path_input_image        = path_input_image
          self.path_preprocessed_npz   = path_preprocessed_npz
          self.mode_input_generator    = mode_input_generator

          self.deal_path_save_preprocessed()
          self.basic_max_shape_inverse = basic_max_shape_inverse
          self.basic_scalar_factor     = basic_scalar_factor
          self.batch_size = -1

          assert len((torch.tensor(basic_scalar_factor)==1).nonzero(as_tuple=True)[0])==0,"[Warning] scalar_factor usually not all 1s"
          self.inversed_Warning()

          self.path_yaml = path_yaml
          self.preprocess_YAML()
          if self.mode_input_generator==self.Mode_Complete:
            self.basic_info ="##################################################\n"+ \
            "# copy from tpu-mlir/python/tools/tools_Superior_MaskRCNN/CMD_Example_MaskRCNN_End2End.md\n"+ \
            "##################################################\n"+ \
            "# Input 0) 'img.1'     shape=[ 1 3 800 1216 ]\n"+ \
            "# Input 1) 'max_shape_RPN'      shape=[batch_size, 1, max_filter_num, num_indexes]=[1,1,4741,4] dtype=int32\n"+ \
            "# Input 2) 'max_shape_GetBboxB' shape=[1,batch_size*{},1,4]                    =[1,1,{},4] dtype=int32\n".format(self.C_getBboxB, self.C_getBboxB)+ \
            "# Input 3) 'scale_factor'       shape=[1,1,{},4] dtype =FLOAT32\n".format(self.C_getBboxB)+ \
            "# Input 4) 'scale_factor_mask_pooler'       shape=[1,1,100,4] dtype =FLOAT32\n"+ \
            "##################################################"
          elif self.mode_input_generator==self.Mode_StartFromRPN2END:
            self.basic_info ="##################################################\n"+ \
            "Input 0) '11' shape=[ {} 256 200 304 ] dtype=FLOAT32\n".format(self.batch_size)+ \
            "Input 1) '12' shape=[ 1 256 100 152 ] dtype=FLOAT32\n"+ \
            "Input 2) '13' shape=[ 1 256 50 76 ] dtype=FLOAT32\n"+ \
            "Input 3) '16' shape=[ 1 256 25 38 ] dtype=FLOAT32\n"+ \
            "<!-- Input 4) '15' shape=[ 1 256 13 19 ] dtype=FLOAT32 -->\n"+ \
            "Input 4) '18' shape=[ 1 3 200 304 ] dtype=FLOAT32\n"+ \
            "Input 5) '19' shape=[ 1 3 100 152 ] dtype=FLOAT32\n"+ \
            "Input 6) '20' shape=[ 1 3 50 76 ] dtype=FLOAT32\n"+ \
            "Input 7) '21' shape=[ 1 3 25 38 ] dtype=FLOAT32\n"+ \
            "Input 8) '22' shape=[ 1 3 13 19 ] dtype=FLOAT32\n"+ \
            "Input 9) '23' shape=[ 1 12 200 304 ] dtype=FLOAT32\n"+ \
            "Input 10) '24' shape=[ 1 12 100 152 ] dtype=FLOAT32\n"+ \
            "Input 11) '25' shape=[ 1 12 50 76 ] dtype=FLOAT32\n"+ \
            "Input 12) '26' shape=[ 1 12 25 38 ] dtype=FLOAT32\n"+ \
            "Input 13) '27' shape=[ 1 12 13 19 ] dtype=FLOAT32\n"+ \
            "Input 14) 'max_shape_RPN'      shape=[batch_size, 1, max_filter_num, num_indexes]=[1,1,4741,4] dtype=int32\n"+ \
            "Input 15) 'max_shape_GetBboxB' shape=[1,batch_size*{},1,4]                    =[1,1,{},4] dtype=int32\n".format(self.C_getBboxB, self.C_getBboxB)+ \
            "Input 16) 'scale_factor_BboxPooler'       shape=[1,4,{},4] dtype =FLOAT32\n".format(self.C_getBboxB)+ \
            "Input 17) 'scale_factor_MaskPooler'       shape=[1,1,100,4] dtype =FLOAT32\n"+ \
            "##################################################"
          elif self.mode_input_generator==self.Mode_StartFromRPN2BBOXPOOLER:
             self.basic_info ="##################################################\n"+ \
            "Input 0) '11' shape=[ 1 256 200 304 ] dtype=FLOAT32\n"+ \
            "Input 1) '12' shape=[ 1 256 100 152 ] dtype=FLOAT32\n"+ \
            "Input 2) '13' shape=[ 1 256 50 76 ] dtype=FLOAT32\n"+ \
            "Input 3) '16' shape=[ 1 256 25 38 ] dtype=FLOAT32\n"+ \
            "<!-- Input 4) '15' shape=[ 1 256 13 19 ] dtype=FLOAT32 -->\n"+ \
            "Input 4) '18' shape=[ 1 3 200 304 ] dtype=FLOAT32\n"+ \
            "Input 5) '19' shape=[ 1 3 100 152 ] dtype=FLOAT32\n"+ \
            "Input 6) '20' shape=[ 1 3 50 76 ] dtype=FLOAT32\n"+ \
            "Input 7) '21' shape=[ 1 3 25 38 ] dtype=FLOAT32\n"+ \
            "Input 8) '22' shape=[ 1 3 13 19 ] dtype=FLOAT32\n"+ \
            "Input 9) '23' shape=[ 1 12 200 304 ] dtype=FLOAT32\n"+ \
            "Input 10) '24' shape=[ 1 12 100 152 ] dtype=FLOAT32\n"+ \
            "Input 11) '25' shape=[ 1 12 50 76 ] dtype=FLOAT32\n"+ \
            "Input 12) '26' shape=[ 1 12 25 38 ] dtype=FLOAT32\n"+ \
            "Input 13) '27' shape=[ 1 12 13 19 ] dtype=FLOAT32\n"+ \
            "Input 14) 'max_shape_RPN'      shape=[batch_size, 1, max_filter_num, num_indexes]=[1,1,4741,4] dtype=int32\n"+ \
            "##################################################"
          elif self.mode_input_generator==self.Mode_BackBone:
            self.basic_info ="##################################################\n"+ \
                "# copy from tpu-mlir/python/transform/README_Superior_MaskRCNN_README.md\n"+ \
                "##################################################\n"+ \
                "# Input 0) 'img.1'     shape=[ 1 3 800 1216 ]\n"+ \
                "##################################################"
          else: assert 0,"[Error] Such Mode-{} is not supported!".format(self.mode_input_generator)
          self.print_debug(self.basic_info)

          if mode_input_generator==self.Mode_Complete:
            self.process_complete_mode()
          elif mode_input_generator==self.Mode_StartFromRPN2END:
            self.process_RPN_mode()
          elif self.mode_input_generator==self.Mode_StartFromRPN2BBOXPOOLER:
            self.process_RPN2BBOXPOOLER_Mode()
          elif self.mode_input_generator==self.Mode_BackBone:
            self.process_BackBone_Mode()

    def inversed_Warning(self):
          self.print_debug("[Warning] please input h-w-inversed_shape!")

    def preprocess_YAML(self):
          with open(self.path_yaml, 'r') as file:
            self.dict_content_yaml = yaml.unsafe_load(file)
          self.num_indexes= self.dict_content_yaml["NUM_INDEXES"]
          self.max_filter_num= self.dict_content_yaml["MAX_LENGTH_STATIC_STRECHED"]
          self.C_getBboxB =  self.dict_content_yaml["MAX_PER_IMG"]*self.dict_content_yaml["NUM_CLASSES_GetBboxB"]
          self.roi_slice_maskpooler =  self.dict_content_yaml["ROI_SLICE_MASK_POOLER"]

    def process_InputIMG_Data(self):
          input_data = np.load(self.path_input_image)
          if isinstance(input_data, np.ndarray):
              self.processed_input_data[str(0)] =  input_data
              self.batch_size =  input_data.shape[0]
          else:
            self.processed_input_data[str(0)] =  input_data[input_data.files[0]]
            self.batch_size =  input_data[input_data.files[0]].shape[0]

    def gen_basic_tenor_and_repeat_twice(self, input_A: np.ndarray) -> np.ndarray:
          input_A = np.array(input_A).reshape([-1,2])
          assert  input_A.shape[0]==self.batch_size
          input_A = input_A.reshape([self.batch_size, 1, 1, 2])
          input_A = np.concatenate([input_A, input_A ],axis=-1)
          input_A = input_A.reshape([self.batch_size, 1, 1, 4])
          self.inversed_Warning()
          return input_A

    def process_PPL_ConstOp(self, num_constOp: int = 3):
          assert self.batch_size>0
          start_idx = len(list(self.processed_input_data.keys()))

          #1/2 max_val for RPN2END & GetBBoxes
          self.basic_max_shape_inverse = self.gen_basic_tenor_and_repeat_twice(self.basic_max_shape_inverse)

          max_shape_RPN = np.repeat( self.basic_max_shape_inverse,self.max_filter_num, axis=2)
          self.processed_input_data[str(start_idx+0)] =  max_shape_RPN.astype(np.float32)

          if num_constOp>=2:
            max_shape_getBboxB =np.repeat( self.basic_max_shape_inverse,self.C_getBboxB, axis=2).reshape([1,1,-1,self.num_indexes])
            self.processed_input_data[str(start_idx+1)] =  max_shape_getBboxB.astype(np.float32)

            #4-scale-factor
            self.basic_scalar_factor = self.gen_basic_tenor_and_repeat_twice(self.basic_scalar_factor)
            scale_factor_getBboxB = np.repeat( self.basic_scalar_factor,self.C_getBboxB, axis=2).reshape([1,1,-1,self.num_indexes])
            self.processed_input_data[str(start_idx+2)] =  scale_factor_getBboxB.astype(np.float32)

            scale_factor_MaskPOoler = np.repeat( self.basic_scalar_factor,self.roi_slice_maskpooler, axis=2).reshape([1,1,-1,self.num_indexes])
            self.processed_input_data[str(start_idx+3)] =  scale_factor_MaskPOoler.astype(np.float32)

          self.print_debug("-----------------Statiscal Info for save_path: {}--------------------".format(self.path_save_preprocessed))
          for idx, per_key in enumerate(self.processed_input_data.keys()):
              self.print_debug("[Key]-{}  shape-{}".format(per_key, self.processed_input_data[per_key].shape))
          self.print_debug("-------------------------------------------------------------------")
          np.savez(self.path_save_preprocessed, **self.processed_input_data)


          recheck_data = np.load(self.path_save_preprocessed)
          assert np.sum(np.abs(recheck_data[recheck_data.files[start_idx+0]]-  self.processed_input_data[recheck_data.files[start_idx+0]])) <1e-6
          assert np.sum(np.abs(recheck_data[recheck_data.files[start_idx+0]]- max_shape_RPN)) <1e-6
          if num_constOp>=2:
            assert np.sum(np.abs(recheck_data[recheck_data.files[start_idx+1]]- max_shape_getBboxB)) <1e-6
            #   for i in range(scale_factor_getBboxB.flatten().shape[0]):
            #       if np.abs(recheck_data[recheck_data.files[start_idx+2]].flatten()[i]- scale_factor_getBboxB.flatten()[i])>1e-6:
            #           print(i,recheck_data[recheck_data.files[start_idx+2]].flatten()[i], scale_factor_getBboxB.flatten()[i])
            assert np.sum(np.abs(recheck_data[recheck_data.files[start_idx+2]]- scale_factor_getBboxB))/scale_factor_getBboxB.flatten().shape[0] <1e-6,np.sum(np.abs(recheck_data[recheck_data.files[start_idx+2]]- scale_factor_getBboxB))
          self.print_debug("[CMP]All stored {}-data Checked&Passed!".format(len(self.processed_input_data.keys())))

    def gen_model_transform_cmd(self):
        self.print_debug("----------------------------[Helper]model_deploy--------------------------------")
        context_info = ""
        context_info += "--test_input  {} \ \n".format(self.path_save_preprocessed)
        test_reference_path = "Ref_Output_"+self.path_save_preprocessed.split("/")[-1].replace("input","").replace(self.SAVE_suffix, "")
        if self.path_preprocessed_npz is None:
          context_info += "--test_reference  {} #using output npz of your last block \ ".format(test_reference_path)
        else:
          context_info += "--test_reference  {} #using output npz of your last block \ ".format(self.path_preprocessed_npz)

        self.print_debug(context_info)

    def process_RPN_Data(self):
        dict_RPN = np.load(self.path_input_image)
        list_keys = list(dict_RPN.keys())
        self.batch_size =  dict_RPN[list_keys[0]].shape[0]
        # 'x_0', 'x_1', 'x_2', 'x_3', 'cls_scores_0', 'cls_scores_1', 'cls_scores_2', 'cls_scores_3', 'cls_scores_4', 'bbox_preds_0', 'bbox_preds_1', 'bbox_preds_2', 'bbox_preds_3', 'bbox_preds_4
        for idx, keys_per in enumerate(list_keys):
            self.processed_input_data[str(idx)] =dict_RPN[keys_per]

    def process_complete_mode(self):
        self.process_InputIMG_Data()
        self.process_PPL_ConstOp()
        self.gen_model_transform_cmd()

    def process_RPN_mode(self):
        self.process_RPN_Data()
        self.process_PPL_ConstOp()
        self.gen_model_transform_cmd()

    def process_RPN2BBOXPOOLER_Mode(self):
        self.process_RPN_Data()
        self.process_PPL_ConstOp(num_constOp=1)
        self.gen_model_transform_cmd()

    def process_BackBone_Mode(self):
        self.process_InputIMG_Data()
        np.savez(self.path_save_preprocessed, **self.processed_input_data)
        self.print_debug("[Save-path]: {}".format(self.path_save_preprocessed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--debug", action="store_true", help='keep middle file if debug')
    parser.add_argument("--path_yaml", type=str, default=None, help="one YAML recording MaskRCNN parameters")

    '''
    | Tester | Stage                           | Files                                              | ID  |Context                                     | Shape                                       |
    |--------|---------------------------------|----------------------------------------------------|-----|--------------------------------------------|---------------------------------------------|
    |--------| MaskRCNN_Reorganize_TOP_Input.py| Superior_IMG.npy                                   | '0' |origin image                                | [ 1, 3, 800, 1216 ]                         |
    | End2End|model_deploy                     | Superior_IMG_SuperiorMaskRCNNInputPreprocessed.npz | '0' |preprocessed image                          | [ 1, 3, 800, 1216 ]                         |
    |        |                                 |                                                    | '1' |max_shape_RPN                               | [batch_size, 1, max_filter_num, num_indexes]|
    |        |                                 |                                                    | '2' |max_shape_GetBboxB                          | [1, batch_size*20000, 1, 4]                 |
    |        |                                 |                                                    | '3' |scale_factor_GetBboxB                       | [1, batch_size, 20000, 4]                   |
    |        |                                 |                                                    | '4' |scale_factor_MaskPooler                     | [batch_size, 1, roi_slice, 4]               |
    | Utest  |model_transform                  | Superior_IMG_BackBone.npz                          | '0' |preprocessed image                          | [ 1, 3, 800, 1216 ]                         |
    '''
    parser.add_argument('--basic_max_shape_inverse', type=str2listInt,  default="1216,800",
                        help="inital h-w-inversed multi-batch img_shape(though it called max_shape, it's not padding_shape!) \
                        ex:1199,800,1100,740  for batch-2 [800,1199,3]&[740,1100,3]")
    parser.add_argument('--basic_scalar_factor', type=str2listFloat,  default="1.8734375,1.8735363",
                        help="inital h-w-inversed multi-batch scalar_factor \
                        ex:1.3,1.2 for 1-batch w-h-inversed")
    parser.add_argument('--mode_input_generator', type=str,  default="Complete", choices=["Complete", "RPN2END", "RPN2BBOXPOOLER"],
                        help="mode of input")
    parser.add_argument('--path_input_image', type=str, default=None,
                        help="path for input img npz, must be compatible with --preprocess in later mlir_transformer!")
    parser.add_argument('--path_preprocessed_npz', type=str, default=None,
                        help="path for preprocessed input npz")

    # yapf: enable
    args = parser.parse_args()

    Superior_MaskRCNN_InputParser = MaskRCNN_InputPreprocessor(args.path_yaml,
                                                               args.path_input_image,
                                                               args.path_preprocessed_npz,
                                                               args.basic_max_shape_inverse,
                                                               args.basic_scalar_factor,
                                                               args.debug,
                                                               args.mode_input_generator)
'''
#only preprocess input data
python3 python/tools/tool_maskrcnn.py \
    --path_yaml /workspace/tpu-mlir/regression/dataset/MaskRCNN/CONFIG_MaskRCNN.yaml \
    --path_input_image     /workspace/tpu-mlir/regression/dataset/MaskRCNN/Superior_IMG_BackBone.npz \
    --path_preprocessed_npz ./Superior_IMG_BackBone_SuperiorMaskRCNNInputPreprocessed.npz \
    --basic_max_shape_inverse 1216,800 \
    --basic_scalar_factor     1.8734375,1.8735363 \
    --mode_input_generator "Complete" \
    --debug
'''
