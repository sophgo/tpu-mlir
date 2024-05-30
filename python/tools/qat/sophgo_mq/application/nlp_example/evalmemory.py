from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
## specify the model you want to train on your device
#model = AutoModel.from_pretrained("t5-large") 
configuration = GPT2Config.from_pretrained('gpt2-medium',resid_pdrop = 0.3 , output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("gpt2-medium", config=configuration)
## estimate the memory cost (both CPU and GPU)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)
estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)
