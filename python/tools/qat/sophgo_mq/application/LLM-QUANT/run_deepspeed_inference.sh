export PYTHONPATH=../../:$PYTHONPATH
deepspeed --num_gpus 2 ptq-llm-OPT-dsinference.py