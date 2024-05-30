export PYTHONPATH=../../:$PYTHONPATH
deepspeed ptq-llm-dszeroinference.py --deepspeed_config config.json
