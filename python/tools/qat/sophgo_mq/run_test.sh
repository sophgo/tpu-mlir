#!/bin/bash
# PTQ test
python ./sophgo_mq/ptq_train_all_model.py

# QAT test
python ./sophgo_mq/qat_train_all_model.py