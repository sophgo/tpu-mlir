import os
import yaml
import torch
import random
import numpy as np
from easydict import EasyDict
from datasets import load_dataset
from transformers import (
    AutoConfig,
    PretrainedConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
    config = EasyDict(config)
    return config

def make_huggingface_training_args(config_train, config_progress):
    training_args = TrainingArguments(
        seed=config_train.seed,
        output_dir=config_train.output_dir,
        overwrite_output_dir=config_train.overwrite_output_dir,
        do_train=config_train.do_train,
        do_eval=config_train.do_eval,
        do_predict=config_train.do_predict,
        evaluation_strategy=config_train.evaluation_strategy,
        eval_steps=config_train.eval_steps,
        per_device_eval_batch_size=config_train.per_device_eval_batch_size,
        log_level=config_progress.log_level,
        log_level_replica=config_progress.log_level_replica,
        logging_dir=config_progress.logging_dir,
        logging_strategy=config_progress.logging_strategy,
        no_cuda=config_progress.no_cuda,
        run_name=config_progress.run_name,
        disable_tqdm=config_progress.disable_tqdm,
        metric_for_best_model=config_progress.metric_for_best_model,
        greater_is_better=config_progress.greater_is_better,
        label_names = ['labels']
    )
    
    config_progress.log_level = training_args.get_process_log_level()
    return training_args

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_dataset_labels(config_data):
    # datasets
    raw_datasets = load_dataset("glue", config_data.task_name)
    # num_labels
    if config_data.task_name=='stsb':
        num_labels = 1
        label_list = None
    else:
        label_list = raw_datasets['train'].features['label'].names
        num_labels = len(label_list)
    return raw_datasets, num_labels, label_list


def load_model(config_model, config_data, num_labels):
    # num_labels first to indentity the classification heads
    tokenizer = AutoTokenizer.from_pretrained(
        config_model.model_name_or_path,
        cache_dir=config_model.cache_dir,
        use_fast=config_model.use_fast_tokenizer,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
    )
    config_tmp = AutoConfig.from_pretrained(
        config_model.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=config_data.task_name,
        cache_dir=config_model.cache_dir,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config_model.model_name_or_path,
        from_tf=bool(".ckpt" in config_model.model_name_or_path),
        config=config_tmp,
        cache_dir=config_model.cache_dir,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
    )
    return tokenizer, model


def preprocess_dataset(config_data, training_args, raw_datasets, label_to_id, tokenizer):
    # tokenize the data
    sentence1_key, sentence2_key = task_to_keys[config_data.task_name]
    if config_data.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False
    max_seq_length = config_data.max_seq_length

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not config_data.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    return raw_datasets
