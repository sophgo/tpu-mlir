import os
import yaml
import logging
from easydict import EasyDict
from datasets import load_dataset
from transformers import (
    AutoConfig,
    TrainingArguments,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
)


logger = logging.getLogger("transformer")


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
        per_device_train_batch_size=config_train.per_device_train_batch_size,
        per_device_eval_batch_size=config_train.per_device_eval_batch_size,
        gradient_accumulation_steps=config_train.gradient_accumulation_steps,
        eval_accumulation_steps=config_train.gradient_accumulation_steps,
        learning_rate=config_train.learning_rate,
        weight_decay=config_train.weight_decay,
        max_grad_norm=config_train.max_grad_norm,
        num_train_epochs=config_train.num_train_epochs,
        max_steps=config_train.max_steps,
        lr_scheduler_type=config_train.lr_scheduler_type,
        warmup_ratio=config_train.warmup_ratio,
        warmup_steps=config_train.warmup_steps,
        gradient_checkpointing=config_train.gradient_checkpointing,
        remove_unused_columns=config_train.remove_unused_columns,
        label_names=config_train.label_names,
        log_level=config_progress.log_level,
        log_level_replica=config_progress.log_level_replica,
        logging_dir=config_progress.logging_dir,
        logging_strategy=config_progress.logging_strategy,
        logging_steps=config_progress.logging_steps,
        save_strategy=config_progress.save_strategy,
        save_steps=config_progress.save_steps,
        save_total_limit=config_progress.save_total_limit,
        save_on_each_node=config_progress.save_on_each_node,
        no_cuda=config_progress.no_cuda,
        run_name=config_progress.run_name,
        disable_tqdm=config_progress.disable_tqdm,
        load_best_model_at_end=config_progress.load_best_model_at_end,
        metric_for_best_model=config_progress.metric_for_best_model,
        greater_is_better=config_progress.greater_is_better
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    config_progress.log_level = training_args.get_process_log_level()
    return training_args


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


def load_image_dataset(config_data, config_model):
    """
    dataset_name: Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub).
    dataset_config_name: The configuration name of the dataset to use (via the datasets library).
    train_dir: A folder containing the training data.
    validation_dir: A folder containing the validation data.
    train_val_split: Percent to split off of train for validation.
    max_train_samples: For debugging purposes or quicker training, truncate the number of training examples to this value if set.
    max_eval_samples: For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.
    """
    if config_data.dataset_name is not None:
        dataset = load_dataset(
            config_data.dataset_name,
            config_data.dataset_config_name,
            cache_dir=config_model.cache_dir,
            task="image-classification",
            use_auth_token=True if config_model.use_auth_token else None,
        )
    else:
        data_files = {}
        if config_data.train_dir is not None:
            data_files["train"] = os.path.join(config_data.train_dir, "**")
        if config_data.validation_dir is not None:
            data_files["validation"] = os.path.join(config_data.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=config_model.cache_dir,
            task="image-classification",
            keep_in_memory=True,
        )
    
    # If we don't have a validation split, split off a percentage of train as validation.
    config_data.train_val_split = None if "validation" in dataset.keys() else config_data.train_val_split
    if isinstance(config_data.train_val_split, float) and config_data.train_val_split > 0.0:
        split = dataset["train"].train_test_split(config_data.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]
    return dataset


def load_model(config_model, num_labels, label2id, id2label):
    config = AutoConfig.from_pretrained(
        config_model.config_name or config_model.model_name_or_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=config_model.cache_dir,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
    )
    model = AutoModelForImageClassification.from_pretrained(
        config_model.model_name_or_path,
        from_tf=bool(".ckpt" in config_model.model_name_or_path),
        config=config,
        cache_dir=config_model.cache_dir,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
        ignore_mismatched_sizes=config_model.ignore_mismatched_sizes,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        config_model.feature_extractor_name or config_model.model_name_or_path,
        cache_dir=config_model.cache_dir,
        revision=config_model.model_revision,
        use_auth_token=True if config_model.use_auth_token else None,
    )
    return model, feature_extractor
