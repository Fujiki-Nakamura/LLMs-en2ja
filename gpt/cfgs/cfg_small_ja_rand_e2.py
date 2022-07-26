from dataclasses import dataclass, field
import os
from typing import Optional


exp_name = "fnakamura/gpt2-small-ja-rand_e2"


@dataclass
class ModelArguments:
    model_name = "rinna/japanese-gpt2-small"
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    config_overrides: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=model_name)
    tokenizer_name: Optional[str] = field(default="fnakamura/gpt2-small-en2ja")
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=True)

    from_pt: bool = field(default=True)
    cache_dir: Optional[str] = field(default=os.getenv("HF_DATASETS_CACHE"))
    dtype: Optional[str] = field(default="float32")

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default="elyza/elyza-pretrain-corpus")
    dataset_config_name: Optional[str] = field(default="v2")
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    block_size: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=1)
    preprocessing_num_workers: Optional[int] = field(default=128)
    keep_linebreaks: bool = field(default=True)

    max_seq_length: Optional[int] = field(default=1024)
    split: str = field(default=None)
    streaming: bool = field(default=False)
    save_to_disk: bool = field(default=False)

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."
