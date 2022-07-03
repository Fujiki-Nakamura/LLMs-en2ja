from dataclasses import asdict, dataclass, field
import os
from typing import Optional


exp_name = "fnakamura/t5-base-en2ja_pt-4"
model_name = "fnakamura/t5-base-en2ja_pt-4"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=model_name)
    from_pt: bool = field(default=False)
    model_type: Optional[str] = field(default="")
    config_name: Optional[str] = field(default=model_name)
    tokenizer_name: Optional[str] = field(default="fnakamura/t5-base-en2ja")
    use_fast_tokenizer: bool = field(default=True)

    cache_dir: Optional[str] = field(default=os.getenv("HF_DATASETS_CACHE"))
    dtype: Optional[str] = field(default="float32")
    use_auth_token: bool = field(default=True)


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    dataset_name: Optional[str] = field(default="oscar")
    dataset_config_name: Optional[str] = field(default="unshuffled_deduplicated_ja")
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    train_ref_file: Optional[str] = field(default=None)
    validation_ref_file: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=1)
    max_seq_length: Optional[int] = field(default=512)
    preprocessing_num_workers: Optional[int] = field(default=32)
    mlm_probability: float = field(default=0.15)
    mean_noise_span_length: float = field(default=3.0)

    streaming: bool = field(default=False)

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.data_files is None:  # noqa
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


@dataclass
class TrainingArguments:
    output_dir: str = field(default=f"/scratch/acd14036hb/outputs/{exp_name}")
    overwrite_output_dir: bool = field(default=True)

    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    # for caching datasets
    # do_train: bool = field(default=False)
    # do_eval: bool = field(default=False)
    resume_from_checkpoint: str = field(default="fnakamura/t5-base-en2ja_pt-1")

    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=16)
    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0.0)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)
    adafactor: bool = field(default=False)
    num_train_epochs: float = field(default=1.0)
    warmup_steps: int = field(default=0)
    logging_steps: int = field(default=10000)
    save_steps: int = field(default=10000)
    eval_steps: int = field(default=10000)
    push_to_hub: bool = field(default=True)
    hub_model_id: str = field(default=None)
    hub_token: str = field(default=os.getenv("HF_TOKEN"))
    seed: int = field(default=42)

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum): d[k] = v.value  # noqa
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum): d[k] = [x.value for x in v]  # noqa
            if k.endswith("_token"): d[k] = f"<{k.uppser()}>"  # noqa
        return d
