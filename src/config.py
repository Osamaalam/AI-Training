import yaml
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for loading the base Vision-Language Model."""
    model_name: str = "../Models/VLM-unsloth-8b"
    load_in_4bit: bool = False
    load_in_16bit: bool = True
    dtype: str = "bfloat16"
    use_gradient_checkpointing: str = "unsloth"

@dataclass
class PEFTConfig:
    """Configuration for Parameter-Efficient Fine-Tuning (LoRA)."""
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    random_state: int = 3407
    use_rslora: bool = True
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True

@dataclass
class DatasetConfig:
    """Configuration for datasets loading and formatting."""
    train_csv_path: str = "train.csv"
    val_csv_path: str = "valid.csv"
    max_length: int = 2048

@dataclass
class TrainingConfig:
    """Configuration for the SFTTrainer and training process."""
    output_dir: str = "outputs-16-32-6-1e4-8b-16bit"
    num_train_epochs: int = 6
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"
    seed: int = 3407
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "epoch"
    report_to: str = "none"
    bf16: bool = True
    torch_compile: bool = False
    remove_unused_columns: bool = False

@dataclass
class PipelineConfig:
    """Root configuration class unifying all pipeline stages."""
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Loads and parses pipeline configuration from a YAML file."""
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Configuration file not found at {yaml_path}. Using default configuration.")
            config_dict = {}

        model_dict = config_dict.get("model", {})
        peft_dict = config_dict.get("peft", {})
        dataset_dict = config_dict.get("dataset", {})
        training_dict = config_dict.get("training", {})

        return cls(
            model=ModelConfig(**model_dict),
            peft=PEFTConfig(**peft_dict),
            dataset=DatasetConfig(**dataset_dict),
            training=TrainingConfig(**training_dict)
        )

    def log_config(self) -> None:
        """Utility to log the loaded configuration attributes."""
        logger.info("=" * 60)
        logger.info("FRACTURE DETECTION VLM TRAINING PIPELINE CONFIGURATION")
        logger.info("=" * 60)
        for section, cfg in [("Model", self.model), ("PEFT/LoRA", self.peft), 
                             ("Dataset", self.dataset), ("Training", self.training)]:
            logger.info(f"[{section} Settings]")
            for k, v in cfg.__dict__.items():
                logger.info(f"  {k:30s}: {v}")
        logger.info("=" * 60)
