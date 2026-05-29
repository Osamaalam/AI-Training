import os
import sys
import argparse
import logging
from transformers import set_seed
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# Adjust path to enable importing from local src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PipelineConfig
from src.dataset import prepare_datasets
from src.model import load_vlm_model, configure_peft_model

# Configure professional logging layout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("train_pipeline")


def parse_args():
    """Parses command-line arguments to override YAML configurations."""
    parser = argparse.ArgumentParser(description="Production-Grade Fracture Detection VLM Finetuning Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to pipeline config YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--train_csv", 
        type=str, 
        help="Override training CSV path"
    )
    parser.add_argument(
        "--val_csv", 
        type=str, 
        help="Override validation CSV path"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        help="Override total training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="Override per-device training batch size"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Override output directory for checkpoints"
    )
    return parser.parse_args()


def main():
    logger.info("Initializing Fracture Detection VLM Training Pipeline...")
    
    # 1. Parse CLI arguments & load configuration
    args = parse_args()
    config = PipelineConfig.from_yaml(args.config)
    
    # Apply command-line overrides if provided
    if args.train_csv:
        config.dataset.train_csv_path = args.train_csv
    if args.val_csv:
        config.dataset.val_csv_path = args.val_csv
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.output_dir:
        config.training.output_dir = args.output_dir
        
    config.log_config()
    
    # 2. Disable torch dynamo & inductor compiler as per notebook specification for stable/fast VLM training
    logger.info("Setting environment variables for stable VLM fine-tuning...")
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"
    
    # 3. Set global seed for deterministic initialization & training
    logger.info(f"Setting seed: {config.training.seed}")
    set_seed(config.training.seed)
    
    # 4. Prepare conversational dataset formats
    try:
        train_dataset, val_dataset = prepare_datasets(
            train_csv_path=config.dataset.train_csv_path,
            val_csv_path=config.dataset.val_csv_path
        )
        logger.info(f"Loaded {len(train_dataset):,} train conversations and {len(val_dataset):,} validation conversations.")
    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}")
        sys.exit(1)
        
    # 5. Load and Patch Vision-Language Model
    try:
        model, tokenizer = load_vlm_model(config.model)
    except Exception as e:
        logger.error(f"Failed to load VLM model: {e}")
        sys.exit(1)
        
    # 6. Apply PEFT/LoRA configuration
    try:
        model = configure_peft_model(model, config.peft)
    except Exception as e:
        logger.error(f"Failed to configure PEFT: {e}")
        sys.exit(1)
        
    # 7. Enable model for training
    logger.info("Enabling model parameters for training via FastVisionModel.for_training...")
    FastVisionModel.for_training(model)
    
    # 8. Configure Trainer and launch training run
    logger.info("Constructing SFTTrainer and matching configurations...")
    
    trainer_config = SFTConfig(
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        warmup_ratio=config.training.warmup_ratio,
        num_train_epochs=config.training.num_train_epochs,
        learning_rate=config.training.learning_rate,
        logging_steps=config.training.logging_steps,
        optim=config.training.optim,
        weight_decay=config.training.weight_decay,
        lr_scheduler_type=config.training.lr_scheduler_type,
        seed=config.training.seed,
        output_dir=config.training.output_dir,
        report_to=config.training.report_to,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_strategy=config.training.save_strategy,
        bf16=config.training.bf16,
        remove_unused_columns=config.training.remove_unused_columns,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=config.dataset.max_length,
        torch_compile=config.training.torch_compile,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=trainer_config
    )
    
    logger.info("==================================================")
    logger.info("🚀 Launching Fine-Tuning Execution...")
    logger.info("==================================================")
    
    try:
        trainer_stats = trainer.train()
        logger.info("=" * 50)
        logger.info("🎉 Fine-Tuning Completed Successfully!")
        logger.info(f"Training parameters: {trainer_stats}")
        logger.info("=" * 50)
    except Exception as e:
        logger.critical(f"Training execution crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
