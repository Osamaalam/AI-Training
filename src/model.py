import logging
from typing import Any
import torch
from unsloth import FastVisionModel
from src.config import ModelConfig, PEFTConfig

logger = logging.getLogger(__name__)

def resolve_dtype(dtype_str: str) -> torch.dtype:
    """
    Helper function to convert string datatype config to actual torch.dtype object.
    
    Args:
        dtype_str (str): Datatype name (e.g. "bfloat16", "float16", "float32").
        
    Returns:
        torch.dtype: Matched PyTorch datatype.
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    resolved = dtype_map.get(dtype_str.lower(), torch.bfloat16)
    logger.info(f"Resolved dtype '{dtype_str}' to {resolved}")
    return resolved


def load_vlm_model(config: ModelConfig) -> tuple[Any, Any]:
    """
    Loads the pretrained Vision-Language Model with optimized memory parameters.
    
    Args:
        config (ModelConfig): Object containing the VLM loading configurations.
        
    Returns:
        tuple: (loaded_model, loaded_tokenizer)
    """
    logger.info(f"Initializing FastVisionModel from path: {config.model_name}")
    
    torch_dtype = resolve_dtype(config.dtype)
    
    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            config.model_name,
            load_in_4bit=config.load_in_4bit,
            load_in_16bit=config.load_in_16bit,
            dtype=torch_dtype,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )
        logger.info("✅ Pretrained FastVisionModel and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model from {config.model_name}: {e}")
        raise e


def configure_peft_model(model: Any, config: PEFTConfig) -> Any:
    """
    Wraps the loaded VLM inside a Parameter-Efficient Fine-Tuning (PEFT) framework using LoRA.
    
    Args:
        model: Pretrained base Vision-Language Model.
        config (PEFTConfig): LoRA config dataclass.
        
    Returns:
        Any: Annotated PEFT/LoRA wrapper around the model.
    """
    logger.info("Configuring PEFT/LoRA modules for parameter-efficient fine-tuning...")
    
    try:
        peft_model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=config.finetune_vision_layers,
            finetune_language_layers=config.finetune_language_layers,
            finetune_attention_modules=config.finetune_attention_modules,
            finetune_mlp_modules=config.finetune_mlp_modules,
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            random_state=config.random_state,
            use_rslora=config.use_rslora,
            loftq_config=None,
        )
        
        # Log parameter stats for debugging
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        logger.info(f"PEFT applied successfully.")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total model parameters: {total_params:,}")
        logger.info(f"Finetuning fraction: {100 * trainable_params / total_params:.4f}% of parameters")
        
        return peft_model
    except Exception as e:
        logger.error(f"Failed to wrap model with PEFT: {e}")
        raise e
