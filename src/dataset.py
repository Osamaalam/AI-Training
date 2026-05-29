import os
import logging
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value

logger = logging.getLogger(__name__)

# Standard instruction template for the VLM model
FRACTURE_INSTRUCTION = """You are a medical imaging assistant specialized in musculoskeletal X-ray interpretation.

You will be given the X-ray images.

Task:
Classify the image as either positive or negative.

Definitions:
- Positive: The image shows abnormality.
- Negative: The image shows no abnormality.

Rules:
- Do not use information outside the given image.
- Do not guess or speculate.
- Do not explain your reasoning.
- Do not output anything except the final label.

Output:
Respond with exactly one word, all lowercase.

Allowed outputs:
positive
negative
"""

def load_dataframes(train_csv_path: str, val_csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train and validation CSV files and logs descriptive statistics.
    
    Args:
        train_csv_path (str): Path to the training dataset CSV.
        val_csv_path (str): Path to the validation dataset CSV.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Loaded pandas DataFrames for training and validation.
    """
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training CSV not found at: {train_csv_path}")
    if not os.path.exists(val_csv_path):
        raise FileNotFoundError(f"Validation CSV not found at: {val_csv_path}")

    logger.info(f"Loading training data from: {train_csv_path}")
    train_df = pd.read_csv(train_csv_path)
    
    logger.info(f"Loading validation data from: {val_csv_path}")
    val_df = pd.read_csv(val_csv_path)

    logger.info("✅ Datasets loaded successfully.")
    logger.info(f"Training samples: {len(train_df):,}")
    logger.info(f"Validation samples: {len(val_df):,}")

    # Log class distributions if 'label' column exists
    if 'label' in train_df.columns:
        train_dist = train_df['label'].value_counts().to_dict()
        logger.info(f"Training label distribution: {train_dist}")
    if 'label' in val_df.columns:
        val_dist = val_df['label'].value_counts().to_dict()
        logger.info(f"Validation label distribution: {val_dist}")

    return train_df, val_df


def convert_to_conversation(sample: dict) -> dict:
    """
    Converts a single dataset sample into a structured standard VLM conversation format.
    
    Args:
        sample (dict): A dictionary containing 'image_path', 'modality', and 'label'.
        
    Returns:
        dict: A dictionary under the key 'messages' formatted for chat-based training.
    """
    user_text = (
        f"{FRACTURE_INSTRUCTION}\n"
        f"Image Modality: {sample.get('modality', 'Unknown')}"
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": sample["image_path"]}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["label"]}
            ]
        }
    ]

    return {"messages": conversation}


def get_hf_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame) -> DatasetDict:
    """
    Converts pandas DataFrames into Hugging Face Datasets with explicit features.
    
    Args:
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.
        
    Returns:
        DatasetDict: A Hugging Face DatasetDict containing 'train' and 'validation' splits.
    """
    features = Features({
        "image_path": Value("string"),
        "label": Value("string"),
        "modality": Value("string"),
    })

    # Ensure all required columns are present and clean
    for df_name, df in [("train", train_df), ("validation", val_df)]:
        missing_cols = [col for col in ["image_path", "label", "modality"] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns {missing_cols} in {df_name} dataframe.")

    train_dataset = Dataset.from_pandas(train_df, features=features)
    val_dataset = Dataset.from_pandas(val_df, features=features)

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })


def prepare_datasets(train_csv_path: str, val_csv_path: str) -> tuple[list[dict], list[dict]]:
    """
    Orchestrates loading CSV data, converting to HF format, and transforming 
    to conversation dictionaries ready for SFTTrainer fine-tuning.
    
    Args:
        train_csv_path (str): Path to training CSV file.
        val_csv_path (str): Path to validation CSV file.
        
    Returns:
        tuple[list[dict], list[dict]]: Formatted lists of conversations for training and validation splits.
    """
    train_df, val_df = load_dataframes(train_csv_path, val_csv_path)
    hf_dataset = get_hf_datasets(train_df, val_df)

    logger.info("Formatting dataset splits into VLM conversation structure...")
    converted_train = [convert_to_conversation(sample) for sample in hf_dataset["train"]]
    converted_val = [convert_to_conversation(sample) for sample in hf_dataset["validation"]]
    
    logger.info("✅ Conversation formatting completed.")
    return converted_train, converted_val
