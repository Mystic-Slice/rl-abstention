from data import get_medmcqa_data, get_politifact_data
from model import get_model

from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
from rewards import format_reward, accuracy_reward
import logging
from constants import *
import constants

# Set up logging
logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.INFO, # Change to DEBUG for experimentations
    datefmt=DATE_FORMAT)

logger = logging.getLogger()

# ======================== CONFIGURATION ========================

# Training type: 'RL' (GRPO) or 'SFT' (Supervised Fine-Tuning)
TRAINING_TYPE = 'RL'  # Options: 'RL', 'SFT'

# Model configuration
BASE_MODEL = QWEN  # Options: GRANITE | QWEN
MODEL_TYPE = LORA  # Options: LORA | FULL
LOAD_SPECIFIC_MODEL = False  # If True, load and merge a specific checkpoint
MODEL_CHECKPOINT_PATH = "rl_medmcqa_abstention/checkpoint-100"  # Path to checkpoint (only used if LOAD_SPECIFIC_MODEL=True)

# Dataset configuration
DATA = MEDMCQA  # Options: MEDMCQA, POLITIFACT, etc.
IDK_ENABLED = True  # Toggle IDK option in dataset. Mostly True in train.py

# Output configuration
# OUTPUT_DIR = "rl_medmcqa_abstention"  # Directory to save model checkpoints and final model
OUTPUT_DIR = "_".join(TRAINING_TYPE, DATA, BASE_MODEL.split("/")[0])
# ======================== LOAD MODEL ========================

logger.info(f"Loading base model: {BASE_MODEL} (type: {MODEL_TYPE})")
model, tokenizer = get_model(BASE_MODEL, MODEL_TYPE)

if LOAD_SPECIFIC_MODEL:
    logger.info(f"Loading specific model checkpoint: {MODEL_CHECKPOINT_PATH}")
    model = merge_lora_model(model, MODEL_CHECKPOINT_PATH)
    logger.info("Checkpoint loaded and merged")

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

# Configure tokenizer and model for SFT training
if TRAINING_TYPE == 'SFT':
    # model.generation_config.eos_token_id = 151645  # Only used for QWEN SFT
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.bos_token = tokenizer.pad_token
    tokenizer.bos_token_id = tokenizer.pad_token_id

logger.info("Model configuration:", model.config)
logger.info("Model", model)
logger.info("Tokenizer", tokenizer)

logger.debug("Generation configuration:", model.generation_config)

logger.info(f"Device: {model.device}")

# ======================== LOAD DATASET ========================

logger.info(f"Loading dataset: {DATA} (IDK enabled: {IDK_ENABLED})")

match DATA:
    case constants.MEDMCQA:
        ds = get_medmcqa_data(idk_enabled=IDK_ENABLED)
    case constants.POLITIFACT:
        ds = get_politifact_data(idk_enabled=IDK_ENABLED)
    case _:
        logger.error("Please select valid dataset")
        raise ValueError("Invalid dataset selected")

logger.info(ds)

train_dataset = ds[TRAIN] # Avoid changing
val_dataset = ds[VAL] # Avoid changing

logger.info('Sample prompt:')
logger.info(train_dataset[0]['prompt'])
logger.info(f"Correct option: {train_dataset[0]['correct_option']}")
if IDK_ENABLED:
    logger.info(f"IDK option: {train_dataset[0].get('idk_option', 'N/A')}")
if TRAINING_TYPE == 'SFT' and 'completion' in train_dataset[0]:
    logger.info(f"Completion: {train_dataset[0]['completion']}")

# ======================== TRAINING CONFIGURATION ========================

if TRAINING_TYPE == 'RL':
    logger.info("Using GRPO (RL) Training Configuration")

    training_args = GRPOConfig(
        # Output directory for checkpoints and logs
        output_dir=OUTPUT_DIR,

        # Learning rate for optimizer
        learning_rate=2e-5,

        # Keep all columns (needed for reward functions to access correct_option)
        remove_unused_columns=False,

        # Gradient accumulation: accumulate gradients over N steps before updating
        gradient_accumulation_steps=64,

        # Maximum number of training steps
        max_steps=500,

        # Enable gradient checkpointing to reduce memory usage
        gradient_checkpointing=True,

        # Use bfloat16 precision for training (faster, less memory)
        bf16=True,

        # Number of completions to generate per prompt for RL
        num_generations=8,

        # Batch size per GPU for training
        per_device_train_batch_size=8,

        # Batch size per GPU for evaluation
        per_device_eval_batch_size=8,

        # Maximum length of generated completions
        max_completion_length=1024,

        # Maximum length of input prompts
        max_prompt_length=256,

        # Report metrics to TensorBoard
        report_to=["tensorboard"],

        # Log metrics every N steps
        logging_steps=5,

        # Save checkpoint strategy
        save_strategy="steps",
        save_steps=20, # Save checkpoint every 20 steps
        save_total_limit=3,  # Keep only last 3 checkpoints

        # Evaluation strategy
        eval_strategy='steps',
        eval_steps=40,

        # Use vLLM for faster inference during generation
        use_vllm=True,
        vllm_mode="colocate",  # Colocate vLLM with training process
        vllm_gpu_memory_utilization=0.5,  # Use 50% of GPU memory for vLLM

        # Sampling parameters for generation
        top_k=40,  # Sample from top-k tokens
        top_p=0.95,  # Nucleus sampling threshold

        # GRPO-specific parameters
        epsilon=0.2,  # KL penalty coefficient
        epsilon_high=0.28,  # Upper bound for adaptive KL
        scale_rewards=False,  # Don't normalize rewards

        # Weights for multiple reward functions [format_reward, accuracy_reward]
        reward_weights=[1, 2],

        # Clear CUDA cache every N steps to prevent OOM
        torch_empty_cache_steps=1,

        # Enable vLLM sleep mode to save memory when not generating
        vllm_enable_sleep_mode=True
    )

    logger.info("Training arguments:")
    logger.info(training_args)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

elif TRAINING_TYPE == 'SFT':
    logger.info("Using SFT (Supervised Fine-Tuning) Configuration")

    training_args = SFTConfig(
        # Output directory for checkpoints and logs
        output_dir=OUTPUT_DIR,

        # Learning rate for optimizer
        learning_rate=2e-5,

        # Gradient accumulation: accumulate gradients over N steps before updating
        gradient_accumulation_steps=8,

        # Maximum number of training steps
        max_steps=1000,

        # Enable gradient checkpointing to reduce memory usage
        gradient_checkpointing=True,

        # Use bfloat16 precision for training (faster, less memory)
        bf16=True,

        # Batch size per GPU for training
        per_device_train_batch_size=8,

        # Batch size per GPU for evaluation
        per_device_eval_batch_size=8,

        # Maximum sequence length (prompt + completion)
        max_length=1024,

        # Report metrics to TensorBoard
        report_to=["tensorboard"],

        # Log metrics every N steps
        logging_steps=5,

        # Save checkpoint strategy
        save_strategy="steps",
        save_steps=50, # Save checkpoint every 50 steps
        save_total_limit=3,  # Keep only last 3 checkpoints

        # Evaluation strategy
        eval_strategy='steps',
        eval_steps=50,

        # Clear CUDA cache every N steps to prevent OOM
        torch_empty_cache_steps=1,
    )

    logger.info("Training arguments:")
    logger.info(training_args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

logger.info(f"Training device: {training_args.device}")

# ======================== TRAIN MODEL ========================

logger.info(f"Starting {TRAINING_TYPE} training...")

# Uncomment to resume from checkpoint
# trainer.train(resume_from_checkpoint=True)

trainer.train()

# ======================== SAVE MODEL ========================

final_model_path = f"{OUTPUT_DIR}/final_model"
trainer.save_model(final_model_path)
logger.info(f"Final model saved to: {final_model_path}")

logger.info("Training completed successfully!")