from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from peft import PeftModelForCausalLM
from data import get_medmcqa_data, get_politifact_data
import datasets
import logging
from tqdm import tqdm
import re
from itertools import islice
from constants import *
import constants
from rewards import extract_answer
from utils import chunked
import time
from datetime import datetime
import os


logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.INFO,
    datefmt=DATE_FORMAT)

logger = logging.getLogger()

# Configuration flags
MODEL = GRANITE # GRANITE | QWEN as baselines
LOAD_SPECIFIC_MODEL = False
MODEL_CHECKPOINT_NAME = "sft_rl_medmcqa_abstention_qwen_chk300_model_idk_plus_0/checkpoint-90" # only useful if loading specific model
if LOAD_SPECIFIC_MODEL:
    EVAL_TYPE = SFT # SFT | RL | SFT_RL. This will only affect the output file name
else:
    EVAL_TYPE = BASELINES

DATA = MEDMCQA # MEDMCQA | POLITIFACT
IDK_ENABLED = False  # Toggle IDK option in dataset
EVAL_TYPE = BASELINES
EVAL_ON = TEST # always keep this test dataset for eval unless really necessary
NUM_SAMPLES = 40000

# parallel processing and checkpointing
USE_BATCH_PROCESSING = True  # Toggle between batch and sequential processing
BATCH_SIZE = 32
CHECKPOINT_INTERVAL_HOURS = 2  # Checkpoint interval in hours
CHECKPOINT_DIR = "eval_checkpoints"  # Directory to save checkpoints



def save_checkpoint(records, num_processed, elapsed_time_hours):
    """Save checkpoint with processed records"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{CHECKPOINT_DIR}/checkpoint_records_{num_processed}_time_{elapsed_time_hours:.2f}h_{timestamp}"

    checkpoint_ds = datasets.Dataset.from_list(records)
    checkpoint_ds.save_to_disk(checkpoint_name)
    logger.info(f"Checkpoint saved: {checkpoint_name} (Records: {num_processed}, Time: {elapsed_time_hours:.2f}h)")
    return checkpoint_name

logger.info("Using model: %s", MODEL)

model = AutoModelForCausalLM.from_pretrained(MODEL)
if LOAD_SPECIFIC_MODEL:
    model = merge_lora_model(model, MODEL_CHECKPOINT_NAME)
    logger.info("Peft model Loaded")
logger.info("Base model loaded")

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side = "left")
logger.info("Tokenizer loaded")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
logger.info("Pipeline device: %s", pipe.device)

# Load dataset with IDK flag
match DATA:
    case constants.MEDMCQA:
        NUM_OPTIONS = 4
        if IDK_ENABLED:
            NUM_OPTIONS += 1
        ds = get_medmcqa_data(idk_enabled=IDK_ENABLED)
    case constants.POLITIFACT:
        NUM_OPTIONS = 6
        if IDK_ENABLED:
            NUM_OPTIONS += 1
        ds = get_politifact_data(idk_enabled=IDK_ENABLED)
    case _:
        logger.error("Please select valid dataset")
        raise ValueError("Invalid dataset selected")

logger.info("Dataset loaded: %s", ds[EVAL_ON])

train_ds = ds[EVAL_ON]
final_records = []

# Checkpoint tracking
start_time = time.time()
last_checkpoint_time = start_time
checkpoint_interval_seconds = CHECKPOINT_INTERVAL_HOURS * 3600
num_processed = 0

if USE_BATCH_PROCESSING:
    logger.info("Using BATCH processing mode")

    for batch in tqdm(chunked(train_ds.select(range(NUM_SAMPLES)), BATCH_SIZE), desc="Evaluation progress", total=(NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE):
        prompts = [s['prompt'] for s in batch]
        outputs = pipe(prompts, max_new_tokens=1024, batch_size=BATCH_SIZE)

        for s, out in zip(batch, outputs):
            generated = out[0]['generated_text'][-1]['content']
            answer = extract_answer(generated)

            logger.info("Prompt: %s", s['prompt'])
            logger.info("Model response: %s", generated)
            logger.info("Model Answer Extracted: %s", answer)
            logger.info("Correct Answer: %s", s['correct_option'])
            logger.info("="*100)

            s['model_response'] = generated
            s['model_answer'] = answer
            final_records.append(s)
            num_processed += 1

        # Check if checkpoint interval has passed
        current_time = time.time()
        if current_time - last_checkpoint_time >= checkpoint_interval_seconds:
            elapsed_time_hours = (current_time - start_time) / 3600
            save_checkpoint(final_records, num_processed, elapsed_time_hours)
            last_checkpoint_time = current_time

else:
    logger.info("Using SEQUENTIAL processing mode")

    for sample in tqdm(train_ds.select(range(NUM_SAMPLES)), desc="Evaluation progress"):
        logger.info("Prompt: %s", sample['prompt'])

        response = pipe(sample['prompt'], max_new_tokens=1024)[0]['generated_text'][-1]['content']
        answer = extract_answer(response)

        logger.info("Model response: %s", response)
        logger.info("Model Answer Extracted: %s", answer)
        logger.info("Correct Answer: %s", sample['correct_option'])
        logger.info("="*100)

        sample['model_response'] = response
        sample['model_answer'] = answer
        final_records.append(sample)
        num_processed += 1

        # Check if checkpoint interval has passed
        current_time = time.time()
        if current_time - last_checkpoint_time >= checkpoint_interval_seconds:
            elapsed_time_hours = (current_time - start_time) / 3600
            save_checkpoint(final_records, num_processed, elapsed_time_hours)
            last_checkpoint_time = current_time


# Save final output
out_ds = datasets.Dataset.from_list(final_records)
idk_suffix = "_idk" if IDK_ENABLED else ""
EVAL_DATA_NAME = f"eval_outputs/{EVAL_TYPE}_{MODEL}_{}_{DATA}_{NUM_SAMPLES}_{NUM_OPTIONS}option{idk_suffix}"

out_ds.save_to_disk(EVAL_DATA_NAME)
logger.info(f"Final evaluation results saved to: {EVAL_DATA_NAME}")

# Log total time
total_time_hours = (time.time() - start_time) / 3600
logger.info(f"Evaluation completed. Total time: {total_time_hours:.2f} hours, Records processed: {num_processed}")
