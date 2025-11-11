import pandas as pd
import logging
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from constants import LOGGING_FORMAT, DATE_FORMAT, TRAIN, VAL, TEST, SFT, IDK_PHRASES, MEDMCQA, MEDMCQA_DATA, POLITIFACT, POLITIFACT_DATA, POLITIFACT_FILE_NAME, GSM8K, GSM8K_DATA
import kagglehub
import re
from datasets import concatenate_datasets
import random
import os

# Set up logging
logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.INFO,
    datefmt=DATE_FORMAT)

logger = logging.getLogger()

# Define base number of options for each dataset (without IDK)
DATASET_OPTIONS = {
    MEDMCQA: 4,      # A-D
    POLITIFACT: 6,   # A-F
    GSM8K: 0, # No options
    # Add more datasets here as needed
}


def process_example_medmcqa(sample, idk_enabled=False):
    TRAINING_TYPE = os.getenv("TRAINING_TYPE") or os.getenv("EVAL_TYPE")
    choices = [sample[f'op{x}'] for x in 'abcd']
    options_dict = {chr(65 + i): choice for i, choice in enumerate(choices)}
    options_str = "\n".join([f'{chr(65 + i)}: {choice}' for i, choice in enumerate(choices)])
    correct_option = chr(65 + sample['cop'])
    idk_option = chr(65 + len(choices) - 1)

    base_content = "Answer the following question. Provide your thoughts between <reasoning> and </reasoning> symbols. Provide the final answer option (letter only) between <answer> and </answer> symbols."

    if idk_enabled:
        base_content += " Answer only if you are certain, else choose I Don't Know."
        choices = choices + ["I Don't Know"]
        if TRAINING_TYPE is not None and TRAINING_TYPE == SFT:
            if sample['correct_answer'] != sample["model_answer"]:
                idk_phrase = random.choice(IDK_PHRASES)
                completion = f"<reasoning>{sample['exp'] + ' ' + idk_phrase}</reasoning><answer>{idk_option}</answer>"
            else:
                completion = f"<reasoning>{sample['exp']}</reasoning><answer>{correct_option}</answer>"

            COMPLETION_MESSAGES = [
                {
                    'role': 'assistant',
                    'content': completion
                }
            ]


    PROMPT_MESSAGES = [
        {
            'role': 'user',
            'content': base_content + \
                f"Question: {sample['question']}\n" \
                f"Options: \n{options_str}"
        } 
    ]

    result = {
        'prompt': PROMPT_MESSAGES,
        'correct_answer': correct_option
    }

    if idk_enabled:
        result['idk_answer'] = chr(65 + len(choices) - 1)
    if TRAINING_TYPE is not None and TRAINING_TYPE == SFT:
        result['completion'] = COMPLETION_MESSAGES

    return result


def process_example_politifact(sample, idk_enabled=True):
    choices = ["true", "mostly-true", "half-true", "mostly-false", "false", "pants-fire"]

    if idk_enabled:
        choices = choices + ["I Don't Know"]

    options_dict = {chr(65 + i): choice for i, choice in enumerate(choices)}
    options_str = "\n".join([f'{chr(65 + i)}: {choice}' for i, choice in enumerate(choices)])
    correct_option = chr(65 + choices.index(sample['verdict']))

    base_content = "Answer the following question. Provide your thoughts between <reasoning> and </reasoning> symbols. Provide the final answer option (letter only) between <answer> and </answer> symbols."

    if idk_enabled:
        base_content += " Answer only if you are certain, else choose I Don't Know."

    PROMPT_MESSAGES = [
        {
            'role': 'user',
            'content': base_content + \
                f"Question: {sample['statement']}\n" \
                f"Options: \n{options_str}"
        }
    ]

    result = {
        'prompt': PROMPT_MESSAGES,
        'correct_answer': correct_option
    }

    if idk_enabled:
        result['idk_answer'] = chr(65 + len(choices) - 1)

    return result

def process_example_gsm8k(sample, idk_enabled=False):
    match = re.search(r"####\s*(-?[\d,]+)", sample['answer'])
    if match:
        correct_numeric_answer = int(match.group(1).replace(",", ""))
    else:
        log.error("GSM8K data not clean")

    base_content = "Answer the following question. Provide your thoughts between <reasoning> and </reasoning> symbols. Provide the final numeric answer (number only) between <answer> and </answer> symbols."
    if idk_enabled:
        base_content += " Answer only if you are certain, else answer I Don't Know."

    PROMPT_MESSAGES = [
        {
            'role': 'user',
            'content': base_content + \
                f"Question: {sample['question']}"
        }
    ]
    result = {
        'prompt': PROMPT_MESSAGES,
        'correct_answer': correct_numeric_answer
    }
    if idk_enabled:
        result['idk_answer'] = "I Don't Know"
    return result


def get_medmcqa_data(idk_enabled=False):
    TRAINING_TYPE = os.getenv("TRAINING_TYPE") or os.getenv("EVAL_TYPE")
    if TRAINING_TYPE is not None and TRAINING_TYPE == SFT:
        ds = load_from_disk('./eval_outputs/baseline_ibm-granite/granite-3.3-2b-instruct_medmcqa_train_40000_4options')
        ds = ds.filter(
            lambda sample: (sample['exp'] is not None) and (len(sample['exp']) < 2250)
        )
        # Split sizes: train=58%, val=2%, test=40%
        # Train=146037, Validation=220, Test=36565
        return get_data(ds, lambda x: process_example_medmcqa(x, idk_enabled),
                        train_size=0.7988, val_size=0.0012, test_size=0.20)
    ds = load_dataset(MEDMCQA_DATA, split=TRAIN)
    # Split sizes: train=79.88%, val=0.12%, test=20%
    # Train=146037, Validation=220, Test=36565
    return get_data(ds, lambda x: process_example_medmcqa(x, idk_enabled),
                    train_size=0.7988, val_size=0.0012, test_size=0.20)


def get_politifact_data(idk_enabled=True):
    path = kagglehub.dataset_download(POLITIFACT_DATA)
    df = pd.read_json(path + POLITIFACT_FILE_NAME, lines=True)
    ds = Dataset.from_pandas(df)
    # Split sizes: train=78.8%, val=1.2%, test=20%
    # Train=16667, Validation=254, Test=4231
    return get_data(ds, lambda x: process_example_politifact(x, idk_enabled),
                    train_size=0.788, val_size=0.012, test_size=0.20)

def get_gsm8k_data(idk_enabled=True):
    ds = load_dataset(GSM8K_DATA, "main")
    ds = concatenate_datasets([ds[TRAIN], ds[TEST]])
    # Split sizes: train=60%, val=10%, test=30%
    # Train=5275, Validation=880, Test=2638
    return get_data(ds, lambda x: process_example_gsm8k(x, idk_enabled),
                    train_size=0.60, val_size=0.10, test_size=0.30)


def get_data(ds, process_example, train_size, val_size, test_size):
    ds.cleanup_cache_files()
    ds = ds.map(
        process_example,
        num_proc=8
    )

    # First split: separate test set (20%)
    ds = ds.train_test_split(test_size=test_size, seed=42)

    # Second split: separate validation from training
    # val_size as proportion of the training set
    val_proportion = val_size / (train_size + val_size)
    val = ds[TRAIN].train_test_split(test_size=val_proportion, seed=42)

    logger.info(f"Length of train: {len(val[TRAIN])}")
    logger.info(f"Length of validation: {len(val[TEST])}")
    logger.info(f"Length of test: {len(ds[TEST])}")

    modified_ds_split = DatasetDict({
        TRAIN: val[TRAIN],
        VAL: val[TEST],
        TEST: ds[TEST]})

    return modified_ds_split
