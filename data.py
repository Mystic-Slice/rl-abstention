import pandas as pd
import logging
from datasets import load_dataset, DatasetDict, Dataset
from constants import IDK, MEDMCQA_DATA, POLITIFACT_DATA, POLITIFACT_FILE_NAME, LOGGING_FORMAT, DATE_FORMAT, TRAIN, VAL, TEST
import kagglehub

# Set up logging
logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.INFO,
    datefmt=DATE_FORMAT)

logger = logging.getLogger()


def process_example_medmcqa(sample, idk_enabled=False):
    choices = [sample[f'op{x}'] for x in 'abcd']

    if idk_enabled:
        choices = choices + ["I Don't Know"]

    options_dict = {chr(65 + i): choice for i, choice in enumerate(choices)}
    options_str = "\n".join([f'{chr(65 + i)}: {choice}' for i, choice in enumerate(choices)])
    correct_option = chr(65 + sample['cop'])

    base_content = "Answer the following question. Provide your thoughts between <reasoning> and </reasoning> symbols. Provide the final answer option (letter only) between <answer> and </answer> symbols."

    if idk_enabled:
        base_content += " Answer only if you are certain, else choose I Don't Know."

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
        'correct_option': correct_option
    }

    if idk_enabled:
        result['idk_option'] = chr(65 + len(choices) - 1)

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
        'correct_option': correct_option
    }

    if idk_enabled:
        result['idk_option'] = chr(65 + len(choices) - 1)

    return result


def get_medmcqa_data(idk_enabled=False):
    ds = load_dataset(MEDMCQA_DATA, split=TRAIN)
    # Split sizes: train=79.88%, val=0.12%, test=20%
    return get_data(ds, lambda x: process_example_medmcqa(x, idk_enabled),
                    train_size=0.7988, val_size=0.0012, test_size=0.20)


def get_politifact_data(idk_enabled=True):
    path = kagglehub.dataset_download(POLITIFACT_DATA)
    df = pd.read_json(path + POLITIFACT_FILE_NAME, lines=True)
    ds = Dataset.from_pandas(df)
    # Split sizes: train=78.8%, val=1.2%, test=20%
    return get_data(ds, lambda x: process_example_politifact(x, idk_enabled),
                    train_size=0.788, val_size=0.012, test_size=0.20)


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