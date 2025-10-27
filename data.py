import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from constants import IDK, MEDMCQA_DATA, POLITIFACT_DATA, POLITIFACT_FILE_NAME
import kagglehub    


def process_example_medmcqa(sample):
    choices = [sample[f'op{x}'] for x in 'abcd']
    options_dict = {chr(65 + i): choice for i, choice in enumerate(choices)}
    options_str = "\n".join([f'{chr(65 + i)}: {choice}' for i, choice in enumerate(choices)])
    correct_option = chr(65 + sample['cop'])

    PROMPT_MESSAGES = [
        {
            'role': 'user',
            'content': "Answer the following question. Provide your thoughts between <reasoning> and </reasoning> symbols. Provide the final answer option (letter only) between <answer> and </answer> symbols."\
                f"Question: {sample['question']}\n" \
                f"Options: \n{options_str}"
        }
    ]

    return {
        'prompt': PROMPT_MESSAGES,
        'correct_option': correct_option
    }

def process_example_politifact(sample):
    choices = ["true", "mostly-true", "half-true", "mostly-false", "false", "pants-fire"] + ["I Don't Know"]
    options_dict = {chr(65 + i): choice for i, choice in enumerate(choices)} #ABCDEF G
    options_str = "\n".join([f'{chr(65 + i)}: {choice}' for i, choice in enumerate(choices)])
    correct_option = chr(65 + choices.index(sample['verdict'])) 

    PROMPT_MESSAGES = [
        {
            'role': 'user',
            'content': "Answer the following question. Provide your thoughts between <reasoning> and </reasoning> symbols. Provide the final answer option (letter only) between <answer> and </answer> symbols. Answer only if you are certain, else choose I Don't Know."\
                f"Question: {sample['statement']}\n" \
                f"Options: \n{options_str}"
        }
    ]
    return {
        'prompt': PROMPT_MESSAGES,
        'correct_option': correct_option,
        'idk_option': chr(65 + len(choices) - 1)
    }

def get_medmcqa_data():
    ds = load_dataset(MEDMCQA_DATA, split = 'train')
    return get_data(ds, process_example_medmcqa, 0.0012)

def get_politifact_data():
    path = kagglehub.dataset_download(POLITIFACT_DATA)
    df = pd.read_json(path + POLITIFACT_FILE_NAME, lines=True)
    ds = Dataset.from_pandas(df)
    return get_data(ds, process_example_politifact, 0.012)


def get_data(ds, process_example, test_size):
    ds.cleanup_cache_files()
    ds = ds.map(
        process_example,
        num_proc=8
    )

    ds = ds.train_test_split(test_size=0.2, seed=42)
    val = ds['train'].train_test_split(test_size=test_size, seed=42)

    print("length of train", str(len(val['train'])))
    print("length of validation", str(len(val['test'])))
    print("length of test", str(len(ds['test'])))

    modified_ds_split = DatasetDict({
        'train': val['train'],
        'validation': val['test'],
        'test': ds['test']})
    return modified_ds_split
