import pandas as pd
from datasets import load_dataset

# mmlu
def process_example(sample):
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

def get_data():
    ds = load_dataset('openlifescienceai/medmcqa')['train']
    ds.cleanup_cache_files()
    ds = ds.map(
        process_example,
        num_proc=8
    )

    ds = ds.train_test_split(test_size=0.001, seed=42)

    return ds
