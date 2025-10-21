from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from peft import PeftModelForCausalLM
from data import get_data
import datasets
from tqdm import tqdm
import re

def extract_answer(completion):
    match = re.search(r"<answer>([A-Ea-e])</answer>", completion)
    if match is not None:
        return match.group(1).strip().upper()
    return None

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
print("Base model loaded")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
print("Tokenizer loaded")

# model = PeftModelForCausalLM.from_pretrained(model, "rl_medmcqa_abstention/checkpoint-40")
# print("Peft model Loaded")

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

print(pipe.device)

ds = get_data()

print(ds['test'])

test_ds = ds['test']

final_records = []

for sample in tqdm(test_ds.select(range(100))):
    print("Prompt: ", sample['prompt'])
    response = pipe(sample['prompt'], max_new_tokens=1024)[0]['generated_text'][-1]['content']
    answer = extract_answer(response)
    print("Model response: ", response)
    print("Model Answer Extracted: ", answer)
    print("Correct Answer:", sample['correct_option'])
    print("="*100)
    sample['model_response'] = response
    sample['model_answer'] = answer
    final_records.append(sample)

out_ds = datasets.Dataset.from_list(final_records)

out_ds.save_to_disk("eval_outputs/base")
