from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from peft import PeftModelForCausalLM
from data import get_data
import datasets
import logging
from tqdm import tqdm
import re

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(name)s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger()

def extract_answer(completion):
    match = re.search(r"<answer>(.*)</answer>", completion)
    if match is not None:
        return match.group(1).strip()
    return None

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
logger.info("Base model loaded")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
logger.info("Tokenizer loaded")

# model = PeftModelForCausalLM.from_pretrained(model, "rl_medmcqa_abstention/checkpoint-40")
# logger.info("Peft model Loaded")

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

logger.info(pipe.device)

ds = get_data()

logger.info(ds['test'])

test_ds = ds['test']

final_records = []

for sample in tqdm(test_ds.select(range(10)), "Evaluation progress"):
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

out_ds = datasets.Dataset.from_list(final_records)

out_ds.save_to_disk("eval_outputs/base")
