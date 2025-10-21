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

NUM_SAMPLES = 14000
MODEL = "ibm-granite/granite-3.3-2b-instruct"
DATA = "medmcqa"
NUM_OPTIONS = 4
def extract_answer(completion):
    match = re.search(r"<answer>\s*([A-Da-d])[^<]*<\/answer>", completion)
    if match is not None:
        return match.group(1).strip()
    return None

logger.info("Using model: %s", MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
logger.info("Base model loaded")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
logger.info("Tokenizer loaded")

# model = PeftModelForCausalLM.from_pretrained(model, "rl_medmcqa_abstention/checkpoint-40")
# logger.info("Peft model Loaded")

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

logger.info(pipe.device)

ds = get_data()

logger.info(ds['test'])

test_ds = ds['test']

final_records = []

for sample in tqdm(test_ds.select(range(NUM_SAMPLES)), "Evaluation progress"):
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
EVAL_DATA_NAME = "eval_outputs/baseline_" + MODEL + "_" + DATA + "_" + str(NUM_SAMPLES) + "_" + str(NUM_OPTIONS) + "options"
out_ds.save_to_disk(EVAL_DATA_NAME)
