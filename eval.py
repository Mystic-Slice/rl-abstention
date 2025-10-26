from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from peft import PeftModelForCausalLM
from data import get_data
import datasets
import logging
from tqdm import tqdm
import re
from itertools import islice
from constants import QWEN, GRANITE, MEDMCQA

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(name)s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger()

NUM_SAMPLES = 14000
MODEL = GRANITE
DATA = MEDMCQA
NUM_OPTIONS = 5
def extract_answer(completion):
    match = re.search(r"<answer>\s*([A-Ea-e])[^<]*<\/answer>", completion)
    if match is not None:
        return match.group(1).strip().upper()
    return None

def chunked(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

logger.info("Using model: %s", MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
logger.info("Base model loaded")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
logger.info("Tokenizer loaded")

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# model = PeftModelForCausalLM.from_pretrained(model, "rl_medmcqa_abstention/checkpoint-40")
# logger.info("Peft model Loaded")

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

logger.info(pipe.device)

ds = get_data()

logger.info(ds['test'])

test_ds = ds['test']

final_records = []
BATCH_SIZE = 32
#for sample in tqdm(test_ds.select(range(NUM_SAMPLES//2)), "Evaluation progress"):
#for batch in tqdm(chunked(test_ds.select(range(NUM_SAMPLES//2)), BATCH_SIZE), desc="Evaluation progress"):
#    prompts = [s['prompt'] for s in batch]
#    outputs = pipe(prompts, max_new_tokens=1024, batch_size=BATCH_SIZE)

#    for s, out in zip(batch, outputs):
#        generated = out[0]['generated_text'][-1]['content']
#        answer = extract_answer(generated)

#        logger.info("Prompt: %s", s['prompt'])
#        logger.info("Model response: %s", generated)
#        logger.info("Model Answer Extracted: %s", answer)
#        logger.info("Correct Answer: %s", s['correct_option'])
#        logger.info("="*100)

#        s['model_response'] = generated
#        s['model_answer'] = answer
#        final_records.append(s)


#out_ds = datasets.Dataset.from_list(final_records)
#EVAL_DATA_NAME = "eval_outputs/baseline_" + MODEL + "_" + DATA + "_" + str(NUM_SAMPLES) + "_" + str(NUM_OPTIONS) + "options_part1"
#out_ds.save_to_disk(EVAL_DATA_NAME)


#for sample in tqdm(test_ds.select(range(NUM_SAMPLES//2, NUM_SAMPLES)), "Evaluation progress"):
for batch in tqdm(chunked(test_ds.select(range(NUM_SAMPLES//2, NUM_SAMPLES)), BATCH_SIZE), desc="Evaluation progress"):
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

out_ds = datasets.Dataset.from_list(final_records)
EVAL_DATA_NAME = "eval_outputs/baseline_" + MODEL + "_" + DATA + "_" + str(NUM_SAMPLES) + "_" + str(NUM_OPTIONS) + "options_part2"
out_ds.save_to_disk(EVAL_DATA_NAME)
