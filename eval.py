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


logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.INFO,
    datefmt=DATE_FORMAT)

logger = logging.getLogger()

MODEL = GRANITE
DATA = MEDMCQA
EVAL_TYPE = BASELINES
EVAL_ON = TEST
NUM_SAMPLES = 40000
NUM_OPTIONS = 4
BATCH_SIZE = 32
LOAD_SPECIFIC_MODEL = False
MODEL_CHECKPOINT_NAME = "sft_rl_medmcqa_abstention_qwen_chk300_model_idk_plus_0/checkpoint-90"

def chunked(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

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

logger.info(pipe.device)

match DATA:
    case constants.MEDMCQA:
        ds = get_medmcqa_data()
    case constants.POLITIFACT:
        ds = get_politifact_data()
    case _:
        logger.error("Please select valid dataset")

logger.info(ds[EVAL_ON])

train_ds = ds[EVAL_ON]
final_records = []

#for sample in tqdm(test_ds.select(range(NUM_SAMPLES//2)), "Evaluation progress"):
for batch in tqdm(chunked(train_ds.select(range(NUM_SAMPLES)), BATCH_SIZE), desc="Evaluation progress"):
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
EVAL_DATA_NAME = "eval_outputs/"+ EVAL_TYPE + "_" + MODEL + "_" + DATA + "_" + str(NUM_SAMPLES) + "_" + str(NUM_OPTIONS) + "option"
out_ds.save_to_disk(EVAL_DATA_NAME)
