LOGGING_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


TRAIN = "train"
VAL = "validation"
TEST = "test"
BASELINE = "baseline"
SFT = "sft"
RL = "rl"
SFT_RL = "sft_rl"
GRANITE = "ibm-granite/granite-3.3-2b-instruct"
QWEN = "qwen/qwen3-4B-Instruct-2507"
LORA = "lora"
FULL = "full"
IDK = "I Don't Know"
MEDMCQA = "medmcqa"
MEDMCQA_DATA = "openlifescienceai/medmcqa" # used in data.py only
POLITIFACT = "politifact"
POLITIFACT_DATA = "rmisra/politifact-fact-check-dataset" # used in data.py only
POLITIFACT_FILE_NAME = "/politifact_factcheck_data.json" # used in data.py only
GSM8K = "gsm8k"
GSM8K_DATA = "openai/gsm8k"  # used in data.py only
PROMPT = "prompt"
ANSWER = "correct_answer"
