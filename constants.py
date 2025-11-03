LOGGING_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

TRAIN = "train"
VAL = "validation"
TEST = "test"
BASELINES = "baselines"
SFT = "sft"
RL = "rl"
SFT_RL = "sft_rl"
GRANITE = "ibm-granite/granite-3.3-2b-instruct"
QWEN = "qwen/qwen3-4B-Instruct-2507"
IDK = "I Don't Know"
MEDMCQA = "medmcqa"
MEDMCQA_DATA = "openlifescienceai/medmcqa" # used in data.py only
POLITIFACT = "politifact"
POLITIFACT_DATA = "rmisra/politifact-fact-check-dataset" # used in data.py only
POLITIFACT_FILE_NAME = "/politifact_factcheck_data.json" # used in data.py only


# This will be set by eval.py or train.py
# Default values shown here
DATA = MEDMCQA  # Current dataset being used
IDK_ENABLED = False  # Whether IDK option is enabled