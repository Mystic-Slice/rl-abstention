# Rewarding Intellectual Humility: Learning When Not to Answer in Large Language Models

## Datasets used:
MEDMCQA: https://huggingface.co/datasets/openlifescienceai/medmcqa

GSMK8: https://huggingface.co/datasets/openai/gsm8k

Hendrycks Math: https://huggingface.co/datasets/EleutherAI/hendrycks_math


## Code architecture
train.py: Used for Training RL or SFT by loading model

eval.py: Used for evaluation of model (base model or SFT-tuned model or RL-model)

rewards.py: Used for RL with idk reward weight, reward systems defined - format reward and accuracy reward

data.py: Used for loading the datasets

utils.py: Utility functions for extracting answers or resolving checkpoints


## Steps to execute:
### training
```bash
pip -r install requirements.txt
# Set appropriate settings in train.py
# Modify job.sh as per the availability and include "python -u train.py" line at the bottom
sbatch job.sh
```
### evaluation
```bash
pip -r install requirements.txt
# Set appropriate settings in eval.py
# Modify job.sh as per the availability and include "python -u eval.py" line at the bottom
sbatch job.sh
```
