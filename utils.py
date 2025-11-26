from itertools import islice
from transformers import trainer_utils
import os
import re
from constants import MEDMCQA, POLITIFACT, GSM8K, MATH

# Define base number of options for each dataset (without IDK)
DATASET_OPTIONS = {
    MEDMCQA: 4,      # A-D
    POLITIFACT: 6,   # A-F
    GSM8K: 0, # No options
    MATH: 0, # No options, latex
    # Add more datasets here as needed
}

# TODO: remove if not used
def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def chunked(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

def resolve_checkpoint(resume_flag, output_dir):
    """Resolve checkpoint path from flag and output directory."""
    if isinstance(resume_flag, str):
        return resume_flag
    if resume_flag:
        try:
            return trainer_utils.get_last_checkpoint(output_dir)
        except Exception as e:
            return None
    return None


def extract_answer(completion):
    """
    Extract answer from completion using dynamically generated pattern.
    Pattern is based on current DATA and IDK_ENABLED settings from constants.
    """
    pattern = get_answer_pattern()
    DATA = os.getenv("DATA")
    match = re.search(pattern, completion)
    if match is None:
        return None
    if DATA == MATH:
        return extract_boxed_contents(match.group(1).strip())
    return match.group(1).strip().upper()


def get_answer_pattern():
    """
    Dynamically generate the answer pattern based on dataset and IDK configuration.
    Returns a regex pattern for extracting answers.
    """
    # Get number of options for current dataset from data.py
    DATA = os.getenv("DATA")
    IDK_ENABLED = os.getenv("IDK_ENABLED").strip().lower() in {"true"}
    num_options = DATASET_OPTIONS.get(DATA)
    if DATA == MATH:
        # The indestructible regex!
        pattern = rf"<answer>([\s\S]*)</answer>"
    elif num_options == 0:
        if IDK_ENABLED:
            pattern = r"<answer>(I Don't Know|-?[\d,]+)</answer>"
        else:
            pattern = rf"<answer>(-?[\d,]+)</answer>"
    else:
        if IDK_ENABLED:
            num_options += 1

        # Generate the pattern: A-D becomes A-D, A-E becomes A-E, etc.
        # chr(65) is 'A', so chr(65 + num_options - 1) gives the last letter
        last_letter = chr(65 + num_options - 1)
        last_letter_lower = last_letter.lower()

        pattern = rf"<answer>([A-{last_letter}a-{last_letter_lower}])</answer>"
    return pattern

def extract_boxed_contents(text: str) -> list[str]:
    """Return list of contents inside each \\boxed{...}, supports nested braces."""
    results = set()
    i = 0
    L = len(text)
    seq = r'\boxed{'
    while True:
        start = text.find(seq, i) # returns lowest index of 1st occurance
        if start == -1:
            break
        # position of the opening brace after '\boxed'
        j = start + len(seq) - 1  # points at '{'
        # scan forward to find matching '}' accounting for nested braces
        depth = 0
        k = j
        while k < L:
            ch = text[k]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    # content between j+1 and k-1 inclusive
                    content = text[j+1:k]
                    results.add(content.replace(" ", ""))
                    i = k + 1
                    break
            k += 1
        else:
            # no matching brace found; stop
            break
    return list(results)
