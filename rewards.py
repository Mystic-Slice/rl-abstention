import logging
import re
from constants import LOGGING_FORMAT, DATE_FORMAT, DATA, IDK_ENABLED
from data import DATASET_OPTIONS

logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.INFO,
    datefmt=DATE_FORMAT)
logger = logging.getLogger()


def get_answer_pattern():
    """
    Dynamically generate the answer pattern based on dataset and IDK configuration.
    Returns a regex pattern for extracting answers.
    """
    # Get number of options for current dataset from data.py
    num_options = DATASET_OPTIONS.get(DATA, 4)  # Default to 4 if dataset not found

    # Add 1 if IDK is enabled
    if IDK_ENABLED:
        num_options += 1

    # Generate the pattern: A-D becomes A-D, A-E becomes A-E, etc.
    # chr(65) is 'A', so chr(65 + num_options - 1) gives the last letter
    last_letter = chr(65 + num_options - 1)
    last_letter_lower = last_letter.lower()

    pattern = rf"<answer>([A-{last_letter}a-{last_letter_lower}])</answer>"
    logger.debug(f"Using answer pattern for {DATA} (IDK={IDK_ENABLED}): {pattern}")

    return pattern


def extract_answer(completion):
    """
    Extract answer from completion using dynamically generated pattern.
    Pattern is based on current DATA and IDK_ENABLED settings from constants.
    """
    pattern = get_answer_pattern()
    match = re.search(pattern, completion)
    if match is not None:
        return match.group(1).strip().upper()
    return None

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    rewards_list = [1.0 if match else -1.0 for match in matches]
    return rewards_list

# mmlu
def accuracy_reward(completions, correct_option, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    answers = [extract_answer(comp) for comp in completion_contents]
    rewards = []
    
    for (ans, correct_ans) in zip(answers, correct_option):
        if ans == correct_ans:
            rewards.append(1)
        else:
            rewards.append(-1)

    format_rewards = format_reward(completions)
    for comp, reward, form_reward in zip(completion_contents, rewards, format_rewards):
        logger.debug(comp)
        logger.debug("Accuracy Reward: %s", reward)
        logger.debug("Format Reward: %s", form_reward)
        logger.debug("="*50)
    logger.info("Accuracy Rewards: %s", rewards)
    logger.info("Format Rewards: %s", format_rewards)
    return rewards
