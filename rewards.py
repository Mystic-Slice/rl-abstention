import logging
import re
from constants import LOGGING_FORMAT, DATE_FORMAT
from utils import extract_answer

logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.INFO,
    datefmt=DATE_FORMAT)
logger = logging.getLogger()

CORRECT_ANSWER_REWARD = 1
IDK_ANSWER_REWARD = 0
INCORRECT_ANSWER_REWARD = -1


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    rewards_list = [1.0 if match else -1.0 for match in matches]
    return rewards_list

# This method does not require to be modified even if idk_answer is not passed
def accuracy_reward(completions, correct_answer, idk_answer=None, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    answers = [extract_answer(comp) for comp in completion_contents]
    rewards = []

    for i, (ans, correct_ans) in enumerate(zip(answers, correct_answer)):
        if ans == correct_ans:
            rewards.append(CORRECT_ANSWER_REWARD)
        elif idk_answer and ans == idk_answer[i].upper():
            rewards.append(IDK_ANSWER_REWARD)
        else:
            rewards.append(INCORRECT_ANSWER_REWARD)

    format_rewards = format_reward(completions)
    for comp, reward, form_reward in zip(completion_contents, rewards, format_rewards):
        logger.info(comp)
        logger.info("Accuracy Reward: %s", reward)
        logger.info("Format Reward: %s", form_reward)
        logger.info("="*50)
    logger.info("Accuracy Rewards: %s", rewards)
    logger.info("Format Rewards: %s", format_rewards)
    return rewards
