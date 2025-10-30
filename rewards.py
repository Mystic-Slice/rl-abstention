import logging
import re

logging.basicConfig(
    format=LOGGING_FORMAT,
    level=logging.INFO,
    datefmt=DATE_FORMAT)
logger = logging.getLogger()


def extract_answer(completion):
    match = re.search(r"<answer>([A-Ga-g])<\answer>", completion)
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
