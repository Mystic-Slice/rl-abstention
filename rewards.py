import re

def extract_answer(completion):
    match = re.search(r"<answer>([A-Ea-e])<\answer>", completion)
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
def accuracy_reward(completions, correct_option, idk_option, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    answers = [extract_answer(comp) for comp in completion_contents]
    rewards = []
    
    for (ans, correct_ans, idk_op) in zip(answers, correct_option, idk_option):
        if ans == correct_ans:
            rewards.append(1)
        elif ans == idk_op:
            rewards.append(0)
        else:
            rewards.append(-1)

    format_rewards = format_reward(completions)
    for comp, reward, form_reward in zip(completion_contents, rewards, format_rewards):
        print(comp)
        print("Accuracy Reward: ", reward)
        print("Format Reward: ", form_reward)
        print("="*50)
    print("Accuracy Rewards: ", rewards)
    print("Format Rewards: ", format_rewards)
    return rewards
