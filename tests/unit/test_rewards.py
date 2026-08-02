from grpo.rewards import extract_final_answer, gsm8k_reward


def test_extract_final_answer_uses_last_boxed_answer():
    assert extract_final_answer("work \\boxed{3}; correction \\boxed{4}") == "4"


def test_reward_accepts_exact_final_answer_only():
    assert gsm8k_reward("Answer: \\boxed{42}", "reasoning #### 42") == 1.0
    assert gsm8k_reward("Answer: 42", "reasoning #### 42") == 0.0

