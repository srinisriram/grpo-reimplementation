import pytest
import torch

from grpo.objective import group_relative_advantages


def test_advantages_are_group_centered_and_normalized():
    rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    advantages = group_relative_advantages(rewards, eps=0.0)
    expected = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(advantages.mean(dim=1), torch.zeros(1))


def test_advantages_are_independent_per_prompt_group():
    rewards = torch.tensor([[0.0, 1.0], [10.0, 8.0]])
    advantages = group_relative_advantages(rewards, eps=0.0)
    torch.testing.assert_close(advantages, torch.tensor([[-1.0, 1.0], [1.0, -1.0]]))


def test_zero_variance_group_is_finite_and_documented_behavior():
    advantages = group_relative_advantages(torch.ones(1, 4))
    assert torch.isfinite(advantages).all()
    torch.testing.assert_close(advantages, torch.zeros_like(advantages))

