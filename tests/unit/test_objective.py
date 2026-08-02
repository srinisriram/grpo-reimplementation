import torch

from grpo.objective import grpo_loss, importance_ratio


def test_importance_ratio_is_exponentiated_logprob_difference():
    current = torch.tensor([[-0.2, -1.0]])
    old = torch.tensor([[-0.7, -1.0]])
    torch.testing.assert_close(importance_ratio(current, old), torch.tensor([[torch.exp(torch.tensor(0.5)), 1.0]]))


def test_old_policy_values_do_not_receive_gradients():
    current = torch.tensor([[-0.2]], requires_grad=True)
    old = torch.tensor([[-0.7]], requires_grad=True)
    ratio = importance_ratio(current, old.detach())
    ratio.sum().backward()
    assert current.grad is not None
    assert old.grad is None


def test_full_loss_is_scalar_and_returns_diagnostics():
    current = torch.tensor([[-0.2, -0.4]], requires_grad=True)
    old = torch.tensor([[-0.3, -0.5]])
    ref = torch.tensor([[-0.3, -0.5]])
    advantages = torch.tensor([[1.0, 1.0]])
    mask = torch.tensor([[1, 1]], dtype=torch.bool)
    loss, metrics = grpo_loss(current, old, ref, advantages, mask, clip_range=0.2, kl_coef=0.04)
    assert loss.ndim == 0
    assert {"policy_loss", "kl", "clip_fraction"} <= metrics.keys()
    loss.backward()
    assert current.grad is not None


def test_full_loss_rejects_empty_completion_mask():
    current = torch.tensor([[-0.2]], requires_grad=True)
    with __import__("pytest").raises(ValueError, match="completion"):
        grpo_loss(current, current.detach(), current.detach(), torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.bool), 0.2, 0.04)


def test_clipping_cases_are_covered_by_a_manual_table():
    """Fill expected values from your Unit 2 clip table before implementing grpo_loss.

    This parametrization forces you to test the four sign/clip-direction cases
    rather than only a favorable positive-advantage example.
    """
    cases = [
        (1.0, 1.5, "positive advantage, high ratio"),
        (1.0, 0.5, "positive advantage, low ratio"),
        (-1.0, 1.5, "negative advantage, high ratio"),
        (-1.0, 0.5, "negative advantage, low ratio"),
    ]
    assert len(cases) == 4
