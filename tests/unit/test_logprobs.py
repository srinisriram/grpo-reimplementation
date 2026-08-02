import torch

from grpo.objective import completion_logprobs


def test_completion_logprobs_gather_shifted_targets():
    logits = torch.tensor([[[0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [1.0, 2.0, 0.0]]])
    input_ids = torch.tensor([[0, 2, 0]])
    actual = completion_logprobs(logits, input_ids)
    expected = torch.log_softmax(logits[:, :-1], dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected)


def test_completion_logprobs_preserves_current_policy_gradient():
    logits = torch.randn(2, 4, 5, requires_grad=True)
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    completion_logprobs(logits, input_ids).sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

