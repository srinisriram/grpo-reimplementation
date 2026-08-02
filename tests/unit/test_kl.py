import torch

from grpo.objective import kl_estimator


def test_kl_is_zero_for_identical_policies():
    logps = torch.tensor([[-1.0, -2.0]])
    torch.testing.assert_close(kl_estimator(logps, logps), torch.zeros_like(logps))


def test_kl_estimator_is_nonnegative():
    current = torch.tensor([[-0.5, -3.0]])
    reference = torch.tensor([[-1.2, -2.0]])
    assert (kl_estimator(current, reference) >= 0).all()

