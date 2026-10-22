import torch

from nicheib.gates import HardConcreteGate, HardConcreteSampler, budget_loss


def test_hard_concrete_gate_shapes_and_ranges():
    gate = HardConcreteGate(8)
    sample = gate()
    expected = gate.expected_l0()
    assert sample.shape == (8,)
    assert expected.shape == (8,)
    assert torch.all(sample >= 0)
    assert torch.all(sample <= 1)
    assert torch.all(expected >= 0)
    assert torch.all(expected <= 1)


def test_budget_loss_is_finite():
    gate = HardConcreteGate(8)
    loss = budget_loss(gate.expected_l0(), target_rate=0.2)
    assert torch.isfinite(loss)


def test_hc_sampler_shapes_and_ranges():
    sampler = HardConcreteSampler()
    logits = torch.randn(100)
    mask, el0 = sampler(logits)
    assert mask.shape == (100,)
    assert el0.shape == (100,)
    assert torch.all(mask >= 0)
    assert torch.all(mask <= 1)
    assert torch.all(el0 >= 0)
    assert torch.all(el0 <= 1)


def test_hc_sampler_produces_exact_zeros():
    """Hard Concrete should produce exact zeros for strongly negative logits."""
    sampler = HardConcreteSampler()
    logits = torch.full((1000,), -10.0)
    sampler.train()
    mask, _ = sampler(logits)
    # Most gates should be exactly zero when logits are very negative
    assert (mask == 0.0).sum() > 800


def test_hc_sampler_deterministic_mode():
    sampler = HardConcreteSampler()
    logits = torch.randn(50)
    sampler.eval()
    mask1, el0_1 = sampler(logits)
    mask2, el0_2 = sampler(logits)
    # Deterministic mode: same logits → same output
    assert torch.allclose(mask1, mask2)
    assert torch.allclose(el0_1, el0_2)


def test_hc_sampler_expected_l0_approximates_open_rate():
    """Expected L0 should approximate the empirical fraction of open gates."""
    sampler = HardConcreteSampler()
    logits = torch.zeros(1000)  # 50% prior
    sampler.train()
    n_trials = 200
    open_counts = torch.zeros(1000)
    for _ in range(n_trials):
        mask, _ = sampler(logits)
        open_counts += (mask > 0).float()
    empirical_rate = open_counts / n_trials
    _, el0 = sampler(logits)
    # Tolerance: within 0.1 on average
    assert (empirical_rate - el0).abs().mean() < 0.1


def test_hc_sampler_budget_loss_integration():
    """Budget loss should work with HardConcreteSampler expected_l0."""
    sampler = HardConcreteSampler()
    logits = torch.randn(50)
    el0 = sampler.expected_l0(logits)
    loss = budget_loss(el0, target_rate=0.2, mode="barrier")
    assert torch.isfinite(loss)
