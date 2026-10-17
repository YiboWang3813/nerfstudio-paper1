from types import SimpleNamespace

import pytest
import torch

from methods.ours.density_control import (
    OursOfflineDensityController,
    attenuation_protection_mask,
    transmission_from_tau,
)


def test_lower_tau_yields_lower_transmission_for_fixed_overlap():
    tau = torch.tensor([0.2, 0.8])
    density = torch.tensor([1.5, 1.5])
    transmission = transmission_from_tau(tau, density)
    assert transmission[0] < transmission[1]


def test_p1_high_tau_protects_weak_attenuators():
    tau = torch.tensor([0.1, 0.2, 0.8, 0.9])
    mask, _ = attenuation_protection_mask(tau, mode="high_tau", quantile=0.75)
    assert mask.tolist() == [False, False, False, True]


def test_p2_low_tau_protects_strong_attenuation_potential():
    tau = torch.tensor([0.1, 0.2, 0.8, 0.9])
    mask, _ = attenuation_protection_mask(tau, mode="low_tau", quantile=0.75)
    assert mask.tolist() == [True, False, False, False]


def test_p3_path_effect_prefers_supported_strong_attenuator():
    tau = torch.tensor([0.1, 0.1, 0.95])
    support = torch.tensor([1.0, 10.0, 20.0])
    mask, stats = attenuation_protection_mask(tau, mode="path_effect", quantile=0.67, support=support)
    assert mask.tolist() == [False, True, False]
    assert stats["path_effect_score_max"] > stats["path_effect_score_mean"]


def test_p3_preserves_weak_local_large_path_effect_gaussian():
    tau = torch.tensor([0.1, 0.1, 0.95])
    support = torch.tensor([1.0, 10.0, 20.0])
    path_protected, _ = attenuation_protection_mask(tau, mode="path_effect", quantile=0.67, support=support)
    weak_local = torch.tensor([True, True, True])
    low_contribution = torch.tensor([True, True, True])

    prune = weak_local & low_contribution & ~path_protected

    assert path_protected[1]
    assert not prune[1]


def test_p3_prunes_weak_local_negligible_path_effect_gaussian():
    tau = torch.tensor([0.1, 0.1, 0.95])
    support = torch.tensor([1.0, 10.0, 20.0])
    path_protected, _ = attenuation_protection_mask(tau, mode="path_effect", quantile=0.67, support=support)
    weak_local = torch.tensor([True, True, True])
    low_contribution = torch.tensor([True, True, True])

    prune = weak_local & low_contribution & ~path_protected

    assert not path_protected[0]
    assert prune[0]


def test_path_contribution_quantile_uses_ag_not_tau():
    tau = torch.tensor([0.99, 0.75, 0.25, 0.01])
    path = torch.tensor([0.0, 1.0, 2.0, 10.0])
    mask, stats = attenuation_protection_mask(
        tau,
        mode="path_contribution",
        quantile=0.75,
        path_contribution=path,
        path_threshold_mode="quantile",
    )
    assert mask.tolist() == [False, False, False, True]
    assert stats["path_contribution_available"] == 1.0

    reversed_tau_mask, _ = attenuation_protection_mask(
        tau.flip(0),
        mode="path_contribution",
        quantile=0.75,
        path_contribution=path,
        path_threshold_mode="quantile",
    )
    assert torch.equal(mask, reversed_tau_mask)


def test_path_contribution_fixed_threshold():
    tau = torch.tensor([0.2, 0.4, 0.6, 0.8])
    path = torch.tensor([0.0, 1.0, 2.0, 10.0])
    mask, stats = attenuation_protection_mask(
        tau,
        mode="path_contribution",
        quantile=0.5,
        path_contribution=path,
        path_threshold_mode="fixed",
        path_threshold=2.0,
    )
    assert mask.tolist() == [False, False, True, True]
    assert stats["attenuation_protect_threshold"] == 2.0


def test_path_contribution_missing_stats_fails_safe():
    tau = torch.tensor([0.2, 0.4, 0.6])
    mask, stats = attenuation_protection_mask(
        tau,
        mode="path_contribution",
        quantile=0.8,
        path_contribution=None,
    )
    assert mask.tolist() == [True, True, True]
    assert stats["path_contribution_available"] == 0.0


def test_path_contribution_invalid_stats_fail_safe():
    tau = torch.tensor([0.2, 0.4, 0.6])
    path = torch.tensor([0.0, float("nan"), float("inf")])
    mask, _ = attenuation_protection_mask(
        tau,
        mode="path_contribution",
        quantile=0.8,
        path_contribution=path,
    )
    assert mask[1]
    assert mask[2]


def test_path_accumulator_detects_population_change_within_interval():
    dc = object.__new__(OursOfflineDensityController)
    dc.resp_path_contribution = torch.empty(0)
    dc.resp_path_support = torch.empty(0)
    dc.path_frames_accumulated = 0
    container = SimpleNamespace(num_points=3, device=torch.device("cpu"))
    dc.update_path_stats(container, torch.tensor([1.0, 2.0, 3.0]), torch.ones(3))
    dc.update_path_stats(container, torch.tensor([0.5, 0.5, 0.5]), torch.ones(3))
    assert dc.path_frames_accumulated == 2
    assert torch.allclose(dc.resp_path_contribution, torch.tensor([1.5, 2.5, 3.5]))

    container.num_points = 4
    with pytest.raises(RuntimeError, match="population changed"):
        dc.update_path_stats(container, torch.ones(4), torch.ones(4))
