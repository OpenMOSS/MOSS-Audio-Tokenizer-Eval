import numpy as np
import pytest

from moss_eval.metrics.core import _sisdr_value


def _reference_sisdr(ref, syn):
    alpha = np.dot(syn, ref) / np.dot(ref, ref)
    target = alpha * ref
    return 10 * np.log10(np.sum(target ** 2) / np.sum((syn - target) ** 2))


@pytest.mark.parametrize("snr_db", [-10.0, -5.0, 0.0, 10.0, 30.0])
def test_sisdr_matches_definition_for_additive_noise(snr_db):
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000)
    noise = rng.standard_normal(16000)
    noise *= np.sqrt(np.sum(ref ** 2) / (np.sum(noise ** 2) * 10 ** (snr_db / 10)))
    syn = ref + noise
    assert _sisdr_value(ref, syn) == pytest.approx(_reference_sisdr(ref, syn), abs=1e-4)


def test_sisdr_is_scale_invariant_and_negative_for_unrelated_signal():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(16000)
    syn = ref + 2.0 * rng.standard_normal(16000)
    assert _sisdr_value(ref, 0.25 * syn) == pytest.approx(_sisdr_value(ref, syn), abs=1e-9)
    unrelated = rng.standard_normal(16000)
    assert _sisdr_value(ref, unrelated) < -20.0
