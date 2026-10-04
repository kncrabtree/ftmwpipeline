"""The window_model epoch guard for unreproducible frozen skirts (ROADMAP D18).

Before analysis epoch 4 a free-tau window with frozen contributors held their
skirt at an unrecorded starting tau, so ``window_model`` must refuse it rather
than draw a model the fit never minimised. The guard is unit-tested on
constructed context / fit objects; it reads only ``ctx.fit_epoch``,
``wf.fixed_parameters`` and ``wf.shared_parameters``.
"""

from types import SimpleNamespace

import pytest

from ftmwpipeline._internal.model_impl import (
    FROZEN_SKIRT_FOLLOWS_TAU_EPOCH,
    _require_reproducible_frozen_skirt,
)
from ftmwpipeline.file_manager import IncompleteProvenanceError
from ftmwpipeline.fitting.model_eval import FROZEN_PEAK_PREFIX

pytestmark = [pytest.mark.unit]


def _wf(*, frozen: bool, tau_fitted):
    fixed = {f"{FROZEN_PEAK_PREFIX}0": {"amplitude": 1.0}} if frozen else {}
    tau = {} if tau_fitted is None else {"fitted": tau_fitted}
    return SimpleNamespace(fixed_parameters=fixed, shared_parameters={"tau_us": tau})


def _ctx(epoch):
    return SimpleNamespace(fit_epoch=epoch)


def test_epoch_3_frozen_free_tau_is_refused():
    # Catches: guard removed or made a no-op (an unreproducible model is drawn).
    with pytest.raises(IncompleteProvenanceError) as exc:
        _require_reproducible_frozen_skirt(
            _ctx(3), _wf(frozen=True, tau_fitted=True), 7
        )
    assert exc.value.missing == ["stage5_fitting.window[7].frozen_skirt_tau_us"]


def test_epoch_4_frozen_free_tau_is_accepted():
    # Catches: epoch comparison dropped or inverted (a current file refused).
    assert FROZEN_SKIRT_FOLLOWS_TAU_EPOCH == 4
    _require_reproducible_frozen_skirt(_ctx(4), _wf(frozen=True, tau_fitted=True), 7)
    _require_reproducible_frozen_skirt(_ctx(5), _wf(frozen=True, tau_fitted=True), 7)


def test_epoch_3_fixed_tau_is_accepted():
    # Catches: tau_free test dropped (a held-tau window is bit-identical under
    # any epoch and must stay readable).
    _require_reproducible_frozen_skirt(_ctx(3), _wf(frozen=True, tau_fitted=False), 7)


def test_epoch_3_without_frozen_contributors_is_accepted():
    # Catches: has_frozen test dropped (every old free-tau window refused).
    _require_reproducible_frozen_skirt(_ctx(3), _wf(frozen=False, tau_fitted=True), 7)


def test_unrecorded_epoch_with_frozen_free_tau_is_refused():
    # Catches: a missing environment (epoch None) treated as current.
    with pytest.raises(IncompleteProvenanceError):
        _require_reproducible_frozen_skirt(
            _ctx(None), _wf(frozen=True, tau_fitted=True), 7
        )
