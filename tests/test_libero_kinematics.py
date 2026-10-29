"""EE path model check (plan P0(c)): the configured lag model must predict where the LIBERO arm goes better
than VLS's constant model, on chunks it was not fitted on.

This is a regression guard, not the accuracy gate: the absolute target (< 1 cm, < 5 deg at chunk end) is
checked by scripts/calibrate_ee_model.py, on random actions and on real rollouts (--from_logs). On random
actions the lag model reached ~1.6 cm vs ~2.0 cm for constant (2026-10-27). Slow (builds a LIBERO env).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from calibrate_ee_model import evaluate, make_adapter  # noqa: E402


@pytest.fixture(scope="module")
def adapter():
    return make_adapter("libero_object")


def test_lag_model_beats_constant_on_held_out_chunks(adapter):
    models = {'constant': {'ee_prediction': 'constant'}, 'lag': {'ee_prediction': 'lag'}}
    errors = evaluate(adapter, np.random.default_rng(123), chunks=15, model_cfgs=models)
    print(f"\nchunk-end error (median): constant {errors['constant']['pos_cm']['median']:.2f} cm / "
          f"{errors['constant']['rot_deg']['median']:.1f} deg, lag {errors['lag']['pos_cm']['median']:.2f} cm / "
          f"{errors['lag']['rot_deg']['median']:.1f} deg")
    assert errors['lag']['pos_cm']['median'] < errors['constant']['pos_cm']['median']
    assert errors['lag']['rot_deg']['median'] < errors['constant']['rot_deg']['median']
