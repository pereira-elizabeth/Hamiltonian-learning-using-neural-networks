# tests/test_utils.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.physics_utilities import eigensys, diagonalize, fidelity, ldos, ldos_map_from_pred

def test_fidelity_bounds_and_identity():
    a = np.array([1.0, -2.0, 3.0])
    b = 2.0 * a
    c = -b

    f_same = fidelity(a, b)
    f_opp  = fidelity(a, c)

    assert 0.999 <= f_same <= 1.0
    assert -1.0 <= f_opp  <= -0.999

def test_ldos_map_from_pred_shape_and_finite():
    N, f = 8, 64
    # simple linear onsite for a tiny chain
    y_pred = np.linspace(-0.8, 0.8, N)

    Z = ldos_map_from_pred(
        y_pred, N=N, f=f,
        eigensys=eigensys, ldos=ldos,
        col=0, freq_min=-1.0, freq_max=1.0
    )

    assert Z.shape == (f, N)
    assert np.all(np.isfinite(Z)), "LDOS map should have finite values"
