# tests/test_noise_sweep.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.ml_models import create_model
from src.physics_utilities import adding_noise, fidelity
from src.fidelity_vs_noise_study import predict_on_noise

def test_fidelity_v_noise_minimal_run():
    rng = np.random.default_rng(0)
    # tiny synthetic dataset
    X_train = rng.normal(size=(64, 12)).astype("float32")
    y_train = rng.normal(size=(64, 1)).astype("float32")
    X_test  = rng.normal(size=(32, 12)).astype("float32")
    y_test  = rng.normal(size=(32, 1)).astype("float32")

    noise_vals, Ftr, Fte y_pred = predict_on_noise(
        0.0, 0.1,
        X_train, y_train, X_test, y_test,
        n_points=3, epochs=1, batch_size=16,
        create_model=create_model,
        adding_noise=adding_noise,
        fidelity=fidelity
    )

    assert noise_vals.shape == (3,)
    assert Ftr.shape == (3,) and Fte.shape == (3,)
    assert np.all(np.isfinite(Ftr)) and np.all(np.isfinite(Fte))
    # fidelity must lie in [-1, 1]
    assert np.all(Ftr <= 1.0) and np.all(Ftr >= -1.0)
    assert np.all(Fte <= 1.0) and np.all(Fte >= -1.0)
