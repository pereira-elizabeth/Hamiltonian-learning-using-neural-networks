# tests/test_models.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from src.ml_models import create_model, create_model_prevent_overfitting

def _forward_pass_ok(make_model):
    X = np.random.randn(5, 10).astype("float32")
    model = make_model(input_shape=X.shape[1:])
    model.compile(optimizer="adam", loss="mse")
    y = model.predict(X, verbose=0)
    assert y.shape[0] == X.shape[0]
    assert np.all(np.isfinite(y))

def test_create_model_forward():
    _forward_pass_ok(create_model)

def test_create_model_prevent_overfitting_forward():
    _forward_pass_ok(create_model_prevent_overfitting)
