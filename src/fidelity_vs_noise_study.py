import numpy as np

def predict_on_noise(min_noise, max_noise, Xtr, Ytr, Xte, Yte,
                     *, n_points=10, epochs=20, batch_size=16,
                     create_model, adding_noise, fidelity):
    """Return arrays: noise_vals, F_train, F_test (no plotting)."""
    noise_vals = np.linspace(min_noise, max_noise, n_points)
    F_train, F_test = [], []

    for s in noise_vals:
        Xtr_n, Ytr_n = adding_noise(Xtr, Ytr, s)
        Xte_n, Yte_n = adding_noise(Xte, Yte, s)

        model = create_model(input_shape=Xtr_n.shape[1:])
        model.compile(optimizer="adam", loss="mse")
        model.fit(Xtr_n, Ytr_n, epochs=epochs, batch_size=batch_size,
                  validation_split=0.2, verbose=0)

        Ytr_p = model.predict(Xtr_n, verbose=0)
        Yte_p = model.predict(Xte_n, verbose=0)

        # cast to scalar if fidelity returns an array
        F_train.append(float(np.asarray(fidelity(Ytr_p, Ytr_n)).mean()))
        F_test.append(float(np.asarray(fidelity(Yte_p, Yte_n)).mean()))
        

    return noise_vals, np.asarray(F_train), np.asarray(F_test), Yte_p
