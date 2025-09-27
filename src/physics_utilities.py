import numpy as np
import scipy
from scipy import linalg

def eigensys(v1): #Hamiltonian of the system of a 1D chain with random real onsite energies
    N = len(v1)
    H1 = np.diag(v1) + np.diag(np.ones(N-1), k=1) + np.diag(np.ones(N-1), k=-1)
    return H1
  
def diagonalize(H):
    eigs, vecs = linalg.eigh(H)
    return eigs, vecs
  
def ldos(w1, vl1, freq): #spectral density of the system
    dos = np.imag(np.sum(np.conj(vl1)*vl1 / (freq - w1 + 1j*0.05), axis = 1))
    dspp = -dos / np.pi
    return dspp

def visualize_HAM(H_):  #A 2d plot of the real and imaginary parts of the Hamiltonian
    if H_.size < 17:
        print("real:")
        print(H_.real)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    c0 = ax[0].imshow(H_.real)
    ax[0].set_title("Real part")
    fig.colorbar(c0, ax=ax[0])  

# Generate input data
def input_data(N,f,l):   #Function to generate the input data.
    X0 = np.zeros((N * l, f), dtype=np.float64)  #the spectral density of states of the Hamiltonian
    Y0 = np.zeros((N * l,1), dtype=np.float64)  #the parameters of the Hamiltonian that are to be predicted

    for i in range(l):
        v1 = np.random.uniform(0, 1, N) #random real onsite energy
        H1 = eigensys(v1)  #Hamiltonian of the 1D chain with 8 sites
        w, vl1 = linalg.eigh(H1)  #eigensystem 
        for ip1, freq in enumerate(np.linspace(-2., 2., f)):
            dspp = ldos(w, vl1, freq)  #spectral density for a grid of frequencies for the Hamiltonian
            X0[i * N: (i + 1) * N, ip1] = dspp
        Y0[i * N: (i + 1) * N,0] = v1  #the output vector is saved as (v1) i.e. (real potential)
    return X0, Y0

def adding_noise(p1, q1, noise_std):  #adding noise to the data from normal distribution
    noise1 = np.random.normal(0, noise_std, size=p1.shape)
    noise2 = np.random.normal(0, noise_std, size=q1.shape)
    p2 = np.concatenate([p1, p1 + noise1], axis=0)
    q2 = np.concatenate([q1, q1 + noise2], axis=0)
    indices = np.arange(len(p2))
    np.random.shuffle(indices)
    return p2[indices], q2[indices]

def fidelity(y_p, y_t): #fidelity i.e. correlation between true and predicted values
    ar = np.array(y_p)
    br = np.array(y_t)
    cr = ar * br
    F = (np.mean(cr) - np.mean(ar) * np.mean(br)) / (
        (np.mean(np.square(ar)) - np.square(np.mean(ar))) *
        (np.mean(np.square(br)) - np.square(np.mean(br)))
        ) ** 0.5    
    return F

import numpy as np
from scipy import linalg

def ldos_map_from_pred(y_pred, N, f, eigensys, ldos,
                       *, col=0, freq_min=-2.0, freq_max=2.0, dtype=np.float64):
    """
    Compute LDOS map Z(f, N) from predicted onsite values.

    Parameters
    ----------
    y_pred : array
        Predicted onsite values; 1D or 2D. If 2D, takes first N rows from column `col`.
    N : int
        Number of sites.
    f : int
        Number of frequency points.
    eigensys : callable
        Function that builds the Hamiltonian: H = eigensys(v).
    ldos : callable
        LDOS function: ldos(eigs, vecs, omega) -> (N,).
    col : int
        Column index if y_pred is 2D. Default 0.
    freq_min, freq_max : float
        Frequency grid range.
    dtype : numpy dtype
        dtype for arrays.

    Returns
    -------
    Z : (f, N) array
        LDOS map (rows = frequencies, cols = sites).
    """
    yp = np.asarray(y_pred)
    v = yp[:N, col] if yp.ndim == 2 else yp[:N]
    v = np.asarray(v, dtype=dtype).squeeze()

    H = eigensys(v)
    w, V = linalg.eigh(H)

    freqs = np.linspace(freq_min, freq_max, f, dtype=dtype)
    Z = np.zeros((f, N), dtype=dtype)
    for i, om in enumerate(freqs):
        Z[i, :] = ldos(w, V, om)

    return Z
