import torch
from scipy.special import erf
from scipy.linalg import orth
import numpy as np


def sample(codeword, basis=None):
    # pseudogaussian = codeword * torch.abs(torch.randn_like(codeword, dtype=torch.float64))
    codeword_np = codeword.numpy()
    pseudogaussian_np = codeword_np * np.abs(np.random.randn(*codeword_np.shape))
    pseudogaussian = torch.from_numpy(pseudogaussian_np).to(dtype=torch.float64)
    if basis is None:
        return pseudogaussian
    return pseudogaussian @ basis.T


def recover_posteriors(z, basis=None, variances=None):
    if variances is None:
        default_variance = 1.5
        denominators = np.sqrt(2 * default_variance * (1+default_variance)) * torch.ones_like(z)
    elif type(variances) is float:
        denominators = np.sqrt(2 * variances * (1 + variances))
    else:
        denominators = torch.sqrt(2 * variances * (1 + variances))

    if basis is None:
        return erf(z / denominators)
    else:
        return erf((z @ basis) / denominators)

def random_basis(n):
    gaussian = torch.randn(n, n, dtype=torch.double)
    return orth(gaussian)

### Gaussian pancakes
def rho(z, s=1.):
    return np.exp(-np.pi * z**2 / s**2)

def discrete_gaussian(s=1., bound=1, shift=0.):
    bound *= s
    ks = np.arange(-bound, bound+1, dtype=int) + shift
    ps = rho(ks, s)
    ps /= ps.sum()
    return np.random.choice(ks, p=ps)

def noisy_discrete_gaussian(beta, gamma, shift=0.):
    k = discrete_gaussian(s=np.sqrt(beta**2+gamma**2), shift=shift)
    return np.random.normal(k * gamma / (beta**2+gamma**2), beta/np.sqrt(2*np.pi*(beta**2+gamma**2)))

def pdf_noisy_discrete_gaussian(beta, gamma, z, bound=1, shift=0.):
    bound *= np.sqrt(beta**2+gamma**2)
    ks = torch.arange(-bound, bound+1, dtype=int) + shift
    return rho(ks - gamma * z, beta).sum()

### You can set gamma = 2 sqrt(n) and beta = 1/sqrt(n)
def pancakes(W, beta, gamma, shifts=None):
    if shifts is None:
        shifts = np.zeros(W.shape[0])
    return torch.tensor([noisy_discrete_gaussian(beta, gamma, shifts[i]) for i in range(W.shape[0])]) @ W

def decode_pancakes(W, beta, gamma, z, two_shifts=None):
    Wz = torch.tensor(W) @ torch.tensor(z)
    bits = []
    for zi in Wz:
        szi = float(zi * gamma)
        score0 = abs(round(szi-two_shifts[0]) - (szi-two_shifts[0]))
        score1 = abs(round(szi-two_shifts[1]) - (szi-two_shifts[1]))
        bits.append(np.argmin([score0,score1]))
        # p0 = pdf_noisy_discrete_gaussian(beta, gamma, zi, shift=two_shifts[0])
        # p1 = pdf_noisy_discrete_gaussian(beta, gamma, zi, shift=two_shifts[1])
        # bits.append(np.argmax([p0,p1]))
    return bits


