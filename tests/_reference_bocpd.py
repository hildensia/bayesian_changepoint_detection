"""Independent NumPy/scipy reference implementations used by the tests."""
import numpy as np
from scipy.stats import multivariate_t

def reference_mv_bocpd(X, lam, dof0, kappa0, mu0, W0):
    """Adams & MacKay with Normal-Wishart predictive, Murphy (2007) eqs 255-258.
    Precision Lambda ~ Wishart(W, nu); posterior W_n^{-1} = W^{-1} + kappa/(kappa+1) (x-mu)(x-mu)^T."""
    T, D = X.shape
    R = np.zeros((T + 1, T + 1)); R[0, 0] = 1.0
    mu = [mu0.copy()]; kappa = [kappa0]; nu = [dof0]; Winv = [np.linalg.inv(W0)]
    H = 1.0 / lam
    for t in range(T):
        x = X[t]
        pred = np.empty(t + 1)
        for r in range(t + 1):
            tdof = nu[r] - D + 1
            shape = Winv[r] * (kappa[r] + 1) / (kappa[r] * tdof)
            pred[r] = multivariate_t.pdf(x, loc=mu[r], shape=shape, df=tdof)
        R[1:t + 2, t + 1] = R[:t + 1, t] * pred * (1 - H)
        R[0, t + 1] = np.sum(R[:t + 1, t] * pred * H)
        R[:, t + 1] /= R[:, t + 1].sum()
        new_mu, new_kappa, new_nu, new_Winv = [mu0.copy()], [kappa0], [dof0], [np.linalg.inv(W0)]
        for r in range(t + 1):
            d = x - mu[r]
            new_mu.append((kappa[r] * mu[r] + x) / (kappa[r] + 1))
            new_kappa.append(kappa[r] + 1); new_nu.append(nu[r] + 1)
            new_Winv.append(Winv[r] + kappa[r] / (kappa[r] + 1) * np.outer(d, d))
        mu, kappa, nu, Winv = new_mu, new_kappa, new_nu, new_Winv
    return R
