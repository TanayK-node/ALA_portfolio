"""Independent numerical audit of Proposition 1 (does not import the repo's theory/simulate modules)."""
import numpy as np
from scipy import stats
rng = np.random.default_rng(20261102)

def draw(N, T, n, Sigma):
    L = np.linalg.cholesky(Sigma)
    out = np.empty((n, 3))
    one = np.ones(N)
    V = 1 / (one @ np.linalg.solve(Sigma, one))
    for lo in range(0, n, 2000):
        b = min(2000, n - lo)
        X = rng.standard_normal((b, T, N)) @ L.T + rng.standard_normal(N)       # arbitrary mean
        Xc = X - X.mean(1, keepdims=True)
        S = Xc.transpose(0, 2, 1) @ Xc / (T - 1)
        y = np.linalg.solve(S, np.ones((b, N, 1)))[..., 0]
        s = y.sum(1); w = y / s[:, None]
        Q = (1 / s) / V
        R = np.einsum('bi,ij,bj->b', w, Sigma, w) / V
        Xte = rng.standard_normal((b, 60, N)) @ L.T
        G = np.einsum('bhi,bi->bh', Xte, w).var(1, ddof=1) / np.einsum('bi,ij,bj->b', w, Sigma, w)  # chi2_{59}/59 if Gaussian
        out[lo:lo + b] = np.column_stack([Q, R, G])
    return out

rows = []
for N, T in [(10, 60), (30, 60), (30, 120), (30, 250), (50, 120)]:
    A = rng.standard_normal((N, N)); Sigma = A @ A.T / N + 0.2 * np.eye(N)     # random non-diagonal Sigma
    n = 60000
    d = draw(N, T, n, Sigma); Q, R, G = d.T
    EQ, ER = (T - N) / (T - 1), (T - 2) / (T - N - 1)
    c = (T - 1) * (T - 2) / ((T - N) * (T - N - 1))
    cq = (T - 1) * (T - 2) / ((T - N - 1) * (T - N - 2))             # E[R/Q]
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    ks_q = stats.kstest((T - 1) * Q, 'chi2', args=(T - N,)).pvalue
    ks_r = stats.kstest((R - 1) * (T - N + 1) / (N - 1), 'f', args=(N - 1, T - N + 1)).pvalue
    corr = np.corrcoef(Q, R)[0, 1]
    # QLIKE of the forecast a*Q*V against realised v = V*R*G  (loss = v/h - log(v/h) - 1); V cancels
    ql = lambda a: np.mean(R * G / (a * Q) - np.log(R * G / (a * Q)) - 1)
    rows.append((N, T, (Q.mean() - EQ) / se(Q), (R.mean() - ER) / se(R), ks_q, ks_r, corr, c, cq,
                 np.mean(R / (c * Q)), ql(1.0), ql(c), ql(cq), np.median(R / (c * Q))))
print(" N    T | zEQ    zER   KS_Q   KS_R  corr(Q,R) |  c      E[R/Q]  | mean R/(cQ)  median | QLIKE raw   c     E[R/Q]")
for r in rows:
    print(f"{r[0]:3d} {r[1]:4d} | {r[2]:+5.2f} {r[3]:+5.2f}  {r[4]:.3f}  {r[5]:.3f}  {r[6]:+.3f}   | {r[7]:6.3f} {r[8]:7.3f} | {r[9]:7.3f}   {r[13]:6.3f} | {r[10]:8.3f} {r[11]:6.3f} {r[12]:7.3f}")
