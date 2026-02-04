import numpy as np


def euclidean_dist(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=-1))


def gaussian_kernel(d, bw):
    # w = exp(-0.5 * (d/bw)^2)
    return np.exp(-0.5 * (d / bw) ** 2)


def build_design_matrix(x1):
    # X = [1, x1]
    return np.column_stack([np.ones_like(x1), x1])


def gwr_fit(X, y, coords, bw):
    """
    Fit GWR where regression points = sample points.

    Returns:
      betas: (n x k) local coefficients
      y_hat: (n,) fitted values
      residuals: (n,) y - y_hat
      S: (n x n) hat matrix (so y_hat = S @ y)
      trS: trace(S)
      trSTS: trace(S.T @ S)
      enp: effective number of parameters = 2 tr(S) - tr(S.T S)
    """
    n, k = X.shape
    S = np.zeros((n, n))
    betas = np.zeros((n, k))

    # Precompute pairwise distances (n x n)
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = euclidean_dist(coords[i], coords)

    for j in range(n):
        d = D[j]  # distances from regression point j to all observations
        w = gaussian_kernel(d, bw)
        W = np.diag(w)

        XtW = X.T @ W
        XtWX = XtW @ X

        # Numerical stability for tiny samples
        XtWX_inv = np.linalg.pinv(XtWX)

        beta_j = XtWX_inv @ (XtW @ y)
        betas[j] = beta_j

        # Hat matrix row: s_j = x_j^T (X' W X)^-1 X' W
        xj = X[j].reshape(1, -1)  # (1 x k)
        s_j = (xj @ XtWX_inv @ XtW)  # (1 x n)
        S[j, :] = s_j

    y_hat = S @ y
    residuals = y - y_hat

    trS = np.trace(S)
    trSTS = np.trace(S.T @ S)
    enp = 2 * trS - trSTS

    return {
        "betas": betas,
        "y_hat": y_hat,
        "residuals": residuals,
        "S": S,
        "trS": trS,
        "trSTS": trSTS,
        "enp": enp
    }


def r2_score(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def aicc_style(y, residuals, enp):
    """
    AICc-style score used for bandwidth comparison.
    Lower is better.

    Uses:
      sigma2 = RSS / n
      AIC  = n*log(sigma2) + n*log(2*pi) + n
      AICc adjustment based on ENP

    This is a practical scoring rule for selecting bandwidth in a toy setup.
    """
    n = len(y)
    rss = np.sum(residuals ** 2)
    sigma2 = rss / n

    # Guard against degenerate values
    sigma2 = max(sigma2, 1e-12)

    aic = n * np.log(sigma2) + n * np.log(2 * np.pi) + n

    denom = (n - 2 - enp)
    if denom <= 0:
        return np.inf

    aicc = aic + (2 * enp * (enp + 1)) / denom
    return aicc


def make_synthetic_data(seed=7):
    """
    Tiny synthetic spatial dataset:
      coords: scattered in a 2D plane
      x1: predictor
      y: outcome where the slope varies with location (so GWR has something to find)
    """
    rng = np.random.default_rng(seed)

    n = 30
    coords = rng.uniform(0, 10, size=(n, 2))
    x1 = rng.normal(0, 1, size=n)

    # True local slope depends on x-coordinate
    local_slope = 1.0 + 0.15 * coords[:, 0]
    intercept = 2.0
    noise = rng.normal(0, 0.5, size=n)

    y = intercept + local_slope * x1 + noise
    return coords, x1, y


def main():
    coords, x1, y = make_synthetic_data()

    X = build_design_matrix(x1)

    # Candidate bandwidths (distance units match coords)
    candidate_bws = [0.8, 1.2, 1.8, 2.5, 3.5, 5.0]

    results = []
    for bw in candidate_bws:
        fit = gwr_fit(X, y, coords, bw)
        score = aicc_style(y, fit["residuals"], fit["enp"])
        r2 = r2_score(y, fit["y_hat"])

        results.append((bw, score, r2, fit["enp"]))
        print(f"bw={bw:>4} | AICc*={score:>10.3f} | R2={r2:>6.3f} | ENP={fit['enp']:.3f}")

    best = min(results, key=lambda t: t[1])
    best_bw = best[0]

    print("\nBest bandwidth:", best_bw)

    best_fit = gwr_fit(X, y, coords, best_bw)
    betas = best_fit["betas"]

    print("\nLocal coefficient preview (first 5 rows):")
    # columns: intercept, slope
    for i in range(5):
        print(f"point {i:02d} | b0={betas[i,0]: .3f} | b1={betas[i,1]: .3f}")

    print(f"\nFinal: R2={r2_score(y, best_fit['y_hat']):.3f} | ENP={best_fit['enp']:.3f} | tr(S)={best_fit['trS']:.3f}")


if __name__ == "__main__":
    main()
