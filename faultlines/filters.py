
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _validate_returns(returns):
    """
    Validate a returns matrix.
    Rows = observations, columns = assets.
    """
    if isinstance(returns, pd.DataFrame):
        columns = returns.columns
        X = returns.to_numpy(dtype=float)
    else:
        X = np.asarray(returns, dtype=float)
        columns = None

    if X.ndim != 2:
        raise ValueError("Returns must be a 2D matrix.")

    if not np.isfinite(X).all():
        raise ValueError("Returns contain NaN or infinite values.")

    if X.shape[0] < 3 or X.shape[1] < 2:
        raise ValueError("Need at least 3 observations and 2 assets.")

    if np.any(np.std(X, axis=0, ddof=1) == 0):
        raise ValueError("Returns contain a constant asset.")

    return X, columns


def rmt_filter(returns):
    """
    Marchenko-Pastur RMT denoising.

    1. Calculate the sample correlation matrix.
    2. Calculate its eigenvalues and eigenvectors.
    3. Estimate the Marchenko-Pastur upper noise boundary.
    4. Replace noise eigenvalues with their average.
    5. Reconstruct and rescale the covariance matrix.

    Returns a denoised covariance matrix.
    """

    X, columns = _validate_returns(returns)

    n, p = X.shape

    # Calculate sample standard deviations
    std = np.std(X, axis=0, ddof=1)

    # Calculate correlation matrix
    corr = np.corrcoef(X, rowvar=False)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Marchenko-Pastur theoretical noise boundaries
    q = p / n
    lambda_min = (1 - np.sqrt(q)) ** 2
    lambda_max = (1 + np.sqrt(q)) ** 2

    # Identify eigenvalues inside the noise spectrum
    noise_mask = ((eigenvalues >= lambda_min) & (eigenvalues <= lambda_max))

    # Start with the original eigenvalues
    filtered_eigenvalues = eigenvalues.copy()

    # Remove tiny negative numerical eigenvalues
    filtered_eigenvalues = np.maximum(filtered_eigenvalues, 0)

    if np.any(noise_mask):
        noise_average = np.mean(filtered_eigenvalues[noise_mask])
        filtered_eigenvalues[noise_mask] = noise_average

    # Reconstruct filtered correlation matrix
    filtered_corr = (
        eigenvectors
        @ np.diag(filtered_eigenvalues)
        @ eigenvectors.T
    )

    # Normalize diagonal to 1
    diagonal = np.sqrt(np.maximum(np.diag(filtered_corr), 1e-12))
    filtered_corr = filtered_corr / np.outer(diagonal, diagonal)

    # Convert correlation back to covariance
    filtered_cov = np.outer(std, std) * filtered_corr
    filtered_cov = (filtered_cov + filtered_cov.T) / 2

    return pd.DataFrame(
        filtered_cov,
        index=columns,
        columns=columns
    )


def ledoit_wolf_filter(returns):
    """
    Ledoit-Wolf covariance shrinkage estimator.

    Returns a shrunk covariance matrix.
    """

    X, columns = _validate_returns(returns)

    estimator = LedoitWolf()
    estimator.fit(X)

    covariance = estimator.covariance_

    return pd.DataFrame(
        covariance,
        index=columns,
        columns=columns
    )

def covariance_to_correlation(covariance):
    """
    Convert covariance matrix into correlation matrix.
    """

    cov = np.asarray(covariance, dtype=float)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance must be square.")

    if not np.isfinite(cov).all():
        raise ValueError("Covariance contains invalid values.")

    std = np.sqrt(np.diag(cov))

    if np.any(std <= 0):
        raise ValueError("Variances must be positive.")

    corr = cov / np.outer(std, std)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)

    return corr