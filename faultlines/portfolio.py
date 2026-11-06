
import numpy as np
import pandas as pd

from scipy.optimize import minimize

from .filters import rmt_filter, ledoit_wolf_filter


def minimum_variance_weights(covariance):
    """
    Calculate long-only minimum variance portfolio weights.

    Constraints:
    - Weights sum to 1
    - No short selling
    - Each weight lies between 0 and 1
    """

    cov = np.asarray(covariance, dtype=float)
    cov = (cov + cov.T) / 2

    n_assets = cov.shape[0]

    initial_weights = np.ones(n_assets) / n_assets

    def portfolio_variance(weights):
        return weights @ cov @ weights

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    ]

    bounds = [(0, 1)] * n_assets

    result = minimize(
        portfolio_variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12}
    )

    if not result.success:
        raise RuntimeError(
            f"Portfolio optimization failed: {result.message}"
        )

    return result.x


def compare_portfolio_variance(train_returns, test_returns):
    """
    Compare raw, RMT-filtered and Ledoit-Wolf portfolios.

    Training data:
        Estimate covariance matrices and optimize weights.

    Testing data:
        Evaluate realized portfolio variance out of sample.

    Returns:
        results_df: Performance comparison
        weights_df: Portfolio weights for each method
    """

    if not isinstance(train_returns, pd.DataFrame):
        raise TypeError("train_returns must be a DataFrame.")

    if not isinstance(test_returns, pd.DataFrame):
        raise TypeError("test_returns must be a DataFrame.")

    if not train_returns.columns.equals(test_returns.columns):
        raise ValueError("Training and testing assets must match.")

    X_train = train_returns.to_numpy(dtype=float)
    X_test = test_returns.to_numpy(dtype=float)

    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Returns contain missing or infinite values.")

    if X_train.shape[0] < 3 or X_test.shape[0] < 2:
        raise ValueError("Insufficient training or testing observations.")

    if X_train.shape[1] < 2:
        raise ValueError("At least two assets are required.")

    # 1. Raw sample covariance
    raw_cov = np.cov(X_train, rowvar=False, ddof=1)

    # 2. RMT-filtered covariance
    rmt_cov = rmt_filter(train_returns).to_numpy()

    # 3. Ledoit-Wolf covariance
    lw_cov = ledoit_wolf_filter(train_returns).to_numpy()

    covariance_matrices = {
        "Raw": raw_cov,
        "RMT": rmt_cov,
        "Ledoit-Wolf": lw_cov
    }

    results = []
    weights_dict = {}

    for method, covariance in covariance_matrices.items():

        # Calculate minimum variance portfolio weights
        weights = minimum_variance_weights(covariance)

        # Estimated daily variance using training covariance
        estimated_variance = weights @ covariance @ weights

        # Calculate out-of-sample portfolio returns
        portfolio_returns = X_test @ weights

        # Realized daily variance on testing data
        realized_variance = np.var(
            portfolio_returns, ddof=1
        )

        # Annualized realized volatility
        annualized_volatility = np.sqrt(
            realized_variance * 252
        )

        results.append({
            "Method": method,
            "Estimated_Daily_Variance": estimated_variance,
            "Realized_Daily_Variance": realized_variance,
            "Annualized_Realized_Volatility": annualized_volatility
        })

        weights_dict[method] = weights

    results_df = pd.DataFrame(results).set_index("Method")

    weights_df = pd.DataFrame(
        weights_dict,
        index=train_returns.columns
    )

    return results_df, weights_df


def rolling_portfolio_validation(
    returns,
    window=252,
    step=21
):
    """
    Rolling out-of-sample portfolio validation.

    window: Number of training observations.
    step: Number of testing observations per period.

    Returns:
        summary_df: Overall out-of-sample performance.
        window_results_df: Results for each rolling window.
        portfolio_returns_df: Daily out-of-sample portfolio returns.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a DataFrame.")

    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("Returns contain missing or infinite values.")

    if window < 3 or step < 1:
        raise ValueError("Invalid window or step size.")

    if len(returns) <= window:
        raise ValueError("Not enough observations for rolling validation.")

    methods = ["Raw", "RMT", "Ledoit-Wolf"]

    all_portfolio_returns = {method: [] for method in methods}
    window_results = []

    # Store previous portfolio weights for turnover calculation
    previous_weights = {method: None for method in methods}


    # Move through the dataset chronologically
    for start in range(0, len(returns) - window - step + 1 , step):

        train_end = start + window
        test_end = min(train_end + step, len(returns))

        train = returns.iloc[start:train_end]
        test = returns.iloc[train_end:test_end]

        if len(test) < 2:
            continue

        # Estimate covariance and optimize portfolios
        results, weights = compare_portfolio_variance(
            train,
            test
        )

        results = results.reset_index()

        # Variance forecast errors
        results["Variance_Forecast_Absolute_Error"] = abs(
            results["Estimated_Daily_Variance"]
            - results["Realized_Daily_Variance"]
        )

        results["Variance_Forecast_Squared_Error"] = (
            results["Estimated_Daily_Variance"]
            - results["Realized_Daily_Variance"]
        )**2

        # Portfolio turnover between consecutive windows
        turnover_values = {}

        for method in methods:
            current_weights = weights[method].to_numpy()

            if previous_weights[method] is None:
                turnover = np.nan
            else:
                turnover = 0.5 * np.sum(
                    np.abs(current_weights - previous_weights[method])
                )

            turnover_values[method] = turnover
            previous_weights[method] = current_weights.copy()

        results["Turnover"] = results["Method"].map(turnover_values)

        # Record training and testing dates
        results["Train_Start"] = train.index[0]
        results["Train_End"] = train.index[-1]
        results["Test_Start"] = test.index[0]
        results["Test_End"] = test.index[-1]


        window_results.append(results)

        # Store daily out-of-sample portfolio returns
        for method in methods:

            portfolio_return = (
                test.to_numpy() @ weights[method].to_numpy()
            )

            series = pd.Series(
                portfolio_return,
                index=test.index,
                name=method
            )

            all_portfolio_returns[method].append(series)

    # Combine all testing periods
    portfolio_returns_df = pd.concat(
        {
            method: pd.concat(all_portfolio_returns[method])
            for method in methods
        },
        axis=1
    )

    # Calculate overall out-of-sample performance
    summary = []

    for method in methods:

        realized_variance = portfolio_returns_df[method].var()

        summary.append({
            "Method": method,
            "OOS_Daily_Variance": realized_variance,
            "OOS_Annualized_Volatility": np.sqrt(
                realized_variance * 252
            )
        })

    summary_df = pd.DataFrame(summary).set_index("Method")

    window_results_df = pd.concat(
        window_results,
        ignore_index=True
    )

    return summary_df, window_results_df, portfolio_returns_df
