import os
import numpy as np
import pandas as pd
from faultlines.pipeline import get_log_return_matrix, get_ticker_metadata


def classify_market_regimes(
    returns_df: pd.DataFrame,
    vol_window: int = 20,
    turbulent_quantile: float = 0.75,
) -> pd.DataFrame:
  """Classifies trading days into 'Calm' and 'Turbulent' regimes using rolling

  realized volatility of an equal-weighted market index.
  """
  market_ret = returns_df.mean(axis=1)
  rolling_vol = market_ret.rolling(window=vol_window).std() * np.sqrt(252)

  # Backfill initial window NaNs using the earliest valid volatility estimate
  rolling_vol = rolling_vol.bfill()
  threshold = rolling_vol.quantile(turbulent_quantile)

  regimes = pd.DataFrame(
      {
          "market_return": market_ret,
          "rolling_vol_ann": rolling_vol,
          "regime": np.where(rolling_vol >= threshold, "Turbulent", "Calm"),
          "vol_threshold": threshold,
      },
      index=returns_df.index,
  )
  return regimes


def compute_forbes_rigobon_adjustment(
    corr_turbulent: pd.DataFrame,
    var_calm: float,
    var_turbulent: float,
) -> pd.DataFrame:
  """Applies the Forbes & Rigobon (2002) heteroskedasticity adjustment to

  conditional turbulent correlations:
      rho* = rho / sqrt(1 + delta * (1 - rho^2))
  where delta = (var_turbulent / var_calm) - 1.
  """
  delta = max((var_turbulent / var_calm) - 1.0, 0.0)
  rho = corr_turbulent.values.copy()

  # Clip strictly inside (-1, 1) on off-diagonals to avoid numerical precision issues
  rho_clipped = np.clip(rho, -0.999999, 0.999999)
  denom = np.sqrt(1.0 + delta * (1.0 - rho_clipped**2))
  adj = rho_clipped / denom
  np.fill_diagonal(adj, 1.0)

  return pd.DataFrame(
      adj, index=corr_turbulent.index, columns=corr_turbulent.columns
  )


def run_regime_analysis(
    start_date: str = "2010-01-01",
    end_date: str = "2017-12-31",
    tickers: list = None,
    vol_window: int = 20,
    turbulent_quantile: float = 0.75,
    output_dir: str = "results",
) -> dict:
  """Runs end-to-end market regime classification and volatility-adjusted

  contagion analysis, exporting summary tables to output_dir.
  """
  os.makedirs(output_dir, exist_ok=True)

  # Use selected_tickers.csv by default if available and no explicit list passed
  if tickers is None and os.path.exists(
      os.path.join(output_dir, "selected_tickers.csv")
  ):
    sel_df = pd.read_csv(os.path.join(output_dir, "selected_tickers.csv"))
    tickers = sel_df["Ticker"].str.upper().tolist()

  returns_df = get_log_return_matrix(
      start_date=start_date,
      end_date=end_date,
      asset_type="Stock",
      min_obs_pct=0.90,
      fill_missing=False,
  )

  if tickers:
    valid_cols = [c for c in returns_df.columns if c.upper() in set(tickers)]
    returns_df = returns_df[valid_cols]

  # Drop rows with all NaNs and fill residual isolated gaps with 0 for pairwise alignment
  returns_df = returns_df.dropna(how="all").fillna(0.0)

  regime_df = classify_market_regimes(
      returns_df,
      vol_window=vol_window,
      turbulent_quantile=turbulent_quantile,
  )

  calm_mask = regime_df["regime"] == "Calm"
  turb_mask = regime_df["regime"] == "Turbulent"

  calm_returns = returns_df.loc[calm_mask]
  turb_returns = returns_df.loc[turb_mask]

  var_calm = float(regime_df.loc[calm_mask, "market_return"].var())
  var_turb = float(regime_df.loc[turb_mask, "market_return"].var())

  corr_calm = calm_returns.corr()
  corr_turb_raw = turb_returns.corr()
  corr_turb_adj = compute_forbes_rigobon_adjustment(
      corr_turb_raw, var_calm, var_turb
  )

  # Extract upper-triangle pairwise correlations (excluding diagonal)
  triu_idx = np.triu_indices_from(corr_calm.values, k=1)
  pairs_df = pd.DataFrame({
      "Ticker_A": corr_calm.index[triu_idx[0]],
      "Ticker_B": corr_calm.columns[triu_idx[1]],
      "corr_calm": corr_calm.values[triu_idx],
      "corr_turbulent_raw": corr_turb_raw.values[triu_idx],
      "corr_turbulent_adj": corr_turb_adj.values[triu_idx],
  })
  pairs_df["raw_shift"] = (
      pairs_df["corr_turbulent_raw"] - pairs_df["corr_calm"]
  )
  pairs_df["true_contagion_shift"] = (
      pairs_df["corr_turbulent_adj"] - pairs_df["corr_calm"]
  )

  # Enrich with SIC sector metadata
  meta = get_ticker_metadata(list(returns_df.columns))
  sector_map = meta.set_index("Ticker")["sicDescription"].to_dict()
  pairs_df["Sector_A"] = pairs_df["Ticker_A"].map(sector_map)
  pairs_df["Sector_B"] = pairs_df["Ticker_B"].map(sector_map)
  pairs_df["same_sector"] = (pairs_df["Sector_A"] == pairs_df["Sector_B"]) & (
      pairs_df["Sector_A"].notna()
  )

  # Summary metrics for reporting & dashboard cards
  summary_df = pd.DataFrame([{
      "start_date": start_date,
      "end_date": end_date,
      "n_assets": returns_df.shape[1],
      "calm_days": int(calm_mask.sum()),
      "turbulent_days": int(turb_mask.sum()),
      "ann_vol_calm": float(np.sqrt(var_calm * 252)),
      "ann_vol_turbulent": float(np.sqrt(var_turb * 252)),
      "variance_ratio_delta": float((var_turb / var_calm) - 1.0),
      "mean_corr_calm": float(pairs_df["corr_calm"].mean()),
      "mean_corr_turbulent_raw": float(pairs_df["corr_turbulent_raw"].mean()),
      "mean_corr_turbulent_adj": float(pairs_df["corr_turbulent_adj"].mean()),
      "pct_pairs_true_contagion": float(
          (pairs_df["true_contagion_shift"] > 0.05).mean() * 100
      ),
  }])

  # Export artifacts
  regime_df.to_csv(os.path.join(output_dir, "regime_timeline.csv"))
  pairs_df.to_csv(
      os.path.join(output_dir, "regime_pairwise_contagion.csv"), index=False
  )
  summary_df.to_csv(
      os.path.join(output_dir, "regime_summary.csv"), index=False
  )
  corr_turb_adj.to_csv(os.path.join(output_dir, "corr_turbulent_adjusted.csv"))

  return {
      "timeline": regime_df,
      "pairwise": pairs_df,
      "summary": summary_df,
      "corr_calm": corr_calm,
      "corr_turbulent_raw": corr_turb_raw,
      "corr_turbulent_adj": corr_turb_adj,
  }


if __name__ == "__main__":
  results = run_regime_analysis()
  print("\n=== Regime & Contagion Summary ===")
  print(results["summary"].to_string(index=False))