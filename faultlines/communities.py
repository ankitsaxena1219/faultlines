
"""Community detection and stock-sector analysis for FaultLines."""

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def build_correlation_graph(
    returns_df: pd.DataFrame,
    threshold: float = 0.30,
) -> nx.Graph:
    """
    Build a weighted stock network from daily stock returns.

    Stocks are nodes. An edge connects two stocks when their
    positive correlation is greater than or equal to threshold.
    The edge weight is the Pearson correlation.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    returns = returns_df.select_dtypes(include=[np.number]).copy()

    if returns.empty:
        raise ValueError("No numeric return columns found.")

    if returns.columns.duplicated().any():
        raise ValueError("Duplicate ticker columns found.")

    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(axis=1, how="all")

    if returns.shape[1] == 0:
        raise ValueError("No valid ticker columns found.")

    correlations = returns.corr(min_periods=30)

    graph = nx.Graph()
    graph.add_nodes_from(correlations.columns)

    tickers = correlations.columns.tolist()

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            ticker_a = tickers[i]
            ticker_b = tickers[j]
            correlation = correlations.loc[ticker_a, ticker_b]

            if pd.notna(correlation) and correlation >= threshold:
                graph.add_edge(
                    ticker_a,
                    ticker_b,
                    weight=float(correlation),
                )

    return graph


def detect_communities(
    graph: nx.Graph,
    resolution: float = 1.0,
    seed: int = 42,
) -> dict:
    """
    Detect stock communities using the Louvain algorithm.

    Returns a dictionary mapping each ticker to its community ID.
    """
    if graph.is_directed():
        raise ValueError("Louvain requires an undirected graph.")

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot detect communities in an empty graph.")

    if resolution <= 0:
        raise ValueError("resolution must be greater than zero.")

    if graph.number_of_edges() == 0:
        communities = [{node} for node in graph.nodes()]
    else:
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=resolution,
            seed=seed,
        )

    assignments = {}

    for community_id, members in enumerate(
        sorted(communities, key=lambda c: sorted(map(str, c))[0])
    ):
        for ticker in members:
            assignments[ticker] = community_id

    return assignments

def compare_communities_to_sectors(
    assignments: dict,
    metadata_df: pd.DataFrame,
) -> tuple:
    """
    Compare detected stock communities with SEC SIC classifications.

    Returns:
        details: Stock-level community and SIC information.
        community_summary: Industry composition of each community.
        ari: Adjusted Rand Index comparing communities with SIC codes.
    """
    if not assignments:
        raise ValueError("Community assignments cannot be empty.")

    required_columns = {"Ticker", "sic", "sicDescription"}
    missing_columns = required_columns - set(metadata_df.columns)

    if missing_columns:
        raise ValueError(
            f"Metadata is missing columns: {sorted(missing_columns)}"
        )

    assignment_df = pd.DataFrame(
        assignments.items(),
        columns=["Ticker", "Community"]
    )

    assignment_df["Ticker"] = assignment_df["Ticker"].astype(str).str.upper()

    metadata = metadata_df.copy()
    metadata["Ticker"] = metadata["Ticker"].astype(str).str.upper()

    details = assignment_df.merge(
        metadata[["Ticker", "sic", "sicDescription"]],
        on="Ticker",
        how="left"
    )

    details["sic"] = details["sic"].astype("string")

    valid = details.dropna(subset=["sic"])

    if valid["sic"].nunique() < 2:
        raise ValueError("At least two SIC categories are required for ARI.")

    ari = adjusted_rand_score(
        valid["sic"],
        valid["Community"]
    )

    # Calculate total community sizes, including stocks without SIC metadata.
    total_counts = (
        details.groupby("Community")
        .agg(Total_Stocks=("Ticker", "count"))
    )

    # Calculate industry composition using only stocks with SIC metadata.
    classified_summary = (
        valid.groupby("Community")
        .agg(
            Stocks_With_SIC=("Ticker", "count"),
            Number_of_Industries=("sic", "nunique"),
            Dominant_Industry=("sicDescription",
                               lambda x: x.value_counts().index[0]),
            Dominant_Industry_Count=("sicDescription",
                                     lambda x: x.value_counts().iloc[0]),
        )
    )

    community_summary = total_counts.join(
        classified_summary, how="left"
    ).reset_index()

    community_summary["Stocks_With_SIC"] = (
        community_summary["Stocks_With_SIC"].fillna(0).astype(int)
    )

    community_summary["Stocks_Without_SIC"] = (
        community_summary["Total_Stocks"]
        - community_summary["Stocks_With_SIC"]
    )

    community_summary["Number_of_Industries"] = (
        community_summary["Number_of_Industries"].fillna(0).astype(int)
    )

    community_summary["Dominant_Industry"] = (
        community_summary["Dominant_Industry"].fillna("No SIC data")
    )

    community_summary["Dominant_Industry_Count"] = (
        community_summary["Dominant_Industry_Count"].fillna(0).astype(int)
    )

    community_summary["Industry_Concentration"] = (
        community_summary["Dominant_Industry_Count"]
        / community_summary["Stocks_With_SIC"].replace(0, pd.NA)
    ).fillna(0)

    return details, community_summary, ari

def match_communities(
    previous: dict,
    current: dict,
) -> pd.DataFrame:
    """
    Match communities between consecutive windows using Jaccard similarity.
    """
    previous_ids = sorted(set(previous.values()))
    current_ids = sorted(set(current.values()))

    rows = []

    for prev_id in previous_ids:
        prev_members = {
            ticker for ticker, cid in previous.items()
            if cid == prev_id
        }

        for curr_id in current_ids:
            curr_members = {
                ticker for ticker, cid in current.items()
                if cid == curr_id
            }

            intersection = len(prev_members & curr_members)
            union = len(prev_members | curr_members)

            similarity = intersection / union if union else 0

            rows.append({
                "Previous_Community": prev_id,
                "Current_Community": curr_id,
                "Shared_Stocks": intersection,
                "Jaccard_Similarity": similarity,
            })

    return pd.DataFrame(rows)


def analyze_temporal_communities(
    returns_df: pd.DataFrame,
    window_size: int = 120,
    step_size: int = 20,
    threshold: float = 0.30,
) -> tuple:
    """
    Detect and track stock communities across rolling time windows.

    Returns:
        temporal_results: Community membership by window.
        stability_results: Community matching between windows.
    """
    if window_size <= 0 or step_size <= 0:
        raise ValueError("Window size and step size must be positive.")

    if len(returns_df) < window_size:
        raise ValueError("Not enough observations for the selected window.")

    temporal_rows = []
    stability_frames = []
    previous_assignments = None
    window_number = 0

    for start in range(0, len(returns_df) - window_size + 1, step_size):
        end = start + window_size
        window_df = returns_df.iloc[start:end]

        graph = build_correlation_graph(
            window_df,
            threshold=threshold,
        )

        assignments = detect_communities(graph)

        window_start = window_df.index[0]
        window_end = window_df.index[-1]

        for ticker, community_id in assignments.items():
            temporal_rows.append({
                "Window": window_number,
                "Window_Start": window_start,
                "Window_End": window_end,
                "Ticker": ticker,
                "Community": community_id,
            })

        if previous_assignments is not None:
            matches = optimal_community_matching(
                previous_assignments,
                assignments,
            )

            matches["Window"] = window_number
            matches["Window_Start"] = window_start
            matches["Window_End"] = window_end

            stability_frames.append(matches)

        previous_assignments = assignments
        window_number += 1

    temporal_results = pd.DataFrame(temporal_rows)

    if stability_frames:
        stability_results = pd.concat(
            stability_frames,
            ignore_index=True,
        )
    else:
        stability_results = pd.DataFrame()

    return temporal_results, stability_results

def optimal_community_matching(
    previous: dict,
    current: dict,
) -> pd.DataFrame:
    """
    Match communities between consecutive windows using
    maximum-weight one-to-one Jaccard similarity.
    """
    from scipy.optimize import linear_sum_assignment

    previous_ids = sorted(set(previous.values()))
    current_ids = sorted(set(current.values()))

    if not previous_ids or not current_ids:
        return pd.DataFrame()

    similarity_matrix = np.zeros(
        (len(previous_ids), len(current_ids))
    )

    for i, prev_id in enumerate(previous_ids):
        prev_members = {
            ticker for ticker, cid in previous.items()
            if cid == prev_id
        }

        for j, curr_id in enumerate(current_ids):
            curr_members = {
                ticker for ticker, cid in current.items()
                if cid == curr_id
            }

            union = len(prev_members | curr_members)
            intersection = len(prev_members & curr_members)

            similarity_matrix[i, j] = (
                intersection / union if union else 0
            )

    row_indices, col_indices = linear_sum_assignment(
        -similarity_matrix
    )

    rows = []

    for i, j in zip(row_indices, col_indices):
        rows.append({
            "Previous_Community": previous_ids[i],
            "Current_Community": current_ids[j],
            "Jaccard_Similarity": similarity_matrix[i, j],
            "Shared_Stocks": sum(
                1 for ticker in previous
                if previous[ticker] == previous_ids[i]
                and current.get(ticker) == current_ids[j]
            ),
        })

    return pd.DataFrame(rows)


def calculate_network_metrics(graph: nx.Graph) -> dict:
    """
    Calculate structural and weighted metrics for a stock network.

    Returns:
        Dictionary containing network-level metrics.
    """
    number_of_nodes = graph.number_of_nodes()
    number_of_edges = graph.number_of_edges()

    if number_of_nodes == 0:
        return {
            "Nodes": 0,
            "Edges": 0,
            "Density": 0.0,
            "Average_Degree": 0.0,
            "Average_Strength": 0.0,
            "Average_Clustering": 0.0,
            "Average_Betweenness": 0.0,
        }

    degrees = dict(graph.degree())

    strengths = dict(
        graph.degree(weight="weight")
    )

    clustering = nx.average_clustering(
        graph,
        weight="weight",
    )

    betweenness = nx.betweenness_centrality(
        graph,
        weight=None,
    )

    return {
        "Nodes": number_of_nodes,
        "Edges": number_of_edges,
        "Density": nx.density(graph),
        "Average_Degree": sum(degrees.values()) / number_of_nodes,
        "Average_Strength": sum(strengths.values()) / number_of_nodes,
        "Average_Clustering": clustering,
        "Average_Betweenness": sum(betweenness.values()) / number_of_nodes,
    }


def analyze_temporal_network_metrics(
    returns_df: pd.DataFrame,
    window_size: int = 120,
    step_size: int = 20,
    threshold: float = 0.30,
) -> pd.DataFrame:
    """
    Calculate network metrics across rolling time windows.
    """
    if window_size <= 0 or step_size <= 0:
        raise ValueError("Window size and step size must be positive.")

    if len(returns_df) < window_size:
        raise ValueError("Not enough observations for the selected window.")

    rows = []

    for start in range(
        0,
        len(returns_df) - window_size + 1,
        step_size,
    ):
        end = start + window_size
        window_df = returns_df.iloc[start:end]

        graph = build_correlation_graph(
            window_df,
            threshold=threshold,
        )

        metrics = calculate_network_metrics(graph)

        metrics["Window"] = len(rows)
        metrics["Window_Start"] = window_df.index[0]
        metrics["Window_End"] = window_df.index[-1]
        metrics["Window_Days"] = window_size
        metrics["Threshold"] = threshold

        rows.append(metrics)

    return pd.DataFrame(rows)


def calculate_stock_centrality(graph: nx.Graph) -> pd.DataFrame:
    """
    Calculate stock-level centrality metrics to identify
    influential hubs and bridge stocks.
    """
    if graph.number_of_nodes() == 0:
        return pd.DataFrame()

    degree = dict(graph.degree())

    strength = dict(
        graph.degree(weight="weight")
    )

    betweenness = nx.betweenness_centrality(
        graph,
        weight=None,
    )

    closeness = nx.closeness_centrality(graph)

    results = pd.DataFrame({
        "Ticker": list(graph.nodes()),
        "Degree": [degree[n] for n in graph.nodes()],
        "Strength": [strength[n] for n in graph.nodes()],
        "Betweenness": [betweenness[n] for n in graph.nodes()],
        "Closeness": [closeness[n] for n in graph.nodes()],
    })

    return results.sort_values(
        "Betweenness",
        ascending=False,
    ).reset_index(drop=True)


def analyze_temporal_stock_centrality(
    returns_df: pd.DataFrame,
    window_size: int = 120,
    step_size: int = 20,
    threshold: float = 0.30,
) -> pd.DataFrame:
    """
    Track stock-level centrality across rolling windows.
    """
    if window_size <= 0 or step_size <= 0:
        raise ValueError("Window size and step size must be positive.")

    if len(returns_df) < window_size:
        raise ValueError("Not enough observations for the selected window.")

    results = []

    window_number = 0

    for start in range(
        0,
        len(returns_df) - window_size + 1,
        step_size,
    ):
        end = start + window_size
        window_df = returns_df.iloc[start:end]

        graph = build_correlation_graph(
            window_df,
            threshold=threshold,
        )

        centrality = calculate_stock_centrality(graph)

        centrality["Window"] = window_number
        centrality["Window_Start"] = window_df.index[0]
        centrality["Window_End"] = window_df.index[-1]

        results.append(centrality)

        window_number += 1

    return pd.concat(results, ignore_index=True)
