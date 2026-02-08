import sqlite3
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
import networkx as nx

import igraph as ig
import leidenalg as la


def cramers_v_matrix(df, label=""):
    """
    Plots and returns Cramers-V-Matrix.
    """

    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        return np.sqrt(chi2 / (n * (min(k-1, r-1))))

    # Build correlation matrix
    cols = df.columns
    matrix = pd.DataFrame(index=cols, columns=cols)

    for i in cols:
        for j in cols:
            matrix.loc[i, j] = cramers_v(df[i], df[j])

    cv_matrix = matrix.astype(float)

    title = f"Cramér's V Attribute Correlation Matrix for {label}" if label != "" \
        else "Cramér's V Attribute Correlation Matrix"

    plt.figure(figsize=(8,6))
    sns.heatmap(cv_matrix, annot=True, cmap="Blues")
    plt.title(title)
    plt.show()

    return cv_matrix


def calculate_global_metrics(G, G_lcc, label='Simulation'):
    """
    Calculates global metrics for graph and prints them. Some metrics require connected graph.
    :param G: Graph
    :param G_lcc: Largest connected component of G
    :return: table of global metrics
    """
    avg_deg = sum(dict(G.degree()).values()) / G.number_of_nodes()
    dens = nx.density(G)

    # largest connected component -> diameter !!!
    diam = nx.diameter(G_lcc.to_undirected())  # Longest shortest path between any two nodes
    print(f"Mean degree: {avg_deg:.2f}\nDensity: {dens:.4f}\nDiameter: {diam}")

    avg_short_path = nx.average_shortest_path_length(G_lcc.to_undirected())
    print(f"Avg. shortest path: {avg_short_path:.3f}")

    ig_graph = ig.Graph.TupleList(G.edges(), directed=True)
    partition = la.find_partition(ig_graph, partition_type=la.ModularityVertexPartition)
    modularity = partition.modularity
    print(f"Modularity score: {modularity:.3f}")

    assortativity = nx.attribute_assortativity_coefficient(G, 'persona')
    print("Persona assortativity:", assortativity)

    metrics = pd.DataFrame({
        'Metric': ['Mean degree', 'Density', 'Diameter', 'Avg. shortest path', 'Modularity', 'Persona assortativity'],
        label: [avg_deg, dens, diam, avg_short_path, modularity, assortativity]
    })

    return metrics



def calculate_local_metrics(G_lcc):
    """
    Calculates local metrics for nodes and saves it in DataFrame. Creates metrics summary.
    :param G_lcc: Largest connected component of G
    :return: table with local metrics and persona attribute, summary
    """
    in_deg = dict(G_lcc.in_degree())
    out_deg = dict(G_lcc.out_degree())
    total_deg = dict(G_lcc.degree())
    bet = nx.betweenness_centrality(G_lcc)
    eig = nx.eigenvector_centrality_numpy(G_lcc.to_undirected())
    pagerank = nx.pagerank(G_lcc, alpha=0.85)
    kcore = nx.core_number(G_lcc.to_undirected())

    local_metrics = pd.DataFrame({
        'node': list(G_lcc.nodes()),
        'persona': [G_lcc.nodes[n].get('persona') for n in G_lcc.nodes()],
        'in_degree': [in_deg[n] for n in G_lcc.nodes()],
        'out_degree': [out_deg[n] for n in G_lcc.nodes()],
        'total_degree': [total_deg[n] for n in G_lcc.nodes()],
        'betweenness': [bet[n] for n in G_lcc.nodes()],
        'eigenvector': [eig[n] for n in G_lcc.nodes()],
        'pagerank': [pagerank[n] for n in G_lcc.nodes()],
        'kcore': [kcore[n] for n in G_lcc.nodes()]
    })

    summary = local_metrics.groupby('persona').agg(
        n_nodes=('node', 'count'),
        mean_in_degree=('in_degree', 'mean'),
        mean_out_degree=('out_degree', 'mean'),
        mean_total_degree=('total_degree', 'mean'),
        mean_betweenness=('betweenness', 'mean'),
        median_betweenness=('betweenness', 'median'),
        mean_eigenvector=('eigenvector', 'mean'),
        mean_pagerank=('pagerank', 'mean'),
        mean_kcore=('kcore', 'mean')
    )

    global_mean_pagerank = local_metrics['pagerank'].mean()
    summary['pagerank_ratio'] = (
            summary['mean_pagerank'] / global_mean_pagerank
    )

    return local_metrics, summary



def statistical_difference_check(metrics):
    """
    Checks statistically differences between personae for each metric and prints results.
    """
    cols = ['in_degree', 'out_degree', 'total_degree', 'betweenness', 'eigenvector', 'pagerank', 'kcore']

    for col in cols:
        groups = [
            metrics[metrics['persona'] == p][col]
            for p in metrics['persona'].unique()
        ]

        stat_diff = kruskal(*groups)
        print(f"{col} statistics: p = {stat_diff.pvalue:.4f}")



def persona_permutation_test(G, persona_summary, n_perm=100, seed=None):
    """
    Null model validation function for persona.
    Calculates local metrics for permutated persona attributes.
    Keeps network structure unchanged.
    Returns df with metrics distributions across permutations.
    """
    rng = np.random.default_rng(seed)

    nodes = list(G.nodes())
    original_personas = np.array([G.nodes[n]["persona"] for n in nodes])

    null_distributions = {
        metric: {p: [] for p in persona_summary.index}
        for metric in persona_summary.columns
    }

    for _ in range(n_perm):
        permuted = rng.permutation(original_personas)
        nx.set_node_attributes(G, dict(zip(nodes, permuted)), "persona")

        _, perm_stats = calculate_local_metrics(G)

        for metric in perm_stats.columns:
            for persona in perm_stats.index:
                null_distributions[metric][persona].append(
                    perm_stats.loc[persona, metric]
                )

    nx.set_node_attributes(G, dict(zip(nodes, original_personas)), "persona")

    return null_distributions


def permutation_pvalues(persona_summary, null_dist):
    """
    Calculates p-value for null distribution across permutations.
    Returns df with p-values for metrics vs persona.
    """
    pvals = pd.DataFrame(index=persona_summary.index, columns=persona_summary.columns)

    for metric in persona_summary.columns:
        for persona in persona_summary.index:
            obs = persona_summary.loc[persona, metric]
            null = np.array(null_dist[metric][persona])

            pvals.loc[persona, metric] = (
                np.sum(null >= obs) + 1
            ) / (len(null) + 1)

    return pvals.astype(float)



def permutation_zscores(persona_summary, null_dist):
    """
    Calculates z-scores for between observed values and null distribution across permutations.
    Returns df with z_scores for metrics vs persona.
    """
    zscores = pd.DataFrame(
        index=persona_summary.index,
        columns=persona_summary.columns,
        dtype=float
    )

    for persona in persona_summary.index:
        for metric in persona_summary.columns:
            obs = pd.to_numeric(
                persona_summary.loc[persona, metric],
                errors='coerce'
            )

            null = pd.to_numeric(
                null_dist[metric][persona],
                errors='coerce'
            )

            if len(null) < 5 or np.isnan(obs):
                zscores.loc[persona, metric] = np.nan
                continue

            mu = null.mean()
            sigma = null.std(ddof=1)

            if sigma == 0:
                zscores.loc[persona, metric] = 0.0
            else:
                zscores.loc[persona, metric] = (obs - mu) / sigma

    return zscores



def persona_metric_stability(persona_tables):
    """
    Calculates mean, std, and cv for metrics for personas across simulations.
    Returns df with results.
    """
    rows = []

    for persona, df in persona_tables.items():
        for metric in df.index:
            vals = df.loc[metric].values

            rows.append({
                'persona': persona,
                'metric': metric,
                'mean': np.mean(vals),
                'std': np.std(vals, ddof=1),
                'cv': np.std(vals, ddof=1) / np.mean(vals)
            })

    return pd.DataFrame(rows)


def persona_rank_stability(persona_tables):
    """
    Creates ranking of metrics-personas across simulations.
    """
    rankings = []

    metrics = next(iter(persona_tables.values())).index
    simulations = next(iter(persona_tables.values())).columns

    for metric in metrics:
        for sim in simulations:
            values = {
                persona: persona_tables[persona].loc[metric, sim]
                for persona in persona_tables
            }

            ranks = (
                pd.Series(values)
                .rank(ascending=False)
                .astype(int)
            )

            for persona, rank in ranks.items():
                rankings.append({
                    'persona': persona,
                    'metric': metric,
                    'simulation': sim,
                    'rank': rank
                })

    return pd.DataFrame(rankings)


def rank_consistency(rank_df):
    """
    Calculates mean and standard deviation for rank of each metric-persona.
    """
    return (
        rank_df
        .groupby(['persona', 'metric'])['rank']
        .agg(['mean', 'std'])
        .rename(columns={'std': 'rank_std'})
    )


