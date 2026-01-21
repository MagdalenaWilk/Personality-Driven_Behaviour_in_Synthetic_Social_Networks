import sqlite3
import pandas as pd
import numpy as np
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

