import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import networkx as nx

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import igraph as ig
import leidenalg as la

def describe_personae(df, feature_cols=['openness', 'conscientiousness', 'extroversion', 'agreeableness', 'neuroticism']):
    """
    Creates and returns crosstab table features vs. persona.
    """
    describtion = pd.crosstab(df['persona'], df[feature_cols[0]], normalize='index')

    for i in range(1, len(feature_cols)):
        tab = pd.crosstab(df['persona'], df[feature_cols[i]], normalize='index')
        describtion.join(tab, on='persona')

    return describtion


def persona_significance(df, feature_cols=['openness', 'conscientiousness', 'extroversion', 'agreeableness', 'neuroticism']):
    """
    Function prints significance of each feature regarding personae.
    """
    print("Features significance regarding personae:")

    for col in feature_cols:
        table = pd.crosstab(df['persona'], df[col])
        chi2, p, dof, expected = chi2_contingency(table)

        print(f"{col}: {p}")
