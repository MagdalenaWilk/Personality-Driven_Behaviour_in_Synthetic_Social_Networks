import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


def describe_personae(df, feature_cols=['openness', 'conscientiousness', 'extroversion', 'agreeableness', 'neuroticism']):
    """
    Creates and returns crosstab table features vs. persona.
    """
    description = pd.crosstab(df['persona'], df[feature_cols[0]], normalize='index')

    for i in range(1, len(feature_cols)):
        tab = pd.crosstab(df['persona'], df[feature_cols[i]], normalize='index')
        description = description.join(tab, on='persona')

    return description


def persona_significance(df, feature_cols=['openness', 'conscientiousness', 'extroversion', 'agreeableness', 'neuroticism']):
    """
    Function prints significance of each feature regarding personae.
    """
    print("Features significance regarding personae:")

    for col in feature_cols:
        table = pd.crosstab(df['persona'], df[col])
        chi2, p, dof, expected = chi2_contingency(table)

        print(f"{col}: {p}")


def compare_persona_across_simulations(summaries, num_persona):
    """
    Compares persona across simulations. Creates summary table for each persona and returns them.
    """
    persona_tables = {}

    for i in range(num_persona):
        tables = []

        for summary in summaries:
            sim_name = summary.loc['Persona_1', 'Simulation']
            tab = summary.loc[f"Persona_{i+1}"].rename(sim_name)
            tables.append(tab)

        table = pd.concat(tables, axis=1)
        persona_tables[f"Persona_{i+1}"] = table.drop('Simulation', axis=0)

    print(f"Number of persona tables: {len(persona_tables)}")

    return persona_tables