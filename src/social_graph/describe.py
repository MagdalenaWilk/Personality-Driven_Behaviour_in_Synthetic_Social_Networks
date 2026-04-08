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


def describe_unfollows(follow_df):
    """
    Function analyses unfollow statistics. Plot figures of follow, unfollow counts. Compare it for distinct personas.
    :param follow_df: follow table already merged with personas.
    :return: simple statistics
    """
    unfollow = follow_df[follow_df['action'] == "unfollow"]
    unfollows_per_follower = (unfollow.groupby('follower_id').size().reset_index(name='n_unfollows'))
    print(f"Number of users who used unfollow: {len(unfollows_per_follower)}")

    plt.hist(unfollows_per_follower['n_unfollows'], bins=100)
    plt.show()

    unfollows_per_persona = (unfollow.groupby('persona').size().reset_index(name='n_unfollows'))

    sns.barplot(x=unfollows_per_persona['persona'], y=unfollows_per_persona['n_unfollows'])
    plt.show()

    follows_per_persona = (follow_df[follow_df['action'] == "follow"].
                           groupby('persona').size().reset_index(name='n_follows'))
    follows_per_persona = follows_per_persona.merge(unfollows_per_persona, on='persona')
    follows_per_persona['ratio'] = follows_per_persona['n_unfollows'] / follows_per_persona['n_follows']
    print(f"Follows & Unfollows per persona:\n{follows_per_persona}")

    return follows_per_persona



def describe_follows_over_time(follow_df, round_step=24):
    """
    Function analyses follow over time. Compare it for distinct personas. Generates plots.
    :param follow_df: follow table already merged with personas.
    :return: Follow statistics per day per persona.
    """
    max_round = follow_df['round'].max()
    print(f"Rounds: {max_round}")
    print(f"Days: {max_round / 24}")

    round_iter = round_step
    counts_daily = []

    while round_iter <= max_round:
        follow_r = follow_df[(follow_df['action'] == 'follow') &
                          (follow_df['round'] <= round_iter)]
        unfollow_r = follow_df[(follow_df['action'] == 'unfollow') &
                              (follow_df['round'] <= round_iter)]

        follows_per_persona = follow_r.groupby('persona').size().reset_index(name='n_follows')
        unfollows_per_persona = unfollow_r.groupby('persona').size().reset_index(name='n_unfollows')

        df = follows_per_persona.merge(unfollows_per_persona, on='persona', how='outer').fillna(0)

        df['day'] = round_iter // 24

        counts_daily.append(df)

        round_iter += round_step

    follows_daily = pd.concat(counts_daily, ignore_index=True)

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=follows_daily,
        x='day',
        y='n_follows',
        hue='persona'
    )
    plt.title('Follows growth over time')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=follows_daily,
        x='day',
        y='n_unfollows',
        hue='persona'
    )
    plt.title('Unfollows growth over time')
    plt.show()

    return follows_daily