import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

def lifespan_analysis(follow_persona_df, agg_by='persona'):
    """
    The function analyses lifespan of users' connections. Returns DF with results.
    """
    lifespan_df = follow_persona_df.sort_values(['follower_id', 'user_id', 'id'])

    lifespans = []
    invalid_unfollows_counts = 0

    for (u, v, persona), group in lifespan_df.groupby(['follower_id', 'user_id', 'persona']):
        start = None

        for _, row in group.iterrows():
            if row['action'] == 'follow':
                start = row['round']
            elif row['action'] == 'unfollow' and start is not None:
                lifespan = (row['round'] - start) / 24
                lifespans.append((persona, u, v, lifespan))
                start = None
            elif row['action'] == 'unfollow' and start is None:
                print(f"Invalid unfollow action for users {u} -> {v}: {row['action']}")
                invalid_unfollows_counts += 1

        if start is not None:
            lifespans.append((persona, u, v, '+'))  # connection still exist

    print(f"Invalid unfollow actions: {invalid_unfollows_counts}")

    lifespans = pd.DataFrame(lifespans, columns=['persona', 'follower_id', 'user_id', 'lifespan(days)'])
    lifespans['is_existing'] = lifespans['lifespan(days)'] == '+'
    lifespans['lifespan_num'] = pd.to_numeric(
        lifespans['lifespan(days)'],
        errors='coerce'  # "+" → NaN
    )
    lifespan_summary = lifespans.groupby(agg_by).agg(
        total_connections=('lifespan(days)', 'count'),
        existing=('is_existing', 'sum'),
        removed=('lifespan_num', 'count'),
        lifespan_mean=('lifespan_num', 'mean'),
        lifespan_std=('lifespan_num', 'std'),
        lifespan_median=('lifespan_num', 'median')
    ).reset_index()

    lifespan_summary['survival_rate'] = lifespan_summary['existing'] / lifespan_summary['total_connections']

    return lifespan_summary



def posts_analysis(posts):
    """
    The function post_counts the number of posts per user and analyses posts' sentiment polarity.
    """
    sia = SentimentIntensityAnalyzer()

    posts['polarity'] = posts['tweet'].apply(
        lambda x: sia.polarity_scores(x)['compound']
    )

    conditions = [
        posts['polarity'] > 0.05,
        posts['polarity'].between(-0.05, 0.05)
    ]

    choices = ['Positive', 'Neutral']

    posts['polarity_category'] = np.select(
        conditions,
        choices,
        default='Negative'
    )

    posts['word_count'] = posts['tweet'].apply(lambda x: len(x.split(' ')))

    sentiment_summary = posts.groupby('user_id').agg(
        positive_posts=('polarity_category', lambda x: (x == 'Positive').sum()),
        neutral_posts=('polarity_category', lambda x: (x == 'Neutral').sum()),
        negative_posts=('polarity_category', lambda x: (x == 'Negative').sum()),
        total_posts=('polarity_category', 'count'),
        avg_word_count=('word_count', 'mean')
    ).reset_index()
    sentiment_summary = pd.DataFrame(sentiment_summary)
    sentiment_summary = sentiment_summary.merge(posts[['user_id', 'persona']].drop_duplicates(inplace=False), on=['user_id'])

    return sentiment_summary


def add_behavioral_features_to_persona_summary(conn, summary, persona_df, simulation_days=60):

    # POST FEATURES
    posts = pd.read_sql("SELECT * FROM post", conn)
    posts = posts.drop_duplicates(
        subset=['user_id', 'tweet', 'round']
    )

    personas = persona_df[['id', 'persona']].rename(columns={'id': 'user_id'})
    posts = posts.merge(personas, on='user_id', how='left')

    sentiment_summary = posts_analysis(posts)

    sentiment_summary['neutral_posts_ratio'] = (
            sentiment_summary['neutral_posts'] /
            sentiment_summary['total_posts']
    )

    sentiment_summary['negative_posts_ratio'] = (
            sentiment_summary['negative_posts'] /
            sentiment_summary['total_posts']
    )

    sentiment_summary['posts_per_day'] = sentiment_summary['total_posts'] / simulation_days

    # LIFESPAN FEATURES
    follow = pd.read_sql("SELECT * FROM follow", conn)
    personas = personas.rename(columns={'user_id': 'follower_id'})
    follow = follow.merge(personas, on='follower_id', how='left')

    lifespan = lifespan_analysis(follow, agg_by='follower_id')
    lifespan.rename(columns={'follower_id': 'user_id'}, inplace=True)

    behavior_summary = lifespan.merge(sentiment_summary, on='user_id', how='left')

    # statistical significance check
    cols = ['posts_per_day', 'avg_word_count', 'neutral_posts_ratio', 'negative_posts_ratio', 'survival_rate', 'lifespan_mean']

    behavior_summary = behavior_summary[cols + ['persona', 'user_id']]
    behavior_summary['lifespan_mean'].fillna(0, inplace=True)

    for col in cols:
        groups = [
            behavior_summary[behavior_summary['persona'] == p][col]
            for p in behavior_summary['persona'].unique()
        ]

        stat_diff = kruskal(*groups)
        print(f"{col} statistics: p = {stat_diff.pvalue:.4f}")

    behavior_summary = behavior_summary.groupby('persona').agg(
        posts_per_day=('posts_per_day', 'mean'),
        avg_word_count=('avg_word_count', 'mean'),
        neutral_posts_ratio=('neutral_posts_ratio', 'mean'),
        negative_posts_ratio=('negative_posts_ratio', 'mean'),
        survival_rate=('survival_rate', 'mean'),
        lifespan_mean=('lifespan_mean', 'mean')
    ).reset_index()

    # final df
    final_summary = summary.merge(
        behavior_summary,
        on='persona',
        how='left'
    ).set_index('persona')

    return final_summary


def comments_analysis(comments):
    """
    Reddit comments analysis.
    """

    sia = SentimentIntensityAnalyzer()

    # --- text length ---
    comments['word_count'] = (
        comments['body']
        .fillna('')
        .astype(str)
        .str.split()
        .str.len()
    )

    # --- sentiment polarity ---
    comments['polarity'] = comments['body'].apply(
        lambda x: sia.polarity_scores(str(x))['compound']
    )

    # --- sentiment categories ---
    conditions = [
        comments['polarity'] > 0.05,
        comments['polarity'].between(-0.05, 0.05)
    ]

    choices = ['Positive', 'Neutral']

    comments['polarity_category'] = np.select(
        conditions,
        choices,
        default='Negative'
    )

    # --- aggregate per author ---
    features = (
        comments.groupby('author')
        .agg(
            # activity
            total_comments=('body', 'count'),

            # sentiment counts
            positive_comments=(
                'polarity_category',
                lambda x: (x == 'Positive').sum()
            ),

            neutral_comments=(
                'polarity_category',
                lambda x: (x == 'Neutral').sum()
            ),

            negative_comments=(
                'polarity_category',
                lambda x: (x == 'Negative').sum()
            ),

            # sentiment ratios
            positive_ratio=(
                'polarity_category',
                lambda x: (x == 'Positive').mean()
            ),

            neutral_ratio=(
                'polarity_category',
                lambda x: (x == 'Neutral').mean()
            ),

            negative_ratio=(
                'polarity_category',
                lambda x: (x == 'Negative').mean()
            ),

            # polarity statistics
            avg_polarity=('polarity', 'mean'),
            polarity_std=('polarity', 'std'),

            # text features
            avg_word_count=('word_count', 'mean')
        )
        .reset_index()
    )

    return features


def plot_pca(plot_df, colour_by=None, title='PCA'):
    # unique interests
    features = plot_df[colour_by].unique()
    # assign colors
    cmap = plt.cm.get_cmap('tab20', len(features))

    plt.figure(figsize=(8, 6))
    for i, colour in enumerate(features):
        subset = plot_df[plot_df[colour_by] == colour]

        plt.scatter(
            subset['PC1'],
            subset['PC2'],
            label=colour,
            alpha=0.7
        )

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()