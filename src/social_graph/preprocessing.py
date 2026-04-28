import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



PROFESSION_MAP = {
    "Professor": "Science_Academia",
    "Researcher": "Science_Academia",
    "Scientist": "Science_Academia",
    "Teacher": "Science_Academia",
    "Tutor": "Science_Academia",
    "Librarian": "Science_Academia",
    "School Counselor": "Science_Academia",
    "Psychologist": "Science_Academia",
    "Archaeologist": "Science_Academia",
    "Biologist": "Science_Academia",
    "Astronomer": "Science_Academia",
    "Lab Technician": "Science_Academia",
    "Special Education Teacher": "Science_Academia",
    "Student": "Science_Academia",

    "Surgeon": "Healthcare",
    "Nurse": "Healthcare",
    "Doctor": "Healthcare",
    "Dentist": "Healthcare",
    "Physiotherapist": "Healthcare",
    "Paramedic": "Healthcare",
    "Veterinarian": "Healthcare",
    "Medical Assistant": "Healthcare",
    "Home Health Aide": "Healthcare",
    "Caregiver": "Healthcare",
    "Pharmacist": "Healthcare",

    "Personal Trainer": "Sport",
    "Yoga Instructor": "Sport",
    "Athlete": "Sport",
    "Sports Coach": "Sport",
    "Referee": "Sport",

    "IT Technician": "Technology",
    "Network Administrator": "Technology",
    "Software Engineer": "Technology",
    "Web Developer": "Technology",
    "Cybersecurity Analyst": "Technology",
    "Data Scientist": "Technology",
    "Robotics Engineer": "Technology",
    "Electrical Engineer": "Technology",
    "Mechanical Engineer": "Technology",
    "Civil Engineer": "Technology",

    "Lawyer": "Law_Finance_Admin",
    "Judge": "Law_Finance_Admin",
    "Paralegal": "Law_Finance_Admin",
    "Accountant": "Law_Finance_Admin",
    "Human Resources Manager": "Law_Finance_Admin",
    "Business Consultant": "Law_Finance_Admin",
    "Financial Analyst": "Law_Finance_Admin",
    "Stockbroker": "Law_Finance_Admin",
    "Bank Teller": "Law_Finance_Admin",
    "Entrepreneur": "Law_Finance_Admin",
    "Real Estate Agent": "Law_Finance_Admin",

    "Dancer": "Arts_Media",
    "Musician": "Arts_Media",
    "Photographer": "Arts_Media",
    "Filmmaker": "Arts_Media",
    "Actor": "Arts_Media",
    "Comedian": "Arts_Media",
    "Tattoo Artist": "Arts_Media",
    "Writer": "Arts_Media",
    "Graphic Designer": "Arts_Media",
    "Clown": "Arts_Media",
    "Fortune Teller": "Arts_Media",
    "Journalist": "Arts_Media",
    "Street Performer": "Arts_Media",

    "Construction Worker": "Skilled_Trades",
    "Mechanic": "Skilled_Trades",
    "Electrician": "Skilled_Trades",
    "Plumber": "Skilled_Trades",
    "Welder": "Skilled_Trades",
    "Blacksmith": "Skilled_Trades",
    "Handyman": "Skilled_Trades",
    "Carpenter": "Skilled_Trades",
    "Textile Worker": "Skilled_Trades",
    "Factory Worker": "Skilled_Trades",
    "Miner": "Skilled_Trades",
    "Painter": "Skilled_Trades",
    "Day Laborer": "Skilled_Trades",
    "Garbage Collector": "Skilled_Trades",

    "Farmer": "Agriculture",
    "Rancher": "Agriculture",
    "Beekeeper": "Agriculture",
    "Agricultural Worker": "Agriculture",
    "Winemaker": "Agriculture",
    "Fisherman": "Agriculture",

    "Truck Driver": "Transport",
    "Taxi Driver": "Transport",
    "Courier": "Transport",
    "Food Delivery Driver": "Transport",
    "Dock Worker": "Transport",
    "Railway Worker": "Transport",
    "Pilot": "Transport",
    "Postal Worker": "Transport",
    "Flight Attendant": "Transport",

    "Cashier": "Retail_Service",
    "Retail Salesperson": "Retail_Service",
    "Barista": "Retail_Service",
    "Waiter": "Retail_Service",
    "Bartender": "Retail_Service",
    "Hotel Receptionist": "Retail_Service",
    "Janitor": "Retail_Service",
    "Housekeeper": "Retail_Service",
    "Call Center Agent": "Retail_Service",
    "Customer Service Representative": "Retail_Service",
    "Fast Food Worker": "Retail_Service",
    "Baker": "Retail_Service",
    "Butcher": "Retail_Service",
    "Chef": "Retail_Service",
    "Babysitter": "Retail_Service",
    "Dog Walker": "Retail_Service",
    "Personal Assistant": "Retail_Service",

    "Police Officer": "Security",
    "Security Guard": "Security",
    "Corrections Officer": "Security",
    "Firefighter": "Security",
    "Military Officer": "Security",
    "Soldier": "Security",

    "Escort": "Alternative",
    "Scavenger": "Alternative",
    "Street Vendor": "Alternative",
    "Busker": "Alternative",
    "Gambler": "Alternative",
    None: "Alternative"
}


def load_data_and_describe_network(connection):
    """
    This function load simulation data and print basic information about the network.
    It returns DataFrames: users, follow, post.
    """
    follow = pd.read_sql("SELECT * FROM follow", connection)
    users = pd.read_sql("SELECT * FROM user_mgmt", connection)
    posts = pd.read_sql("SELECT * FROM post", connection)

    users.drop([0], inplace=True)  # delete Admin

    users = users.rename(columns={
        'oe': 'openness',
        'co': 'conscientiousness',
        'ex': 'extroversion',
        'ag': 'agreeableness',
        'ne': 'neuroticism'
    })

    n_users = users["id"].nunique()
    n_actions = len(follow)
    n_follow = (follow["action"] == "follow").sum()
    n_unfollow = (follow["action"] == "unfollow").sum()

    n_rounds = pd.read_sql("SELECT COUNT(DISTINCT day) FROM rounds;", connection)
    n_posts = pd.read_sql("SELECT COUNT(DISTINCT id) FROM post;", connection)

    print(
        f"Number of users: {n_users}\n"
        f"Total actions: {n_actions}\n"
        f"Follow actions: {n_follow}\n"
        f"Unfollow actions: {n_unfollow}\n"
        f"Number of rounds: {n_rounds['COUNT(DISTINCT day)'].iloc[0]}\n"
        f"Number of posts: {n_posts['COUNT(DISTINCT id)'].iloc[0]}\n"
    )

    return users, follow, posts


def create_graph(df):
    """
    This function creates a follow graph from a dataframe.
    It checks if graph is connected or not.
    It prints basic information about the graph.
    If graph is not connected, it creates also the biggest connected component.
    It returns created graphs.
    """
    #G = nx.from_pandas_edgelist(df, source="follower_id", target="user_id", create_using=nx.DiGraph())

    G = nx.DiGraph()

    df = df.sort_values(by=['id'])

    for _, row in df.iterrows():
        u = row['follower_id']
        v = row['user_id']

        if row['action'] == 'follow':
            G.add_edge(u, v)

        elif row['action'] == 'unfollow':
            if G.has_edge(u, v):
                G.remove_edge(u, v)

    print(f"Number of nodes: {len(G)}")
    print(f"Number of edges: {G.size()}")

    components = list(nx.weakly_connected_components(G))
    sizes = [len(c) for c in components]

    print("Number of connective components:", len(components))
    print("Components sizes:", sizes)

    if len(components) > 1:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G_lcc = G.subgraph(largest_cc).copy()

        print(f"Number of nodes (LCC): {len(G_lcc)}")
        print(f"Number of edges (LCC): {G_lcc.size()}")

    else:
        G_lcc = G

    return G, G_lcc


def check_profession_with_map(df):
    not_in_map = set()

    for p in df['profession']:
        if p not in list(PROFESSION_MAP.keys()):
            not_in_map.add(p)

    print(f"Number of professions not in map: {len(not_in_map)}\n"
          f"Professions: {not_in_map}")

    return not_in_map


def map_professions(df, map=PROFESSION_MAP):
    """
    Function maps professions for users.
    """
    df['profession'] = df['profession'].map(PROFESSION_MAP)


def create_persona_df(df,  professions_map=PROFESSION_MAP):
    """
    Creates a persona DataFrame. Encode age. Map professions using map.
    Returns personae DataFrame.
    """
    cols = ['id', 'openness', 'conscientiousness', 'extroversion', 'agreeableness', 'neuroticism',
            'age', 'profession', 'gender', 'leaning', 'education_level']

    personae = df[cols].copy()

    personae['age'] = personae['age'].astype(int)
    personae['age'] = personae['age'].apply(encode_age)

    map_professions(personae, map=professions_map)

    return personae


def create_persona_features_df(df, feature_cols=['openness', 'conscientiousness', 'extroversion', 'agreeableness', 'neuroticism']):
    """
    Creates encoded features DataFrame with selected features. Display size of encoded features DF.
    Returns encoded features DataFrame.
    """
    features_df = df[feature_cols].copy()
    features_df_encoded = pd.get_dummies(features_df, columns=feature_cols, drop_first=False)

    print(f"Features encoded size: {features_df_encoded.shape}")

    return features_df_encoded



def encode_age(age):
    if age < 30: return "young"
    elif age < 50: return "middle"
    else: return "old"


def search_best_number_of_personae(features_encoded):
    """
    Function searches best number of persona clusters using Kmeans.
    It prints silhouette scores and plots them.
    """
    scores = []

    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features_encoded)
        score = silhouette_score(features_encoded, labels)
        scores.append(score)
        print(f"k={k} silhouette={score:.4f}")

    plt.plot(range(2, 10), scores)
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.show()


def create_personae(k, df, features_encoded):
    """
    Creates k persona clusters using KMeans. Map numeric labels to strings. Counts persona values.
    Returns df updated with persona label.
    :param k: number of clusters
    :param df: any df with users, e.g. personae
    :param features_encoded: encoded features DataFrame
    :return: df with new column 'persona'
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['persona'] = kmeans.fit_predict(features_encoded)

    persona_map = {}
    for i in range(k):
        persona_map[i] = f"Persona_{i+1}"

    df['persona'] = df['persona'].map(persona_map)

    print(df['persona'].value_counts())

    return df


def add_persona_to_follow(follow_df, personas_df):
    """
    Returns follows merged with persona type.
    """
    # follow_df = follow_df.drop(columns=['id'], inplace=False)
    follow_df.drop_duplicates(inplace=True)

    personas_df = personas_df[['id', 'persona']].reset_index()
    personas_df = personas_df.rename(columns={'id': 'follower_id'}, inplace=False)

    follow_df = follow_df.merge(personas_df, on='follower_id', how='left')
    follow_df.drop(columns=['index'], inplace=True)

    return follow_df