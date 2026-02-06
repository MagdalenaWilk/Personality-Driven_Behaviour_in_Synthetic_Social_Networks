from social_graph.preprocessing import *
from social_graph.describe import *
from social_graph.metrics import *

import networkx as nx


FEATURE_COLS = ['openness', 'conscientiousness', 'extroversion', 'agreeableness', 'neuroticism', 'age', 'profession']


def run_simulation_preprocessing(conn, label='Simulation', feature_cols = FEATURE_COLS):
   """
   Function run whole simulation preprocessing.
   Display basic information.
   Returns personae, encoded features, and follow DataFrames.
   """
   if label != 'Simulation':
       print(f"====================================================\n"
             f"Running {label}...\n")

   print("Data loading ...\n")
   users, follow, posts = load_data_and_describe_network(conn)

   print("Checking professions consistency ...\n")
   profession_not_in_map = check_profession_with_map(users)

   print("\nCreating persona and features DataFrames ...\n")
   personae = create_persona_df(users)
   features_df = create_persona_features_df(personae, feature_cols)

   print("\nSearching best number of persona ...\n")
   search_best_number_of_personae(features_df)

   print("\nDone!")

   return personae, features_df, follow


# These functions are separated, because we want to manually set the best number of personae


def cluster_persona_and_analyse(persona_df, features_df, k, label='Simulation', feature_cols = FEATURE_COLS):
    """
    Function creates and analysis personae.
    Display basic information.
    Returns description of personae and mapping by users id.
    """
    if label != 'Simulation':
        print(f"====================================================\n"
              f"Running {label}...\n")

    print("Persona creation ...\n")
    personae = create_personae(k, persona_df, features_df)
    persona_dict = personae.set_index('id')['persona'].to_dict()

    print("\nPersona analysis ...\n")
    description = describe_personae(personae, feature_cols)
    persona_significance(personae, feature_cols)
    cv_matrix = cramers_v_matrix(personae[feature_cols + ['persona']], label='Persona')

    return description, persona_dict



def build_graph_and_analyse(follow, persona_dict, label='Simulation'):
    """
     Creates follow graph with persona attribute and calculates global and local metrics.
     Returns table with global metrics and local metrics aggregated by persona.
    """
    if label != 'Simulation':
        print(f"====================================================\n"
              f"Running {label}...\n")

    print("Graph creation ...\n")
    G, G_lcc = create_graph(follow)

    nx.set_node_attributes(G, persona_dict, "persona")
    nx.set_node_attributes(G_lcc, persona_dict, "persona")

    print("\nGlobal metrics ...\n")
    global_metrics = calculate_global_metrics(G, G_lcc, label=label).set_index('Metric')

    print("\nLocal metrics ...\n")
    local_metrics, summary = calculate_local_metrics(G_lcc)
    if label != 'Simulation':
        summary['Simulation'] = label

    print("\nStatistical check (metrics vs. persona) ...\n")
    statistical_difference_check(local_metrics)

    return global_metrics, summary