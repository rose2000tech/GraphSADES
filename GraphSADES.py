import networkx as nx
import numpy as np
import random

def GetImportant(Att):
    # Calculate the matrix of importance coefficients between the nodes of the graph
    U = np.array(Att)
    return U

def sort_edges_by_importance(E_activate, U):
    # Sort edges based on importance
    importance = {edge: U[edge[0], edge[1]] for edge in E_activate}
    sorted_edges = sorted(E_activate, key=lambda edge: importance[edge], reverse=True)
    return sorted_edges

def GraphSADES(G, E_activate, E_backup, Att):
    # Step 1: Calculate the matrix of importance coefficients
    U = GetImportant(Att)

    # Step 2: Sort edges based on importance
    E_activate = sort_edges_by_importance(E_activate, U)

    if not np.all(U == 1):
        # Step 4: Move low-importance edges to backup
        len_E = len(E_activate) + len(E_backup)
        lambda_value = 0.6
        theta = len(E_activate) / len_E
        num_edges_to_move = int((1 - lambda_value) * len_E)
        E_backup += E_activate[-num_edges_to_move:]
        E_activate = E_activate[:-num_edges_to_move]

        if theta < lambda_value:
            # Step 7: Add randomly selected edges from backup to activate set
            num_edges_to_add = int((lambda_value - theta) * len(E_activate))
            random.shuffle(E_backup)
            E_activate += E_backup[:num_edges_to_add]
            E_backup = E_backup[num_edges_to_add:]

    # Step 10: Update sets and generate subgraph
    E_new_activate = E_activate
    E_new_backup = E_backup
    G_sub = G.edge_subgraph(E_new_activate).copy()

    return E_new_activate, E_new_backup, G_sub
