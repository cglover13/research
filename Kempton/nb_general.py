import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import networkx as nx
from networkx.algorithms.connectivity import edge_connectivity, average_node_connectivity

def create_k(G):
    # Get necessary matrices
    A = nx.adjacency_matrix(G).todense()
    D = np.diag(list(dict(G.degree()).values()))
    I = np.eye(D.shape[0])
    return np.block([[A,D-I],[-I,np.zeros((D.shape[0],D.shape[0]))]])

def create_s_t(G):
    direct = G.to_directed()
    # Find S and T
    S = np.zeros((len(direct.edges),len(G.nodes)))
    T = np.zeros((len(G.nodes),len(direct.edges)))
    for i,a in enumerate(direct.edges):
        for j,b in enumerate(G.nodes):
#             print(a,b)
            if a[1] == b:
                S[i,j] = 1
#                 print('S Here')
            if a[0] == b:
#                 print('T Here')
                T[j,i] = 1
    return S, T


def to_edge_space(G, B=False, graph=True, ret_tau = False):
    direct = G.to_directed()
    # Find S and T
    S = np.zeros((len(direct.edges),len(G.nodes)))
    T = np.zeros((len(G.nodes),len(direct.edges)))
    for i,a in enumerate(direct.edges):
        for j,b in enumerate(G.nodes):
#             print(a,b)
            if a[1] == b:
                S[i,j] = 1
#                 print('S Here')
            if a[0] == b:
#                 print('T Here')
                T[j,i] = 1
    # Create tau
        tau = np.zeros((len(direct.edges),len(direct.edges)))
        for i,a in enumerate(direct.edges):
            for j,b in enumerate(direct.edges):
                if a[0]==b[1] and a[1]==b[0]:
                    tau[i][j] = 1
    # Create edge matrix
    if B:
        if graph:
            if ret_tau:
                return nx.Graph(S@T), nx.Graph(S@T-tau), nx.Graph(tau)
            return nx.Graph(S@T), nx.Graph(S@T-tau)
        if ret_tau:
            return S@T, S@T - tau, tau
        return S@T, S@T - tau
    if graph:
        if ret_tau:
            return nx.Graph(S@T), nx.Graph(tau)
        return nx.Graph(S@T)
    if ret_tau:
        return  S@T, tau
    return S@T

def rank(x):
    """
    Computes standardize ranking of vector which accounts for ties
    
    Parameters:
        x (ndarray)
    
    Returns:
        rank (ndarray)
    """
    # Create ranking array and sorted array
    copy = np.copy(x)
    sorted_x = np.sort(copy)
    rank = np.zeros_like(copy)
    # Check rank of each node
    for i, val in enumerate(copy):
        # Account for ties by using np.min
        rank[i] = np.min(np.where(np.around(sorted_x,decimals=12)==round(val,12))[0])
        
    return rank






