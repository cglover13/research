import numpy as np
import networkx as nx
import numpy.linalg as la
from itertools import permutations
# All functions assume the graph is networkx

def vol(G):
	"""Find volume of a graph"""
	return sum([G.degree(i) for i in G.node])

def deg_matrix(G):
	# Get degree matrix D
    return np.diag([G.degree(v) for v in G.nodes()])

def trans_prob_matrix(G):
	# Get transition probability matrix
    A=nx.adjacency_matrix(G)
    D=deg_matrix(G)
    P=la.inv(D)@A
    return P 

def edge_ordering_dictionary(G):
	"""Assign list to transition from vertices to edges in 
	adjacency matrix"""

	# Ask Dr. Kempton about self directing edges!!!

	return sorted(list(G.edges) + [i[::-1] for i in list(G.edges) if i[::-1] != i])

def trans_prob_matrix_nb(G):
	"""Create transition probability matrix with regards to edges
	as to include non-backtracking"""

	# Get edge ordering_dictionary
	valid_pairs = [i for i in list(permutations(edge_ordering_dictionary(G),2)) if i[0][1] == i[1][0] and i[0][0] != i[1][1]]
	D = edge_ordering_dictionary(G)
	print(D)
