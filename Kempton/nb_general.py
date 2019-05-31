

import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import networkx as nx
from networkx.algorithms.connectivity import edge_connectivity, average_node_connectivity


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


def to_edge_space(G, B=False, graph=True):
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
	# Create edge matrix
	if B:
		# Create tau
		tau = np.zeros((len(direct.edges),len(direct.edges)))
		for i,a in enumerate(direct.edges):
			for j,b in enumerate(direct.edges):
				if a[0]==b[1] and a[1]==b[0]:
					tau[i][j] = 1
		if graph:
			return nx.Graph(S@T), nx.Graph(S@T-tau)
		return S@T, S@T - tau
	if graph:
		return nx.Graph(S@T)
	return S@T





