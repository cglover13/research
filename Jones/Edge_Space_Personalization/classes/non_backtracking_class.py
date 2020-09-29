"""
Non-Backtracking Graphs Class
"""

import numpy as np
import networkx as nx
import numpy.linalg as la


class NonBacktrackingGraph():
	"""
	Defines a non-backtracking graph
	Contains functions relating to movements on this graph
	"""
	def __init__(self, G):
		"""
		Initialize graph

		Parameters
		-----------
			G : (nxgraph) networkx graph

		Attributes
		-----------
			G : original graph
			A : adjacency matrix
			D : degree matrix
			P : probability matrix
		"""
		# Set graph
		self.G = G

		# Create degree matrix
		self.D = np.diag([self.G.degree(v) for v in self.G.nodes()])

		# Create adjacency matrix
		self.A = nx.adjacency_matrix(self.G)

		# Create probability matrix
		self.P = la.inv(self.D)@self.A

	# def add_node(self,n):
	# 	"""Update all attributes after adding node

	# 	Parameters
	# 	----------
	# 		n () : node
	# 	"""
	# 	self.G.add_node(n)

	# 	# Update graphs
	# 	self.D = np.diag([G.degree(v) for v in G.nodes()])
	# 	self.A = nx.adjacency_matrix(G)
	# 	self.P = la.solve(D,A)

	# def add_nodes(self,n):
	# 	"""Update all attributes after adding nodes

	# 	Parameters
	# 	----------
	# 		n () : various nodes
	# 	"""
	# 	self.G.add_nodes_from(n)

	# 	# Update graphs
	# 	self.D = np.diag([G.degree(v) for v in G.nodes()])
	# 	self.A = nx.adjacency_matrix(G)
	# 	self.P = la.solve(D,A)

	# def add_edge(self,a,b):
	# 	"""Update all attributes after adding edge

	# 	Parameters
	# 	----------
	# 		a () : first node
	# 		b () : second node
	# 	"""
	# 	self.G.add_edge(a,b)

	# 	# Update graphs
	# 	self.D = np.diag([G.degree(v) for v in G.nodes()])
	# 	self.A = nx.adjacency_matrix(G)
	# 	self.P = la.solve(D,A)

	# def add_edges_from(self,edges):
	# 	"""Update all attributes after adding edges

	# 	Parameters:
	# 		edges () : collection of edges
	# 	"""
	# 	self.G.add_edges_from(edges)

	# 	# Update graphs
	# 	self.D = np.diag([G.degree(v) for v in G.nodes()])
	# 	self.A = nx.adjacency_matrix(G)
	# 	self.P = la.solve(D,A)

	def vol(self):
		"""Find the total volume of a graph"""
		# total = 0
		# for i in self.G.node:
		# 	print(i)
		# 	total += self.G.degree(i)
		# return total
		return sum([self.G.degree(i) for i in self.G.node])

	def trans_prob_matrix_nb(self):
		'''Create nonbacktracking probability matrix
		'''
		# Get volume
		vol = self.vol()
		# Initialize matrix
		self.P_nb = np.zeros((vol,vol))
		# Get edge orders
		self.edge_ordering_dictionary = sorted(list(self.G.edges) + [i[::-1] for i in list(self.G.edges) if i[::-1] != i])
		# Get probability of each edge
		for i,a in enumerate(self.edge_ordering_dictionary):
			for j,b in enumerate(self.edge_ordering_dictionary):
				u = a[0]
				v = a[1]
				x = b[0]
				y = b[1]
				if v == x and y != u:
					self.P_nb[i][j] = float(1/(self.G.degree(v)-1))

def test_1():
	G = nx.Graph()
	G.add_nodes_from([1,2,3,4])
	G.add_edges_from([(1,3),(1,2),(1,4),(2,2),(3,2),(3,4),(4,3)])
	graph = NonBacktrackingGraph(G)
	graph.trans_prob_matrix_nb()
	return graph.P_nb





