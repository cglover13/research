# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import networkx as nx

def vis_eig(G):
    """ Displays the eigenvalues of a given graph or numpy array 
    
        Input:
            G (numpy array or networkx graph): Adjacency matrix or numpy array
            
        Returns:
            r (list): Tuple of eigenvalues of given matrix
    """
        
    if isinstance(G, nx.classes.graph.Graph):
        G = nx.to_numpy_matrix(G)
        
        
    lambda_ = la.eig(G)[0]
    
    plt.figure(figsize=(6, 6))
    plt.grid()
    plt.title("Spectrum of Graph $G$, $\sigma(G)$", fontsize=20)
    plt.scatter([np.real(x) for x in lambda_], [np.imag(x) for x in lambda_], color='r')
    plt.xlabel('$\Re(\lambda)$', fontsize=15)
    plt.ylabel('$\Im(\lambda)$', fontsize=15)
    plt.show()