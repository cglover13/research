# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import classes.nb_general as NB_G
import scipy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def get_elements(G, eps=.9):
    """ Given a nx graph G, find each component for backtracking and non-backtracking random walks. """
    
    # Find L, R, and B
    A = nx.to_numpy_array(G)
    L, T = NB_G.create_s_t(G)
    R = T.T
    C, B = NB_G.to_edge_space(G,graph=False,B = True)

    # Normal page rank - dual
    W_ = R@L.T
    D_ = np.diag(W_@np.ones(W_.shape[0]))
    
    pi = R@la.solve(R.T@R,np.eye(R.shape[1]))@np.ones(R.shape[1])
    pr_x = la.solve(np.eye(R.shape[0])-eps*W_.T@la.inv(D_),((1-eps)/la.norm(pi,ord=1))*pi)
    
    """ HERE """ 
    # Add correct staring distribution (np.dot(R, np.ones(R.shape[1]))) 
    nx_x = nx.pagerank(nx.from_numpy_array(C), eps, tol=1e-8, max_iter=250)
    
    nx_x = np.fromiter(nx_x.values(), dtype=float)
    vertex_pr = R.T@pr_x
    vertex_pr_nx = R.T@np.array(list(nx.pagerank(nx.Graph(C),alpha=eps,tol=1e-8,max_iter=250).values()))
    vertex_pr_primal_x = np.array(list(nx.pagerank(nx.Graph(G),alpha=eps,max_iter=250, tol=1e-8).values()))

    # Tuple of: analytical dual, numerical dual, analytical vertex, numerical vertex, numerical base
    NORMAL = (pr_x, nx_x, vertex_pr, vertex_pr_nx, vertex_pr_primal_x)
    
    # NB page rank - dual
    D = np.sum(A, axis=0)
    nu = (L@la.inv(np.diag(D)))@np.ones(L.shape[1])
    B_ = W_-np.multiply(W_,W_.T)
    D_f = np.diag(B_@np.ones(B_.shape[0]))
    pr_y = la.solve(np.eye(B_.shape[0])-eps*B_.T@la.inv(D_f), ((1-eps)/L.shape[1])*nu)
    nx_y = nx.pagerank(nx.from_numpy_array(B), eps, tol=1e-8, max_iter=250)
    nx_y = np.fromiter(nx_y.values(), dtype=float)
    vertex_pr_nb = L.T@pr_y
    vertex_pr_nb_nx = L.T@np.array(list(nx.pagerank(nx.Graph(B),alpha=eps, tol=1e-8, max_iter=250).values()))
    NON_BACK = (pr_y, nx_y, vertex_pr_nb, vertex_pr_nb_nx)
    
    return NORMAL, NON_BACK

def fitness(G, eps=.9, method_=0):
    
    try:
        NORM, NB = get_elements(G, eps)
    
        if method_ == 0:
            # Method 0 - Analytical Solution (Primal)
            return np.var(NB[2])/np.var(NORM[2])
        
        elif method_ == 1:
            # Method 1 - Numerical Solution (Primal)
            return np.var(NB[3])/np.var(NORM[3])
     
        elif method_ == 2:
            # Method 2 - Analytical Solution (Dual)
            return np.var(NB[0])/np.var(NORM[0])
            
        else:
            # Method 3 - Numerical Solution (Dual)
            return np.var(NB[1])/np.var(NORM[1])
    except:
        return 0
    
def init_graph(V = 20):
    """ Makes a random graph """
    # V = np.random.randint(4, V)
    E = np.random.randint(4, int(V*(V-1)/2)+1)
    
    G = nx.gnm_random_graph(V,E)
    
    
    return G

def fill_dangling_nodes(G):
    """ Fills any dangling nodes randomly in graph """
    
    A = nx.to_numpy_array(G)
    
    # Identify dangling nodes
    deg_ = A@np.ones(A.shape[0])
    good_nodes = []
    bad_nodes = []
    for k, i in enumerate(deg_>0):
        if i:
            good_nodes.append(k)
        else:
            bad_nodes.append(k)
            
    for n in bad_nodes:
        for i in range(1, np.random.randint(2, A.shape[0])):
            G.add_edge(n, np.random.choice(np.arange(0,A.shape[0],1)))
            
    return G
            
def simulate(v_, children=100, generations=5, pr=True):
    
    BF = []
    BG = []

    CHILDREN = [init_graph(V=v_) for i in range(children)]
    
    for _ in range(generations):
        
        if pr:
            print("Starting generation {}".format(_))
            
        best_fitness = []
        best_graph = []
        
        next_gen_seed = [] # Place to store our graphs to mutate
        
        F = [fitness(c) for c in CHILDREN]
        
        # Take the top 25% of children
        for i in range(int(children*.25)):
            
            # Keep top 25%
            k = np.argmax(F)
            
            # Add top performer to seed generation
            next_gen_seed.append(CHILDREN[k])
            
            # Save top 10% for visual inspection
            if i < int(children*.1):
                best_fitness.append(F[k])
                best_graph.append(CHILDREN[k])
                
            # Delete score
            del F[k]
            # Remove child
            del CHILDREN[k]
        
        if pr:
            print("Best Score: {}".format(best_fitness[0]))
            print("Starting mutations...")
                
        # Mutation 1 - Randomly add edges
        def add_edge(G):
            """ Given graph G, adds a random amount of random edges to it """
            G_ = G.copy()
            non_edges = list(nx.non_edges(G_))
            k = int(np.random.uniform(high=len(non_edges)))
            for _ in range(k):
                idx = np.random.choice(len(non_edges))
                e = non_edges[idx]
                G_.add_edge(e[0], e[1])
                non_edges = list(nx.non_edges(G_))
                
            return G_
                
        # Mutation 2 - Randomly take away edges
        def remove_edge(G):
            """ Given graph G, deletes a random amount of edges from it """
            G_ = G.copy()
            edges = list(G_.edges)
            k = int(np.random.uniform(high=len(edges)))
            for _ in range(k):
                idx = np.random.choice(len(edges))
                e = edges[idx]
                G_.remove_edge(e[0], e[1])
                edges = list(G_.edges)
            return G_
        
        # Mutation 3 - Randomly mix two graphs together
        def mix_graphs(A, B):
            """ Given two network x graphs, randomly mixes them and returns a new graph """
            A_ = nx.to_numpy_array(A).flatten()
            B_ = nx.to_numpy_array(B).flatten()
            
            X = np.vstack((A_,B_))
            idx = np.random.choice([0,1], size=len(A_))
            
            C = np.array([X[i[0], i[1]] for i in zip(idx, np.arange(0,len(A_),1))])
            
            return nx.from_numpy_array(C.reshape((int(np.sqrt(len(A_))), int(np.sqrt(len(A_))))))
        
        
        # Run mutation 1
        added_edges = [add_edge(next_gen_seed[idx]) for idx in np.random.choice(len(next_gen_seed), size=int(children*.25))]
            
        # Run mutation 2
        removed_edges = [fill_dangling_nodes(remove_edge(next_gen_seed[idx])) for idx in np.random.choice(len(next_gen_seed), size=int(children*.25))]
        
        # Run mutation 3
        mixed_graphs = [fill_dangling_nodes(mix_graphs(next_gen_seed[idx[0]], next_gen_seed[idx[1]])) for idx in np.random.choice(len(next_gen_seed), size=(int(children*.25),2))]
    
        # Now we have our next generation
        CHILDREN = [*next_gen_seed, *added_edges, *removed_edges, *mixed_graphs]
        
        # Save our top results for visual inspection
        BF.append(best_fitness)
        BG.append(best_graph)
        
        if pr:
            plt.title("Generation {}, Score {}".format(_, best_fitness[0]))
            nx.draw(best_graph[0])
            plt.show()
        
    return BF, BG
        
def check_metrics():
    
    DOMAIN = np.arange(4,16,1)
    G = [nx.complete_graph(k) for k in DOMAIN]
    
    METRICS = [get_elements(g)[1] for g in G]
    
    print(METRICS)
        
        
        