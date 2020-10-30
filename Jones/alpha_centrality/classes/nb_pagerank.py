# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 12:49:08 2017

@author: Mark
"""

import networkx as nx
import non_backtracking_tools as nb
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# #G = nx.Graph()
# #G.add_edges_from([(0,1),(1,2),(2,0),(0,3),(1,3),(2,3),(3,4),(0,4),(1,4),(2,4),
# #                  (4,5),(5,6),(8,6),(8,7),(5,7),(6,7),(5,8),(5,9),(6,9),(7,9),(8,9)])

# G = nx.random_partition_graph([50,50],.25,.05)



# F = list(nx.connected_component_subgraphs(G))

# #print F

# G = F[0]

# A = nx.adjacency_matrix(G).todense()
# EA,VA = la.eigh(A)
# #print EA
# #print VA.T

# B = nb.edge_adjacency_nb_matrix(G)

# #print B

# E,V = la.eig(B)

# #print E
# #print V.T

# #for l in E:
# #    print (l,abs(l))

# print '------------------------------------------------------'

# #V1 = [nb.edge_distr_to_vertex_distr(G,V.T[i]) for i in range (len(E))]
# #for i in range(len(E)):
# #    print V1[i]

# print '------------------------------------------------------'

def pr(G,s,a): #Inputs graph G, initial distribution s on the vertices, and jumping constant a between 0 and 1
    return a*np.dot(s,la.inv(np.eye(len(G.nodes())) - (1-a)*nb.trans_prob_matrix(G)))

def q(G,s,a):
    p = pr(G,s,a)
    return [p[v]/G.degree(v) for v in G.nodes()]

def pr_tilde(G,s,a):
    s_prime = nb.vertex_distr_to_edge_distr(G,s)
    pr_prime = a*np.dot(s_prime,la.inv(np.eye(2*len(G.edges())) - (1-a)*nb.trans_prob_matrix_nb(G)))
    # return nb.edge_distr_to_vertex_distr(G,pr_prime)
    return pr_prime
    
def q_tilde(G,s,a):
    p = pr_tilde(G,s,a)
    return [p[v]/G.degree(v) for v in G.nodes()]
    
# s = [0 for v in G.nodes()]
# s[0]=1
# a=.1

# print pr(G,s,a)
# print pr_tilde(G,s,a)


# seed = {}
# for v in G.nodes():
#     seed[v] = 0
# seed[0] = 1

# p = nx.pagerank(G,alpha = 1-a,personalization=seed).values()

# print p



# """
# p= q(G,s,a)
# p[0]=p[1]

# p_tilde = q_tilde(G,s,a)
# p_tilde[0]=p_tilde[1]

# #print pr(G,s,a)
# #print q(G,s,a)
# #print nx.pagerank(G,alpha = a,personalization=seed)
# #print pr_tilde(G,s,a)
# #print(q_tilde(G,s,a))
    
# plt.plot([l.real for l in E],[l.imag for l in E],'o')
# plt.grid()
# #plt.savefig('nb_spect.png')
# plt.show()

# pos = nx.spring_layout(G)

# plt.figure(figsize=(16,9))
# nx.draw_networkx_edges(G,pos)
#                        #nodelist=[ncenter],
#                        #width=edge_widths, 
#                        #edge_color=weight_list.values(),                        
#                        #edge_cmap=plt.cm.Blues)
                       
                       
# nx.draw_networkx_nodes(G,pos,
#                        #nodelist=p.keys(),
#                        node_size=100,
#                        node_color=p,
#                        cmap=plt.cm.Greens,
#                        linewidths=.3)
                       
# #nx.draw_networkx_nodes(G,pos,nodelist=p.keys(),
# #                       node_size=100,
# #                       node_color=p.values(),
# #                       cmap=plt.cm.Greens)
                       
# plt.xlim(-0.05,1.05)
# plt.ylim(-0.05,1.05)
# plt.axis('off')
# plt.savefig('pr_cluster0.png')
# plt.show()    

# plt.figure(figsize=(16,9))
# nx.draw_networkx_edges(G,pos)
#                        #nodelist=[ncenter],
#                        #width=edge_widths, 
#                        #edge_color=weight_list.values(),                        
#                        #edge_cmap=plt.cm.Blues)
                       
                       
# nx.draw_networkx_nodes(G,pos,
#                        #nodelist=p.keys(),
#                        node_size=100,
#                        node_color=p_tilde,
#                        cmap=plt.cm.Greens,
#                        linewidths=.3)
                       
# #nx.draw_networkx_nodes(G,pos,nodelist=p.keys(),
# #                       node_size=100,
# #                       node_color=p.values(),
# #                       cmap=plt.cm.Greens)
                       
# plt.xlim(-0.05,1.05)
# plt.ylim(-0.05,1.05)
# plt.axis('off')
# plt.savefig('pr_cluster0_nb.png')
# plt.show()    
# """