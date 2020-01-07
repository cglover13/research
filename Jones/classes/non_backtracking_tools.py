# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:09:50 2015

@author: Owner
"""

import numpy as np
import networkx as nx
import numpy.linalg as la
#import matplotlib.pyplot as plt
#import random as rand


def vol(G):
    V=0
    for v in G.nodes():
        V+=G.degree(v)
    return V


def trans_prob_matrix(G):
    A=nx.adjacency_matrix(G)
    D=np.diag([G.degree(v) for v in G.nodes()])
    P=la.inv(D)*A
    return P

def deg_matrix(G):
    return np.diag([G.degree(v) for v in G.nodes()])

def edge_ordering_dictionary(G):
    D=dict()
    i=0
    for u in G.nodes():
        for v in G.neighbors(u):
            D.setdefault((u,v),i)
            i=i+1

    return D
    
def trans_prob_matrix_nb(G):
    P=np.zeros((vol(G),vol(G)))
    D=edge_ordering_dictionary(G)
    for a in D.keys():
        i=D[a]
        for b in D.keys():
            j=D[b]
            u=a[0]
            v=a[1]
            x=b[0]
            y=b[1]
            if v==x and y!=u:
                P[i][j]=float(1)/(G.degree(v)-1)
    return P

def trans_prob_matrix(G):
    P=np.zeros((vol(G),vol(G)))
    D=edge_ordering_dictionary(G)
    for a in D.keys():
        i=D[a]
        for b in D.keys():
            j=D[b]
            u=a[0]
            v=a[1]
            x=b[0]
            y=b[1]
            if v==x:
                P[i][j]=float(1)/(G.degree(v)-1)
    return P
    
def edge_adjacency_nb_matrix(G):
    B=np.zeros((vol(G),vol(G)))
    D=edge_ordering_dictionary(G)
    for a in D.keys():
        i=D[a]
        for b in D.keys():
            j=D[b]
            u=a[0]
            v=a[1]
            x=b[0]
            y=b[1]
            if v==x and y!=u:
                B[i][j]=1
    return B

def edge_adjacency_matrix(G):
	C = np.zeros((vol(G),vol(G)))
	D = edge_ordering_dictionary(G)
	for a in D.keys():
		i = D[a]
		for b in D.keys():
			j = D[b]
			v = a[1]
			x = b[0]
			if v == x:
				C[i][j] = 1
	return C

def dir_nb_laplacian(G):
    I = np.eye(vol(G))
    P = trans_prob_matrix_nb(G)
    return (I - (P+np.transpose(P))/2)

def vect_to_dict(G,x): #inputs vector of length 2m
    d=dict()
    D=edge_ordering_dictionary(G)
    for a in D.keys():
        d.setdefault(a,x[D[a]])
    return d
    
def dict_to_vect(G,d): #inputs dictionary whose keys are the edges
    x=[0 for i in range(vol(G))]
    D=edge_ordering_dictionary(G)
    for a in D.keys():
        i=D[a]
        x[i]=d[a]
    return x
    
def vertex_distr_to_edge_distr(G,x): #takes a distribution on the vertices and outputs a distrubution on directed edges
    d=dict()
    D=edge_ordering_dictionary(G)
    for a in D.keys():
        u=a[0]
        d.setdefault(a,float(x[u])/G.degree(u))
    y=dict_to_vect(G,d)
    return y
        
def edge_distr_to_vertex_distr(G,y): #takes a distribution on the set of directed edges and returns a distribution on the vertex set
    D=edge_ordering_dictionary(G)
    d=vect_to_dict(G,y)
    x=[0 for i in G.nodes()]
    for a in D.keys():
        u=a[0]
        x[u]+=d[a]
    return x

def nb_rand_steps(G,x,k): #inputs a graph G, initial distribution x, and number of steps k
    P=trans_prob_matrix_nb(G)
    y=vertex_distr_to_edge_distr(G,x)
    yk = np.dot(y,la.matrix_power(P,k))
    return edge_distr_to_vertex_distr(G,yk)
    
def nb_vertex_matrix(G,k): #Outputs the k-step transition prob matrix for a nb random walk on the vertices
    e=list()
    for v in G.nodes():
        x = [0 for u in G.nodes()]
        x[v]=1
        e.append(x)
    P=[nb_rand_steps(G,e[v],k) for v in G.nodes()]
    return P

def in_out_subspace(G):
    P = np.zeros([2*len(G.edges()),2*len(G.nodes())])
    d = edge_ordering_dictionary(G)
    for u in G.nodes():
        for v in G.neighbors(u):
            P[d[(u,v)],u]=1
            P[d[(u,v)],len(G.nodes())+v]=1
    return P
    
def full_out(G):
    P = np.zeros([2*len(G.edges()),2*len(G.edges())])
    d = edge_ordering_dictionary(G)
    for u in G.nodes():
        v = G.neighbors(u)[0]
        for x in G.neighbors(u):
            P[d[(u,v)],d[(u,x)]]=1; P[d[(u,x)],d[(u,v)]]=1
            if x!=v:
                P[d[(u,x)],d[(u,x)]]=-1
    return P
    
def full_in(G):
    P = np.zeros([2*len(G.edges()),2*len(G.edges())])
    d = edge_ordering_dictionary(G)
    for u in G.nodes():
        v = G.neighbors(u)[0]
        for x in G.neighbors(u):
            P[d[(v,u)],d[(x,u)]]=1; P[d[(x,u)],d[(v,u)]]=1
            if x!=v:
                P[d[(x,u)],d[(x,u)]]=-1
    return P
            
    
def ihara_matrix(G):
    n=len(G.nodes())
    C = np.zeros([2*n,2*n])
    for u in G.nodes():
        C[u,n+u]=-1; C[n+u,u]=-1
        for v in G.neighbors(u):
            C[n+u,n+v]=1
            C[n+u,u]+=1
    return C