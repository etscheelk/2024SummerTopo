'''
Generator functions for various random simplicial complexes to use as null models.

Right now, this file has
 - `erdos_renyi_complex`: Creates a simplicial complex based (losely) one the Erdos
 Renyi random graph model
'''

# preliminaries
from scipy import sparse
import networkx as nx
import numpy as np


def erdos_renyi_complex(N: int,
                        density: float | None = 1.,
                        edge_weight_gen: callable = lambda size: 1-np.random.random(size),
                        num_edges: int | None = None,
                        edge_weights: np.ndarray[float] | None = None,
                        return_graph: bool = False,
                        edge_attribute: str = 'weight',
                        seed: int | None = None
                        ) -> sparse.csr_matrix | nx.Graph:
    '''
    Makes a simplicial complex based (kinda) on the Erdos Renyi random graph.

    Essentially, creates an adjacency matrix for an undirected graph with the inputted
    density or number fo edges and random or inputted weights for each of the edges

    Args:
        `N` (int): The number of nodes in the complex
        `density` (float | None): The density of the graph. Used if both `num_edges`
        and `edge_weights` are None. Default 1 (meaning the graph is complete)
        `edge_weight_gen` (callable | None): The function used to generate edge weights.
        Should take a size and return a 1D array of length size with the edges. Edges
        returned with weight 0 will be ignored. Ignored if `edge_weights` is defined. 
        Default 1-np.random.random(size), which randomly and uniformly samples from the
        interval (0, 1]
        `num_edges` (int | None): The number of edges in the simplicial complex if an
        int, otherwise use the density to calculate this. Used if `edge_weights` is
        None. Default None (so we use the density)
        `edge_weights` (np.ndarray[float] | None): The weights of edges in the graph.
        Used to find the number of edges and the edge weight if defined. Ignore if None.
        Default None
        `return_graph` (bool): If True, returns a networkx graph os the generated complex.
        Otherwise, returns a scipy CSR sparse matrix. Default False
        `edge_attribute` (str): The name of the attribute used in the graph. Ignored if
        `return_graph` is false. Default "weight".
        `seed` (int | None): Seed in the numpy random number generator. Default None
    
    Returns:
        `adj` (sparse.csr_matrix): Distance matrix for nodes in the simplicial complex
        (returned if `return_graph` is false)
        `G` (nx.Graph): Graph of the simplicial complex (returned if `return_graph` is
        true)
    '''
    np.random.seed(seed)

    # find number of edges
    if edge_weights is not None: # if we're using a list of edge weights, use that number of edges
        num_edges = len(edge_weights)
    elif num_edges is not None: # if the number of edges is set, use that
        pass
    else:
        num_edges = int(density * N*(N-1)/2) # number of edges in the graph
    
    # find edge weights
    if edge_weights is None:
        edge_weights = edge_weight_gen(num_edges)
    
    # edge coordinates
    i, j = np.triu_indices(N, 1) # all coordinates above the diagnal in the upper triangular matrix
    filter = np.random.choice(int(N*(N-1)/2), num_edges, replace=False) # the indicies of the coordinates to keep
    i, j = i[filter], j[filter] # keep them

    # adjacency matrix
    adj = sparse.csr_matrix((N, N)) # N by N adjacency matrix
    adj[i, j] = edge_weights # set the found coordinates to the values in the edge_weights list
    adj += adj.T # make it symetric

    # graph
    if return_graph:
        G = nx.from_scipy_sparse_array(adj, edge_attribute=edge_attribute)
        return G

    # format adjacency matrix
    adj = adj.tocsr()
    adj.setdiag(0) # oat needs diagnal to be defined and smaller than entries, this is an easy way to do that
    adj = adj.sorted_indices()
    return adj
