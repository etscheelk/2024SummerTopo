'''
Generator functions for various random simplicial complexes to use as null models.

Right now, this file has
 - `erdos_renyi_complex`: Creates a simplicial complex based (losely) one the Erdos
 Renyi random graph model
 - `gen_gaussian`: Creates a functions for a gaussian with a given center, weight,
 and spread. Works in any dimension
 - `gen_stacked_gaussians`: Creates a function for a bunch of added gaussians with
 a given center, weight, and spread. Works in any dimension
 - `sample_from_pdf`: Rejection sample to sample according to a user defined pdf.
 Works faster for vectorized PDFs, but can work for any through a parameter
 - `complex_from_points`: Creates a simplicial complex from a set of points. Edge
 weights are the distances between points
 - `assign_edge_weights`: Assigned either random or user defined edge weights to a
 network for use as a simplicial complex
'''

# preliminaries
from scipy import sparse, spatial
import networkx as nx
import numpy as np
import warnings


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
    adj.setdiag(0) # oat needs diagnal to be defined and smaller than entries, this is an easy way to do that
    adj = adj.sorted_indices()

    return adj


def gen_gaussian(A: float,
                 x0: float | np.ndarray[float],
                 sigma: float,
                 ) -> callable:
    '''
    Creates a function that follows a Gaussian with the inputted parameters

    To plot a 1D gaussian, use
    ```
    import matplotlib.pyplot as plt
    import numpy as np

    res = 250 # number of points we calculate at
    gaussian = gen_gaussian(A, x0, sigma) # gaussian function
    x = np.linspace(0, 1, res) # points to calculate plot on

    plt.plot(x, gaussian(x))
    ```

    To plot a 2D gaussian, use
    ```
    import matplotlib.pyplot as plt
    import numpy as np

    res = 250 # plot resolution
    gaussian = gen_gaussian(A, np.array([x0, y0]), sigma) # gaussian function, needs to be 2 dimensional to plot it
    x = np.meshgrid(*[np.linspace(0, 1, res) for _ in range(2)]) # grid to calculate plot on

    cm = plt.pcolormesh(*x, gaussian(np.stack(x, axis=-1)))
    plt.colorbar(cm)
    ```

    Args:
        `A` (float): Weight of the gaussian
        `x0` (float | np.ndarray[float]): The location of the gaussian. The dimension of
        the function is calculated based on the length of the list. If just a number,
        creates a 1 dimensional function
        `sigma` (float): The spread of the gaussian.
    
    Returns:
        `gaussian` (callable): Gausian function of weight `A` and spread `sigma`
        centered at `x0`. The function takes a numpy array x of the coordiante you want to
        find the value of the gaussian at and returns the value. It does work with
        matricies with each row as a different coordinate as the input
    '''
    var = 2*sigma**2 # not actual variance, idc, its what the function needs

    # multidimensional gaussian
    if isinstance(x0, np.ndarray):
        if len(x0) == 1: # 1d in a list
            x0 = x0[0]
        else:
            gaussian = lambda x: A * np.exp(-np.sum((x - x0)**2, axis=-1) / var) # follows gaussian formula
    
    # 1d gaussian
    if isinstance(x0, float):
        gaussian = lambda x: A * np.exp(-(x - x0)**2 / var) # follows gaussian formula
        return gaussian

    return gaussian


def gen_stacked_gaussians(As: list[float],
                          x0s: list[float] | list[np.ndarray[float]],
                          sigmas: list[float],
                          ) -> callable:
    '''
    Creates a function that sums gaussians with the inputted parameters at a point

    Args:
        `A` (list[float]): Weight of each gaussian
        `x0` (list[float] | list[np.ndarray[float]]): The location of each gaussian.
        The dimension of the function is calculated based on the length of the list.
        If just a number, creates a 1 dimensional function
        `sigma` (list[float]): The spread of each gaussian.

    Returns:
        `gaussians` (callable): Function that takes a numpy array coordinate and
        returns the sum of the gaussians at that point. Can also work with matricies
        where each row is a different coordinate
    '''
    assert len(As) == len(x0s) == len(sigmas) # make sure the number of gaussians is the same for all the lists

    gaussian_funcs = []
    for A, x0, sigma in zip(As, x0s, sigmas): # make each of the gaussians
        gaussian = gen_gaussian(A, x0, sigma)
        gaussian_funcs.append(gaussian)
    
    gaussians = lambda x: sum([g(x) for g in gaussian_funcs]) # add the functions together

    return gaussians


def sample_from_pdf(n: int,
                    dim: int,
                    pdf: callable,
                    bounds: np.ndarray[float] | None = None,
                    pdf_max: float = 1,
                    vectorized: bool = True,
                    max_iter: int = np.inf,
                    supress_warnings: bool = False,
                    seed: int | None = None
                    ) -> np.ndarray[float]:
    '''
    Samples n points in R^d according to the given pdf

    Uses Rejection Sampling to pick n random points in R^(d+1) and check that the
    last of the randomly chosen points is less than the pdf at that point. If it is,
    the first d coords in R^d are added to the sample. If it's not, the point is
    excluded

    Args:
        `n` (int): Number of points in the final sample
        `dim` (int): Dimension fo the final sample
        `pdf` (callable): PDF you want to sample from. Should be a function from
        R^dim -> R which returns numbers in [0, 1). The closer the max is to 1, the
        faster the sampling will be
        `bounds` (np.ndarray[float] | None): A dim*2 array of bounds. The first element
        in each row is the lower bound for that dimension and the second element in
        each row is the upper bound for that dimension. If none, points in each
        dimension are generated in [0, 1). Default None.
        `pdf_max` (float): The max value the pdf reaches within the bounds. Random
        numbers are generated in [0, `pdf_max`) to decide whether a point should be
        kept in the sample or not. If it's too low, some low probability areas will
        be oversampled. If it's too high, the function takes a long time. Default 1
        `vectorized` (bool): Whether the PDF function works on a vector of points or
        if it needs to be done 1 at a time
        `max_iter` (int): The maximum number of iterations to do to get the sample.
        Default inf
        `supress_warnings` (bool): Whether you want to stop the maximum iteration
        warning. Default False
        `seed` (int | None): The seed for the numpy random number generator. Default
        None
    
    Returns:
        `sample` (np.nsarray[float]): A n by dim array of the randomly generated points
    '''
    np.random.seed(seed)

    if bounds is not None:
        scale = np.concatenate((bounds[:, 1] - bounds[:, 0], np.array([pdf_max]))) # upper bound minus lower bound, last ones goes to random checker
        shift = np.concatenate((bounds[:, 0], np.zeros(1))) # lower bound, last one goes to random checker
    else: # generate in a [0, 1) box
        scale = np.concatenate((np.ones(dim), np.array([pdf_max]))) # scale by 1, scale pdf by pdfmax
        shift = np.zeros(dim+1) # shift by nothing

    if not vectorized: # turn the function into something that can be applied to a vector of points
        pdff = lambda s: np.array([pdf(r) for r in s])
        pdf = pdff
    # else:
        # pdff = pdf # use pdff, otherwise pdf becomes recursive with vectorize off

    sample = np.full((n, dim), np.nan) # initialize the sample

    i = 0 # number of iterations
    r = 0 # number of filled rows
    while r < n and i < max_iter: # while the sample has less points than you want
        possible_sample = np.random.random((n-r, dim+1)) * scale[None, :] + shift[None, :] # points to check if they should be in the sample
        filter = possible_sample[:, dim] <= pdf(possible_sample[:, :dim]) # check if points are less than the pdf
        dr = sum(filter) # number of rows we fill this iteration
        sample[r:r+dr, :] = possible_sample[filter, :dim] # fill the rows

        i += 1 # increase iteration count
        r += dr # increase row count

    if i >= max_iter and r < n: # if iteration count exceeded, keep only successful samples
        sample = sample[:r, :]

        if not supress_warnings:
            warnings.warn(f'Max iteration count exceeded, returning sample with n={r}')

    return sample


def complex_from_points(pts: np.ndarray[float],
                        density: None | float = None,
                        num_edges: None | int = None,
                        norm: int = 2,
                        normalize: bool = False,
                        return_graph: bool = False,
                        edge_attribute: str = 'weight'
                        ) -> sparse.csr_matrix | nx.Graph:
    '''
    Create a simplicial complex from a set of points

    Args:
        `pts` (np.ndarray[float]): Array of points. Each row should be a different point
        `density` (None | float): The desired density of the simplicial complex. If set,
        only the shortest edges are kept to keep the network at the desired density. Only
        used if `num_edges` is None. Default None
        `num_edges` (None | int):  The number of edges in the simplicial complex. If set,
        only the shortest edges are kept to have the desired number of edges. Default None
        `norm` (int): The norm to use to find the distace between points
        `normalize` (bool): Whether to normalize the maximum distance between points to 1.
        Default None.
        `return_graph` (bool): If True, returns a networkx graph os the generated complex.
        Otherwise, returns a scipy CSR sparse matrix. Default False
        `edge_attribute` (str): The name of the attribute used in the graph. Ignored if
        `return_graph` is false. Default "weight".

    Returns:
        `adj` (sparse.csr_matrix): Distance matrix for nodes in the simplicial complex
        (returned if `return_graph` is false)
        `G` (nx.Graph): Graph of the simplicial complex (returned if `return_graph` is
        true)
    '''
    n = len(pts) # number of points
    dist_mat = spatial.distance_matrix(pts, pts, p=norm) # distances
    dist_mat[np.tril_indices(n, -1)] = dist_mat[np.triu_indices(n, 1)] # make symetrical (only keep upper triangular part)
    dist_mat[np.diag_indices(n)] = 0 # make diagnal 0 (avoid machine error issues)

    # if no filtering has to be done, return now
    if density is None and num_edges is None:
        dist_mat = sparse.csr_matrix(dist_mat) # oat likes sparse matricies
    else: # keep only the first few edges
        if num_edges is None: # density is defined
            num_edges = int(density * n*(n-1)/2)

        i, j = np.triu_indices(n, 1)
        dist_arr = dist_mat[i, j] # elemnts in distance matrix, each array corresponds to i, j
        dist_order = np.argsort(dist_arr) # sort the indicies, argsort so we can reference i, j at that point
        keep = dist_order[:num_edges] # keep num_edges elements (as edge weights)

        dist_mat = sparse.csr_array((n, n)) # n by n empty sparse array
        dist_mat[i[keep], j[keep]] = dist_arr[keep] # set elements
        dist_mat += dist_mat.T # symetrical

    if normalize:
        dist_mat = dist_mat / dist_mat.max() # make longest length 1

    # graph
    if return_graph:
        G = nx.from_scipy_sparse_array(dist_mat, edge_attribute=edge_attribute)
        return G

    # format it
    dist_mat.setdiag(0) # make diagnal 0 (oat needs it)
    dist_mat = dist_mat.sorted_indices()
    
    return dist_mat


def assign_edge_weights(G: nx.Graph,
                        edge_attribute: str = 'weight',
                        edge_weight_gen: callable = lambda size: 1-np.random.random(size),
                        edge_weights: np.ndarray[float] | None = None,
                        shuffle: bool = False,
                        return_adj: bool = False,
                        seed: int | None = None
                        ) -> sparse.csr_matrix | nx.Graph:
    '''
    Assigns random or user defined weights to edges in a networkx graph

    Args:
        `G` (nx.Graph): Networkx graph you want edge weights assigned to
        `edge_attribute` (str): The name of attribute to be added to the edges. Default
        "weight"
        `edge_weight_gen` (callable): Function to generate edge weights. Takes an int and
        returns that many numbers which will be used as weights for each edge. Default
        creates a random number in (0, 1]
        `edge_weights` (np.ndarray[float] | None): If defined, the edge weights that will
        used in the network. Edge weights will be assigned in the order generated by G.edges
        and randomly ordered if `shuffle` is True. Any extra weights will be ignored and
        missing weights will be set to 0. Default None
        `shuffle` (bool): Whether to randomly order the edge weights array before defining
        them. Default False
        `return_adj` (bool): Whether to return an adjacency matrix instead of a network.
        Default False
        `seed` (int | None): Seed in the numpy random number generator
    
    Returns:
        `G` (nx.Graph): Identical graph to G with edge weights added. Returned if `return_adj`
        is False
        `adj` (sparse.csr_matrix): Scipy sparse adjacency matrix for the graph. Has defined 0s
        accross the diagnal and generated edge weights filled in for nodes that have a weight 
        and empty values when no edge exists. Returned only if `return_adj` is True
    '''
    np.random.seed(seed)
    G = G.copy() # don't modify initial graph
    n = len(G.edges) # number of edges we need to define

    # create the edge weights
    if edge_weights is None:
        edge_weights = edge_weight_gen(n) # edge weights to use
    else:
        edge_weights = np.concatenate((edge_weights, np.zeros(n)))[:n] # keep the first n elements in edge weights and fill in anything extra with zeros
    
    if shuffle: # reorder if set
        np.random.shuffle(edge_weights)
    
    # nx.set_edge_weights takes a dict of edge: {attribute name: value} pairs
    # create that dict
    edge_weight_dict = {e: {edge_attribute: w} for e, w in zip(G.edges, edge_weights)}
    nx.set_edge_attributes(G, edge_weight_dict)

    # return adj
    if return_adj:
        adj = nx.adjacency_matrix(G, weight=edge_attribute) # adjacency matrix
        adj.setdiag(0)
        adj = adj.sorted_indices(0)

        return adj
    
    # return G
    return G


def shuffle_edge_weights(G: nx.Graph,
                         seed: int | None = None
                         ) -> nx.Graph:
    '''
    Create a new graph with the same edges as G but the edge weights randomly reassigned

    Args:
        `G` (nx.Graph): Networkx graph you want edge weights reassigned for
        `seed` (int | None): Seed for the numpy random number generator

    Returns:
        `H` (nx.Graph): Networkx graph thats a copy of G with each edge having a random other
        edge's edge weight
    '''
    # setup
    np.random.seed(seed) # amke predictable

    # new graph with same nodes
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True)) # keep node attributes

    # get edges
    edges = np.array(G.edges)
    weights = np.array(list(dict(G.edges).values()))

    # scramble
    np.random.shuffle(weights)

    # pairs edges and weights
    edge_weight_pairs = [(u, v, attr) for (u, v), attr in zip(edges, weights)]
    H.add_edges_from(edge_weight_pairs) # add edges
    
    return H
