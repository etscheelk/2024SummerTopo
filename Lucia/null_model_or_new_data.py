# Making null model or new data

import numpy as np
import networkx as nx
import random
import pandas as pd
import oatpy as oat
from ripser import ripser
import sys
import os


utils_path = '/Users/luohanzhi/Desktop/大三上/2024SummerTopo/Gavin/utils'
if utils_path not in sys.path:
    sys.path.append(utils_path)


def format_ripser_output(ripser_res: dict[str: np.ndarray]
                         ) -> pd.DataFrame:
    '''
    Takes the results from a ripser function and formats it as a dataframe
    to be used to compare to other barcodes.

    Args:
        `ripser_res` (dict[str: np.ndarray]): The results from the `ripser`
        function within the `ripser` library
    
    Returns:
        `res` (Dataframe): Formatted and sorted dataframe containing the
        barcode information in "dimension", "birth", and "death" columns
    '''
    dimension = [] # feature dimention
    birth = [] # birth time
    death = [] # death time

    for dim, bc in enumerate(ripser_res['dgms']):
        for b, d in bc:
            dimension.append(dim)
            birth.append(b)
            death.append(d)

    # collect results
    res = pd.DataFrame(data={'dimension': dimension, 'birth': birth, 'death': death})
    res = res.sort_values(['dimension', 'birth', 'death'], ignore_index=True)
    return res


def persistance_image(barcode: pd.DataFrame,
                      dim: int | list[int] | None = None,
                      sigma: float = 0.1,
                      res: int = 20,
                      weight_func: callable = lambda x, y: y,
                      return_vec: bool = False
                      ) -> np.ndarray | dict[int: np.ndarray]:
    '''
    Makes a persitance image from a homology dataframe

    The coordinates for the peristance image are normalized within [0, 1]. The function
    treats features that never die as having a death coordiate of 1, so features that die
    in the last period aren't distingushed from features that never die. If you want to
    plot the perstance image, use
    ```
        import matplotlib.pyplot as plt

        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        x, y = np.meshgrid(x, y)
        res = persistance_image(
                homology,
                res=resolution,
                return_vec=False # return_vec defaults false, so you can leave it out, but not make it true
            )

        plt.axis('equal')
        plt.pcolormesh(x, y, res[dimension])
    ```

    Args:
        `barcode` (Dataframe): A dataframe with homological features of the simplicial
        complex. Should have a "birth" column for when features were born, a "death"
        column with when features died, and a "dimension" column with the feature
        dimension
        `dim` (int | list[int] | None): The dimensions to create a persistance image for.
        If an int, creates an image for only that dimension. If a list (or other iterable),
        creates an image for all the dimensions in the list. If None, creates an image for
        all unique dimensions in `barcode`.
        Default None.
        `sigma` (float): Standard deviation of the Gaussians used for the persistance
        image. Default 0.1
        `res` (int): Resolution of the returned peristance image. Default 20
        `weight_func` (callable): Function in f(x, y) form that takes the birth, lifetime
        coordinates of the feature and returns the weight of the feature in the gaussians.
        f(x, 0) should be 0 for all x values. Defaults to linear scaling based only on the
        lifetime 
        `return_vec` (bool): Whether to return an matrix or vector. If True, returns a
        vector. Typically, a matrix will work better to visualize the persistance image but
         a vector is better to compare two barcodes. Default False
    
    Returns:
        `persistance_images` (list[np.ndarray]): A list of matricies of vectors (depening on
        the return_vec value) with the peristance images. Each value in the list corresponds
        to a different dimension, with the ith value corresponding to the persistance image
        for i dimensional features
    '''
    # we want it in a list
    if isinstance(dim, int):
        dim = [dim]

    # maximum lifetime
    # we want to normalize everything to [0, 1], this is the value we divide by to do that
    max_lifetime = oat.barcode.max_finite_value(barcode['death'])

    # points on the persistance diagram
    persistance_diagram = pd.DataFrame()
    persistance_diagram['dimension'] = barcode.reset_index(drop=True)['dimension'] # Each unique dimension will be a different diagram
    persistance_diagram['birth'] = barcode.reset_index(drop=True)['birth'] / max_lifetime # normalize birth times
    persistance_diagram['death'] = barcode.reset_index(drop=True)['death'] / max_lifetime # normalize death times
    persistance_diagram.loc[persistance_diagram['death'] == np.inf, 'death'] = 1 # set inf death times to the max (1)
    persistance_diagram['lifetime'] = persistance_diagram['death'] - persistance_diagram['birth'] # birth, lifetime create persistance image basis
    persistance_diagram['weight'] = weight_func(persistance_diagram['birth'], persistance_diagram['lifetime'])

    # remove rows after max dimension
    if dim is not None:
        persistance_diagram = persistance_diagram[persistance_diagram['dimension'].isin(dim)]
    else:
        dim = list(persistance_diagram['dimension'].unique()) # loop over dim later, needs to be defined

    # matricies to calculate peristance image
    # x, y coordinate of every entry i, j in the persistance image is x[0][i, j], x[1][i, j]
    x = np.meshgrid(*[np.linspace(0, 1, res) for _ in range(2)])

    # apply gaussians and get images
    persistance_images = []
    for d in dim: # create a different persisance image for each dimension
        d_persistance_diagram = persistance_diagram[persistance_diagram['dimension'] == d] # peristance diagram fo dimension
        gaussians = rc.gen_stacked_gaussians(
                As=np.array(d_persistance_diagram['weight']),
                x0s=np.array(d_persistance_diagram[['birth', 'lifetime']]),
                sigmas=np.full(len(d_persistance_diagram), sigma)
            )
        persistance_image = np.full((res, res), 0) + gaussians(np.stack(x, axis=-1)) # full makes sure we have all 0s if dimension has no features
        persistance_images.append(persistance_image)
    
    # make a vector (if we should)
    if return_vec:
        persistance_images = [np.reshape(pi, -1) for pi in persistance_images]
    
    # reshape returned value
    if len(persistance_images) == 1: # take it out of the list if there's only one thing
        persistance_images = persistance_images[0]
    else: # make a dict with the dim values
        persistance_images = {d: pi for d, pi in zip(dim, persistance_images)}

    return persistance_images

# filter the data
def filter(article_concept_df, NUM_ARTICLE_MIN, NUM_ARTICLE_MAX):
    '''
    Filters the input DataFrame of articles and concepts based on frequency and relevance criteria.
    
    Parameters:
    article_concept_df (pd.DataFrame): A DataFrame containing articles and associated concepts.
    NUM_ARTICLE_MIN (int): Minimum number of articles a concept must appear in to be retained.
    NUM_ARTICLE_MAX (int): Maximum number of articles a concept can appear in to be retained.
    
    Returns:
    pd.DataFrame: The filtered DataFrame with renamed columns for clarity.
    '''
    RELEVANCE_CUTOFF = 0.7
    sizes = article_concept_df.groupby('concept').transform('size') # match each concept to the number of times it appears
    article_concept_df = article_concept_df[(sizes >= NUM_ARTICLE_MIN)
            & (sizes <= NUM_ARTICLE_MAX)]
    article_concept_df = article_concept_df[article_concept_df['mean'] > RELEVANCE_CUTOFF] # filter relevance   
    article_concept_df = article_concept_df.rename(columns={ # rename for clarity
        "mean":"relevance_mean",
        "size":"concept_freq_in_abstract"
    })
    return(article_concept_df)



#Adding edges from the data
def add_edges(article_concept_df):
    '''
    Creates a list of edges representing relationships between concepts based on their co-occurrence in articles.
    
    Parameters:
    article_concept_df (pd.DataFrame): A DataFrame containing columns 'article_id', 'year', and 'concept'.
    
    Returns:
    np.array: An array of edge tuples, where each tuple represents an edge between two concepts, 
              with associated attributes 'article_id' and 'year'.
    '''
    
    edge_df = article_concept_df[['article_id', 'year', 'concept']].merge( # combine to make edge list
        right=article_concept_df[['article_id', 'year', 'concept']],
        on=['article_id', 'year'],
        how='outer'
    )
    edge_df = edge_df[edge_df['concept_x'] < edge_df['concept_y']] # remove rows where the concepts are equal and make sure theres only one of each row
    edge_df = edge_df.sort_values( # sort to get the earlier year to the start
            ['concept_x', 'year'] # technially don't need to sort by concept_x, but it makes it easier for a human to understand
        ).drop_duplicates( # Keep only the first occurance if a row is duplicated
            subset=['concept_x', 'concept_y']
        )
    edge_tuple_gen  = lambda row: ( # map to make edge tuples
        row['concept_x'], # edge source
        row['concept_y'], # edge target
        { # dict of attributes
            'article_id': row['article_id'], # article edge is from
            'year': row['year'] # year article was published in
        }
    )
    edges = np.array(edge_df.apply(edge_tuple_gen, axis=1)) # make edge tuples
    return(edges)


#Adding nodes from the data
def add_nodes(article_concept_df):
    """
    Creates a list of nodes representing the concepts in articles.
    
    Parameters:
    article_concept_df (pd.DataFrame): A DataFrame containing columns 'article_id', 'year', and 'concept'.
    
    Returns:
    np.array: An array of nodes tuples.
    """
    node_tuple_gen = lambda row: ( # map to make node tuples
        row['concept'], # node
        { # attribute dict
            'article_id': row['article_id'], # article edge is from
            'year': row['year'] # year article was published in
        }
    )
    article_concept_df = article_concept_df.sort_values( # sort to get the earlier year to the start
        ['concept', 'year'] # technially don't need to sort by concept_x, but it makes it easier for a human to understand
    ).drop_duplicates( # Keep only the first occurance if a concept
        subset='concept'
    )
    nodes = np.array(article_concept_df.apply(node_tuple_gen, axis=1)) # make edge tuples
    return(nodes)


def get_G(article_concept_df, B, lower_concept, up):
    '''
    enerates a graph of concepts based on the co-occurrence of concepts in articles.
    
    Parameters:
    article_concept_df (pd.DataFrame): The dataset containing article information with columns 'article_id', 'year', and 'concept'.
    B (list): A list of numbers representing concepts to exclude from consideration.
    lower_concept (int): The lower bound for concept numbers to consider.
    up (int): The upper bound for concept numbers to consider.
    
    Returns:
    list: A list containing the generated graph and the chosen minimum article number (NUM_ARTICLE_MIN).
    '''
    number_of_nodes = 2000
    n = 40

    A = list(range(lower_concept, up))
    A_filtered = list(set(A) - set(B))
    
    NUM_ARTICLE_MIN=np.random.choice(A_filtered)
    print(NUM_ARTICLE_MIN)

    if NUM_ARTICLE_MIN< 40:
        n = 20

    while number_of_nodes>1000:
        NUM_ARTICLE_MAX = NUM_ARTICLE_MIN+n
        concept_df = filter(article_concept_df, NUM_ARTICLE_MIN, NUM_ARTICLE_MAX)

        edges = add_edges(concept_df)
        nodes = add_nodes(concept_df)

        number_of_nodes = len(nodes)
        n = n-2


    concept_G = nx.Graph()
    concept_G.add_nodes_from(nodes)
    concept_G.add_edges_from(edges)

    print(concept_G)

    for (u, v, d) in concept_G.edges(data=True):
        year = d['year']
        weight = (year - 1920) / (2021 - 1920)
        concept_G[u][v]['weight'] = weight
    
    return[concept_G, NUM_ARTICLE_MIN]


def get_G_one(article_concept_df, lower_concept, up):
    """
    Generates a graph of concepts based on the co-occurrence of concepts in articles, within a specified range of concept frequencies.
    
    Parameters:
    article_concept_df (pd.DataFrame): The dataset containing article information with columns 'article_id', 'year', and 'concept'.
    lower_concept (int): The lower threshold for the number of articles a concept must appear in to be included.
    up (int): The upper threshold for the number of articles a concept can appear in to be included.
    
    Returns:
    nx.Graph: A NetworkX graph where nodes represent concepts and edges represent co-occurrence of concepts in articles.
    """
    concept_df = filter(article_concept_df, lower_concept, up)

    edges = add_edges(concept_df)
    nodes = add_nodes(concept_df)


    concept_G = nx.Graph()
    concept_G.add_nodes_from(nodes)
    concept_G.add_edges_from(edges)

    print(concept_G)

    for (u, v, d) in concept_G.edges(data=True):
        year = d['year']
        weight = (year - 1920) / (2021 - 1920)
        concept_G[u][v]['weight'] = weight
    
    return concept_G



# Function to sample a random number with distribution F by using triangle-based higher-order stub
def sample_from_distribution(F):
        return np.random.choice(F)

def generate_network(n, F):
    """
    Generates a network with `n` nodes where each node's degree is sampled from a given distribution `F`,
    and ensures that the total degree is divisible by 3 to form a valid graph structure.

    Parameters:
    n (int): Number of nodes in the network.
    F (function): A function that defines the distribution from which node degrees are sampled.

    Returns:
    nx.Graph: A NetworkX graph where nodes are connected to form triangles based on the sampled degrees.
    """
    V = list(range(1, n + 1))
    totalDeg = 1
    S1 = []


    # Step 1: Ensure totalDeg is divisible by 3
    while totalDeg % 3 != 0:
        S1 = []
        totalDeg = 0
        while len(S1) < n:
            X = sample_from_distribution(F)
            X_floor = int(np.floor(X))
            if 1 <= X_floor <= (n-1)*(n-2)//2:
                S1.append(X_floor)
                totalDeg += X_floor

    # Step 2: Create stubs for each node
    S2 = []
    for i in range(n):
        S2.extend([i] * S1[i])
    
    # Step 3: Form triangles from stubs
    G = nx.Graph()
    G.add_nodes_from(V)
    while len(S2) > 0:
        h1 = np.random.choice(S2)
        S2.remove(h1)
        h2 = np.random.choice(S2)
        S2.remove(h2)
        h3 = np.random.choice(S2)
        S2.remove(h3)
        
        G.add_edge(h1+1, h2+1)
        G.add_edge(h1+1, h3+1)
        G.add_edge(h2+1, h3+1)
    
    return G



# Local edge swapping method to make a null model
def local_edge_swap(graph, k, num_swaps_per_node):
    """
    Performs local edge swaps within a k-neighborhood of each node in the graph to randomize connections while preserving the node degree distribution.

    Parameters:
    graph (nx.Graph): The input graph on which to perform edge swaps.
    k (int): The radius of the neighborhood around each node within which edge swaps are performed.
    num_swaps_per_node (int): The number of edge swaps to perform for each node.

    Returns:
    nx.Graph: A new graph with edges swapped within the k-neighborhoods of nodes.
    """
    # Make a copy of the original graph
    H = graph.copy()

    
    # Iterate through each node in the graph
    for node in H.nodes:
        # Get the k-neighborhood of the node
        k_neighborhood = nx.ego_graph(H, node, radius=k)
        
        # Get the edges within the k-neighborhood
        edges = list(k_neighborhood.edges())
        
        # Perform edge swaps within the k-neighborhood
        for _ in range(num_swaps_per_node):
            if len(edges) < 2:
                break
            # Choose two random edges to swap
            edge1, edge2 = random.sample(edges, 2)
            (u1, v1) = edge1
            (u2, v2) = edge2
            
            # Avoid creating self-loops or duplicate edges
            if len({u1, v1, u2, v2}) == 4:
                if not H.has_edge(u1, v2) and not H.has_edge(u2, v1):
                    # Perform the edge swap
                    H.remove_edge(u1, v1)
                    H.remove_edge(u2, v2)
                    H.add_edge(u1, v2)
                    H.add_edge(u2, v1)
                    # Update the edges list
                    edges.remove(edge1)
                    edges.remove(edge2)
                    edges.append((u1, v2))
                    edges.append((u2, v1))
                    
    return H


#Enter a Graph and a diction, count the number of occurance of each edge weight
def weight_distribution(G, diction):
    """
    Computes the distribution of edge weights in the graph and updates the provided dictionary with the counts of each weight.

    Parameters:
    G (nx.Graph): The input graph with edges that have 'weight' attributes.
    diction (dict): A dictionary to store the count of each weight.

    Returns:
    dict: The updated dictionary with the count of each weight in the graph.
    """
    weight_count = diction
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        if weight in weight_count:
            weight_count[weight] += 1
        else:
            weight_count[weight] = 1    
    return(weight_count)



#Randomly choose different real networks
def data_barcode(l, dim, lower_concept, up, article_concept_df, diction, mylist):
    """
    Generates a list of barcodes for 'l' different graphs created from the input data. Each graph is constructed
    by randomly selecting a threshold within the given range. Computes the persistent homology of each graph and
    updates the provided list and dictionary with the results.

    Parameters:
    l (int): The number of graphs to generate.
    dim (int): The maximum dimension for computing the persistent homology.
    lower_concept (int): The lower threshold for the number of articles a concept must appear in to be included.
    up (int): The upper threshold for the number of articles a concept can appear in to be included.
    article_concept_df (pd.DataFrame): The dataset containing article information with columns 'article_id', 'year', and 'concept'.
    diction (dict): A dictionary to store the count of each weight.
    mylist (list): A list to store the barcodes of the graphs.

    Returns:
    list: A list containing the updated list of barcodes, the updated dictionary of weight counts, the list of generated graphs,
          the list of node counts for each graph, and the list of edge counts for each graph.
    """

    g_list = []
    b_list = []
    edge_list = []
    node_list = []

    for i in range(l):

        #Get the graph G
        list_g_b = get_G(article_concept_df, b_list, lower_concept, up)
        G = list_g_b[0]
        b = list_g_b[1]
        b_list.append(b)

        node_number = G.number_of_nodes()
        edge_number = G.number_of_edges()

        node_list.append(node_number)
        edge_list.append(edge_number)


     # Find the distance matrix
        dist_matrix_G = nx.adjacency_matrix(G, weight='weight')

        # Compute the Barcode Using Ripser
        G_res = ripser(dist_matrix_G, maxdim=dim, distance_matrix=True)
        G_res = format_ripser_output(G_res)

        # Update diction
        diction = weight_distribution(G, diction)
        

        #Add barcode into the list
        
        mylist.append(G_res)
        g_list.append(G)

    return [mylist, diction, g_list, node_list, edge_list]


#One network
def data_barcode_one(l, dim, lower_concept, up, article_concept_df, diction, mylist):
    """
    Generates a list of barcodes for a single graph created from the input data with specified thresholds.
    Repeats the same graph 'l' times. Computes the persistent homology of the graph and updates the provided list 
    and dictionary with the results.

    Parameters:
    l (int): The number of times to repeat the barcode computation.
    dim (int): The maximum dimension for computing the persistent homology.
    lower_concept (int): The lower threshold for the number of articles a concept must appear in to be included.
    up (int): The upper threshold for the number of articles a concept can appear in to be included.
    article_concept_df (pd.DataFrame): The dataset containing article information with columns 'article_id', 'year', and 'concept'.
    diction (dict): A dictionary to store the count of each weight.
    mylist (list): A list to store the barcodes of the graph.

    Returns:
    list: A list containing the updated list of barcodes, the updated dictionary of weight counts, and the list of the repeated graph.
    """
    G_list = []

    #Get the graph G
    G = get_G_one(article_concept_df, lower_concept, up)


    # Find the distance matrix
    dist_matrix_G = nx.adjacency_matrix(G, weight='weight')

    # Compute the Barcode Using Ripser
    G_res = ripser(dist_matrix_G, maxdim=dim, distance_matrix=True)
    G_res = format_ripser_output(G_res)

    # Update diction
    diction = weight_distribution(G, diction)
        

    #Add barcode into the list
    for i in range(l):
        mylist.append(G_res)
        G_list.append(G)
    

    return [mylist, diction, G_list]




def local_swap_barcode(l, dim, k, n, G_list, mylist):
    """
    Performs local edge swaps on a list of graphs and computes the persistent homology of the modified graphs.
    Appends the resulting barcodes to the provided list.

    Parameters:
    l (int): The number of graphs to process from G_list.
    dim (int): The maximum dimension for computing the persistent homology.
    k (int): The radius of the neighborhood for the local edge swaps.
    n (int): The number of edge swaps to perform per node.
    G_list (list): A list of graphs to perform edge swaps on.
    mylist (list): A list to store the barcodes of the modified graphs.

    Returns:
    list: The updated list of barcodes after performing edge swaps and computing persistent homology.
    """
    for i in range(l):
        G = G_list[i]

        num_swaps_per_node = n
        
        # Get the local edge swap graph
        G_l = local_edge_swap(G, k, num_swaps_per_node)
        print(i)

        dist_matrix_G_l = nx.adjacency_matrix(G_l, weight='weight')

        G_l_res = ripser(dist_matrix_G_l, maxdim=dim, distance_matrix=True)
        G_l_res = format_ripser_output(G_l_res)

        mylist.append(G_l_res)

    return mylist

def double_swap_barcode(l, dim, n, m, G_list, mylist):
    """
    Performs double edge swaps on a list of graphs and computes the persistent homology of the modified graphs.
    Appends the resulting barcodes to the provided list.

    Parameters:
    l (int): The number of graphs to process from G_list.
    dim (int): The maximum dimension for computing the persistent homology.
    n(int): Number of double-edge swaps to perform
    m (int): Maximum number of attempts to swap edges, greater than n
    G_list (list): A list of graphs to perform edge swaps on.
    mylist (list): A list to store the barcodes of the modified graphs.

    Returns:
    list: The updated list of barcodes after performing edge swaps and computing persistent homology.
    """


    for i in range(l):
        G0 = G_list[i]
        G = G0.copy()
        
        # Get the local edge swap graph
        G_d = nx.double_edge_swap(G, nswap=n, max_tries=m, seed=None)

        dist_matrix_G_d = nx.adjacency_matrix(G_d, weight='weight')

        G_d_res = ripser(dist_matrix_G_d, maxdim=dim, distance_matrix=True)
        G_d_res = format_ripser_output(G_d_res)

        mylist.append(G_d_res)

    return mylist

def double_swap_graph(l, n, m, G_list):
    """
    Performs double edge swaps on a list of graphs and return the graph list

    Parameters:
    l (int): The number of graphs to process from G_list.
    n(int): Number of double-edge swaps to perform
    m (int): Maximum number of attempts to swap edges, greater than n
    G_list (list): A list of graphs to perform edge swaps on.

    Returns:
    list: The list of graphs after edge swap.
    """

    graph_list = []

    for i in range(l):
        G0 = G_list[i]
        G = G0.copy()
        
        # Get the local edge swap graph
        G_d = nx.double_edge_swap(G, nswap=n, max_tries=m, seed=None)

        graph_list.append(G_d)

    return graph_list