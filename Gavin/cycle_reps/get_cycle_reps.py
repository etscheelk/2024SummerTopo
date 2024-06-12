'''
Calculate homology for a network and get a bunch of representative cycles

The challenge here is that there are a lot of cycles and some cycles can
 - Throw errors
 - Take extremely long to calculate bc of degenerate simplicies
and, for now at least, we only want cycle reps that can be feasibly solved in
a reasonable amount of time. To solve that, we use a multiprocessed pool of
cycles which means 1) everything happens faster, 2) errors only shutdown a
single thread, and 3) we can timeout a single thread without having to
recalculate everything.
'''

# load some packages
import sys; sys.path.append("/Users/gavinengelstad/Documents/Documents - Gavin’s MacBook Pro/School/Summer '24/Research/2024SummerTopo/Gavin/utils")
from concurrent.futures import TimeoutError
from multiprocessing import Process
from pebble import ProcessPool
import make_network as mn
import networkx as nx
import oatpy as oat
import numpy as np
import pickle
import time
import os

# FactoredBoundryMatrixVR is unpickleable, meaning it can't be sent between objects
# instead, we use "fork" to create a copy of the global enviorment and make factored
# a global variable. This means the optimize_cycle function can access it
# this is an awful way of doing it, and not at all advisable, but it works and other
# options dont so
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

# config
DATA_PATH = 'datasets/concept_network/'
CONCEPT_FILE = 'articles_category_for_2l_abstracts_concepts_processed_v1_EX_102.csv.gz' # Applied Mathematics
RESULT_PATH = 'results/applied_math_6-12' # save files here
TIMEOUT_LEN = 2 # seconds
RELEVANCE_FILTER = 0.7
FREQ_MIN_FILTER = 0.0006 # 0.006%
FREQ_MAX_FILTER = 0.005 # 0.05%
MIN_YEAR = 1920
NUM_PROCESSES = 5 # number of processed to do with multithreading
DIM_CONDITION = lambda dim: dim == 1 # dimension of rows which we optimize a cycle for


def optimize_cycle(cycle, nodes):
    '''
    Cycle is a row of the `homology` dataframe. That means it has columns
        - "dimension": The cycle dimension
        - "birth": The cycle birth filtration level
        - "death": The cycle death filtration level
        - "birth simplex": The final simplex that makes the cycle a cycle (dim)
        - "death simplex": The simplex that closes the cycle (dim+1)
        - "cycle representative"": Dataframe with an unoptimized cycle rep
        - "cycle nnz": The number of simplexes in the unoptimized cycle rep
        - "bounding chain": Dataframe with the simplicies that fill the
        unoptimized cycle
        - "bounding nnz": The number of simplicies in the bounding chain
    It also has a `cycle.name` attribute with the index of the cycle in homology

    Factored is the FactoredBoundryMatrixVR object. We use it to solve for the
    cycle rep

    Nodes is a numpy array of the concepts each node represents. We use it to
    convert each simplex index into a concept
    '''
    # optimize cycle
    start = time.time()
    optimal = factored.optimize_cycle( # optimial cycle rep
            birth_simplex=cycle['birth simplex'], 
            problem_type='preserve PH basis'
        )
    time_to_solve = time.time() - start
    print(f'Cycle {cycle.name} finished in {time_to_solve} secs')

    # filter optimal cycle
    # we want all coefficients to be -1 or 1, the optimization problem has machine error
    # so some can be ~-1, ~0, or ~1. Round everything and keep only the ones near -1 or 1
    cycle_rep = optimal.loc['optimal cycle', 'chain']
    filter = round(cycle_rep['coefficient'].astype(float)) != 0
    cycle_rep = cycle_rep[filter]

    # get nodes represented in cycle
    cycle_nodes = cycle_rep['simplex' # simplicies in the cycle
        ].explode( # split simplex lists into nodes
        ).drop_duplicates( # keep only one occurance of each
        ).tolist() # collect them to use as indicies
    cycle_nodes = nodes[cycle_nodes] # get nodes at these indexes

    # save the result
    with open(f'{RESULT_PATH}/cycle_{cycle.name}.pickle', 'wb') as cycle_file:
        # serialize and save
        pickle.dump(
            {
                'id': cycle.name, # cycle index (unique for homology calcuation)
                'dimension': cycle['dimension'],
                'birth': cycle['birth'],
                'birth simplex': cycle['birth simplex'],
                'death': cycle['death'],
                'death simplex': cycle['death simplex'],
                'cycle rep': cycle_rep,
                'cycle nodes': cycle_nodes,
                'num degenerate': sum(1-filter),
                'time': time_to_solve
            },
            cycle_file
        )
    return 1


def callback(future):
    try:
        result = future.result()
        print(f'Cycle succeeded')
    except TimeoutError as err:
        print(f'Cycle timed out')
    except Exception as err:
        print(f'Cycle errored out')


def main():
    # make sure we have a place to save the file
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    ## setup the network
    G = mn.gen_concept_network(
            DATA_PATH+CONCEPT_FILE,
            relevance_cutoff=RELEVANCE_FILTER, # 0.7
            min_article_freq=FREQ_MIN_FILTER, # 0.006%
            max_article_freq=FREQ_MAX_FILTER, # 0.05%
            normalize_year=True,
            year_min=MIN_YEAR # 1920
        ) # use a filtered data file to make the network
    adj = nx.adjacency_matrix(G, weight='norm_year') # adjacency matrix
    node_births = np.array(list(nx.get_node_attributes(G, 'norm_year').values())) # node orgin years, these break the cycle reps (idk why)
    adj.setdiag(node_births) # format for oat
    adj = adj.sorted_indices() # needed on some computers (not others tho which is confusing)

    # save the graph (so we can look back at nodes)
    with open(f'{RESULT_PATH}/graph.pickle', 'wb') as graph_file:
        # serialize and save
        pickle.dump(G, graph_file)

    ## solve homology
    global factored
    start = time.time()
    factored = oat.rust.FactoredBoundaryMatrixVr( # two functions that do this, idk what the other one is
            dissimilarity_matrix=adj,
            homology_dimension_max=2
        )
    homology = factored.homology( # solve homology
            return_cycle_representatives=True, # These need to be true to be able to make a barcode, makes the problem take ~30% longer (1:30ish)
            return_bounding_chains=True
        )
    time_to_solve = time.time() - start
    print(f'Homology calculation finished in {time_to_solve} secs')
    
    # save the results
    with open(f'{RESULT_PATH}/homology.pickle', 'wb') as homology_file:
        # serialize and save
        pickle.dump(
                {
                    'homology': homology,
                    'time': time_to_solve
                },
                homology_file
            )
        
    # optimize cycles
    # use multithreading to deal with
    #   1. The number of calculations
    #   2. Timeouts
    #   3. OAT errors
    print(f'Optimizing {sum(DIM_CONDITION(homology['dimension']))} cycles')
    nodes = np.array(G.nodes) # index -> node key
    with ProcessPool(max_workers=NUM_PROCESSES) as pool: # run NUM_PROCESSES workers at once
        for i in homology[DIM_CONDITION(homology['dimension'])].index: # calculate it for every cycle
            future = pool.schedule(optimize_cycle, (homology.loc[i], nodes), timeout=TIMEOUT_LEN) # 
            future.add_done_callback(callback)


if __name__ == '__main__':
    main()
