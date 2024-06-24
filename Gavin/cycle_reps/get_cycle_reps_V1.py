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
from pebble import ProcessPool
import make_network_v1 as mn
from time import time
import networkx as nx
import pandas as pd
import oatpy as oat
import numpy as np
import pickle
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
RESULT_PATH = 'results/' # save files here
RESULT_FILE = 'applied_math_6-12.pickle'
TIMEOUT_LEN = 30 # seconds
RELEVANCE_FILTER = 0.7
FREQ_MIN_FILTER = 0.00006 # 0.006%
FREQ_MAX_FILTER = 0.0005 # 0.05%
MIN_YEAR = 1920
NUM_PROCESSES = 16 # number of processes to do with multithreading
DIM_CONDITION = lambda dim: dim == 1 # dimension of rows which we optimize a cycle for


def optimize_cycle(cycle):
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
    start = time()
    optimal = factored.optimize_cycle( # optimial cycle rep
            birth_simplex=cycle['birth simplex'], 
            problem_type='preserve PH basis'
        )
    time_to_solve = time() - start
    print(f'Cycle {cycle.name} optimized in {time_to_solve} secs')


    return optimal, time_to_solve


def main():
    main_start = time()

    # make sure we have a place to save the file
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    ## setup the network
    start = time()
    G = mn.gen_concept_network(
            DATA_PATH+CONCEPT_FILE,
            relevance_cutoff=RELEVANCE_FILTER, # 0.7
            min_article_freq=FREQ_MIN_FILTER, # 0.006%
            max_article_freq=FREQ_MAX_FILTER, # 0.05%
            normalize_year=True,
            year_min=MIN_YEAR # 1920
        ) # use a filtered data file to make the network
    time_for_graph = time() - start
    print(f'Graph construction finished in {time_for_graph} secs')
    concepts = np.array(G.nodes) # index -> node key
    adj = nx.adjacency_matrix(G, weight='norm_year') # adjacency matrix
    node_births = np.array(list(nx.get_node_attributes(G, 'norm_year').values())) # node orgin years, these break the cycle reps (idk why)
    adj.setdiag(node_births) # format for oat
    adj = adj.sorted_indices() # needed on some computers (not others tho which is confusing)

    ## solve homology
    global factored # bc of the "fork" multiprocessing start method this means factored is accessable in other threads
    start = time()
    factored = oat.rust.FactoredBoundaryMatrixVr( # two functions that do this, idk what the other one is
            dissimilarity_matrix=adj,
            homology_dimension_max=2
        )
    homology = factored.homology( # solve homology
            return_cycle_representatives=True, # These need to be true to be able to make a barcode, makes the problem take ~30% longer (1:30ish)
            return_bounding_chains=True
        )
    time_for_homology = time() - start
    print(f'Homology calculation finished in {time_for_homology} secs')
        
    # optimize cycles
    # use multithreading to deal with
    #   1. The number of calculations
    #   2. Timeouts
    #   3. OAT errors
    print(f"Optimizing {sum(DIM_CONDITION(homology['dimension']))} cycles")
    with ProcessPool(max_workers=NUM_PROCESSES) as pool: # run NUM_PROCESSES workers at once
        futures = pool.map(
                optimize_cycle,
                [homology.loc[id] for id in homology[DIM_CONDITION(homology['dimension'])].index],
                timeout=TIMEOUT_LEN
            )
    it = futures.result()

    # collect results
    # this relies on lists, so multiprocesing it would cause more problems then it's worth
    # with sharing the list and error handeling. Therefore, we just use this
    results = []
    errors = []
    for id in homology[DIM_CONDITION(homology['dimension'])].index: # go for every cycle
        try:
            optimal, time_to_solve = next(it) # get next element in iteratior

            # filter optimal cycle
            # we want all coefficients to be -1 or 1, the optimization problem has machine error
            # so some can be ~-1, ~0, or ~1. Round everything and keep only the ones near -1 or 1
            dirty_cycle_rep = optimal.loc['optimal cycle', 'chain']
            filter = round(dirty_cycle_rep['coefficient'].astype(float)) != 0
            cycle_rep = dirty_cycle_rep[filter]

            # get nodes represented in cycle
            cycle_nodes = cycle_rep['simplex' # simplicies in the cycle
                ].explode( # split simplex lists into nodes
                ).drop_duplicates( # keep only one occurance of each
                ).tolist() # collect them to use as indicies
            cycle_nodes = concepts[cycle_nodes] # get nodes at these indexes

            results.append({ # save the result
                    'id': id, # cycle index in homology dataframe
                    'dirty optimal cycle representative': dirty_cycle_rep, # cycle rep with bad coefficicents
                    'dirty optimal cycle nnz': len(dirty_cycle_rep), # length of cycle rep with bad cofficients
                    'optimal cycle representative': cycle_rep, # cycle rep
                    'optimal cycle nnz': len(cycle_rep), # number of simplicies in the cycle rep
                    'optimal cycle rounded': sum(1-filter), # number of degenerate simplicies that had to be filtered
                    'optimal cycle time': time_to_solve, # time to get optimal cycle
                    'cycle nodes': cycle_nodes # deindexed (aka string) nodes in the cycle
                })
        except StopIteration:
            print('No more results')
        except Exception as err:
            errors.append({'id': id, 'error': err})
    
    # add results to homology
    homology = homology.join( # add successfully optimized cycles to homology
            pd.DataFrame(
                    data=results,
                    columns=['id', 'dirty optimal cycle representative', 'dirty optimal cycle nnz',
                             'optimal cycle representative', 'optimal cycle nnz', 'optimal cycle rounded',
                             'optimal cycle time', 'cycle nodes'] # if empty, don't throw error
                ).set_index('id')
        )
    homology = homology.join( # add errors to homology
            pd.DataFrame(
                    data=errors,
                    columns=['id', 'error'] # if empty, don't throw error
                ).set_index('id')
        )
    
    # save the results
    with open(RESULT_PATH+RESULT_FILE, 'wb') as results_file:
        # serialize and save
        pickle.dump(
                {
                    'graph': G,
                    'time for graph': time_for_graph,
                    'concepts': concepts,
                    'homology': homology,
                    'time for homology': time_for_homology,
                    'total time': time() - main_start
                },
                results_file
            )


if __name__ == '__main__':
    main()
