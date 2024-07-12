'''
We want to show we're getting meaningful results, not results that are a function of
our specific network structure/filtration parameters

Here, we focus on the second part (filtration paramters), and say given the same network
and filtration distribution, just with the filtrations values "shuffled" to be on
different edges to see if this creates a network with different homology

The goal is to show that our results are a function of the specific filtration values in
our network
'''

# load some packages
import sys; sys.path.append("/Users/gavinengelstad/Documents/Documents - Gavin’s MacBook Pro/School/Summer '24/Research/2024SummerTopo/Gavin/utils")
import random_complexes as rc
from pebble import ProcessPool
import make_network as mn
import pandas as pd
import oatpy as oat
import numpy as np
import pickle
import time
import os

# config
CONCEPT_FILE = 'datasets/concept_network/concepts_Zoology_608.csv' # Zoology
RESULT_PATH = 'results/shuffle'
MIN_RELEVANCE= 0.7
MIN_FREQ = 0.00006 # 0.006%
MAX_FREQ = 0.0005 # 0.05%
MIN_YEAR = 1920
MAX_DIM = 2
N = 200 # number of shuffles we do
GLOBAL_TIMEOUT = 20 # stop any new processes from starting after this point
NUM_PROCESSES = 10 # number of processes to do with multithreading
SEED = 10 # make it replicable
MAX_SEED = 2**32 - 1 # max numpy seed


def make_graph(file, min_relevance, min_freq, max_freq, min_year):
    '''
    Makes a graph with given filters

    Wrapper for `gen_concept_network` that also returns the amount of time it takes
    '''
    start = time.time()
    G = mn.gen_concept_network(
            file,
            min_relevance=min_relevance, # 0.7
            min_year=min_year, # 1920
            min_articles=min_freq, # 0.006%
            max_articles=max_freq, # 0.05%
            normalize_year=True
        ) # use a filtered data file to make the network
    time_for_graph = time.time() - start

    return G, time_for_graph


def graph_homology(G, max_dim, res_file):
    # distance matrix to find homology of
    adj = mn.adj_matrix(G, weight='norm_year', fill_diag=True, diag_val=0) # we fill the diag with 0
    # this means theres no interesting 0 dimension homology, but is needed for there the nodes to exist before the edges post shuffle

    # calculate homology
    start = time.time() # keep track of how long it takes
    factored = oat.rust.FactoredBoundaryMatrixVr( # umatch factorizaion
            dissimilarity_matrix=adj,
            homology_dimension_max=max_dim
        )
    homology = factored.homology( # solve homology
            return_cycle_representatives=False, # we just are going to compare a barcode, this just makes it take longer and the files larger
            return_bounding_chains=False
        )
    time_for_homology = time.time() - start
    print(f"Homology calulation finished in {round(time_for_homology, 2)} secs. Saving to '{res_file}'")

    # create results dictionary (what we save)
    res = {
            'graph': G,
            'homology': homology,
            'time': time_for_homology
        }

    # save results
    with open(res_file, 'wb') as file:
        pickle.dump(res, file)


def main():
    # setup
    np.random.seed(SEED)
    if not os.path.exists(RESULT_PATH): # make sure we have a place to save the file
        os.makedirs(RESULT_PATH)
    
    # create the graph
    G = mn.gen_concept_network(
        CONCEPT_FILE,
        min_relevance=MIN_RELEVANCE, # 0.7
        min_year=MIN_YEAR, # 1920
        min_articles=MIN_FREQ, # 0.0006%
        max_articles=MAX_FREQ, # 0.005%
        normalize_year=True
    )

    # run the processes
    with ProcessPool(NUM_PROCESSES) as pool:
        start = time.time() # make sure we dont exceed timout

        # calculate initial homology
        pool.schedule(
                graph_homology,
                (G.copy(), MAX_DIM, RESULT_PATH+'/original.pickle') # copy makes it so there are no pickling errors later
            )

        # calculate shuffled networks
        for i in range(N):
            shuffled_G = rc.shuffle_edge_weights(G, seed=np.random.randint(0, MAX_SEED+1)) # randint is exculsive of the upper bound
            res_file = RESULT_PATH + f'/shuffle_{i+1}.pickle'
            pool.schedule(
                    graph_homology,
                    (shuffled_G, MAX_DIM, res_file)
                )
            
            # if timeout is exceeded, don't start anymore processes (existing ones can finish)
            if time.time() - start >= GLOBAL_TIMEOUT:
                print(f'Processes timed out. Ran {i+1} shuffles. {N-i} skipped')
                break
    

if __name__ == '__main__':
    main()
