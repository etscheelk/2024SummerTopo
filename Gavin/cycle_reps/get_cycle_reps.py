'''
This is why we need a serializable FactoredBoundryMatrixVR

Python multithreading is slow for what we want to do and multiprocessing can't take 
factored as an input since it's not pickelable. In the past, we've used a `fork` as
a janky workaround to get access to the factored matrix in each process, but that
breaks with the Gurboi optimizer. Therefore, we'll switch to calcuating the 
FactoredBoundryMatrixVR on each process and doing the work in a loop there

This should avoid most of the issues while staying fast through multithreading
'''

# load some packages
import sys; sys.path.append("/Users/gavinengelstad/Documents/Documents - Gavin’s MacBook Pro/School/Summer '24/Research/2024SummerTopo/Gavin/utils")
from multiprocessing import Pool, Manager
from queue import Empty
import make_network as mn
import pandas as pd
import oatpy as oat
import numpy as np
import pickle
import time

# config
CONCEPT_FILE = 'datasets/concept_network/concepts_Zoology_608.csv' # Applied Mathematics
RESULT_FILE = 'test.pickle'
GLOBAL_TIMEOUT_LEN = 15 # seconds, 2 hours in seconds
MIN_RELEVANCE= 0.7
MIN_FREQ = 0.001 # 0.006%
MAX_FREQ = 0.005 # 0.05%
MIN_YEAR = 1920
MAX_DIM = 2
NUM_PROCESSES = 12 # number of processes to do with multithreading
OPTIMIZE_CONDITION = lambda h: h['dimension'] == 1 # rows which we optimize a cycle for
CYCLE_REP = True # whether to find a cycle rep
BOUNDING_CHAIN = False # whether to find a bounding chain rep


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


def clean_chain(dirty_chain):
    '''
    Cleans a chain by removing any cofficeints that round to 0
    '''
    filter = round(dirty_chain['coefficient'].astype(float)) != 0 # true for anything that doesn't round to zero
    chain = dirty_chain[filter] # keep only where filter is true

    return chain


def optimize_cycle_rep(factored, cycle):
    '''
    Get an optimal cycle rep

    Saves to the cycle dictionary and returns it
    '''
    # solve problem
    start = time.time()
    optimal = factored.optimize_cycle( # optimial cycle rep
            birth_simplex=cycle['birth simplex'], 
            problem_type='preserve PH basis'
        )
    time_to_solve = time.time() - start
    print(f'Cycle rep for {cycle['id']} optimized in {time_to_solve} secs')

    # store results
    # remove degenerates
    dirty_optimal_cycle = optimal.loc['optimal cycle', 'chain'] # dataframe of the simplicies and coefficeints in the optimal cycle
    optimal_cycle = clean_chain(dirty_optimal_cycle) # remove coeficicents that round to 0

    # get nodes represented in cycle
    cycle_nodes = optimal_cycle['simplex' # simplicies in the cycle
        ].explode( # split simplex lists into nodes
        ).drop_duplicates( # keep only one occurance of each
        ).tolist() # collect them to use as indicies
    
    # add everything to the cycle dictionary
    cycle['optimal cycle representative'] = optimal_cycle # cycle info
    cycle['optimal cycle nnz'] = len(optimal_cycle) # number of nonzero entries in the cycle
    cycle['cycle nodes'] = cycle_nodes # nodes in the cycle
    cycle['dirty optimal cycle representative'] = dirty_optimal_cycle # cycle without rounding
    cycle['dirty optimal cycle nnz'] = len(dirty_optimal_cycle) # length of cycle prorounding
    cycle['optimal cycle nrz'] = len(dirty_optimal_cycle) - len(optimal_cycle) # number of coefficeints rounded to 0
    cycle['optimal cycle time'] = time_to_solve # time (seconds) to optimize cycle

    return cycle


def optimize_bounding_chain(cycle):
    '''
    UNFINISHED, currently doesn't do anything

    Get an optimal cycle rep

    Saves to the cycle dictionary and returns it
    '''
    return cycle


def optimize_cycles(factored, q):
    results = [] # store cycle results
    cycle = q.get() # first cycle we use. get will wait until something is added to the cycle to pull
    start = time.time() # start time (for timeout)
    while True:
        # optiimze cycle rep
        if CYCLE_REP:
            cycle = optimize_cycle_rep(factored, cycle)
        
        # optimize bounding chain
        if BOUNDING_CHAIN:
            cycle = optimize_bounding_chain(factored, cycle)


        # end conditions
        # timout
        if time.time() - start > GLOBAL_TIMEOUT_LEN: # if timeout has passed
            break

        # no more cycles left
        try:
            cycle = q.get(False) # (defines next cycle if one is left)
        except Empty:
            break
    
    return results


def worker(adj, q):
    '''
    This is a worker that factors the matrix and optimizes cycles

    It doesn't calculate homology, and instead just gets the cycles to optimize from the
    queue `q`. This means we only calculate homology once, in the "homology_worker"
    '''
    # calculate factored. This can't be shared accross processes since it's not pickleable
    start = time.time()
    factored = oat.rust.FactoredBoundaryMatrixVr( # umatch factorizaion
            dissimilarity_matrix=adj,
            homology_dimension_max=MAX_DIM
        )
    print(f'FactoredBoundaryMatrixVr found in {time.time() - start} secs, waiting for homology to optimize cycles')

    return optimize_cycles(factored, q)


def homology_worker(adj, q):
    '''
    This is the worker that calculates homology on top of optimizing cycles

    This worker does everything the other workers do, and also calculates homology to setup
    the queue. The queue stores the cycles that still need to be optimized, to be shared
    between the other processes
    '''
    # calculate homology
    start = time.time()
    factored = oat.rust.FactoredBoundaryMatrixVr( # umatch factorizaion
            dissimilarity_matrix=adj,
            homology_dimension_max=MAX_DIM
        )
    homology = factored.homology( # solve homology
            return_cycle_representatives=True, # These need to be true to be able to make a barcode, makes the problem take ~30% longer (1:30ish)
            return_bounding_chains=True
        )
    time_for_homology = time.time() - start
    print(f'Homology calculation finished in {time_for_homology} secs')

    # setup queue
    opt_homology = homology[OPTIMIZE_CONDITION(homology)].reset_index() # get cycles we want to optimize and reset index
    print(f"Optimizing {len(opt_homology)} cycles")
    # Index reset means id in the dicts for the next part
    for i in opt_homology.index[::-1]: # start at the end (faster ones)
        q.put(dict(opt_homology.loc[i]))

    return homology, time_for_homology, optimize_cycles(factored, q)


def main():
    ## setup process
    global_start = time.time() # global start time

    ## create the graph
    G, time_for_graph = make_graph(CONCEPT_FILE, MIN_RELEVANCE, MIN_FREQ, MAX_FREQ, MIN_YEAR)
    adj = mn.adj_matrix(G, weight='norm_year', fill_diag=True, diag_val=None)
    print(f'Graph construction finished in {time_for_graph} secs')

    ## start processes
    # each process:
    # 1. Calcuates the FactoredBoundryMatrixVR
    # (1.5). One process calculates homology and adds all the cycles to a queue that's shared to optimize cycles
    # 2. Optimizes the cycles for however much time is left under the global timeout
    # 3. Collects the results 
    # 4. Returns the collected results to be turned into a dataframe
    with Pool(NUM_PROCESSES) as pool:
        q = Manager().Queue() # queue will keep the rows that are yet to be calculated

        homology_res = pool.apply_async(homology_worker, (adj, q)) # this will return homology, time_for_homology, and a bunch of optimized cycles
        results = []
        for _ in range(NUM_PROCESSES-1):
            results.append(pool.apply_async(worker, (adj, q))) # start a regular optimize cycle iteration
        
        homology, time_for_homology, cycles = homology_res.get() # get homology info from the first thread
        cycles += sum([r.get() for r in results], []) # add all the results lists together and combine them to get the optimized cycles
        
    ## collect results
    print('Finished, collecting results')
    concepts = np.array(G.nodes) # list of concepts, index -> node key in network (and simplicial complex)
    for c in cycles:
        c['cycle nodes'] = concepts[c['cycle nodes']]
    optimized = pd.DataFrame(cycles)

    # save results
    with open(RESULT_FILE, 'wb') as results_file:
        # serialize and save
        pickle.dump(
                {
                    'graph': G,
                    'time for graph': time_for_graph,
                    'concepts': concepts,
                    'homology': homology,
                    'time for homology': time_for_homology,
                    'optimized': optimized,
                    'total time': time.time() - global_start
                },
                results_file
            )
    print(f"Results saved to '{RESULT_FILE}'")

if __name__ == '__main__':
    main()
