'''
Calculate homology for the network and get a bunch of 2D cycle/bounding
chain reps

The problem that makes this hard is that
    1. There are a lot of cycles we want cycle reps for
    2. Some cycles, especially those born later, take a long time

To solve this, we multithread the process so we can calculate a bunch of
cycle reps simultaniously starting with the ones we know can be solved. 
When a global timout couter is finished, we save all results before the
program is done, so we can get as many cycle reps as possible.
'''

# load some packages
import sys; sys.path.append("/Users/gavinengelstad/Documents/Documents - Gavin’s MacBook Pro/School/Summer '24/Research/2024SummerTopo/Gavin/utils")
# make_network folder, we can just move it to work better
from pebble import ProcessPool
import make_network_v1 as mn
import pandas as pd
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
import multiprocessing as mp
mp.set_start_method('fork', force=True)

# config
DATA_PATH = 'datasets/concept_network/'
CONCEPT_FILE = 'articles_category_for_2l_abstracts_concepts_processed_v1_EX_102.csv.gz' # Applied Mathematics
RESULT_PATH = 'results/' # save files here
RESULT_FILE = 'applied_math_test.pickle'
GLOBAL_TIMEOUT_LEN = 10 # seconds
PROCESS_TIMEOUT_LEN = np.inf # seconds
MIN_RELEVANCE= 0.7
MIN_FREQ = 0.002 # 0.006%
MAX_FREQ = 0.005 # 0.05%
MIN_YEAR = 1920
MAX_DIM = 2
NUM_PROCESSES = 16 # number of processes to do with multithreading
OPTIMIZE_CONDITION = lambda h: h['dimension'] == 1 # rows which we optimize a cycle for
CYCLE_REP = True # whether to find a cycle rep
BOUNDING_REP = True # whether to find a bounding chain rep


def make_graph(file, min_relevance, min_freq, max_freq, min_year):
    '''
    Makes a graph with given filters

    Wrapper for `gen_concept_network` that also returns the amount of time
    it takes
    '''
    start = time.time()
    G = mn.gen_concept_network(
            file,
            relevance_cutoff=min_relevance, # 0.7
            min_article_freq=min_freq, # 0.006%
            max_article_freq=max_freq, # 0.05%
            normalize_year=True,
            year_min=min_year # 1920
        ) # use a filtered data file to make the network
    time_for_graph = time.time() - start

    return G, time_for_graph


def solve_homology(adj, max_dim):
    '''
    Calculates homology given an adjacency matrix

    Saves the `FactoredBoundryMatrixVR` as a global variable for use at other
    points in the file (namely when multithreaded given the fork start method)
    '''
    global factored # bc of the "fork" multiprocessing start method this means factored is accessable in other threads
    start = time.time()
    factored = oat.rust.FactoredBoundaryMatrixVr( # umatch factorizaion
            dissimilarity_matrix=adj,
            homology_dimension_max=max_dim
        )
    homology = factored.homology( # solve homology
            return_cycle_representatives=True, # These need to be true to be able to make a barcode, makes the problem take ~30% longer (1:30ish)
            return_bounding_chains=True
        )
    time_for_homology = time.time() - start

    return homology, time_for_homology


def optimal_cycle_rep(cycle):
    '''
    Get an optimal cycle rep
    '''
    start = time.time()
    optimal = factored.optimize_cycle( # optimial cycle rep
            birth_simplex=cycle['birth simplex'], 
            problem_type='preserve PH basis'
        )
    time_to_solve = time.time() - start
    print(f'Cycle rep for {cycle.name} optimized in {time_to_solve} secs')
    
    return optimal, time_to_solve


def optimal_bounding_chain_rep(cycle):
    '''
    Get an optimal boudning chain rep
    '''
    start = time.time()
    optimal = factored.optimize_bounding_chain( # optimial bounding chain
            birth_simplex=cycle['birth simplex']
        )
    time_to_solve = time.time() - start
    print(f'Boudning chain for {cycle.name} optimized in {time_to_solve} secs')
    
    return optimal, time_to_solve


def optimize_cycle(cycle, cycle_rep=CYCLE_REP, bounding_rep=BOUNDING_REP):
    '''
    Optimizes a cycle rep and bounding chain rep for a cycle. Returns
    a dictionary with the optimized cycle and some information about it

    Cycle is a row of the `homology` dataframe. That means it has columns
        - "dimension": The cycle dimension
        - "birth": The cycle birth filtration level
        - "death": The cycle death filtration level
        - "birth simplex": The final simplex that makes the cycle a cycle
        (dim)
        - "death simplex": The simplex that closes the cycle (dim+1)
        - "cycle representative"": Dataframe with an unoptimized cycle
        rep
        - "cycle nnz": The number of simplexes in the unoptimized cycle
        rep
        - "bounding chain": Dataframe with the simplicies that fill the
        unoptimized cycle
        - "bounding nnz": The number of simplicies in the bounding chain
    It also has a `cycle.name` attribute with the index of the cycle in
    the `homology` dataframe

    We also use `factored`, which is a global variable set in `main` that
    is the FactoredBoundryMatrixVR for the network we're finding homology
    for
    '''
    res = ()

    # optimize cycle
    if cycle_rep:
        res += optimal_cycle_rep(cycle)
    
    # optmize bounding chain
    if bounding_rep:
        res += optimal_bounding_chain_rep(cycle)
    
    return res


def optimize_cycles(homology, num_processes, process_timeout, global_timeout):
    '''
    Optimize all cycles in the homology dataframe

    Uses multithreading to make everything run faster and handle timeouts/
    errors. If a process takes longer than the timeout length, it ends. We
    also have a global timeout after which all processes timeout and we save
    results

    Returns a list of "futures" for the multithreaded processes. Each `future`
    has a `future.result()` method which either returns what was returned by
    the process or throws and errors thrown by the process (mainly a timout 
    error)
    '''
    # loop we optimize cycles in
    homology = homology.sort_values('birth') # sort by birth time
    # processes born earlier finish faster since the optimization
    # problem includes all cycles/simplicies born before the cycle
    # Therefore, we start with the earliest born cycles, since those 
    # are the ones we know we can solve for

    # solve cycles
    futures = []
    start = time.time() # if the whole process takes longer than global_timeout we stop it all
    with ProcessPool(max_workers=num_processes) as pool: # run NUM_PROCESSES workers at once
        # this loops over everything and adds it to futures (almost) immediately
        # that means we get through the whole loop at the very start of this running,
        # then get stuck in the while loop for however long the processes take to run
        for id in homology.index: # loop over the indicies
            # start the process
            future = pool.schedule(
                    optimize_cycle,
                    (homology.loc[id],),
                    timeout=process_timeout # timout process after either the process_timout or we run out of time in the global timout
                )
            futures.append((id, future)) # a dictionary breaks, but this essentially works like a dictionary
        
        # pool exists until the processes are done, that means this has to be indented, otherwise it waits until all processes finished before getting to this block
        while not np.all([f.done() for id, f in futures]): # wait until everything is done
            time.sleep(1) # check again after a second

            if time.time() - start >= global_timeout: # wait until global timeout
                [f[1].cancel() for f in futures] # cancel all remaining processes if timout finishes

    return futures


def clean_chain(dirty_chain):
    '''
    Cleans a chain by removing any cofficeints that round to 0
    '''
    filter = round(dirty_chain['coefficient'].astype(float)) != 0 # true for anything that doesn't round to zero
    chain = dirty_chain[filter] # keep only where filter is true

    return chain


def collect_cycles(futures, cycle_rep, bounding_rep, concepts):
    '''
    Collects the optmized cycles from the futures list and turns it into a pandas
    dataframe

    `futures` should be a list of (id, future) pairs, where I can call future.result()
    to get the return from a function

    Columns in the returned dataframe are what is returned by `optimize_cycle` and
    an error column with the result of any cycles that throw an error
    '''
    results = [] # returns
    for id, f in futures: # collect the results
        cyc_res = {'id': id} # results just for the cycle
        try:
            opt_res = f.result() # results from optimization

            i = 0 # make every permutation of cycle_rep, bounding_rep work
            if cycle_rep: # if we have optimized cycles
                # cycle info
                dirty_optimal_cycle = opt_res[i].loc['optimal cycle', 'chain'] # dataframe of the simplicies and coefficeints in the optimal cycle
                optimal_cycle = clean_chain(dirty_optimal_cycle) # remove coeficicents that round to 0

                # get nodes represented in cycle
                cycle_nodes = optimal_cycle['simplex' # simplicies in the cycle
                    ].explode( # split simplex lists into nodes
                    ).drop_duplicates( # keep only one occurance of each
                    ).tolist() # collect them to use as indicies
                cycle_nodes = concepts[cycle_nodes] # get nodes at these indexes
                
                cyc_res['optimal cycle representative'] = optimal_cycle # cycle info
                cyc_res['optimal cycle nnz'] = len(optimal_cycle) # number of nonzero entries in the cycle
                cyc_res['cycle nodes'] = cycle_nodes # nodes in the cycle
                cyc_res['dirty optimal cycle representative'] = dirty_optimal_cycle # cycle without rounding
                cyc_res['dirty optimal cycle nnz'] = len(dirty_optimal_cycle) # length of cycle prorounding
                cyc_res['optimal cycle nrz'] = len(dirty_optimal_cycle) - len(optimal_cycle) # number of coefficeints rounded to 0
                cyc_res['optimal cycle time'] = opt_res[i+1] # time (seconds) to optimize cycle

                i += 2 # make bounding chain work
            
            if bounding_rep: # if we have optimized bounding chains
                # bounding chain info
                dirty_bounding_chain = opt_res[i].loc['optimal bounding chain', 'chain'] # dataframe of the simplicies and coefficeints in the optimal bounding chain
                bounding_chain = clean_chain(dirty_bounding_chain) # remove coefficients that round to 0

                cyc_res['optimal bounding chain representative'] = bounding_chain # cycle info
                cyc_res['optimal bounding chain nnz'] = len(bounding_chain) # number of nonzero entries in the cycle
                cyc_res['dirty optimal bounding chain representative'] = dirty_bounding_chain # cycle without rounding
                cyc_res['dirty optimal bounding chain nnz'] = len(dirty_bounding_chain) # length of cycle prorounding
                cyc_res['optimal bounding chain nrz'] = len(dirty_bounding_chain) - len(dirty_bounding_chain) # number of coefficeints rounded to 0
                cyc_res['optimal bounding chain time'] = opt_res[i+1] # time (seconds) to optimize cycle

            results.append(cyc_res)
        except Exception as err: # collect the errors
            cyc_res['error'] = err
        results.append(cyc_res)
    
    columns = ['id', 'error']
    if bounding_rep:
        columns = ['optimal bounding chain representative', 'optimal bounding chain nnz', # bounding chain info
                   'dirty optimal bounding chain representative', 'dirty optimal bounding chain nnz', 'optimal bounding chain nrz', # cleaning info, nrz -> number rounded to zero
                   'optimal bounding chain time'] + columns # extra
    if cycle_rep:
        columns = ['optimal cycle representative', 'optimal cycle nnz', 'cycle nodes', # cycle info
                   'dirty optimal cycle representative', 'dirty optimal cycle nnz', 'optimal cycle nrz', # cleaning info, nrz -> number rounded to zero
                   'optimal cycle time'] + columns # extra
        
    df = pd.DataFrame(results, columns=columns) # make a dataframe from the results
    # the columns will be all the keys in the optimize_cycles return dictionaries and an error one
    df = df.set_index('id') # make joinable with homology (id is the index in homology)

    return df


def main():
    ## setup process
    gloabl_start = time.time() # global start time
    if not os.path.exists(RESULT_PATH): # make sure we have a place to save the file
        os.makedirs(RESULT_PATH)

    ## create the graph
    G, time_for_graph = make_graph(DATA_PATH+CONCEPT_FILE, MIN_RELEVANCE, MIN_FREQ, MAX_FREQ, MIN_YEAR)
    adj = mn.adj_matrix(G, weight='norm_year', fill_diag=True)
    print(f'Graph construction finished in {time_for_graph} secs')

    ###solve homology
    homology, time_for_homology = solve_homology(adj, MAX_DIM)
    print(f'Homology calculation finished in {time_for_homology} secs')

    ## optimize cycles
    print(f"Optimizing {sum(OPTIMIZE_CONDITION(homology))} cycles")
    futures = optimize_cycles(homology[OPTIMIZE_CONDITION(homology)], NUM_PROCESSES, PROCESS_TIMEOUT_LEN, GLOBAL_TIMEOUT_LEN)

    ## collect results
    concepts = np.array(G.nodes) # list of concepts, index -> node key in network (and simplicial complex)
    results = collect_cycles(futures, CYCLE_REP, BOUNDING_REP, concepts)
    homology = homology.join(results) # add to homology dict (I don't wanna save extra stuff)

    # save results
    with open(RESULT_PATH+RESULT_FILE, 'wb') as results_file:
        # serialize and save
        pickle.dump(
                {
                    'graph': G,
                    'time for graph': time_for_graph,
                    'concepts': concepts,
                    'homology': homology,
                    'time for homology': time_for_homology,
                    'total time': time.time() - gloabl_start
                },
                results_file
            )
    print('Results saved')


if __name__ == '__main__':
    main()