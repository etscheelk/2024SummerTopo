'''
Use the python optimizer to get all cycle reps and bounding chains for a given network
'''

# load some packages
from multiprocess.context import TimeoutError
import multiprocess as mp
import make_network as mn
import optimizers as op
import pandas as pd
import oatpy as oat
import numpy as np
import pickle
import time
import os

# config
FIELD = os.environ['FIELD']
CONCEPT_FILE = os.environ['CONCEPT_FILE']
RESULTS_FILE = f'results/{FIELD}_results.pickle'
GLOBAL_TIMEOUT_LEN = int(os.environ['TIMEOUT'])  # seconds
MIN_RELEVANCE= float(os.environ['MIN_RELEVANCE'])
MIN_FREQ = float(os.environ['MIN_FREQ'])
MAX_FREQ = float(os.environ['MAX_FREQ'])
MIN_YEAR = int(os.environ['MIN_YEAR'])
MAX_DIM = int(os.environ['MAX_DIM'])
NUM_PROCESSES = int(os.environ['SLURM_CPUS_PER_TASK']) # number of processes to do with multithreading
OPTIMIZE_DIM = int(os.environ['CYCLE_DIM']) # rows which we optimize a cycle for


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


def solve_homology(adj, max_dim):
    '''
    Calculates homology given an adjacency matrix
    '''
    start = time.time()
    factored = oat.rust.FactoredBoundaryMatrixVr( # umatch factorizaion
            dissimilarity_matrix=adj,
            homology_dimension_max=max_dim
        )
    homology = factored.homology( # solve homology
            return_cycle_representatives=False, # it goes faster like this
            return_bounding_chains=False
        )
    time_for_homology = time.time() - start

    return homology, factored, time_for_homology


def create_optimizer(factored, dim, type):
    '''
    Creates an optimizer for an integer problem
    '''
    start = time.time()
    optimizer = type(
            factored,
            dim,
            integer=True,
            supress_gurobi=True
        )
    time_for_optimizer = time.time() - start

    return optimizer, time_for_optimizer


def main():
    ## setup process
    global_start = time.time() # global start time
    print(f'{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}: Starting {FIELD}')

    ## create the graph
    G, time_for_graph = make_graph(CONCEPT_FILE, MIN_RELEVANCE, MIN_FREQ, MAX_FREQ, MIN_YEAR)
    concepts = np.array(G.nodes) # list of concepts, index -> node key in network (and simplicial complex)
    adj = mn.adj_matrix(G, weight='norm_year', fill_diag=True, diag_val=None)
    print(f'Graph construction finished in {time_for_graph} secs')

    ## solve homology
    homology, factored, time_for_homology = solve_homology(adj, MAX_DIM)
    print(f'Homology calculation finished in {time_for_homology} secs')

    ## setup optimizers
    cycle_optimizer, time_for_optimizer = create_optimizer(factored, OPTIMIZE_DIM, op.CycleOptimizer)
    print(f'Cycle optimizer construction finished in {time_for_optimizer} secs')
    bounding_chain_optimizer, time_for_optimizer = create_optimizer(factored, OPTIMIZE_DIM, op.BoundingChainOptimizer)
    print(f'Bounding chain optimizer construction finished in {time_for_optimizer} secs')

    # function to get cycle and bounding chain
    def optimize_cycle(id):
        # get the cycle
        birth_simplex = homology.loc[id, 'birth simplex']
        cycle, cycle_obj, cycle_time = cycle_optimizer.optimize_cycle(
                birth_simplex,
                return_objective=True,
                return_time=True,
            )
        
        # get the bounding chain
        death = homology.loc[id, 'death']
        if death < np.inf:
            bounding_chain, bounding_chain_obj, bounding_chain_time = bounding_chain_optimizer.optimize_bounding_chain(
                    cycle,
                    death,
                    return_objective=True,
                    return_time=True,
                )
            bounding_chain_nnz = len(bounding_chain)  # these can't be done later since it could be none
            bounding_chain_node = concepts[bounding_chain['simplex'].explode().drop_duplicates().to_list()]
        else:
            bounding_chain = None
            bounding_chain_obj = np.nan
            bounding_chain_time = np.nan
            bounding_chain_nnz = np.nan
            bounding_chain_node = None
        
        # collect results
        res = {
                'id': id,
                'birth': homology.loc[id, 'birth'],
                'death': death,
                'birth_simplex': birth_simplex,
                'cycle': cycle,
                'cycle_objective': cycle_obj,
                'cycle_nnz': len(cycle),
                'cycle_nodes': concepts[cycle['simplex'].explode().drop_duplicates().to_list()],
                'cycle_time': cycle_time,
                'bounding_chain': bounding_chain,
                'bounding_chain_objective': bounding_chain_obj,
                'bounding_chain_nnz': bounding_chain_nnz,
                'bounding_chain_nodes': bounding_chain_node,
                'bounding_chain_time': bounding_chain_time,
            }
        
        return res

    ## optimize cycles and bounding chains
    # for speed, we run it on multiple, multithreaded processes
    relevant_cycles = homology[homology['dimension'] == OPTIMIZE_DIM]
    num_to_optimize = len(relevant_cycles)
    print(f'Optimizing {num_to_optimize} cycles')
    finished = True
    with mp.get_context('spawn').Pool(NUM_PROCESSES) as pool:
        # start optimization
        futures = []
        for id in relevant_cycles.index:
            f = pool.apply_async(optimize_cycle, (id,))
            futures.append(f)
        
        # get optimization results
        all_res = []
        num_optimized = 0
        for f in futures:
            try:
                res = f.get(max(GLOBAL_TIMEOUT_LEN - (time.time()-global_start), 0))  # get the results (if less time than timout has passed)
            except TimeoutError:
                finished = False
            else:
                num_optimized += 1
                print(f'Optimized {res['id']} ({num_optimized}/{num_to_optimize})')
                all_res.append(res)
    
    ## collect results
    optimized = pd.DataFrame(all_res).set_index('id')

        # save results
    with open(RESULTS_FILE, 'wb') as results_file:
        # serialize and save
        pickle.dump(
                {
                    'graph': G,
                    'time for graph': time_for_graph,
                    'concepts': concepts,
                    'homology': homology,
                    'optimized': optimized,
                    'time for homology': time_for_homology,
                    'total time': time.time() - global_start
                },
                results_file
            )
    print(f"Finished. Optimized bounding chains and cycles for {'all' if finished else f'{num_optimized}/{num_to_optimize}'} holes. "
          f"Results saved to '{RESULTS_FILE}'")


if __name__ == '__main__':
    main()

