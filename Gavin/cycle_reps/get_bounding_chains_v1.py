'''
Use the python optimizer to get all cycle reps and bounding chains for a given network

This was all run on MSI and won't work here
'''

# load some packages
from multiprocessing.context import TimeoutError
import multiprocessing as mp
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
RESULT_FILE = f'results/{FIELD}_results.pickle'
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

    ## optimize cycles and bounding chains
    # for speed, we run it on multiple, multithreaded processes
    relevant_cycles = homology[homology['dimension'] == OPTIMIZE_DIM]
    num_to_optimize = len(relevant_cycles)
    print(f'Optimizing {num_to_optimize} cycles')
    start = time.time()
    finished = True
    with mp.get_context('spawn').Pool(NUM_PROCESSES) as pool:
        # find cycles
        cycle_futures = []
        for id in relevant_cycles.index:
            f = pool.apply_async(
                    cycle_optimizer.optimize_cycle,
                    (relevant_cycles.loc[id, 'birth simplex'],),
                    kwds={'return_objective': True, 'return_time': True}
                )
            cycle_futures.append((id, f))
        
        # find bounding chains
        bounding_chain_optimizer, time_for_optimizer = create_optimizer(factored, OPTIMIZE_DIM, op.BoundingChainOptimizer)
        print(f'Bounding Chain optimizer construction finished in {time_for_optimizer} secs')
        bounding_chain_futures = []
        cycles_optimized = 0
        for id, f in cycle_futures:
            try:
                cycle, cycle_obj, cycle_time = f.get(max(GLOBAL_TIMEOUT_LEN - (time.time()-start), 0))
            except TimeoutError:
                finished = False
            else:
                # finished cycle optimization
                cycles_optimized += 1
                print(f'Cycle {id} optimized ({cycles_optimized}/{num_to_optimize})')

                # start bounding chain optmization (if bounding chain finished)
                death = homology.loc[id, 'death']
                f = None
                if death < np.inf:
                    f = pool.apply_async(
                        bounding_chain_optimizer.optimize_bounding_chain,
                        (cycle, death),
                        kwds={'return_objective': True, 'return_time': True}
                    )
                bounding_chain_futures.append((id, cycle, cycle_obj, cycle_time, f))

        # collect results
        indexs = []
        birth_simplexes = []
        births = []
        deaths = []
        cycles = []
        cycle_objs = []
        cycle_nnzs = []
        cycle_nodes = []
        cycle_times = []
        bounding_chains = []
        bounding_chain_objs = []
        bounding_chain_nnzs = []
        bounding_chain_nodes = []
        bounding_chain_times = []
        bounding_chains_optimized = 0
        for id, cycle, cycle_obj, cycle_time, f in bounding_chain_futures:
            try:
                if f is None:
                    bounding_chain = bounding_chain_obj = bounding_chain_time = None
                else:
                    bounding_chain, bounding_chain_obj, bounding_chain_time = f.get(
                            max(GLOBAL_TIMEOUT_LEN - (time.time()-start), 0)
                        )
            except TimeoutError:
                finished = False
            else:
                # cycle info
                cycle_nnz = len(cycle)  # number of nonzero coefficenits
                cycle_node = concepts[cycle['simplex'].explode().drop_duplicates().to_list()]

                # bounding chain info
                bounding_chains_optimized += 1
                if bounding_chain is None:
                    bounding_chain_nnz = bounding_chain_node = None
                    print(f"Bounding Chain {id} doesn't exist ({bounding_chains_optimized}/{num_to_optimize})")
                else:
                    bounding_chain_nnz = len(bounding_chain)
                    bounding_chain_node = concepts[bounding_chain['simplex'].explode().drop_duplicates().to_list()]
                    print(f'Bounding Chain {id} optimized ({bounding_chains_optimized}/{num_to_optimize})')

                # collect results
                indexs.append(id)
                birth_simplexes.append(homology.loc[id, 'birth simplex'])
                births.append(homology.loc[id, 'birth'])
                deaths.append(homology.loc[id, 'death'])
                cycles.append(cycle)
                cycle_objs.append(cycle_obj)
                cycle_nnzs.append(cycle_nnz)
                cycle_nodes.append(cycle_node)
                cycle_times.append(cycle_time)
                bounding_chains.append(bounding_chain)
                bounding_chain_objs.append(bounding_chain_obj)
                bounding_chain_nnzs.append(bounding_chain_nnz)
                bounding_chain_nodes.append(bounding_chain_node)
                bounding_chain_times.append(bounding_chain_time)


    ## collect results
    optimized = pd.DataFrame({
                'birth_simplex': birth_simplexes,
                'birth': births,
                'death': deaths,
                'cycle': cycles,
                'cycle_objective': cycle_objs,
                'cycle_nnz': cycle_nnzs,
                'cycle_nodes': cycle_nodes,
                'cycle_time': cycle_times,
                'bounding_chain': bounding_chains,
                'bounding_chain_objective': bounding_chain_objs,
                'bounding_chain_nnz': bounding_chain_nnzs,
                'bounding_chain_nodes': bounding_chain_nodes,
                'bounding_chain_time': bounding_chain_times,
            },
            index=indexs
        )
    
    # save results
    with open(RESULT_FILE, 'wb') as results_file:
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
    print(f"Finished. Optimized {'all' if cycles_optimized == num_to_optimize else f'{cycles_optimized}/{num_to_optimize}'} cycles "
          f"and {'all' if finished else f'{bounding_chains_optimized}/{num_to_optimize}'} bounding chains. "
          f"Results saved to '{RESULT_FILE}'")


if __name__ == '__main__':
    main()
