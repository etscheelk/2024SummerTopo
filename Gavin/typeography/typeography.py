'''
Find typeography of nodes in the network

This code is meant to run on MSI
'''

## load some packages
import make_network as mn
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import os

# config
OPTIMIZED_FILE = os.environ['OPTIMIZED_FILE']
ARTICLE_CONCEPT_FILE = os.environ['ARTICLE_CONCEPT_FILE']
CITATION_FILE = os.environ['CITATION_FILE']
FIELD = os.environ['FIELD']
OUTDIR = os.environ['OUTDIR']
MIN_RELEVANCE = 0.7  # these should be the same as what was used in the optimization network
MIN_YEAR = 1920
MIN_ARTICLE_FREQ = 0.0001
MAX_ARTICLE_FREQ = 0.001


def get_optimization_info(optimized_file):
    '''
    Gets info about the optimization process
    '''
    # load the file
    with open(optimized_file, 'rb') as file:
        res = pickle.load(file)

    # info from res
    G = res['graph']
    optimized = res['optimized']
    concepts = res['concepts']

    return G, optimized, concepts  # we'll need all of these later


def cycle_typography(idx, G, optimized, concepts):
    '''
    Function to get types from the cycle
    '''
    # hole info
    birth = optimized.loc[idx, 'birth']
    death = optimized.loc[idx, 'death']

    # cycle nodes and edges
    cycle = optimized.loc[idx, 'cycle']
    cycle_nodes = concepts[
            cycle['simplex'].explode()
                .drop_duplicates()
                .to_list()
        ]
    cycle_edges = (
            cycle['simplex'].apply(lambda s: concepts[s])  # convert from numbers to nodes
                .apply(tuple)  # hashable type
                .to_numpy()
        )
    birth_edges = np.array([e for e in cycle_edges if G.edges[e]['norm_year'] == birth])
    
    # bounding chain nodes and edges
    if optimized.loc[idx, 'bounding_chain'] is None:
        return cycle_nodes, cycle_edges, birth_edges, np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    bounding_chain = optimized.loc[idx, 'bounding_chain']
    bounding_chain_nodes = concepts[
            bounding_chain['simplex'].explode()
                .drop_duplicates()
                .to_list()
        ]
    bounding_chain_nodes = bounding_chain_nodes[~np.isin(bounding_chain_nodes, cycle_nodes)]
    bounding_chain_edges = (
            bounding_chain['simplex'].apply(lambda s: concepts[s])  # convert from numbers to nodes
                .apply(lambda s: [s[[i, j]] for i in range(len(s)) for j in range(i+1, len(s))])
                .explode()
                .apply(tuple)  # hashable type
                .drop_duplicates()
                .to_numpy()  # create a 1d numpy array, otherwise the isin breaks later since it becomes a 2d array
        )
    bounding_chain_edges = bounding_chain_edges[~np.isin(bounding_chain_edges, cycle_edges)]
    tentpole_edges = np.array([e for e in bounding_chain_edges if (e[0] in bounding_chain_nodes and e[1] in cycle_nodes) or (e[0] in cycle_nodes and e[1] in bounding_chain_nodes)])
    arch_edges = np.array([e for e in bounding_chain_edges if e[0] in bounding_chain_nodes and e[1] in bounding_chain_nodes])
    death_edges = np.array([e for e in bounding_chain_edges if G.edges[e]['norm_year'] == death])
    
    return cycle_nodes, cycle_edges, birth_edges, bounding_chain_nodes, bounding_chain_edges, tentpole_edges, arch_edges, death_edges


def type_all_cycles(G, optimized, concepts):
    '''
    Gets the types for all cycles in the network. Returns a dataframe with info on
    edge and node types
    '''
    # don't modify intitial graph
    G = G.copy()

    # intialize counts to 0
    # we keep track of the results as an edge/node attribute which increments each time an edge/node is a certain type
    nx.set_node_attributes(G, {n: {'cycle': 0, 'bounding_chain': 0} for n in G.nodes})
    nx.set_edge_attributes(G, {e: {'cycle': 0, 'birth': 0, 'bounding_chain': 0, 'tentpole': 0, 'arch': 0, 'death': 0} for e in G.edges})

    # get types from each cycle
    this_cycle_typography = lambda i: cycle_typography(i, G, optimized, concepts)  # find cycle typeography on this network specifically
    for i in optimized.index:
        # get nodes/edges of each type in the cycle
        cycle_nodes, cycle_edges, birth_edges, bounding_chain_nodes, bounding_chain_edges, tentpole_edges, arch_edges, death_edges = this_cycle_typography(i)

        # # ph basis checks (comment out to make it faster)
        # assert birth_edges.size > 0
        # assert optimized.loc[i, 'death'] == np.inf or death_edges.size > 0

        # keep track of types
        for n in cycle_nodes:
            G.nodes[n]['cycle'] += 1
        for e in cycle_edges:
            G.edges[e]['cycle'] += 1
        for e in birth_edges:
            G.edges[e]['birth'] += 1
        for n in bounding_chain_nodes:
            G.nodes[n]['bounding_chain'] += 1
        for e in bounding_chain_edges:
            G.edges[e]['bounding_chain'] += 1
        for e in tentpole_edges:
            G.edges[e]['tentpole'] += 1
        for e in arch_edges:
            G.edges[e]['arch'] += 1
        for e in death_edges:
            G.edges[e]['death'] += 1

    # store node results to dataframe
    node_type_df = pd.DataFrame(
            dict(G.nodes(data=True))
        ).T.reset_index().rename(columns={'index': 'concept'})
    node_type_df = node_type_df.rename(columns={
            'cycle': 'cycle_count', 'bounding_chain': 'bounding_chain_count'
        })
    node_type_df['in_cycle'] = node_type_df['cycle_count'] > 0  # in cycle if count has incremented
    node_type_df['in_bounding_chain'] = node_type_df['bounding_chain_count'] > 0  # in bounding chain if count has incremented
    node_type_df[['cycle_count', 'bounding_chain_count']] = node_type_df[['cycle_count', 'bounding_chain_count']].astype(int)
    node_type_df = node_type_df.drop(columns=['year', 'norm_year', 'article_id'])

    # store edge results to dataframe
    edge_type_df = nx.to_pandas_edgelist(G, source='concept_s', target='concept_t')
    edge_type_df = edge_type_df.rename(columns={
            'cycle': 'cycle_count', 'birth': 'birth_count', 'bounding_chain': 'bounding_chain_count',
            'tentpole': 'tentpole_count', 'arch': 'arch_count', 'death': 'death_count'
        })
    edge_type_df['in_cycle'] = edge_type_df['cycle_count'] > 0  # in cycle if count has incrmeented
    edge_type_df['in_birth'] = edge_type_df['birth_count'] > 0
    edge_type_df['in_bounding_chain'] = edge_type_df['bounding_chain_count'] > 0
    edge_type_df['in_tentpole'] = edge_type_df['tentpole_count'] > 0
    edge_type_df['in_arch'] = edge_type_df['arch_count'] > 0
    edge_type_df['in_death'] = edge_type_df['death_count'] > 0
    flip_mask = edge_type_df['concept_s'] > edge_type_df['concept_t']  # align with concept list we make in dataframe so we can merege
    edge_type_df.loc[flip_mask, ['concept_s', 'concept_t']] = edge_type_df.loc[flip_mask, ['concept_t', 'concept_s']].values
    # .values makes it a numpy array so it ignores indicies
    # otherwise, flip doens't do anything since it matches column lables instead of locations
    num_cols = ['cycle_count', 'birth_count', 'bounding_chain_count', 'tentpole_count', 'arch_count', 'death_count']
    edge_type_df[num_cols] = edge_type_df[num_cols].astype(int)
    edge_type_df = edge_type_df.drop(columns=['year', 'norm_year', 'article_id'])

    return node_type_df, edge_type_df


def make_article_edge_df(article_concept_df):
    '''
    Makes the dataframe of all edges in the network
    '''
    # merge on itself to get edges
    article_edge_df = article_concept_df[['article_id', 'year', 'concept']].merge(
            article_concept_df[['article_id', 'concept']],
            on='article_id',
            how='outer',
            suffixes=['_s', '_t']
        )
    article_edge_df = article_edge_df[article_edge_df['concept_s'] < article_edge_df['concept_t']].reset_index(drop=True)

    return article_edge_df


def get_article_info(article_df, citation_df, type_df, type_idxs):
    '''
    Gets info about the articl's role in the network.

    `article_df` can be either an edge or node file
    '''
    # add in occurance count
    article_df['in_network'] = True
    article_df['rank'] = article_df.groupby(type_idxs)['year'].rank('min')
    article_df['first_in_network'] = article_df['rank'] == 1
    article_df['second_in_network'] = article_df['rank'] == 2

    # merge concept role info
    article_df = article_df.merge(
            type_df,
            on=type_idxs,
            how='outer',
        )
    
    # create variables of interest
    cols_of_interest = type_df.columns.drop(type_idxs)  # all columns with information about the type
    article_df['first_' + cols_of_interest] = article_df[cols_of_interest].multiply(article_df['first_in_network'], axis=0)
    article_df['second_' + cols_of_interest] = article_df[cols_of_interest].multiply(article_df['second_in_network'], axis=0)

    # aggregate
    num_cols = np.hstack(('in_network', type_df.columns[type_df.dtypes == int]))
    num_cols = np.concatenate((num_cols, 'first_'+num_cols, 'second_'+num_cols))
    bool_cols = np.hstack(('in_network', type_df.columns[type_df.dtypes == bool]))
    bool_cols = np.concatenate((bool_cols, 'first_'+bool_cols, 'second_'+bool_cols))
    article_groups = article_df.groupby('article_id')
    article_df = pd.concat((
                article_groups[num_cols].sum().rename(columns={
                        'in_network': 'network_count',
                        'first_in_network': 'first_network_count',
                        'second_in_network': 'second_network_count'
                    }),
                article_groups[bool_cols].any()
            ),
            axis=1,
        ).reset_index()
    
    # append citation info
    citation_df = citation_df.merge(
            article_df,
            on='article_id',
            how='left',
        )
    # unmerged, na values -> 0 or false
    num_cols = np.hstack(('network_count', type_df.columns[type_df.dtypes == int]))  # update num_cols to have network_count instead of in_network
    num_cols = np.concatenate((num_cols, 'first_'+num_cols, 'second_'+num_cols))
    citation_df[bool_cols] = citation_df[bool_cols].astype('boolean').fillna(False).astype(bool)
    citation_df[num_cols] = citation_df[num_cols].astype('Int64').fillna(0).astype(int)

    return citation_df


def main():
    # setup process
    print(f'{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}: Starting {FIELD}')

    # get the optimization results
    G, optimized, concepts = get_optimization_info(OPTIMIZED_FILE)
    print('Retrieved optimization results')

    # get types
    node_type_df, edge_type_df = type_all_cycles(G, optimized, concepts)
    print('Finished typing cycles')

    # get citation and article info
    citation_df = pd.read_csv(
            CITATION_FILE,
            compression='gzip'  # need to specify compression if you're reading from a dropbox link
        )
    article_concept_df = mn.filter_article_concept_file(
            ARTICLE_CONCEPT_FILE,
            min_relevance=MIN_RELEVANCE,
            min_articles=MIN_ARTICLE_FREQ,
            max_articles=MAX_ARTICLE_FREQ,
            min_year=MIN_YEAR
        )
    article_edge_df = make_article_edge_df(article_concept_df)
    print('Created relevant dataframes')

    # get citation dataframes
    node_citation_df = get_article_info(article_concept_df, citation_df, node_type_df, 'concept')
    edge_citation_df = get_article_info(article_edge_df, citation_df, edge_type_df, ['concept_s', 'concept_t'])

    # save results
    os.mkdir(OUTDIR)
    node_citation_df.to_parquet(OUTDIR + 'node.parquet')
    edge_citation_df.to_parquet(OUTDIR + 'edge.parquet')

    # finishing things
    print(f'Finished {FIELD}. Results saved in {OUTDIR}')


if __name__ == '__main__':
    main()
