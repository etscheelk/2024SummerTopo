'''
The dataset is (at times) big, and writing code uses a trial and error
process where it's helpful to be able to quickly write, test, and rewrite
code without the process of running it being super computationally
intensive. Therefore, I have a couple sampleing methods to take make the
dataset into a smaller, more workable form.

The sampling methods are:
 - `row_sample`: Sample rows randomly from the dataframe
 - `article_sample`: Sample random articles from the dataframe. Includes all
 concepts found in these articles
 - `concept_sample`: Sample random concepts from the dataframe. Includes all
 articles that contain these concepts
 - `propagation_sample`: Sample by starting at a node in the network and
 including all articles within a certain radius
 - `journal_sample`: Sample by picking all articles within a single journal
'''

# preliminaries
import Gavin.utils.make_network_v1 as mn
import networkx as nx
import pandas as pd
import numpy as np


def row_sample(df: pd.DataFrame,
               n_rows: int,
               seed: int | None = None
               ) -> pd.DataFrame:
    '''
    Samples rows from df. Essentially just runs
    ```
        df.sample(n_rows)
    ```
    This is a really awful way of sampling since it creates a very sparse graph
    without many homological features, so the alternative methods should be
    prefered for most testing cases

    Args:
        `df` (Dataframe): Pandas Dataframe you want to sample from
        `n_rows` (int): The number of rows from the dataframe to sample
        `seed` (int | None): The seed for the randomization process. Default None
    
    Returns:
        `sampled_df` (Dataframe): A sampled dataframe with only certain rows from
        the intial dataframe
    '''
    np.random.seed(seed) # set the seed, make everything replicable
    sampled_df = df.sample(n_rows, ignore_index=True) # sample

    return sampled_df


def article_sample(df: pd.DataFrame,
                   n_articles: int,
                   seed: int | None = None
                   ) -> pd.DataFrame:
    '''
    Samples articles from the dataframe, and returns all rows that include that
    article. This creates a better network than `row_sample`, but still is more
    disconnected than alternatives

    Args:
        `df` (Dataframe): Pandas Dataframe you want to sample from
        `n_articles` (int): The number of articles from the dataframe to sample
        `seed` (int | None): The seed for the randomization process. Default None
    
    Returns:
        `sampled_df` (Dataframe): A sampled dataframe with only certain rows from
        the intial dataframe
    '''
    np.random.seed(seed) # set the seed, make everything replicable
    articles = df['article_id'].drop_duplicates().sample(n_articles) # articles we include
    sampled_df = df[df['article_id'].isin(articles)] # include rows based on the article sample

    return sampled_df.reset_index(drop=True)


def concept_sample(df: pd.DataFrame,
                   n_concepts: int,
                   seed: int | None = None
                   ) -> pd.DataFrame:
    '''
    Samples concepts from the dataframe and returns all rows that include that
    concept. This method creates the best view of the overall network

    Args:
        `df` (Dataframe): Pandas Dataframe you want to sample from
        `n_concepts` (int): The number of concepts from the dataframe to sample
        `seed` (int | None): The seed for the randomization process. Default None
    
    Returns:
        `sampled_df` (Dataframe): A sampled dataframe with only certain rows from
        the intial dataframe
    '''
    np.random.seed(seed) # set the seed, make everything replicable
    concepts = df['concept'].drop_duplicates().sample(n_concepts) # concepts we include
    sampled_df = df[df['concept'].isin(concepts)] # include rows based on the concept sample

    return sampled_df.reset_index(drop=True)


def propagation_sample(df: pd.DataFrame,
                       n_layers: int,
                       source: any = None,
                       seed: int | None = None
                       ) -> pd.DataFrame:
    '''
    Propegates out from a node to get a subset of the network. Not technically a
    sample, since it's only looking at a micro-scale piece of the network, and is
    a breadth-first search instead a but I'm still calling it one bc reasons. This
    is the best method to get a small-scale view of the network, but won't replicate
    any of the global network properites (like the number of disconected components).
    It doesn't work very well on unfiltered data, since there will be nodes that
    connect to too many other nodes (hubs), but works well on getting a smaller bit
    of a filtered network

    Args:
        `df` (Dataframe): Pandas Dataframe you want to sample from
        `n_layers` (int): The number of layers around the source to look at
        `source` (any | None): The node to start at. If `None`, picks, a random node
        from the largest connected component
        `seed` (int | None): The seed for the randomization process. Default None
    
    Returns:
        `sampled_df` (Dataframe): A sampled dataframe with only certain rows from
        the intial dataframe
    '''
    np.random.seed(seed) # set the seed, make everything replicable

    # create a graph
    nodes = mn.get_nodes(df)
    edges = mn.get_edges(df)
    G = mn.make_network_from_nodes_edges(nodes, edges)

    # pick source
    if source is None:
        lcc_nodes = max(nx.connected_components(G), key=len) # choose from lcc to make sure we can propegate around it
        source = np.random.choice(list(lcc_nodes)) # define source
    
    # get concepts to include
    concepts = nx.bfs_tree(G, source=source, depth_limit=n_layers).nodes # concepts to include
    sampled_df = df[df['concept'].isin(concepts)] # include rows based on the concepts

    return sampled_df.reset_index(drop=True)


def journal_sample(df: pd.DataFrame,
                   journal: str,
                   journals_df: pd.DataFrame,
                   ) -> pd.DataFrame:
    '''
    Takes a subset of the journal-concept pairs by only looking at artciles within
    a single journal. The idea is that by non-arbitrarily restricting the suze of
    our "universe" we can replciation both some of the macro- and micro-scale
    properties of the network. Like `propagation_sample`, technically not a sample,
    but I'm still calling it one bc I'm not a statistician

    Args:
        `df` (Dataframe): Pandas Dataframe you want to sample from
        `journal` (str): Journal to sample from
        `journals_df (Dataframe): Dataframe of articles and the journal they're from

    Returns:
        `sampled_df` (Dataframe): A sampled dataframe with only certain rows from
        the intial dataframe
    '''
    # find the valid articles
    valid_articles = journals_df[journals_df['journal_title'] == journal]['article_id']

    # keep only those articles
    sampled_df = df[df['article_id'].isin(valid_articles)]

    return sampled_df
