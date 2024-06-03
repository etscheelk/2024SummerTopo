'''
Some basic functions to create the network that will make my life easier.

Right now, this document has
 - `filter_article_concept_file`: Takes a article-concept file and filters it to exclude irrelevant rows
 - `get_nodes`: Takes a article-concept dataframe and generates a list of nodes
 - `get_edges`: Takes a article-concept dataframe and generates a list of edges
 - `make_network_from_nodes_edges`: Takes nodes and edges for a network and generates a networkx network
 - `gen_concept_network`: Takes a article-concept file and filtering parameters and returns a concept network
'''

# preliminaries
import networkx as nx
import pandas as pd
import numpy as np


def filter_article_concept_file(file: str | pd.DataFrame,
                                relevance_cutoff: float = 0.,
                                min_article_freq: float = 0.,
                                max_article_freq: float = 1.,
                                normalize_year: bool = False,
                                year_min: int = 1600,
                                file_reader: callable = pd.read_csv
                                ) -> pd.DataFrame:
    '''
    Filters a article-concept file to remove concepts that aren't relevant
    to articles and that don't occur enough.

    Args:
        `file` (str of Dataframe): The file you want filtered. If a dataframe,
        filters the dataframe directly. If a string, reads it into a dataframe
        first then filters it.
        `relevance_cutoff` (float): The relevance score a concept must have within
        and article to be included in the dataset. Inclusive, default 0..
        `min_article_freq` (float): The minimum percentage of articles a concept
        needs to appear in to be included int the dataset. Inclusive, default 0.
        `max_article_freq` (int): The maximum percent of artciles a concept needs
        to appear in to be included in the dataset. Inclusive, default 1.
        `year_min` (int): Also, the `year_min` value in the year normalization process.
        All articles from before this year are excluded. Exlcusive, default 1600.
        `normalize_year` (bool): Whether or not the years should be normalized or left
        as the actual time. If `False`, the years are left as is. If `True`, a new
        column `norm_year` is added to the returned dataframe which is calculated as
            (year - year_min) / (year_max - year_min).
        The variable `year_min` is an argument to the function and `year_max` is the max year found
        in the dataset. Default False.
        `file_reader` (callable): Function used to read the dataframe from the file.
        Required if the file is anything that's not a csv.

    Returns:
        `df` (Dataframe): A dataframe containg all valid concept-article pairs. The
        dataframe has the columns
            `article_id`: A unique identifier for each article
            `category_for_2l_code`: The 2l code for the topic of the article
            `year`: The year the article was released
            `concept`: The concept found in the article
            `relevance_mean`: The mean relevance score for all occurences of the
            concept in the article abstact
            `concept_freq_in_abstract`: The number of times the concept shows up
            in the abstract
            `concept_no`: Unique identifier for the concept in the abstract.
            `dfreq_in_category_for_2l`: The number of articles within the topic
            that have the concept
            `dfreq_in_category_for_2l_year`: The number of articles within the topic
            in the same year that have the concept
            `norm_year`: The normalized year. Only in the dataframe if `normalize_year`
            is set to `True`.
    '''
    # read the file (if inputted as a string)
    df = file if type(file) == pd.DataFrame else file_reader(file)

    # basic filters
    df = df.rename(columns={ # rename for clarity
            "mean": "relevance_mean",
            "size": "concept_freq_in_abstract"
        })
    df = df[df['relevance_mean'] > relevance_cutoff] # filter relevance
    df = df[df['year'] > year_min] # filter rows representing too old of articles
    # has to be exclusive bc most persistance libraries break when an edge exists at time 0

    # concept frequency filters
    num_articles = df['article_id'].nunique() # number of articles
    min_articles = num_articles * min_article_freq # lower bound, minimum percentage of articles we want * the number of artciles
    max_articles = num_articles * max_article_freq # upper bound, maximum percentage of articles we want * the number of artciles
    concept_nums = df.groupby('concept').size() # concept frequencies, csv has no repeating article-concept pairs
    concept_nums = concept_nums[concept_nums >= min_articles] # remove concepts that are too rare
    concept_nums = concept_nums[concept_nums <= max_articles] # remove concepts that are too ubiquitous
    df = df[df['concept'].isin(concept_nums.index)] # keep concepts that have "goldilocks" frequency

    # normalize the year
    if normalize_year:
        year_max = df['year'].max() # last year (1 in normalization)
        df['norm_year'] = (df['year'] - year_min) / (year_max - year_min)
    
    return df.reset_index(drop=True)


def get_nodes(article_concept_df: pd.DataFrame
              ) -> np.ndarray[tuple[str, dict[str: any]]]:
    '''
    Generates a list of all the concept nodes in the network

    Args:
        `article_concept_df` (Dataframe): A pandas dataframe with the artcile-concept
        pairs. Should be formatted similarly to the result from
        `filter_article_concept_file`

    Returns:
        `nodes` (array[tuple[str, dict[str: any]]]): A list of the nodes in the network.
        Each node has a name, the concept, which is the first element in the tuple, and
        a dictionary of attributes, the second element in the tuple. The attributes are
            `article_id`: The first article the concept appeared in.
            `year`: The first year the concept was discussed in.
            `norm_year`: The normalized first year the concept was discussed in. Only
            included if `article_concept_df` includes `norm_year` as one of it's columns
        Formatting is chosen to make sure it works with networkx's `nx.add_nodes_from`
    '''
    # decide whether to include 'norm_year' as a node attribute
    normalize_year = 'norm_year' in article_concept_df.columns

    # generator for the node tuples
    if normalize_year:
        node_tuple_gen = lambda row: ( # map includes 'norm_year'
                row['concept'], # node
                { # attribute dict
                    'article_id': row['article_id'], # article edge is from
                    'year': row['year'], # year article was published in
                    'norm_year': row['norm_year'] # normalized year the article was published in
                }
            )
    else:
        node_tuple_gen = lambda row: ( # map doesn't include 'norm_year'
                row['concept'], # node
                { # attribute dict
                    'article_id': row['article_id'], # article edge is from
                    'year': row['year'] # year article was published in
                }
            )
    
    # create node list
    article_concept_df = article_concept_df.sort_values( # sort to get the earlier year to the start
        ['concept', 'year'] # technially don't need to sort by concept, but it makes it easier for a human to understand
    ).drop_duplicates( # Keep only the first occurance if a concept
        subset='concept'
    )
    nodes = np.array(article_concept_df.apply(node_tuple_gen, axis=1)) # make node tuples

    return nodes


def get_edges(article_concept_df: pd.DataFrame
              ) -> np.ndarray[tuple[str, str, dict[str: any]]]:
    '''
    Generates a list of all the concept-concept edges in the network that share an article

    Args:
        `article_concept_df` (Dataframe): A pandas dataframe with the artcile-concept
        pairs. Should be formatted similarly to the result from
        `filter_article_concept_file`

    Returns:
        `edges` (array[tuple[str, str, dict[str: any]]]): A list of the edges in the network.
        Each esge has two nodes, the concepts, which are the first two elements in the tuple,
        and a dictionary of attributes, the third element in the tuple. The attributes are
            `article_id`: The first article the concepts were linked in.
            `year`: The first year the concepts were linked in.
            `norm_year`: The normalized first year the concepts were linked in. Only
            included if `article_concept_df` includes `norm_year` as one of it's columns
        Formatting is chosen to make sure it works with networkx's `nx.add_edges_from`
    '''
    # decide whether to include 'norm_year' as an edge attribute
    normalize_year = 'norm_year' in article_concept_df.columns

    # generator for the edge tuples
    if normalize_year:
        edge_tuple_gen  = lambda row: ( # map to make edge tuples
                row['concept_x'], # edge source
                row['concept_y'], # edge target
                { # dict of attributes
                    'article_id': row['article_id'], # article edge is from
                    'year': row['year'], # year article was published in
                    'norm_year': row['norm_year'] # normalized year the article was published in
                }
            )
        keys = ['article_id', 'year', 'concept', 'norm_year']
        merge_keys = ['article_id', 'year', 'norm_year']
    else:
        edge_tuple_gen  = lambda row: ( # map to make edge tuples
                row['concept_x'], # edge source
                row['concept_y'], # edge target
                { # dict of attributes
                    'article_id': row['article_id'], # article edge is from
                    'year': row['year'] # year article was published in
                }
            )
        keys = ['article_id', 'year', 'concept']
        merge_keys = ['article_id', 'year']
    
    # find concept-concept pairs
    edge_df = article_concept_df[keys].merge( # combine on articles to make edge list
            right=article_concept_df[keys],
            on=merge_keys,
            how='outer'
        )
    edge_df = edge_df[edge_df['concept_x'] < edge_df['concept_y']] # remove rows where the concepts are equal and make sure theres only one of each row
    edge_df = edge_df.sort_values( # sort to get the earlier year to the start
            ['concept_x', 'year'] # technially don't need to sort by concept_x, but it makes it easier for a human to understand
        ).drop_duplicates( # Keep only the first occurance if a row is duplicated
            subset=['concept_x', 'concept_y']
        )
    
    # make edge tuples
    edges = np.array(edge_df.apply(edge_tuple_gen, axis=1)) # make edge tuples

    return edges


def make_network_from_nodes_edges(nodes: list[tuple[str, dict[str, any]]],
                                  edges: list[tuple[str, str, dict[str, any]]],
                                  ) -> nx.Graph:
    '''
    Takes a formatted list of nodes and edges and makes a networkx graph.

    Args:
        `nodes` (list[tuple[str, dict[str, any]]]): A list of node tuples in the
        (node, attibute_dict) form to use as the nodes of the network
        `edges` (list[tuple[str, str, dict[str, any]]]): A list of edge tuples in
        the (source, target, attribute_dict) form to use as the edges of the network
    
    Returns:
        `G` (nx.Graph): A networkx graph with nodes and edges from the inputted lists
    '''
    G = nx.Graph() # create the graph
    G.add_nodes_from(nodes) # add nodes
    G.add_edges_from(edges) # add edges

    return G


def gen_concept_network(file: str | pd.DataFrame,
                        relevance_cutoff: float = 0.7,
                        min_article_freq: float = 0.,
                        max_article_freq: float = 1.,
                        normalize_year: bool = False,
                        year_min: int = 1600,
                        file_reader: callable = pd.read_csv
                        ) -> nx.Graph:
    '''
    Takes a concept-article dataframe or file and returns a networkx
    graph of the concept-concept network based on the dataframe.

    This essentially acts as a wrapper for
        ```
        df = filter_article_concept_file(args)
        nodes = get_nodes(df)
        edges = get_edges(df)
        G = make_network_from_nodes_edges(nodes, edges)
        ```
    and that approach is prefered if you want to be able to direcly access
    elements of the dataframe or node/edge tuples, but otherwise just use
    this function.

    Args:
        `file` (str of Dataframe): The file you want to create a network from. If
        a dataframe, uses the dataframe directly. If a string, reads it into a dataframe
        first then filters it.
        `relevance_cutoff` (float): The relevance score a concept must have within
        and article to be included in the network. Inclusive, default 0.7.
        `min_article_freq` (float): The minimum percentage of articles a concept
        needs to appear in to be included int the dataset. Inclusive, default 0.
        `max_article_freq` (int): The maximum percent of artciles a concept needs
        to appear in to be included in the dataset. Inclusive, default 1.
        `year_min` (int): Also, the `year_min` value in the year normalization process.
        All articles from before this year are excluded. Exlcusive, default 1600.
        `normalize_year` (bool): Whether or not the years should be normalized or left
        as the actual time. If `False`, the years are left as is. If `True`, a new
        column `norm_year` is added to the returned dataframe which is calculated as
            (year - year_min) / (year_max - year_min).
        The `year_min` variable is an argument to the function and `year_max` is the
        max year found in the dataset. Default False.

        `file_reader` (callable): Function used to read the dataframe from the file.
        Required if the file is anything that's not a csv.

    Returns:
        `G` (nx.Graph): Networkx graph of all the concept-concept pairs. Each node and
        edge has the attributes
            `article_id`: The article the concept or concept pair first appeared
            `year`: The year the concept or concept pair first appeared
            `norm_year`: The normalized year the concept or concept pair first appeared
    '''
    df = filter_article_concept_file(file=file, # get the relevant file to name the network from 
                                     relevance_cutoff=relevance_cutoff,
                                     min_article_freq=min_article_freq,
                                     max_article_freq=max_article_freq,
                                     normalize_year=normalize_year,
                                     year_min=year_min,
                                     file_reader=file_reader)
    nodes = get_nodes(df) # nodes in the network
    edges = get_edges(df) # edges in the network
    G = make_network_from_nodes_edges(nodes, edges) # make the network

    return G
