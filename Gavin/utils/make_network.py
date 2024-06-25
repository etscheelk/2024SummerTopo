'''
Updated version of `make_network_v1`. Allows for edges to be filtered based on
PMI and/or cooccurances and changes some keywords up.

Right now, this document has
 - `filter_article_concept_file`: Filters a article-concept file by year, relevance,
 and number of occurances
 - `concept_pairs_from_article_concepts`: Creates an edgelist with no filtering
 - `filter_cooccurrences`: Creates an edgelist with filtered edges
 - `concepts_from_article_concepts`: Creates a nodelist based only on an article-
 concept list
 - `concepts_from_pairs`: Creates a nodelist based on an article-concept list and
 filtered edgelist. Excludes isolate nodes
 - `network_from_dfs`: Creates a network from a dataframe node and edgelist
 - `adj_matrix`: Creates an adjacency matrix from a network
 - `gen_concept_network`: Creates a filtered concept network
'''

# preliminaries
from scipy.sparse import csr_matrix
import networkx as nx
import pandas as pd
import numpy as np


def _filter_year(df: pd.DataFrame, min_year: int | None, max_year: int | None, year_str: str = 'year') -> pd.DataFrame:
    '''
    Remove rows representing too old or new of articles. Filters to artciles published in (min, max]
    '''
    if min_year is not None:
        df = df[df[year_str] > min_year]
    if max_year is not None:
        df = df[df[year_str] <= max_year]

    return df
    # has to be exclusive bc most persistance libraries break when an edge exists at time 0


def _normalize_year(year: pd.Series | np.ndarray | int, min_year: int | None, max_year: int | None) -> pd.DataFrame:
    '''
    Add a column to the dataframe with a normalized year
    '''
    # figure out what min and max year are
    if min_year is None:
        min_year = year.min()
    if max_year is None:
        max_year = year.max()

    # calculate normalized year
    norm_year = (year-min_year) / (max_year-min_year)

    return norm_year


def filter_article_concept_file(file: str | pd.DataFrame,
                                min_relevance: float = 0.,
                                min_year: int | None = 1600,
                                max_year: int | None = None,
                                min_articles: float | int = 0,
                                max_articles: float | int = 1.,
                                normalize_year: bool = False,
                                file_reader: callable = pd.read_csv
                                ) -> pd.DataFrame:
    '''
    Filters a article-concept file to remove concepts that aren't relevant
    to articles and that don't occur enough. Defaults are set to not filter
    anything out

    Args:
        `file` (str of Dataframe): The file you want filtered. If a dataframe,
        filters the dataframe directly. If a string, reads it into a dataframe
        first then filters it.
        `min_relevance` (float): The relevance score a concept must have within
        and article to be included in the dataset. Inclusive, default 0..
        `min_year` (int | None): If set, all articles from this year and before
        are excluded. Also, the `year_min` value in the year normalization process.
        Exlcusive, default 1600.
        `max_year` (int | None): If set, all articles from after this year are
        excluded. Also, the `year_max` value in the year normalization process.
        Inclusive, default None.
        `min_articles` (float | int): The minimum articles a concept needs to appear
        in to be included in the dataset. If an int, used as a number of articles. If
        a float, used as a percentage of articles. Inclusive, default 0.
        `max_article_freq` (float | int): The maximum articles a concept needs to appear
        in to be included in the dataset. If an int, used as a number of articles. If
        a float, used as a percentage of articles. Inclusive, default 1..
        `normalize_year` (bool): Whether or not the years should be normalized or left
        as the actual time. If `False`, the years are left as is. If `True`, a new
        column `norm_year` is added to the returned dataframe which is calculated as
            (year - year_min) / (year_max - year_min).
        If `year_min` and `year_max` aren't none, that's the year used. Otherwise, it's
        the min and max year in the dataset, respecively. Default False.
        `file_reader` (callable): Function used to read the dataframe from the file.
        Required if the file is anything that's not a csv. Default pd.read_csv.

    Returns:
        `df` (Dataframe): A dataframe containg all valid concept-article pairs. The
        dataframe has the columns
            "article_id": A unique identifier for each article
            "year": The year the article was released
            "concept": The concept found in the article
            "relevance_mean": The mean relevance score for all occurences of the
            concept in the article abstact
            "norm_year": The normalized year. Only in the dataframe if `normalize_year`
            is set to `True`
    '''
    # read the file (if inputted as a string)
    df = file.copy() if type(file) == pd.DataFrame else file_reader(file)
    df = df.rename(columns={ # rename for clarity
            'mean': 'relevance_mean'
        })
    df = df[['article_id', 'year', 'concept', 'relevance_mean']]

    # basic filters
    df = df[df['relevance_mean'] >= min_relevance]
    df = _filter_year(df, min_year, max_year)

    # normalize the year
    if normalize_year:
        df['norm_year'] = _normalize_year(df['year'], min_year, max_year)

    # number of articles
    n_articles = df['article_id'].nunique()
    if isinstance(min_articles, float): # filter as percent, not number of articles
        min_articles = int(min_articles * n_articles)
    if isinstance(max_articles, float): # filter as percent, not number of articles
        max_articles = int(max_articles * n_articles)
    sizes = df.groupby('concept').transform('size') # match each concept to the number of times it appears
    df = df[(sizes >= min_articles)
            & (sizes <= max_articles)]

    return df.reset_index(drop=True)


def concept_pairs_from_article_concepts(article_concept_df: pd.DataFrame
                                        ) -> pd.DataFrame:
    '''
    Create a dataframe of concept pairs to be used as an edgelist from an article-
    concept pairs dataframe

    Args:
        `article_concept_df` (pd.Dataframe): Pandas dataframe of article-concept pairs
        with columns for "article_id", "year", and "concept", and, if you want norm_year
        or prob_t/prob_s/count_st included in the result, "norm_year" and "prob"

    Returns:
        `concept_pairs_df` (pd.Dataframe): Pandas edgelist with all filtered edges that
        should be included. Has the columns:
            "concept_s": The source concept for the edge
            "concept_t": The target concept for the edge
            "year": The first year the edge shows up
            "norm_year": The first normalized year the edge shows up. Included only if in
            the original dataframe
            "article_id": The first article the edge shows up in
            "prob_s": The probability of the source concept being in a randomly chosen
            article. Used in the PMI calcultion. Only included if prob is in the initial
            Dataframe
            "prob_t": The probability of the traget concept being in a randomly chosen
            article. Used in the PMI calcultion. Only included if prob is in the initial
            Dataframe
            "count_st": The number of times the two concepts are connected. Only included
            if prob is in the initial Dataframe
    '''
    # figure out what columns to use
    merge_cols = ['article_id', 'concept'] # whether to include prob or not
    prob_cols = []
    if 'prob' in article_concept_df.columns:
        merge_cols.append('prob')
        prob_cols = ['prob_s', 'prob_t', 'count_st']
    year_cols = ['year'] # whether to include norm year or not
    if 'norm_year' in article_concept_df.columns:
        year_cols.append('norm_year')
    
    # merge
    article_concept_pairs_df = pd.merge( # merge article-concept file on itself to get edges
            right=article_concept_df[merge_cols+year_cols], 
            left=article_concept_df[merge_cols], # year is the same across articles, only need it once
            on='article_id', # get concepts that share an article
            suffixes=("_s","_t") # s => source, t => target
        )
    article_concept_pairs_df = article_concept_pairs_df[article_concept_pairs_df['concept_s'] < article_concept_pairs_df['concept_t']] # remove duplicated/same concept edges

    # filter to one occurance
    concept_pairs_df = article_concept_pairs_df.groupby( # group by concepts
            ['concept_s', 'concept_t'], # edge
            sort=False # make idxmin work
        ).agg(
            count_st=('article_id', 'count'), # count is faster than n_unique
            min_year_i=('year', 'idxmin') # index of minimum year, use to get min year and first article id
        ).reset_index() # take concept pairs out of index
    concept_pairs_df[
            article_concept_pairs_df.columns.drop(['concept_s', 'concept_t']) # include all columns
        ] = article_concept_pairs_df.loc[
            concept_pairs_df['min_year_i'], # index of min year
            article_concept_pairs_df.columns.drop(['concept_s', 'concept_t']) # take all columns from the earliest instance
        ].reset_index(drop=True)  # make assignment work correctly
    concept_pairs_df = concept_pairs_df[['concept_s', 'concept_t']+year_cols+['article_id']+prob_cols] # reorder/drop min_year_i

    return concept_pairs_df


def filter_cooccurrences(article_concept_df: pd.DataFrame,
                         min_cooccurrences: int | float = 0.,
                         max_cooccurrences: int | float = 1.,
                         min_pmi: float = -np.inf,
                         k: int | float = 1
                         ) -> pd.DataFrame:
    '''
    Filters an `article_concept_df` and creates an edgelist that removes edges that
    aren't connected enough and have enough cooccurrences

    NOTE: The calculated PMI uses natural log instead of log base 2. This is easy to
    change, but seems to have precident (even the example on the PMI Wikipedia page
    uses the natual log) and has better vibes

    Args:
        `article_concept_df` (pd.Dataframe): Dataframe of articles and concepts to be
        filtered. Should have a 'article_id', 'concept', 'year', and 'relevance_mean'
        column with an optional 'norm_year' column
        `min_cooccurrences` (int | float): The minimum number of cooccurrences two
        articles should have to be included. If an int, treated as a number. If a float,
        treated as a percent. Default 0. (float)
        `max_cooccurrences` (int | float): The maximum number of cooccurrences two
        articles should have to be included. If an int, treated as a number. If a float,
        treated as a percent. Default 1. (float)
        `min_pmi` (float): The minimum PMI to be included as an edge. Default -inf
        `k` (int | float): The k used in the PMI^k calculation

    Returns:
        `concept_pairs_df` (pd.Dataframe): Pandas edgelist with all filtered edges that
        should be included. Has the columns:
            "concept_s": The source concept for the edge
            "concept_t": The target concept for the edge
            "year": The first year the edge shows up
            "norm_year": The first normalized year the edge shows up. Included only if in
            the original dataframe
            "article_id": The first article the edge shows up in
            "prob_s": The probability of the source concept being in a randomly chosen
            article. Used in the PMI calcultion
            "prob_t": The probability of the traget concept being in a randomly chosen
            article. Used in the PMI calcultion
            "prob_st": The probability of the two concepts being connected in a randomly
            chosen article
            "pmi^k": The PMI value used for filtering
    '''
    article_concept_df = article_concept_df.copy() # don't change original dataframe

    # concept probabilities 
    n_articles = article_concept_df['article_id'].nunique() # denominator for probabilities
    article_concept_df['prob'] = article_concept_df.groupby('concept').transform('size') / n_articles # calculate
    # transform calls `size` on the groupby, then reindexes everything to realign with the rest of the dataframe
    # this means it's there when we merge and avaoids and extra merge

    # edges
    concept_pairs_df = concept_pairs_from_article_concepts(article_concept_df)
    
    # filter cooccurances
    if isinstance(min_cooccurrences, float): # filter as percent, not number of articles
        min_cooccurrences = int(min_cooccurrences * n_articles)
    if isinstance(max_cooccurrences, float): # filter as percent, not number of articles
        max_cooccurrences = int(max_cooccurrences * n_articles)
    concept_pairs_df = concept_pairs_df[(concept_pairs_df['count_st'] >= min_cooccurrences) # filter cooccurances
                                        & (concept_pairs_df['count_st'] <= max_cooccurrences)]
    
    # add concept/article info
    concept_pairs_df['prob_st'] = concept_pairs_df['count_st'] / n_articles # make probability (from count)
    concept_pairs_df = concept_pairs_df.drop(columns='count_st')

    # pmi filter
    concept_pairs_df[f'pmi^{k}'] = np.log(concept_pairs_df['prob_st']**k / (concept_pairs_df['prob_s']*concept_pairs_df['prob_t'])) # calculate pmi^k
    concept_pairs_df = concept_pairs_df[concept_pairs_df[f'pmi^{k}'] >= min_pmi]

    return concept_pairs_df


def concepts_from_article_concepts(article_concept_df: pd.DataFrame,
                                   ) -> pd.DataFrame:
    '''
    Get a list of concepts from a dataframe of article-concept pairs

    Args:
        `article_concept_df` (pd.Dataframe): Pandas dataframe of article-concept pairs
        with columns for "article_id", "year", "concept", and, if you want norm_year
        included in the result, "norm_year"

    Returns:
        `concept_df` (pd.Dataframe): A dataframe of concepts to be nodes in the
        network. Includes the columns:
            "concept": The concept
            "year": The first year the concept showed up in
            "norm_year": The normalized first year the concept showed up in
            "article_id": The first article the concept was in
    '''
    # concept info (from article_concept_df)
    year_cols = ['year'] # whether to include norm year or not
    if 'norm_year' in article_concept_df.columns:
        year_cols.append('norm_year')
    min_year_is = article_concept_df[['concept', 'year']].groupby('concept', sort=False).idxmin()['year']
    concept_df = article_concept_df.loc[
            min_year_is, ['concept']+year_cols+['article_id']
        ]
    
    return concept_df.reset_index(drop=True)


def concepts_from_pairs(article_concept_df: pd.DataFrame,
                        concept_pairs_df: pd.DataFrame
                        ) -> pd.DataFrame:
    '''
    Creates a list of concepts to be used as nodes in the network from an edgelist

    Args:
        `article_concept_df` (pd.Dataframe): A dataframe to use to get information
        about each article
        `concept_pairs_df` (pd.Dataframe): A dataframe of concept pairs. All concepts
        in this dataframe are included in the result

    Returns:
        `concept_df` (pd.Dataframe): A dataframe of concepts to be nodes in the
        network. Includes the columns:
            "concept": The concept
            "year": The first year the concept showed up in
            "norm_year": The normalized first year the concept showed up in
            "article_id": The first article the concept was in
    '''
    # concepts we want to include
    concepts = pd.concat((concept_pairs_df['concept_s'], concept_pairs_df['concept_t'])).drop_duplicates()

    # concept info (from article_concept_df)
    concept_df = concepts_from_article_concepts(article_concept_df)
    
    # keep relevant concepts
    concept_df = concept_df[concept_df['concept'].isin(concepts)]

    return concept_df.reset_index(drop=True)


def _get_nodes(concept_df: pd.DataFrame
               ) -> np.ndarray[tuple[str, dict[str: any]]]:
    '''
    Turn a dataframe of nodes/node info into a format useable to create a networkx graph
    '''
    node_tuple_gen = lambda row: ( # map includes 'norm_year'
            row['concept'], # node
            {k: row[k] for k in concept_df.columns.drop('concept')} # attribute dict
        )
    
    nodes = np.array(concept_df.apply(node_tuple_gen, axis=1)) # make node tuples

    return nodes


def _get_edges(concept_pairs_df: pd.DataFrame
               ) -> np.ndarray[tuple[str, dict[str: any]]]:
    '''
    Turn a dataframe of nodes/node info into a format useable to create a networkx graph
    '''
    edge_tuple_gen = lambda row: ( # map includes 'norm_year'
            row['concept_s'], # source
            row['concept_t'], # target
            {k: row[k] for k in concept_pairs_df.columns if k in ['year', 'norm_year', 'article_id']} # attribute dict
        )
    
    edges = np.array(concept_pairs_df.apply(edge_tuple_gen, axis=1)) # make node tuples

    return edges


def network_from_dfs(concept_pairs_df: pd.DataFrame,
                     concept_df: pd.DataFrame
                     ) -> nx.Graph:
    '''
    Creates a network from a edgelist and nodelist

    Args:
        `concept_pairs_df` (pd.DataFrame): The edgelist. Should have columns for
        "concept_s", "concept_t", "year", "article_id", and, if you want it included,
        "norm_year"
        `concept_df` (pd.DataFrame): The nodelist. Should have columns for "concept",
        "year", "article_id", and, if you want it included, "norm_year"
    
    Returns:
        `G` (nx.Graph): A networkx graph with nodes and edges from the inputted dataframes
    '''
    # get pieces
    nodes = _get_nodes(concept_df)
    edges = _get_edges(concept_pairs_df)

    # make graph
    G = nx.Graph() # create the graph
    G.add_nodes_from(nodes) # add nodes
    G.add_edges_from(edges) # add edges

    return G


def _should_filter__cooccurrences(min_cooccurrences: int | float, max_cooccurrences: int | float, min_pmi: float) -> bool:
    '''
    Takes inputs to filter__cooccurrences, returns True if filter_cooccurances
    would do anything
    '''
    # if theres a minimum number of cooccurances
    if min_cooccurrences > 0:
        return True
    
    # if the max number of cooccurances is a percent and less than 100%
    if isinstance(max_cooccurrences, float) and max_cooccurrences < 1:
        return True
    
    # if the max number of cooccurances is set a number of cooccurances
    if isinstance(max_cooccurrences, int):
        return True
    
    # if theres a set minimum pmi
    if min_pmi > -np.inf:
        return True
    
    return False


def adj_matrix(G: nx.Graph,
               weight: str | None = None,
               fill_diag: bool = True,
               diag_val: float | None = 0
               ) -> csr_matrix:
    '''
    Creates an adjacency matrix of a networkx graph to use for homology
    calculations

    This does, essentially, the same thing as `nx.adjacency_matrix` (and
    is a wrapper for it), just also fills in the diagnal and calls 
    `sorted_indices()` at the end to avoid OAT issues

    Args:
        `G` (nx.Graph): Networkx graph you want an adjacency matrix for
        `weight` (str | None): Key to use as the weight in the adjacency
        matrix. If none, creates an unweighted matrix
        `fill_diag` (bool): Whether to fill the diagnal of the matrix. If
        True, uses either the same key as the edges (`weight`) to get a
        value from each node or the `diag_val` input to fill the diagnal
        `diag_val` (float | None): The number put accross the whole diagnal.
        If set to None, uses a key from each node to get the value
    
    Returns:
        `adj` (csr_matrix): Sparse adjacency matrix for the network
    '''
    ## create the adjacency matrix
    adj = nx.adjacency_matrix(G, weight=weight) # adjacency matrix
    if fill_diag:
        if diag_val is None:
            node_births = list(nx.get_node_attributes(G, weight).values()) # node orgin times
            adj.setdiag(node_births)
        else:
            adj.setdiag(diag_val)
    adj = adj.sorted_indices() # needed on some computers for oat (not others tho which is confusing)

    return adj


def gen_concept_network(file: str | pd.DataFrame,
                        min_relevance: float = 0.,
                        min_year: int | None = 1600,
                        max_year: int | None = None,
                        min_articles: float | int = 0,
                        max_articles: float | int = 1.,
                        normalize_year: bool = False,
                        file_reader: callable = pd.read_csv,
                        min_cooccurrences: int | float = 0.,
                        max_cooccurrences: int | float = 1.,
                        min_pmi: float = -np.inf,
                        k: int | float = 1,
                        return_adj: bool = False
                        ) -> nx.Graph | csr_matrix:
    '''
    Takes a concept-article dataframe or file and returns a networkx graph of the
    concept-concept network based on the dataframe.

    Args:
        `file`, `min_relevance`, `min_year`, `max_year`, `min_articles`, `max_articles`,
        `normalize_year`, `file_reader`: Passed to `filter_article_concept_file`. See docs
        there
        `min_cooccurrences`, `max_cooccurrences`, `min_pmi`, `k`: Passed to
        `filter_cooccurrences`. See docs there
        `return_adj` (bool): Whether to return a graph or adjacency matrix. If True, 
        `normalize_year` is set to True and the adjacency matrix has the norm years filled in

    Returns:
        `G` (nx.Graph): Networkx graph created based on the files. Returned if `return_adj` is
        False
        `adj` (csr_matrix): Adjacency matrix for the graph to do persistance on
    '''
    # if you want an adjacency matrix, we normalize the year
    normalize_year = True if return_adj else normalize_year

    # article concept pairs
    article_concept_df = filter_article_concept_file(
            file,
            min_relevance=min_relevance,
            min_year=min_year,
            max_year=max_year,
            min_articles=min_articles,
            max_articles=max_articles,
            normalize_year=normalize_year,
            file_reader=file_reader
        )
    
    # get node and edge lists
    if _should_filter__cooccurrences(min_cooccurrences, max_cooccurrences, min_pmi):
        concept_pairs_df = filter_cooccurrences(
                article_concept_df,
                min_cooccurrences=min_cooccurrences,
                max_cooccurrences=max_cooccurrences,
                min_pmi=min_pmi,
                k=k
            )
        concept_df = concepts_from_pairs(article_concept_df, concept_pairs_df)
    else:
        concept_pairs_df = concept_pairs_from_article_concepts(article_concept_df)
        concept_df = concepts_from_article_concepts(article_concept_df)
    
    # create the graph
    G = network_from_dfs(concept_pairs_df, concept_df)

    # return
    if return_adj:
        return adj_matrix(G, 'norm_year', True, None)
    return G
