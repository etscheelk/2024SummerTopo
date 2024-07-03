'''
Useful functions to help compare barcodes to other barcodes

Right now, the document has
 - `format_ripser_output`: Formats the ripser barcode function to a
 format that can be used in other functions
 - `format_guhdi_output`: Formats the gudhi barcode function to a
 format that can be used in other functions
 - `format_oat_output`: Formats the oat barcode function to a format
 that can be used in other functions
 - `persistance_image`: Makes a peristance image for an inputted
 homology
 - `bottleneck_distance`: A wrapper for a hera command to find the
 bottleneck distance between barcodes
 - `wasserstein_distance`: A wrapper for a hera command to find the
 Wasserstein Distance between barcodes
 - `barcode_distance_matrix`: A functions that returns the distance
 between a bunch of barcodes in a matrix
'''

# preliminaries
import Gavin.utils.random_complexes as rc
# import random_complexes as rc
import oatpy as oat
import pandas as pd
import numpy as np


def format_ripser_output(ripser_res: dict[str: np.ndarray]
                         ) -> pd.DataFrame:
    '''
    Takes the results from a ripser function and formats it as a dataframe
    to be used to compare to other barcodes.

    Args:
        `ripser_res` (dict[str: np.ndarray]): The results from the `ripser`
        function within the `ripser` library
    
    Returns:
        `res` (Dataframe): Formatted and sorted dataframe containing the
        barcode information in "dimension", "birth", and "death" columns
    '''
    dimension = [] # feature dimention
    birth = [] # birth time
    death = [] # death time

    for dim, bc in enumerate(ripser_res['dgms']):
        for b, d in bc:
            dimension.append(dim)
            birth.append(b)
            death.append(d)

    # collect results
    res = pd.DataFrame(data={'dimension': dimension, 'birth': birth, 'death': death})
    res = res.sort_values(['dimension', 'birth', 'death'], ignore_index=True)
    return res


def format_guhdi_output(guhdi_res: list[int, tuple[float, float]]
                        ) -> pd.DataFrame:
    '''
    Takes the results from a ripser function and formats it as a dataframe
    to be used to compare to other barcodes.

    Args:
        `gudhi_res` (list[int, tuple[float, float]): The results from the
        `persistence` function within the `gudhi` library
    
    Returns:
        `res` (Dataframe): Formatted and sorted dataframe containing the
        barcode information in "dimension", "birth", and "death" columns
    '''
    dimension = [h[0] for h in guhdi_res] # feature dimention
    birth = [h[1][0] for h in guhdi_res] # birth time
    death = [h[1][1] for h in guhdi_res] # death time

    # collect results
    res = pd.DataFrame(data={'dimension': dimension, 'birth': birth, 'death': death})
    res = res.sort_values(['dimension', 'birth', 'death'], ignore_index=True)
    return res


def format_oat_output(oat_res: pd.DataFrame
                      ) -> pd.DataFrame:
    '''
    Takes the results from a oat homology calculation and formats it as
    a dataframe to be used to compare to other barcodes. Don't technially
    need to do this, since the oat output has all the needed columns
    labeled correctly, but this gets rid of extra information in the barcode

    Args:
        `oat_res` (Dataframe): The results from the `homology`
        function within the `oatpy` library
    
    Returns:
        `res` (Dataframe): Formatted and sorted dataframe containing the
        barcode information in "dimension", "birth", and "death" columns
    '''
    res = oat_res[['dimension', 'birth', 'death']]
    res = res.sort_values(['dimension', 'birth', 'death'], ignore_index=True)
    return res


def persistance_image(barcode: pd.DataFrame,
                      dim: int | list[int] | None = None,
                      sigma: float = 0.1,
                      res: int = 20,
                      weight_func: callable = lambda x, y: y,
                      return_vec: bool = False
                      ) -> np.ndarray | dict[int: np.ndarray]:
    '''
    Makes a persitance image from a homology dataframe

    The coordinates for the peristance image are normalized within [0, 1]. The function
    treats features that never die as having a death coordiate of 1, so features that die
    in the last period aren't distingushed from features that never die. If you want to
    plot the perstance image, use
    ```
        import matplotlib.pyplot as plt

        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        x, y = np.meshgrid(x, y)
        res = persistance_image(
                homology,
                res=resolution,
                return_vec=False # return_vec defaults false, so you can leave it out, but not make it true
            )

        plt.axis('equal')
        plt.pcolormesh(x, y, res[dimension])
    ```

    Args:
        `barcode` (Dataframe): A dataframe with homological features of the simplicial
        complex. Should have a "birth" column for when features were born, a "death"
        column with when features died, and a "dimension" column with the feature
        dimension
        `dim` (int | list[int] | None): The dimensions to create a persistance image for.
        If an int, creates an image for only that dimension. If a list (or other iterable),
        creates an image for all the dimensions in the list. If None, creates an image for
        all unique dimensions in `barcode`.
        Default None.
        `sigma` (float): Standard deviation of the Gaussians used for the persistance
        image. Default 0.1
        `res` (int): Resolution of the returned peristance image. Default 20
        `weight_func` (callable): Function in f(x, y) form that takes the birth, lifetime
        coordinates of the feature and returns the weight of the feature in the gaussians.
        f(x, 0) should be 0 for all x values. Defaults to linear scaling based only on the
        lifetime 
        `return_vec` (bool): Whether to return an matrix or vector. If True, returns a
        vector. Typically, a matrix will work better to visualize the persistance image but
         a vector is better to compare two barcodes. Default False
    
    Returns:
        `persistance_images` (list[np.ndarray]): A list of matricies of vectors (depening on
        the return_vec value) with the peristance images. Each value in the list corresponds
        to a different dimension, with the ith value corresponding to the persistance image
        for i dimensional features
    '''
    # we want it in a list
    if isinstance(dim, int):
        dim = [dim]

    # maximum lifetime
    # we want to normalize everything to [0, 1], this is the value we divide by to do that
    max_lifetime = oat.barcode.max_finite_value(barcode['death'])

    # points on the persistance diagram
    persistance_diagram = pd.DataFrame()
    persistance_diagram['dimension'] = barcode.reset_index(drop=True)['dimension'] # Each unique dimension will be a different diagram
    persistance_diagram['birth'] = barcode.reset_index(drop=True)['birth'] / max_lifetime # normalize birth times
    persistance_diagram['death'] = barcode.reset_index(drop=True)['death'] / max_lifetime # normalize death times
    persistance_diagram.loc[persistance_diagram['death'] == np.inf, 'death'] = 1 # set inf death times to the max (1)
    persistance_diagram['lifetime'] = persistance_diagram['death'] - persistance_diagram['birth'] # birth, lifetime create persistance image basis
    persistance_diagram['weight'] = weight_func(persistance_diagram['birth'], persistance_diagram['lifetime'])

    # remove rows after max dimension
    if dim is not None:
        persistance_diagram = persistance_diagram[persistance_diagram['dimension'].isin(dim)]
    else:
        dim = list(persistance_diagram['dimension'].unique()) # loop over dim later, needs to be defined

    # matricies to calculate peristance image
    # x, y coordinate of every entry i, j in the persistance image is x[0][i, j], x[1][i, j]
    x = np.meshgrid(*[np.linspace(0, 1, res) for _ in range(2)])

    # apply gaussians and get images
    persistance_images = []
    for d in dim: # create a different persisance image for each dimension
        d_persistance_diagram = persistance_diagram[persistance_diagram['dimension'] == d] # peristance diagram fo dimension
        gaussians = rc.gen_stacked_gaussians(
                As=np.array(d_persistance_diagram['weight']),
                x0s=np.array(d_persistance_diagram[['birth', 'lifetime']]),
                sigmas=np.full(len(d_persistance_diagram), sigma)
            )
        persistance_image = np.full((res, res), 0) + gaussians(np.stack(x, axis=-1)) # full makes sure we have all 0s if dimension has no features
        persistance_images.append(persistance_image)
    
    # make a vector (if we should)
    if return_vec:
        persistance_images = [np.reshape(pi, -1) for pi in persistance_images]
    
    # reshape returned value
    if len(persistance_images) == 1: # take it out of the list if there's only one thing
        persistance_images = persistance_images[0]
    else: # make a dict with the dim values
        persistance_images = {d: pi for d, pi in zip(dim, persistance_images)}

    return persistance_images


def persistance_image_distance(barcode_1: pd.DataFrame,
                               barcode_2: pd.DataFrame,
                               dim: int | None = None,
                               norm: float = 2.,
                               **kwargs: dict[str: any]
                               ) -> float:
    '''
    Calculates the distance between two barcodes using peristance images.

    Args:
        `barcode_1` (Dataframe): Dataframe with the information for the first barcode.
        Should have "birth" and "death" columns
        `barcode_2` (Dataframe): Dataframe with the information for the second barcode.
        Should have "birth" and "death" columns
        `dim` (int | None): The dimension to look at if an int. Doesn't filter dimensions
        at all otherwise. Will throw an error if None and the barcodes haven't already
        been filtered to only a single dimension.
        `norm` (float): The norm to calculate distance between persistance images using.
        Should be >= 1. Default 2
        `**kwargs` (dict[str: any]): Options passed to the `persistance_image` function
    
    Returns
        `dist` (Float): The Persistance Image Distance between the two barcodes
    '''
    barcode_1, barcode_2 = barcode_1.copy(), barcode_2.copy() # don't change the dataframes

    # filter dimension
    if dim is not None:
        barcode_1 = barcode_1[barcode_1['dimension'] == dim]
        barcode_2 = barcode_2[barcode_2['dimension'] == dim]

    # setup
    persistance_image_1 = persistance_image(barcode_1, return_vec=True, **kwargs)
    persistance_image_2 = persistance_image(barcode_2, return_vec=True, **kwargs)

    # distance function
    if norm == np.inf:
        dist = np.max(np.abs(persistance_image_1 - persistance_image_2))
    else:
        dist = (np.sum(np.abs(persistance_image_1 - persistance_image_2)**norm))**(1/norm)

    return dist


def _hera_format(barcode, dim=None, normalize=True):
    # dimension
    if dim is not None:
        barcode = barcode[barcode['dimension'] == dim]

    # numpy array
    barcode = np.array(barcode[['birth', 'death']])
    
    # normalize and truncate infs
    max_lifetime = oat.barcode.max_finite_value(np.reshape(barcode, -1)) # normalization parameter
    if normalize:
        barcode = barcode / max_lifetime # normalization
        max_lifetime = 1
    barcode = np.minimum(barcode, 1) # truncate inf

    return barcode


def bottleneck_distance(barcode_1: pd.DataFrame,
                        barcode_2: pd.DataFrame,
                        dim: int | None = None,
                        normalize = True
                        ) -> float:
    '''
    Calculates the bottleneck distance between two barcodes. The bottleneck distance is
    the maximum infinity norm difference between feature lifetimes in the minimal
    matching. I suggest either filtering the barcodes by dimension before using this
    function or passing a `dim` argument so it only compares features of the same dimension

    Points with infinite lifetimes are mapped to the highest finite value. If the two
    barcodes have a different number of features, unmatched points are mapped to the
    nearest diagnal

    This is essentially a wrapper for the Bottleneck distance function in 
    [Hera](https://bitbucket.org/grey_narn/hera/src/master/), it just formats everything
    first. That means you need to have the `hera` library installed and on the path (It's
    a bit of a process)

    Args:
        `barcode_1` (Dataframe): Dataframe with the information for the first barcode.
        Should have "dimension", "birth", and "death" columns
        `barcode_2` (Dataframe): Dataframe with the information for the second barcode.
        Should have "dimension", "birth" and "death" columns
        `dim` (int | None): The dimension to look at if an int. Doesn't filter dimensions
        at all otherwise
        `normalize` (bool): Whether to normalize lifetimes to [0, 1] or leave them as it.
        Default true

    Returns:
        `dist` (float): The bottleneck distance between the two barcodes
    '''
    try:
        import hera
    except:
        raise ImportError("Hera package not found. Make sure it's installed and findable on PATH")

    # formatting (filter dimension, normalize, tuncate infs, turn to np array)
    barcode_1 = _hera_format(barcode_1, dim, normalize)
    barcode_2 = _hera_format(barcode_2, dim, normalize)

    # calcualte
    dist = hera.bottleneck_dist(barcode_1, barcode_2)

    return dist


def wasserstein_distance(barcode_1: pd.DataFrame,
                         barcode_2: pd.DataFrame,
                         dim: int | None = None,
                         wasserstein_norm: float = 1.,
                         internal_norm: float = np.inf,
                         normalize = True
                         ) -> float:
    '''
    Calcualtes the Wasserstein Distance between two barcodes. The Wasserstein Distance
    is essentially the minimum length total distance between feature lifetimes in the
    optimal matching. I suggest either filtering the barcodes by dimension before using
    this function or passing a `dim` argument so it only compares features of the same
    dimension
    
    Points with infinite lifetimes are mapped to the highest finite value. If the two
    barcodes have a different number of features, unmatched points are mapped to the
    nearest diagnal

    This converges to the Bottleneck Distance when `wasserstein_norm` and `internal_norm`
    are both set to infinity (but don't do that, it doesn't like `wasserstein_norm` as
    infinity)

    This is essentially a wrapper for the Wasserstein distance function in 
    [Hera](https://bitbucket.org/grey_narn/hera/src/master/), it just formats everything
    first. That means you need to have the `hera` library installed and on the path (It's
    a bit of a process)

    Args:
        `barcode_1` (Dataframe): Dataframe with the information for the first barcode.
        Should have "dimension", "birth", and "death" columns
        `barcode_2` (Dataframe): Dataframe with the information for the second barcode.
        Should have "dimension", "birth" and "death" columns
        `dim` (int | None): The dimension to look at if an int. Doesn't filter dimensions
        at all otherwise
        `wasserstein_norm` (float): The norm to use when calculating the length of the
        vector of feature lifetime differences. Should be >= 1 and != inf. Default 1
        `internal_norm` (float): The norm to use when finding the difference between
        feature lifetimes. Should be >= 1. Default inf.
        `norm` (float): The norm to calculate distance using. Should be >= 1. Default 2
    
    Returns:
        `dist` (Float): The Wasserstein Distance between the two barcodes
    '''
    try:
        import hera
    except:
        raise ImportError("Hera package not found. Make sure it's installed and findable on PATH")

    # hera treats -1 as inf
    if internal_norm == np.inf:
        internal_norm = -1

    # formatting (filter dimension, normalize, tuncate infs, turn to np array)
    barcode_1 = _hera_format(barcode_1, dim, normalize)
    barcode_2 = _hera_format(barcode_2, dim, normalize)

    # options
    params = hera.WassersteinParams()
    params.internal_p = internal_norm # norm used to find distance between features
    params.wasserstein_power = wasserstein_norm # norm used to add feature distances

    # solve
    dist = hera.wasserstein_dist(barcode_1, barcode_2, params)
    
    return dist


def barcode_distance_matrix(barcodes: list[pd.DataFrame],
                            barcode_distance_func: callable,
                            dim: int | None = None,
                            labels: list[str] | None = None,
                            asymetric: bool = False,
                            calc_diagnal: bool = False,
                            **kwargs: dict[str: any]
                            ) -> pd.DataFrame:
    '''
    Creates a matrix of distances between barcodes based on an inputted distance function

    Args:
        `barcodes` (list[Dataframe]): A list of barcodes for find distances between
        `barcode_distance_func` (callable): A function that finds the distance between
        barcodes
        `dim` (int | None): The dimension to look at if an int. Doesn't filter dimension at
        all if None. Default None
        `labels` (list[str] | None): A list of labels for the distance matrix. If None, uses
        default pandas labeling (numbers 0-n)
        `asymetric` (bool): Whether to calcaulte the lower triangle of the distance matrix.
        If this is False, it assumes the distance matrix will be symetric (so the distance
        from barcode 1 to barcode 2 is that same as the distance from barcode 2 to barcode 1)
        and fills in the lower triangle with the values calculated in the upper triangle. If
        True, it calculates all entries, except those on the diagnal. Default False
        `calc_diagnal` (bool): Whether to calcuate entries along the diagnal. Otherwise,
        assumes it's 0. Generally, this should be False. Default False
        `**kwargs` (dict[str: any]): Options passed to the `barcode_distance_func` function
    
    Returns
        `dist_matrix` (pd.Dataframe): A matrix of distances between barcodes
    '''
    # filter dimension
    if dim is not None:
        for i, b in enumerate(barcodes):
            b = b[b['dimension'] == dim]
            barcodes[i] = b
    
    # setup distance
    n = len(barcodes)
    dist_matrix = pd.DataFrame(data=np.full((n, n), 0.), index=labels, columns=labels)

    # calculate distances
    for i in range(n):
        if asymetric:
            for j in range(i): # lower triangle
                dist_matrix.iloc[i, j] = barcode_distance_func(barcodes[i], barcodes[j], **kwargs)

        if calc_diagnal: # diagnal
            dist_matrix.iloc[i, i] = barcode_distance_func(barcodes[i], barcodes[i], **kwargs)

        for j in range(i+1, n): # upper triangle
            dist_matrix.iloc[i, j] = barcode_distance_func(barcodes[i], barcodes[j], **kwargs)

    # fill in lower triangle
    if not asymetric:
        dist_matrix += dist_matrix.T

    return dist_matrix
