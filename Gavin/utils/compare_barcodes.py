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
 - `wasserstein_distance`: An extremely slow algrothim for calculating
 the Wasserstein Distance between barcodes. I strongly suggest not
 using it and using `approx_wasserstein_distance` instead
 - `approx_wasserstein_distance`: A huristic Wasserstein Distance that's
 much faster to calculate
'''

# preliminaries
from itertools import permutations
import oatpy as oat
import pandas as pd
import numpy as np
import warnings


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
    # x, y coordinate of every entry i, j in the persistance image is x[i, j], y[i, j]
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    x, y = np.meshgrid(x, y)

    # apply gaussians and get images
    var = 2 * sigma**2 # actualy 2*variance but this is what we use so
    persistance_images = []
    for d in dim: # create a different persisance image for each dimension
        d_persistance_diagram = persistance_diagram[persistance_diagram['dimension'] == d] # peristance diagram fo dimension
        gaussians = [w * np.exp(-((x-b)**2 + (y-l)**2)/var) for w, b, l in zip(d_persistance_diagram['weight'], d_persistance_diagram['birth'], d_persistance_diagram['lifetime'])]
        # use gaussian formula weighting by the calculated weight and using the point location for all x and y in the matrix
        persistance_image = np.full((res, res), 0) + sum(gaussians) # full makes sure we have all 0s if dimension has no features
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


def wasserstein_distance(barcode_1: pd.DataFrame,
                         barcode_2: pd.DataFrame,
                         dim: int | None = None,
                         norm: float = 2.
                         ) -> float:
    '''
    **NOTE: The use of this function is STRONGLY discouraged due to time requrements**

    Calcualtes the Wasserstein Distance between two barcodes. I suggest filtering the
    barcodes by dimension before using this function so it only compares features of
    the same dimension. If the lengths are unequal, unmatched points are mapped to the
    nearest diagnal

    This is really slow (O(n!) time), and not reccemended for large barcodes

    Args:
        `barcode_1` (Dataframe): Dataframe with the information for the first barcode.
        Should have "birth" and "death" columns
        `barcode_2` (Dataframe): Dataframe with the information for the second barcode.
        Should have "birth" and "death" columns
        `dim` (int | None): The dimension to look at if an int. Doesn't filter dimensions
        at all otherwise
        `norm` (float): The norm to calculate distance using. Should be >= 1. Default 2
    
    Returns:
        `dist` (Float): The Wasserstein Distance between the two barcodes
    '''
    barcode_1, barcode_2 = barcode_1.copy(), barcode_2.copy() # don't change the dataframes

    # filter dimension
    if dim is not None:
        barcode_1 = barcode_1[barcode_1['dimension'] == dim]
        barcode_2 = barcode_2[barcode_2['dimension'] == dim]

    # setup
    max_lifetime_1 = oat.barcode.max_finite_value(barcode_1['death']) # remove infs 
    barcode_1.loc[barcode_1['death'] == np.inf, 'death'] = max_lifetime_1
    max_lifetime_2 = oat.barcode.max_finite_value(barcode_2['death'])
    barcode_2.loc[barcode_2['death'] == np.inf, 'death'] = max_lifetime_2
    barcode_1 = barcode_1.reset_index(drop=True) # need to be able to reorder
    barcode_2 = barcode_2.reset_index(drop=True)
    if len(barcode_2) > len(barcode_1): # we swap the ordering for barcode 1, which needs to be the longer one
        barcode_1, barcode_2 = barcode_2, barcode_1
    n = len(barcode_1)

    # distance function between two features
    if norm == np.inf:
        d = lambda x1, y1, x2, y2: np.maximum(np.abs(x1-x2), np.abs(y1-y2))
    else:
        d = lambda x1, y1, x2, y2: ((np.abs(x1-x2)**norm + np.abs(y1-y2))**norm)**(1/norm)

    # distance function for one ordering
    def distance(bc1, bc2):
        if len(bc1) != len(bc2): # fill in rows with nearest diagnal
            extra_rows = bc1[len(bc2):].copy() # rows that need to be filled in
            extra_rows['birth'] = extra_rows['death'] = (extra_rows['birth']+extra_rows['death']) / 2 # point along diagnal close to point
            bc2 = pd.concat((bc2, extra_rows))
        
        # get distance
        dist_arr = d(bc1['birth'], bc1['death'], bc2['birth'], bc2['death']) # dist between features
        dist = sum(dist_arr) # dist between barcodes

        return dist
    
    # calculate the distance
    dist = np.inf
    for order in permutations(np.arange(n)): # for each permuation of the longer barcode
        # reorder barcode_1 based on the permuations
        bc1_reordered = barcode_1.copy()
        bc1_reordered['order'] = order
        bc1_reordered = bc1_reordered.set_index('order')
        bc1_reordered = bc1_reordered.sort_index()

        # get the distance
        order_dist = distance(bc1_reordered, barcode_2) # distance between reordered barcode 1 and barcode 2
        dist = min(dist, order_dist)

    return dist


def approx_wasserstein_distance(barcode_1: pd.DataFrame,
                                barcode_2: pd.DataFrame,
                                dim: int | None = None,
                                norm: float = 2.,
                                max_iter: int = 10,
                                return_iteration_count: bool = False,
                                supress_warnings: bool = False
                                ) -> float | tuple[float, int]:
    '''
    Approximates the Wasserstein Distance between two barcodes. I suggest filtering the
    barcodes by dimension before using this function so it only compares features of
    the same dimension. If the lengths are unequal, the shortest lifetime features are
    matched with the diagnal (at the start)

    This works by sorting both arrays, by birth/death time, then matching them up. It
    then calcultates the distance between the current order and orders that are swapped
    by 1 row. Repeats this process until there are no distance decreasing row swaps.

    Args:
        `barcode_1` (Dataframe): Dataframe with the information for the first barcode.
        Should have "birth" and "death" columns
        `barcode_2` (Dataframe): Dataframe with the information for the second barcode.
        Should have "birth" and "death" columns
        `dim` (int | None): The dimension to look at if an int. Doesn't filter dimensions
        at all otherwise
        `norm` (float): The norm to calculate distance using. Should be >= 1. Default 2
        `max_iter` (int): The maximum number of iterations to use in the row swap. Default
        10
        `return_iteration_count` (bool): Returns the number of row swapping iterations if
        True. Otherwise, just returns the distance. Default False
        `supress_warnings` (bool): Warns user if maximum iteration count exceeded if False.
        Skips this if True. Default False
    
    Returns:
        `dist` (Float): The estimated Wasserstein Distance between the two barcodes
        `num_iter` (int): Only returned if `return_iteration_count` is True. The number of
        row swapping iterations we go through
    '''
    barcode_1, barcode_2 = barcode_1.copy(), barcode_2.copy() # don't change the dataframes

    # filter dimension
    if dim is not None:
        barcode_1 = barcode_1[barcode_1['dimension'] == dim]
        barcode_2 = barcode_2[barcode_2['dimension'] == dim]

    # setup
    max_lifetime_1 = oat.barcode.max_finite_value(barcode_1['death']) # remove infs 
    barcode_1.loc[barcode_1['death'] == np.inf, 'death'] = max_lifetime_1
    max_lifetime_2 = oat.barcode.max_finite_value(barcode_2['death'])
    barcode_2.loc[barcode_2['death'] == np.inf, 'death'] = max_lifetime_2
    barcode_1 = barcode_1.sort_values(['birth', 'death']).reset_index(drop=True) # sort (to line everything up)
    barcode_2 = barcode_2.sort_values(['birth', 'death']).reset_index(drop=True)
    if len(barcode_2) > len(barcode_1): # we swap the ordering for barcode 1, which needs to be the longer one
        barcode_1, barcode_2 = barcode_2, barcode_1
    n = len(barcode_1)
    barcode_2['added'] = 0 # indicator if row is added

    # fill in entries (if the lengths are unequal)
    if len(barcode_1) != len(barcode_2):
        # figure out how to fill in rows
        num_missing = len(barcode_1) - len(barcode_2) # number of values to add to bc2
        lifetimes = barcode_1['death'] - barcode_2['birth'] # lifetime
        lifetimes = lifetimes.sort_values()
        shortest_lifetimes = lifetimes[:num_missing].sort_index().index

        # create the dataframe to fill with
        concat = []
        last_i = 0
        row_to_add = barcode_2[:1].copy() # row we add where missing values are
        row_to_add['added'] = 1
        row_to_add['birth'] = row_to_add['death'] = 0
        for i in shortest_lifetimes: # for each row we add
            concat += [barcode_2[last_i: i], row_to_add]
            last_i = i
        concat += [barcode_2[i:]]
        barcode_2 = pd.concat(concat).reset_index(drop=True)

    # distance function between two barcodes
    if norm == np.inf:
        d = lambda x1, y1, x2, y2, a: (1-a) * np.maximum(np.abs(x1-x2), np.abs(y1-y2)) + a * np.abs(x1-y1)/2
    else:
        d = lambda x1, y1, x2, y2, a: (1-a) * ((np.abs(x1-x2)**norm + np.abs(y1-y2))**norm)**(1/norm) + a * 2**(1/norm) * np.abs(x1-y1)/2

    # initial distance
    dist = sum(d(barcode_1['birth'], barcode_1['death'], barcode_2['birth'], barcode_2['death'], barcode_2['added']))

    # row swaps
    num_iter = 0 # check that we're under the max number of iterations
    swapped_row = True # indicator if a row has been swapped this iteration
    while swapped_row and num_iter < max_iter:
        swapped_row = False
        num_iter += 1
        for i in range(1, n):
            swapped_bc1 = barcode_1.copy() # swapped barcode 1
            swapped_bc1.loc[i-1, :], swapped_bc1.loc[i, :] = swapped_bc1.loc[i, :], swapped_bc1.loc[i-1, :] # swap rows
            swapped_dist = sum(d(swapped_bc1['birth'], swapped_bc1['death'], barcode_2['birth'], barcode_2['death'], barcode_2['added']))

            # if better than current, swap rows perminantly
            if swapped_dist < dist:
                dist = swapped_dist
                barcode_1 = swapped_bc1
                swapped_row = True

    # tell user if maximum iteration count exceeded
    if num_iter == max_iter and not supress_warnings:
        warnings.warn('Maximum Iteration count exceeded. Returning current value.')
    
    # return iteration count
    if return_iteration_count:
        return dist, num_iter
    
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
