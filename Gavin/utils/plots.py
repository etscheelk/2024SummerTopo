'''
Functions to help make plots of homological features

Functions here are:
    - `plot_persistance_diagram`: Create a persisatnce diagram
    - `plot_barcode`: Create a barcode
    - `get_relevant_ts`: Get times where the number of homological features
    changes
    - `alive_at`: Get the number of cycles alive at a given time
    - `get_alive_path`: Get the number of cycles alive at all relevant times
    - `plot_num_open`: Plot the number of alive cycles over time

The plots are made on pyplot subplot objects. The easiest way to customize them
is to access the objects on the plot using `ax.get_lines()` or similar methods.
'''

## load some packages
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


def plot_persistance_diagram(homology: pd.DataFrame,
                             ax: mpl.axes.Axes,
                             min_filtration: float = 0.,
                             max_filtration: float = 1.,
                             inf_multiplier: float = 1.05,
                             margin_multiplier: float = 0.05,
                             num_ticks: int = 6,
                             colors = plt.get_cmap('tab10'),
                             markersize: float = 3,
                             ) -> None:
    '''
    Creates a persistance diagram

    Args:
        `homology` (pd.DataFrame): Pandas dataframe for the homology.
        Should have 'birth', 'dimension', and 'death' columns following
        OAT style.
        `ax`: (mpl.axes.Axes): Axis object to plot on
        `min_filtration` (float): The start of the persistance
        `max_filtration` (float): The end of the persistance
        `inf_multiplier` (float): The height of the infinity line
        `margin_multiplier`: The amount of padding to add along axises
        `num_ticks` (int): The number of ticks to plot on both axises
        exlcuing the infinity tick
        `colors` (Colormap): The colors to use. Should have a method
        colors(i) to get the ith color in the sequence
        `markersize` (float): The width of the points in the diagram

    Returns:
        None
    '''
    # config
    inf_val = min_filtration + inf_multiplier * (max_filtration - min_filtration)
    margin = margin_multiplier * (inf_val - min_filtration)

    # plot styles
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_aspect('equal')
    xbounds = (min_filtration - margin, max_filtration + margin)
    ybounds = (min_filtration - margin, inf_val + margin)
    ax.set_xlim(xbounds)
    ax.set_ylim(ybounds)
    ax.set_xticks(np.linspace(min_filtration, max_filtration, num_ticks))
    ax.set_yticks(np.hstack((np.linspace(min_filtration, max_filtration, num_ticks), inf_val)))
    yticks = ax.get_yticklabels()
    yticks[-1].set_text(r'$\infty$')
    ax.set_yticklabels(yticks)

    # helper lines
    ax.plot(xbounds, xbounds, 'k--')  # 45 degree line
    ax.axhline(inf_val, ls='--', c='k')

    # plot
    max_dim = homology['dimension'].max()
    for dim in range(max_dim+1):
        dim_homology = homology[homology['dimension'] == dim]
        ax.plot(
                dim_homology['birth'],
                dim_homology['death'].replace(np.inf, inf_val),
                '.',
                c=colors(dim),
                markersize=markersize,
                label=f'$H_{dim}$'
            )
        
    # final things
    ax.legend(loc='lower right')  # legend in bottom right below 45 degree line


def plot_barcode(homology: pd.DataFrame,
                 ax: mpl.axes.Axes,
                 min_filtration: float = 0.,
                 max_filtration: float = 1.,
                 num_ticks: int = 6,
                 margin_multiplier: float = 0.05,
                 colors = plt.get_cmap('tab10'),
                 linewidth: float = 1,
                 ) -> None:
    '''
    Creates a barcode diagram

    Args:
        `homology` (pd.DataFrame): Pandas dataframe for the homology.
        Should have 'birth', 'dimension', and 'death' columns following
        OAT style.
        `ax`: (mpl.axes.Axes): Axis object to plot on
        `min_filtration` (float): The start of the persistance
        `max_filtration` (float): The end of the persistance
        `num_ticks` (int): The number of ticks to plot on the x axis
        exlcuing the infinity tick
        `margin_multiplier`: The amount of padidng to add along the x axis
        `colors` (Colormap): The colors to use. Should have a method
        colors(i) to get the ith color in the sequence
        `linewidth` (float): The width of the lines in the barcode

    Returns:
        None
    '''
    # config
    inf_val = min_filtration + (1 + margin_multiplier) * (max_filtration - min_filtration)
    margin = margin_multiplier * (inf_val - min_filtration)
    homology = homology.sort_values(['dimension', 'birth', 'death']).reset_index(drop=True)  # get it in the right order
    homology = homology.replace(np.inf, inf_val)

    # plot styles
    ax.set_xlabel('T')
    xbounds = (min_filtration - margin, inf_val)
    ax.set_xlim(xbounds)
    ax.set_xticks(np.hstack((np.linspace(min_filtration, max_filtration, num_ticks), inf_val)))
    xticks = ax.get_xticklabels()
    xticks[-1].set_text(r'$\infty$')
    ax.set_xticklabels(xticks)

    # plot
    max_dim = homology['dimension'].max()
    for dim in range(max_dim+1):
        dim_homology = homology[homology['dimension'] == dim]
        x = np.column_stack((dim_homology[['birth', 'death']], np.full(len(dim_homology), None))).ravel()  # goes birth_1, death_1, None, birth_2, death_2, None,...
        y = np.column_stack((dim_homology.index, dim_homology.index, np.full(len(dim_homology), None))).ravel()  # goes 0, 0, None, 1, 1, None, 2, 2, None, ..
        ax.plot(
                x, y,
                c=colors(dim),
                linewidth=linewidth,
                label=f'$H_{dim}$',
            )
        
    # final things
    ax.axvline(max_filtration, ls='--', c='k')
    ax.legend(loc='upper left')  # legend in upper left since typically nothing is there


def get_relevant_ts(homology: pd.DataFrame,
                    min_filtration: float = 0.,
                    max_filtration: float = 1.,
                    ) -> np.ndarray:
    '''
    Gets the time periods relevant to the homology. This means we can do a much
    shorter loop to make the plot, since only time periods
        1. At the start
        2. At the end
        3. Where a cycle is born
        4. Where a cycle dies
    are relevant to the number of open holes

    Args:
        `homology` (pd.DataFrame): Pandas dataframe for the homology. Should have
        'birth' and 'death' columns following OAT style. Should already be filtered
        by dimension
        `min_filtration` (float): The start of the persistance
        `max_filtration` (float): The end of the persistance

    Returns:
        `ts` (np.ndarray): A numpy array with times to check
    '''
    ts = np.unique(
            np.hstack((min_filtration, max_filtration, np.ravel(homology[['birth', 'death']])))
        )
    ts = ts[ts <= max_filtration]  # remove inf
    return ts


def alive_at(t: float,  # time we care about
             homology: pd.DataFrame,  # homology to look at, should already be dimension filtered
             ) -> int:
    '''
    Counts the number of cycles in the homology dataframe alive at
    time t.

    Args:
        `t` (float): Time we want the count at
        `homology` (pd.DataFrame): Pandas dataframe for the homology.
        Should have 'birth' and 'death' columns following OAT style.
        Should already be filtered by dimension

    Returns:
        `count` (int): The number of cycles alive at that time
    '''
    born_before = homology['birth'] <= t  # nonstrict inqueality since born at t => alive at t
    dead_after = homology['death'] > t  # strict inequality since death at t => not alive at t
    alive_at = born_before & dead_after
    return alive_at.sum()


def get_alive_path(homology: pd.DataFrame,
                   ratio: bool = False,
                   min_filtration: float = 0.,
                   max_filtration: float = 1.,
                   ) -> tuple[np.ndarray, np.ndarray]:
    '''
    Gets the number of cycles alive at each relevant time period

    Args:
        `homology` (pd.Dataframe): Pandas dataframe for the homology.
        Should have 'birth' and 'death' columns following OAT style.
        Should already be filtered by dimension
        `ratio` (bool): Whether you want the number of cycles alive
        (false) or the percent of cycles alive (true). Default False
        `min_filtration` (float): The start of the persistance
        `max_filtration` (float): The end of the persistance
    
    Returns:
        `ts` (np.ndarray): The relevant time periods to look at
        `alive_counts` (np.ndarray): The number of cycles alive at each
        of the time periods
    '''
    # get time periods to check
    ts = get_relevant_ts(homology, min_filtration, max_filtration)

    # count number alive at each t
    alive_counts = np.empty_like(ts, dtype=float)
    for i, t in enumerate(ts):
        alive_counts[i] = alive_at(t, homology)

    # normalize (if ratio)
    if ratio:
        alive_counts /= len(homology)
    
    return ts, alive_counts


def plot_betti_curve(homology: pd.DataFrame,
                     ax: mpl.axes.Axes,
                     ratio: bool = False,
                     min_filtration: float = 0.,
                     max_filtration: float = 1.,
                     colors = plt.get_cmap('tab10'),
                     ) -> None:
    '''
    Creates a plot of the number of cycles alive at each time period

    Args:
        `homology` (pd.DataFrame): Pandas dataframe for the homology.
        Should have 'birth', 'dimension', and 'death' columns following
        OAT style.
        `ax`: (mpl.axes.Axes): Axis object to plot on
        `ratio` (bool): Whether you want the number of cycles alive
        (false) or the percent of cycles alive (true). Default False
        `min_filtration` (float): The start of the persistance
        `max_filtration` (float): The end of the persistance
        `colors` (Colormap): The colors to use. Should have a method
        colors(i) to get the ith color in the sequence

    Returns:
        None
    '''
    # plot formatting
    ax.set_xlabel('T')
    if ratio:
        ax.set_ylabel('% of Cycles Alive')
    else:
        ax.set_ylabel('# of Cycles Alive')

    # plot
    max_dim = homology['dimension'].max()
    for dim in range(max_dim+1):
        dim_homology = homology[homology['dimension'] == dim]
        ts, alive_counts = get_alive_path(dim_homology, ratio, min_filtration, max_filtration)
        ax.step(
                ts,
                alive_counts,
                color=colors(dim),
                label=f'$H_{dim}$'
            )
              
    # final things
    ax.legend(loc='upper left')  # legend in upper left since typically nothing is there
