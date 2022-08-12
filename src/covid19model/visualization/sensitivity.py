#############################
## Load necessary packages ##
#############################

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#####################################################
## Bar plot of first order and total Sobol indices ##
#####################################################

def plot_sobol_indices_bar(S1ST, ax, labels=None):

    # Bar plot
    ax = S1ST.plot(kind = "bar", y = ["S1", "ST"], yerr=S1ST[['S1_conf', 'ST_conf']].values.T, capsize=8,
                  color=['white','black'], alpha=0.4, edgecolor='black', ax=ax)
    # Legend
    ax.legend(bbox_to_anchor=(1, 1), fontsize=18)
    # Labels
    x_pos = np.arange(len(labels))
    plt.xticks(x_pos, labels, rotation=0, size=18)
    plt.yticks([0, 0.5, 1], [0, 0.5, 1], size=18)
    # Axes limit
    ax.set_ylim([0,1])
    # Grid
    ax.grid(False)

    return ax

###################
## Circular plot ##
###################

def normalize(x, xmin, xmax):
    return (x-xmin)/(xmax-xmin)

def plot_circles(ax, locs, names, max_s, stats, smax, smin, fc, ec, lw,
                 zorder):
    s = np.asarray([stats[name] for name in names])
    s = 0.01 + max_s * np.sqrt(normalize(s, smin, smax))

    fill = True
    for loc, name, si in zip(locs, names, s):
        if fc=='w':
            fill=False
        else:
            ec='none'

        x = np.cos(loc)
        y = np.sin(loc)

        circle = plt.Circle((x,y), radius=si, ec=ec, fc=fc, transform=ax.transData._b,
                            zorder=zorder, lw=lw, fill=True)
        ax.add_artist(circle)
        
from matplotlib.legend_handler import HandlerPatch
class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = plt.Circle(xy=center, radius=orig_handle.radius)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def legend(ax):
    some_identifiers = [plt.Circle((0,0), radius=5, color='k', fill=False, lw=1),
                        plt.Circle((0,0), radius=5, color='k', fill=True),
                        plt.Line2D([0,0.5], [0,0.5], lw=8, color='darkgray')]
    ax.legend(some_identifiers, ['ST', 'S1', 'S2'],
              loc=(1,1), borderaxespad=0.1, #mode='expand',
              handler_map={plt.Circle: HandlerCircle()}, fontsize=18)
    
def plot_sobol_indices_circular(S1ST, S2, labels=None):
    '''plot sobol indices on a radial plot

    Parameters
    ----------
    sobol_indices : dict
                    the return from SAlib
    criterion : {'ST', 'S1', 'S2', 'ST_conf', 'S1_conf', 'S2_conf'}, optional
    threshold : float
                only visualize variables with criterion larger than cutoff

    '''
    max_linewidth_s2 = 10
    max_s_radius = 0.2

    # Compute minimum and maximum ST
    smax = S1ST.max().max()
    smin = S1ST.min().min()
    
    # Compute minimum and maximum S2
    S2[S2<0.0]=0. #Set negative values to 0 (artifact from small sample sizes)
    s2max = S2.max().max()
    s2min = S2.min().min()
    
    # Names and ticks
    names = S1ST.index
    if not labels:
        labels=names
    n = len(names)
    ticklocs = np.linspace(0, 2*np.pi, n+1)
    locs = ticklocs[0:-1]
    
    # setup figure
    fig = plt.figure(figsize=(8.25, 11.75))
    ax = fig.add_subplot(111, polar=True)
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_xticks(locs)

    ax.set_xticklabels(labels, size=18)
    ax.set_yticklabels([])
    ax.set_ylim(top=1.4)
    legend(ax)

    # plot ST
    plot_circles(ax, locs, names, max_s_radius,
                 S1ST['ST'], smax, smin, 'w', 'k', 1, 9)

    # plot S1
    plot_circles(ax, locs, names, max_s_radius,
                 S1ST['S1'], smax, smin, 'k', 'k', 1, 10)

    # plot S2
    for name1, name2 in itertools.combinations(zip(names, locs), 2):
        name1, loc1 = name1
        name2, loc2 = name2

        weight = S2.loc[name1, name2]
        lw = 0.5+max_linewidth_s2*normalize(weight, s2min, s2max)
        ax.plot([loc1, loc2], [1,1], c='darkgray', lw=lw, zorder=1)

    return fig