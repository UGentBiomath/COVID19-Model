import matplotlib.pyplot as plt

# From Color Universal Design (CUD): https://jfly.uni-koeln.de/color/
colorscale_okabe_ito = {"orange" : "#E69F00", "light_blue" : "#56B4E9",
                        "green" : "#009E73", "yellow" : "#F0E442",
                        "blue" : "#0072B2", "red" : "#D55E00",
                        "pink" : "#CC79A7", "black" : "#000000"}

def _apply_tick_locator(ax):
    """support function to apply default ticklocator settings"""
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    return ax
