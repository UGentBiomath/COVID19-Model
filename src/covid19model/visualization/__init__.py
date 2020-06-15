from .output import population_status,infected
from .utils import colorscale_okabe_ito
import matplotlib.pyplot as plt

__all__ = ["population_status", "infected"]

# covid 19 specific parameters
plt.rcParams.update({
    "axes.prop_cycle": plt.cycler('color',
                                  list(colorscale_okabe_ito.values())),
    "font.size": 15,
    "lines.linewidth" : 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "ytick.major.left": True,
    "axes.grid": True
})