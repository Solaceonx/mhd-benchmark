"""Shared Matplotlib typography for every project figure and movie."""
import matplotlib


PLOT_STYLE = {
    "mathtext.fontset": "cm",
    "font.family": "STIXGeneral",
    "font.size": 14,
}


def apply_plot_style():
    matplotlib.rcParams.update(PLOT_STYLE)


# ---------------------------------------------------------------- colour maps of images and movies
# One map for each kind of quantity, the same on every page, so that a density image looks like a density image
# wherever it is. All run from dark (low, or zero for a signed quantity) to light: seaborn's mako and maps of cmasher.
FIELD_MAPS = {
    "density": "mako",            # gas: density, pressure, energy
    "magnetic": "cmr.freeze",     # field strength, B^2
    "signed": "cmr.iceburn",      # psi, vorticity, a difference: blue negative, black zero, red positive
    "current": "cmr.wildfire",    # a second signed quantity next to the first: green negative, black zero, orange positive
}
ARROW_ON_MAGNETIC = "#ffcf8a"     # field-direction arrows on a "magnetic" image: warm on the blue map

try:        # importing them registers the names above with Matplotlib
    import seaborn  # noqa: F401
    import cmasher  # noqa: F401
    HAVE_MAPS = True
except ImportError:
    HAVE_MAPS = False


def field_map(kind):
    """Name of the colour map for a kind of quantity (a key of FIELD_MAPS)."""
    if not HAVE_MAPS:
        raise RuntimeError("the standard colour maps need seaborn and cmasher in this Python: pip install seaborn cmasher")
    return FIELD_MAPS[kind]


# The movies of the test pages (progress/tests.html): one set of maps per family of tests, so that a page is
# recognisable at a glance. Within a page a quantity keeps its map. Sequential maps run dark to light; the signed ones
# are black at zero. Families are those of progress/gpu_tests.json.
FAMILY_MAPS = {   # family: (density, magnetic, signed)
    "Exact waves": ("cmr.ocean", "cmr.arctic", "cmr.iceburn"),
    "Shock tubes": ("cmr.sapphire", "cmr.freeze", "cmr.iceburn"),
    "Vortices and sheets": ("mako", "cmr.gem", "cmr.redshift"),        # gem keeps colour at its bright end, where most of |B| sits on a log scale
    "Reconnection": ("cmr.amethyst", "cmr.cosmic", "cmr.seaweed"),
    "Kelvin–Helmholtz": ("cmr.rainforest", "cmr.emerald", "cmr.watermelon"),
    "Field loops": ("cmr.gothic", "cmr.ghostlight", "cmr.wildfire"),
    "Blast waves": ("cmr.ember", "cmr.sunburst", "cmr.redshift"),
    "Turbulence": ("cmr.torch", "cmr.amber", "cmr.wildfire"),
}


def family_map(family, kind):
    """Colour map of a kind of quantity ("density", "magnetic", "signed") on the test pages; the notebook's own map for a family not listed."""
    if not HAVE_MAPS:
        raise RuntimeError("the colour maps need seaborn and cmasher in this Python: pip install seaborn cmasher")
    maps = FAMILY_MAPS.get(family)
    return maps[("density", "magnetic", "signed").index(kind)] if maps else FIELD_MAPS[kind]


# ---------------------------------------------------------------- movies and their stills
# The same look as the figures of the pages: white ground, the notebook's inks, its fonts, its colour maps.
INK, MUTED, RULE, PANEL = "#202a2a", "#64716f", "#d9dedd", "#f3f5f4"       # as scripts/paper_common.py
# Hue = the code, on every page (checked for every pair under normal, protan and deutan vision: OKLab distance >= 15
# and >= 8). AREPO's ochre was chosen the same way against the five others (18.7; 10.9 protan, 9.1 deutan).
FAMILY = dict(ct="#eb6834", glm="#199e70", athenak="#2a78d6", ramses="#4a3aa7", athenapp="#a72a68", arepo="#7b5e1c")
RUN = FAMILY["glm"]        # the green of Tom Abel's kernels: a run made with them
REFERENCE = "#52514e"      # the grey of reference runs: the branch or code another is compared with
COLUMN = {"glm": FAMILY["glm"], "ct": FAMILY["ct"], "truth": REFERENCE}     # the three columns of the first pages (progress/style.css)


def apply_movie_style(size=11):
    """Style for a movie frame or a stand-alone still: call before making the figure."""
    apply_plot_style()
    matplotlib.rcParams.update({
        "font.size": size, "axes.titlesize": size + 1, "axes.labelsize": size, "legend.fontsize": size - 1,
        "xtick.labelsize": size - 1, "ytick.labelsize": size - 1,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": MUTED, "axes.linewidth": .8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": RULE, "grid.linewidth": .6, "legend.frameon": False,
    })


def frame(ax):
    """All four spines for an image panel (line plots keep left and bottom only)."""
    for side in ("top", "right"):ax.spines[side].set_visible(True)


def ffmpeg_path():
    """The ffmpeg that encodes the movies: the one imageio-ffmpeg ships (here, or in the project's .venv-movie), or one on PATH."""
    import glob
    import shutil
    from pathlib import Path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    found = shutil.which("ffmpeg") or next(iter(sorted(glob.glob(str(
        Path(__file__).resolve().parents[1]/".venv-movie"/"lib"/"python*"/"site-packages"/"imageio_ffmpeg"/"binaries"/"ffmpeg*")))), None)
    if not found:raise RuntimeError("no ffmpeg: pip install imageio-ffmpeg (scripts/requirements-movie.txt)")
    return found
