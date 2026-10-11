"""Shared pieces of the paper figures (scripts/paper_figures.py): identity of each method, axes, images, the output list."""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_data  # noqa: E402
from plot_style import ARROW_ON_MAGNETIC, FAMILY, apply_plot_style  # noqa: E402

SITE = sweep_data.SITE
OUT = SITE/"media"/"papers"
INK, MUTED, RULE, PANEL = "#202a2a", "#64716f", "#d9dedd", "#f3f5f4"
REFERENCE_INK = "#52514e"

# Hue = the code (FAMILY, from scripts/plot_style.py); line style and marker = the variant within a code, as on the
# other pages of the site.
STYLE = {   # method: (family, label, short label, line style, marker)
    "ct": ("ct", "CT prototype", "CT", "-", "o"),
    "glm": ("glm", "GLM tiled LLF (original path)", "GLM LLF", (0, (1, 1)), "x"),
    "glm_hll16": ("glm", "GLM f16/HLL (production)", "GLM f16/HLL", "-", "o"),
    "glm_hlld": ("glm", "GLM HLLD", "GLM HLLD", "--", "s"),
    "glm_ppm": ("glm", "GLM PPM", "GLM PPM", "-.", "D"),
    "glm_constch": ("glm", "GLM constant $c_h$", "GLM const. $c_h$", (0, (6, 2, 1, 2, 1, 2)), "v"),
    # the three matched set-ups share one dash pattern at different offsets, so coincident curves show all their colours
    "athenak_matched": ("athenak", "AthenaK PLM/HLLE/RK2", "AthenaK PLM/HLLE", (0, (4.5, 4.5)), "s"),
    "athenak_best": ("athenak", "AthenaK PPM/HLLD/RK3", "AthenaK PPM/HLLD", "-", "o"),
    "ramses_matched": ("ramses", "RAMSES HLL", "RAMSES HLL", (3, (4.5, 4.5)), "s"),
    "ramses_best": ("ramses", "RAMSES HLLD", "RAMSES HLLD", "-", "o"),
    "athenapp_matched": ("athenapp", "Athena++ PLM/HLLE/VL2", "Athena++ PLM/HLLE", (4.5, (4.5, 4.5)), "s"),
    "athenapp_best": ("athenapp", "Athena++ PPM/HLLD/RK3", "Athena++ PPM/HLLD", "-", "o"),
    "arepo_moving": ("arepo", "AREPO moving mesh, Powell/HLLD", "AREPO moving mesh", "-", "o"),
    "arepo_static": ("arepo", "AREPO static mesh, Powell/HLLD", "AREPO static mesh", (1.5, (4.5, 4.5)), "s"),
    "athenak_reference": ("reference", "AthenaK reference", "Reference", "-", ""),
    "athenapp_reference": ("reference", "Athena++ check", "Check", (0, (4, 2)), ""),
}
GPU = ("ct", "glm", "glm_hll16", "glm_hlld", "glm_ppm", "glm_constch")
OTHER = ("athenak_matched", "athenak_best", "ramses_matched", "ramses_best", "athenapp_matched", "athenapp_best")
GROUPS = (("gpu", "GLM kernels and CT prototype (T. Abel), H100", GPU), ("other", "AthenaK, RAMSES and Athena++", OTHER))
ALL = GPU + OTHER
AREPO = ("arepo_moving", "arepo_static")      # run after the sweep on the planar tests (sherlock_arepo/); not part of ALL


def color(method):
    fam = STYLE[method][0];return REFERENCE_INK if fam == "reference" else FAMILY[fam]


def label(method, short=False):
    return STYLE[method][2 if short else 1]


def plain(text):
    """A label for HTML: without the mathtext of the figure labels."""
    return text.replace("$c_h$", "cₕ")


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x, digits=1):
    """A number for running text: 4.3×10⁻³ (plain decimals between 0.01 and 1000)."""
    if x == 0 or not np.isfinite(x):return "0" if x == 0 else "–"
    if 1e-2 <= abs(x) < 1e3:return f"{x:.{max(digits, 2)}g}" if abs(x) < 1 else f"{x:.3g}"
    e = int(np.floor(np.log10(abs(x))));m = x/10.0**e
    if round(m, digits) >= 10:m /= 10;e += 1
    return f"{m:.{digits}f}×10{str(e).translate(SUPERSCRIPT)}"


def spread(values, fmt=sci):
    """'a' or 'a to b' for the range of a list of numbers ('' when empty)."""
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:return ""
    lo, hi = fmt(min(v)), fmt(max(v))
    return lo if lo == hi else f"{lo} to {hi}"


def two(x):
    return f"{x:.2f}"


def by_family(values):
    """Split {method: value} into the lists the findings speak of: ct, glm, athena (AthenaK and Athena++), ramses, other (the six set-ups)."""
    out = dict(ct=[], glm=[], athena=[], ramses=[], other=[])
    for m, v in values.items():
        fam = STYLE[m][0]
        if fam == "ct":out["ct"].append(v)
        elif fam == "glm":out["glm"].append(v)
        elif fam in ("athenak", "athenapp"):out["athena"].append(v);out["other"].append(v)
        elif fam == "ramses":out["ramses"].append(v);out["other"].append(v)
    return out


def line_kw(method, **kw):
    out = dict(color=color(method), linestyle=STYLE[method][3], lw=1.5);out.update(kw);return out


def handles(methods, markers=False):
    return [Line2D([], [], label=label(m), marker=STYLE[m][4] if markers else "", ms=4, **line_kw(m)) for m in methods]


def style_axes(ax, grid=True):
    for side in ("top", "right"):ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):ax.spines[side].set_color(MUTED);ax.spines[side].set_linewidth(.8)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=3, width=.8)
    if grid:ax.grid(True, color=RULE, lw=.6);ax.set_axisbelow(True)


def blank(ax, text):
    """A panel with no data: say why."""
    ax.set_facecolor(PANEL);ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)
    ax.text(.5, .5, text, ha="center", va="center", transform=ax.transAxes, color=MUTED, fontsize=7.5, wrap=True)


def image_axes(ax):
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_color(MUTED);s.set_linewidth(.6)


class Figures:
    def __init__(self, out=OUT):
        self.sw = sweep_data.Sweep();self.out = Path(out);self.out.mkdir(parents=True, exist_ok=True)
        self.entries = [];self.notes = []
        apply_plot_style();matplotlib.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "axes.labelcolor": INK,
                                                       "axes.titlecolor": INK, "text.color": INK, "legend.fontsize": 8, "xtick.labelsize": 8,
                                                       "ytick.labelsize": 8, "figure.dpi": 100, "savefig.dpi": 200, "legend.frameon": False})

    # ---- why a part is not drawn ----
    def why(self, test, method, n, seed=42, frame=None):
        sw = self.sw;state = sw.state_of(test, method, n, seed)
        if state is None:return "not in the plan"
        if state == "skipped":return "taken out\nof the sweep"
        rec = sw.recorded(test, method, n, seed)
        if state in ("stopped", "redo") and rec is not None:
            return f"solver stopped\nat t = {float(rec.get('reached') or 0):.3g}"
        if state == "open":return "not finished\non Sherlock"
        bad = sw.went_bad(test, method, n, seed)
        if bad is not None:return f"state not finite\nafter t = {bad:.3g}"
        return "not fetched"

    def miss(self, test, method, n, seed=42):
        self.notes.append(f"{label(method, True)} at N = {n}: {self.why(test, method, n, seed).replace(chr(10), ' ')}")

    def series(self, test, method, n, column, seed=42):
        """(time, values) of a diagnostics column, or None."""
        d = self.sw.diag(test, method, n, seed)
        if d is None or column not in d:return None
        k = self.sw.finite_rows(test, method, n, seed)          # outputs after the state stopped being finite are dropped
        return d["time"][:k], d[column][:k]

    def stopped(self, test, method, n, seed=42):
        """The run ended early: its solver stopped, or its state stopped being finite."""
        return self.sw.state_of(test, method, n, seed) in ("stopped", "redo") or self.sw.went_bad(test, method, n, seed) is not None

    def save(self, fig, ident, **entry):
        path = self.out/f"{ident}.png";fig.savefig(path, bbox_inches="tight", facecolor="white");plt.close(fig)
        notes = sorted(set(self.notes));self.notes = []
        self.entries.append(dict(id=ident, file=f"media/papers/{ident}.png", missing=notes, **entry))
        print(f"wrote {path.name}" + (f"  ({len(notes)} parts missing)" if notes else ""), flush=True)


# ---------------------------------------------------------------- shared pieces
def bmag(u):
    return np.sqrt((u[..., 5:8].astype(np.float64)**2).sum(-1))


def pressure(u, gamma):
    rho = u[..., 0].astype(np.float64)
    ke = (u[..., 1:4].astype(np.float64)**2).sum(-1)/(2*rho);me = (u[..., 5:8].astype(np.float64)**2).sum(-1)/2
    return (gamma - 1)*(u[..., 4] - ke - me)


def potential(u):
    """A_z at cell centres of a planar state (Bx = dA/dy, By = -dA/dx), mean field included."""
    n = u.shape[0];bx = u[..., 5].astype(np.float64);by = u[..., 6].astype(np.float64)
    k = 2*np.pi*np.fft.fftfreq(n, 1.0/n);kx, ky = np.meshgrid(k, k, indexing="ij");k2 = kx**2 + ky**2;k2[0, 0] = 1.0
    ah = -(1j*ky*np.fft.fft2(bx - bx.mean()) - 1j*kx*np.fft.fft2(by - by.mean()))/k2;ah[0, 0] = 0.0
    x = (np.arange(n) + .5)/n;X, Y = np.meshgrid(x, x, indexing="ij")
    return np.fft.ifft2(ah).real + bx.mean()*Y - by.mean()*X


def show(ax, a, **kw):
    """Image of a field indexed [ix, iy] on the unit square."""
    return ax.imshow(np.asarray(a).T, origin="lower", extent=(0, 1, 0, 1), interpolation="nearest", **kw)


def arrows(ax, u, every=None, **kw):
    """Field direction (unit vectors) on a coarse grid."""
    n = u.shape[0];step = every or max(1, n//14);s = slice(step//2, None, step)
    bx = u[s, s, 5].astype(np.float64);by = u[s, s, 6].astype(np.float64);m = np.hypot(bx, by);m[m == 0] = 1
    x = (np.arange(n) + .5)[s]/n;X, Y = np.meshgrid(x, x, indexing="ij")
    ax.quiver(X, Y, bx/m, by/m, color=kw.pop("color", ARROW_ON_MAGNETIC), scale=22, width=.006, headwidth=3.2, headlength=3.6, pivot="mid", **kw)


def nice_limit(x):
    """Round up to 1, 2 or 5 times a power of ten."""
    if not np.isfinite(x) or x <= 0:return 1.0
    e = 10.0**np.floor(np.log10(x));m = x/e
    return (1 if m <= 1 else 2 if m <= 2 else 5 if m <= 5 else 10)*e


def shades(method, count):
    """`count` steps of one hue, light to dark (resolution is a magnitude)."""
    base = np.array(matplotlib.colors.to_rgb(color(method)));white = np.ones(3);dark = base*.55
    if count == 1:return [tuple(base)]
    out = []
    for i in range(count):
        t = i/(count - 1)
        out.append(tuple(white*(1 - (.45 + .55*min(1, 2*t))) + base*(.45 + .55*min(1, 2*t))) if t <= .5 else tuple(base*(1 - (2*t - 1)) + dark*(2*t - 1)))
    return out
