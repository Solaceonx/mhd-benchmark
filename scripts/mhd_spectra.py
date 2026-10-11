"""Parseval-normalized spectra for periodic, unit-square, cell-centered MHD data.

Arrays are (y, x[, component]). Shells are sums, not averages over modes.
All modes (including DC and the square-grid corners) are kept in the tables.
"""
import numpy as np


def mode_grid(n):
    m = np.fft.fftfreq(n, d=1 / n)
    kx, ky = np.meshgrid(m, m)
    radius = np.hypot(kx, ky)
    return kx, ky, np.floor(radius).astype(int)


def transform(q):
    return np.fft.fft2(q, axes=(0, 1), norm="forward")


def spectrum(q, factor=1.0):
    """Return shell-integrated power; sum equals factor * mean(|q|²)."""
    _, _, shells = mode_grid(q.shape[0])
    power = abs(transform(q)) ** 2
    if q.ndim == 3:
        power = power.sum(axis=-1)
    return factor * np.bincount(shells.ravel(), weights=power.ravel())


def mean_square(q):
    a = abs(q) ** 2
    return float(np.mean(a.sum(axis=-1) if q.ndim == 3 else a))


def energies(state):
    rho, mom, b = (state[k] for k in ("density", "momentum", "magnetic"))
    kinetic = float(np.mean(np.sum(mom * mom, axis=-1) / (2 * rho)))
    magnetic = 0.5 * mean_square(b)
    total = float(state["energy"].mean())
    return dict(kinetic=kinetic, magnetic=magnetic,
                thermal=total - kinetic - magnetic, total=total)


def physical_fields(state):
    return {
        "density_fluctuation": (state["density"] - state["density"].mean(), 1.0),
        "kinetic_energy": (state["momentum"] / np.sqrt(state["density"])[..., None], 0.5),
        "magnetic_energy": (state["magnetic"], 0.5),
    }


def divergence(b, operator="spectral"):
    """Common collocated divergence; neither operator is CT's native constraint.

    Spectral derivative uses 2π*m and zero derivative of an even-grid Nyquist
    mode (the standard real-valued trigonometric interpolant convention).
    Centered derivative uses the symbol N*sin(2π*m/N), matching central FD.
    """
    n = b.shape[0]
    kx, ky, _ = mode_grid(n)
    if operator == "spectral":
        dx, dy = 2 * np.pi * kx, 2 * np.pi * ky
        if n % 2 == 0:
            dx[:, n // 2] = 0
            dy[n // 2, :] = 0
    elif operator == "centered":
        dx, dy = n * np.sin(2 * np.pi * kx / n), n * np.sin(2 * np.pi * ky / n)
        if n % 2 == 0:
            dx[:, n // 2] = 0
            dy[n // 2, :] = 0
    else:
        raise ValueError(operator)
    bh = transform(b)
    return np.fft.ifft2(1j * (dx * bh[..., 0] + dy * bh[..., 1]), norm="forward").real


def face_divergence(bx, by):
    """Finite-volume divergence of lower-face fields at cell centers."""
    n = bx.shape[0]
    return n * (np.roll(bx, -1, axis=1) - bx + np.roll(by, -1, axis=0) - by)


def error_norms(actual, reference):
    diff = actual - reference
    e = np.linalg.norm(diff, axis=-1) if diff.ndim == 3 else abs(diff)
    r = np.linalg.norm(reference, axis=-1) if reference.ndim == 3 else abs(reference)
    values = {}
    for key, a, b in (("L1", e.mean(), r.mean()),
                      ("L2", np.sqrt(np.mean(e * e)), np.sqrt(np.mean(r * r))),
                      ("Linf", e.max(), r.max())):
        values["absolute_" + key] = float(a)
        values["relative_" + key] = float(a / b) if b else float("nan")
    return values
