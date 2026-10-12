"""The set-up of every test problem as typeset equations, for the notes page and the test pages.

  python3 scripts/build_notes_setups.py

SETUPS below holds, per test, lines of text with LaTeX between dollar signs, taken from sherlock_sweep/src/problems.jl
and the sweep's configs. Each formula is drawn once by Matplotlib's mathtext as a small SVG (progress/media/setups/, no
web fonts and no script from elsewhere, which matters on a page that holds a key), and the lines go
 - into progress/notes.html (between the SETUPS markers), where a test's page shows them under the note, with the
   LaTeX source to copy;
 - into progress/setups.json, which build_test_pages.py puts on the public test pages.
The formulas use only what mathtext knows (no \\text, no environments): text stays outside the dollar signs.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import mathtext  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/"progress"
OUT = SITE/"media"/"setups"
SIZE = 12.0            # points: the size of the page's text
INK = "#202a2a"
BOX = "Periodic unit box"
FLOW_KH = [r"$\rho = 1,\quad p = 10$",
           r"$v_x = \tanh\frac{y - 1/4}{a} - \tanh\frac{y - 3/4}{a} - 1,\quad a = 0.05$",
           r"$v_y = 0.01\,\sin 2\pi x\,\left[e^{-(y-1/4)^2/\sigma^2} + e^{-(y-3/4)^2/\sigma^2}\right],\quad \sigma = 0.2$"]
KH_BAND = r"$B_x = b\,\sum_{s}\mathrm{sech}^2\frac{y - 1/4 + s}{w},\quad b = 0.01,\quad w = 0.0945$ (the sum is over the periodic images): field only around the lower layer."
OT_STATE = [r"$\rho = \frac{25}{36\pi},\quad p = \frac{5}{12\pi}$",
            r"$\mathbf{B} = B_0\,(-\sin 2\pi y,\ \sin 4\pi x,\ 0),\quad B_0 = 1/\sqrt{4\pi}$",
            r"set from $A_z = B_0\left[\frac{\cos 2\pi y}{2\pi} + \frac{\cos 4\pi x}{4\pi}\right]$"]
LOOP = r"$|\mathbf{B}| = a_0$ inside the loop and zero outside; $\beta = 2p/a_0^2 = 2\times10^{6}$, so the field is a passive tracer."
TURB = [r"each mode an Ornstein–Uhlenbeck process with memory time $T = L/(2\,\mathcal{M}\,c_s)$; the field is rescaled to $a_{\mathrm{rms}}$ at every forcing step, with no ramp.",
        r"Power taken up: $\varepsilon = d\langle E\rangle/dt$. There is no cooling, so in a steady state it is the dissipation.",
        r"$\mathrm{Re}_{\mathrm{eff}} = \frac{v\,L}{\nu},\quad \nu = \frac{\varepsilon}{\langle\rho\,|\nabla\times\mathbf{v}|^2\rangle + \frac{4}{3}\langle\rho\,(\nabla\cdot\mathbf{v})^2\rangle}$, with $v$ the mass-weighted rms velocity",
        r"$b = \sqrt{e^{\sigma_s^2} - 1}\,/\,\mathcal{M},\quad s = \ln(\rho/\bar{\rho})$"]


def blast(dim, b):
    shape = "a disc" if dim == 2 else "a ball"
    field = {0: r"$\mathbf{B} = 0$", 50: r"$B_x = 50/\sqrt{4\pi} = 14.1$: outside, $\beta = 2p/B_x^2 = 1.0\times10^{-3}$",
             100: r"$B_x = 100/\sqrt{4\pi} = 28.2$: outside, $\beta = 2p/B_x^2 = 2.5\times10^{-4}$"}[b]
    return [f"{BOX}{'' if dim == 2 else ' (a cube)'}, " + r"$\gamma = 7/5$, run to $t = 0.01$.",
            r"$\rho = 1,\quad \mathbf{v} = 0$",
            rf"$p = 1000$ for $r < 0.1$, $\quad p = 0.1$ outside; $r$ is the distance from the centre ({shape})",
            field, r"A pressure jump of $10^{4}$. These runs use a pressure floor."]


SETUPS = {
    "mhd": [r"$\partial_t\rho + \nabla\cdot(\rho\mathbf{v}) = 0$",
            r"$\partial_t(\rho\mathbf{v}) + \nabla\cdot\left(\rho\mathbf{v}\mathbf{v} - \mathbf{B}\mathbf{B} + P^{*}\,\mathbf{I}\right) = 0,\quad P^{*} = p + \frac{B^2}{2}$",
            r"$\partial_t E + \nabla\cdot\left[(E + P^{*})\,\mathbf{v} - \mathbf{B}\,(\mathbf{B}\cdot\mathbf{v})\right] = 0,\quad E = \frac{p}{\gamma - 1} + \frac{\rho v^2}{2} + \frac{B^2}{2}$",
            r"$\partial_t\mathbf{B} - \nabla\times(\mathbf{v}\times\mathbf{B}) = 0,\quad \nabla\cdot\mathbf{B} = 0$",
            r"Dedner (GLM) cleaning adds $\psi$: $\quad\partial_t\mathbf{B} - \nabla\times(\mathbf{v}\times\mathbf{B}) + \nabla\psi = 0,\quad \partial_t\psi + c_h^2\,\nabla\cdot\mathbf{B} = -\frac{c_h^2}{c_p^2}\,\psi$",
            r"$\mathbf{B}$ is in units where the magnetic pressure is $B^2/2$."],
    "test/uniform": [BOX + r", $\gamma = 5/3$, run to $t = 0.1$.",
                     r"$\rho = 1,\quad p = 1,\quad \mathbf{v} = (0.2,\ -0.1,\ 0.05),\quad \mathbf{B} = (0.3,\ 0.2,\ 0)$",
                     "Exact answer: nothing changes. Any drift is an error of the implementation, not of the method."],
    "test/entropy": [BOX + r", $\gamma = 5/3$, run to $t = 1$ (one period).",
                     r"$\rho = 1 + 0.2\,\sin 2\pi(x + y - t)$",
                     r"$\mathbf{v} = (0.5,\ 0.5,\ 0),\quad p = 1,\quad \mathbf{B} = (0.3,\ 0.2,\ 0)$",
                     r"Exact answer: the density pattern is carried by the flow and is back at its start at $t = 1$. Cell averages carry the factor $\left[\sin(\pi/N)/(\pi/N)\right]^2$ on the amplitude.",
                     "The field is uniform, so this tests the order of accuracy and not the control of the divergence."],
    "test/alfven": [BOX + r", $\gamma = 5/3$, run to $t = 1$ (one period).",
                    r"$\rho = 1,\quad p = 0.1,\quad B_x = 1,\quad \varphi = 2\pi(x - t)$",
                    r"$(B_y,\ B_z) = 0.1\,(\cos\varphi,\ \sin\varphi),\quad (v_y,\ v_z) = -(B_y,\ B_z)/\sqrt{\rho},\quad v_x = 0$",
                    r"Alfvén speed $v_A = B_x/\sqrt{\rho} = 1$: a circularly polarised wave moving right, an exact nonlinear solution. Along the grid, so $\nabla\cdot\mathbf{B} = 0$ trivially."],
    "test/alfven_oblique": [BOX + r", $\gamma = 5/3$, run to $t = 1/\sqrt{5}$ (one period).",
                            r"The same wave along $\hat{\mathbf{k}} = (1,\ 2)/\sqrt{5}$: $\quad\varphi = 2\pi(x + 2y - \sqrt{5}\,t)$, wavelength $1/\sqrt{5}$",
                            r"$\rho = 1,\quad p = 0.1,\quad \mathbf{B}_0 = \hat{\mathbf{k}}$",
                            r"$\delta\mathbf{B} = 0.1\,(\cos\varphi\ \hat{\mathbf{t}} + \sin\varphi\ \hat{\mathbf{z}}),\quad \hat{\mathbf{t}} = (-2,\ 1)/\sqrt{5},\quad \delta\mathbf{v} = -\delta\mathbf{B}$",
                            "Oblique to the grid, so the update of the field in two dimensions is exercised."],
    "test/brio_wu": [BOX + r", $\gamma = 2$, run to $t = 0.05$. Left state on $0 \leq x < \frac{1}{2}$, right state on $\frac{1}{2} \leq x < 1$: two interfaces.",
                     r"Left: $\rho = 1,\quad p = 1,\quad B_y = +1$",
                     r"Right: $\rho = 0.125,\quad p = 0.1,\quad B_y = -1$",
                     r"$B_x = 0.75,\quad \mathbf{v} = 0$",
                     "Fast rarefactions, a slow compound wave, a contact, a slow shock. No closed-form answer: the reference is a much finer run."],
    "test/brio_wu_rotated": [BOX + r", $\gamma = 2$, run to $t = 0.05/\sqrt{2}$.",
                             r"The Brio–Wu states as functions of $\xi = (x + y)\ \mathrm{mod}\ 1$: left for $\xi < \frac{1}{2}$, right otherwise.",
                             r"Left: $\rho = 1,\quad p = 1$",
                             r"Right: $\rho = 0.125,\quad p = 0.1$",
                             r"$\mathbf{v} = 0$",
                             r"Field along the normal $(1,\ 1)/\sqrt{2}$: $B_n = 0.75$; along the interface: $B_t = \pm 1$",
                             r"The same problem at 45° to the grid: any error in $\nabla\cdot\mathbf{B}$ shows as a wrong normal field."],
    "test/orszag_tang": [BOX + r", $\gamma = 5/3$, run to $t = 0.5$.", OT_STATE[0], r"$\mathbf{v} = (-\sin 2\pi y,\ \sin 2\pi x,\ 0)$", OT_STATE[1], OT_STATE[2],
                         r"Sound speed 1, peak Mach number 1, $\beta = 2p/B_0^2 = 10/3$."],
    "test/rotor": [BOX + r", $\gamma = 7/5$, run to $t = 0.15$. $r$ is the distance from the centre $(\frac{1}{2},\ \frac{1}{2})$; $r_0 = 0.1,\ r_1 = 0.115,\ u_0 = 2$.",
                   r"$r \leq r_0$: $\quad\rho = 10,\quad \mathbf{v} = \frac{u_0}{r_0}\,\left(-(y - \frac{1}{2}),\ x - \frac{1}{2}\right)$",
                   r"$r_0 < r < r_1$: $\quad\rho = 1 + 9f,\quad \mathbf{v} = \frac{f\,u_0}{r}\,\left(-(y - \frac{1}{2}),\ x - \frac{1}{2}\right),\quad f = \frac{r_1 - r}{r_1 - r_0}$",
                   r"$r \geq r_1$: $\quad\rho = 1,\quad \mathbf{v} = 0$",
                   r"$p = 1,\quad B_x = 5/\sqrt{4\pi}$ everywhere",
                   "A dense spinning disc in a uniform field: the field is wound up and brakes the disc."],
    "test/field_loop": [BOX + r", $\gamma = 5/3$, run to $t = 2$.",
                        r"$\rho = 1,\quad p = 1,\quad \mathbf{v} = (2,\ 1,\ \sqrt{5})$",
                        r"$A_z = a_0\,\max(R - r,\ 0),\quad a_0 = 10^{-3},\quad R = 0.3$, with $r$ the distance from the centre", LOOP,
                        r"Exact answer: the loop is carried unchanged. With $v_z \neq 0$, growth of $B_z$ measures the error in $\nabla\cdot\mathbf{B}$."],
    "test/field_loop_novz": [BOX + r", $\gamma = 5/3$, run to $t = 2$.",
                             r"$\rho = 1,\quad p = 1,\quad \mathbf{v} = (2,\ 1,\ 0)$",
                             r"$A_z = a_0\,\max(R - r,\ 0),\quad a_0 = 10^{-3},\quad R = 0.3$, with $r$ the distance from the centre", LOOP,
                             "Exact answer: the loop is carried unchanged (Gardiner & Stone 2005). What is lost of the magnetic energy is numerical diffusion."],
    "test/blast_2d_b0": blast(2, 0), "test/blast_2d_b50": blast(2, 50), "test/blast_2d_b100": blast(2, 100),
    "test/kh_uniform": [BOX + r", $\gamma = 5/3$, run to $t = 3$. Two shear layers, at $y = \frac{1}{4}$ and $y = \frac{3}{4}$.", *FLOW_KH,
                        r"$B_x = \sqrt{2p/\beta} = 0.01,\quad \beta = 2\times10^{5}$: a weak field everywhere"],
    "test/kh_localized": [BOX + r", $\gamma = 5/3$, run to $t = 3$. Two shear layers, at $y = \frac{1}{4}$ and $y = \frac{3}{4}$.", *FLOW_KH, KH_BAND,
                          r"Leakage: field appearing in the upper vortex, around $y = \frac{3}{4}$, before the two layers mix."],
    "test/kh_localized_dtjump": [BOX + r", $\gamma = 5/3$, run to $t = 3.2$. Flow and field as in the localized test:", *FLOW_KH, KH_BAND,
                                 r"At $t = 3$ the timestep is cut to $\Delta t/40$ for 100 steps. With $c_h \propto \Delta x/\Delta t$ the cleaning speed jumps 40-fold; with a constant $c_h$ it does not."],
    "test/current_sheet": [BOX + r", $\gamma = 5/3$, run to $t = 2.5$.",
                           r"$\rho = 1,\quad p = 0.1,\quad \mathbf{v} = (u_0\,\sin 2\pi y,\ 0,\ 0),\quad u_0 = 0.1$",
                           r"$B_y = +1$ for $|x - \frac{1}{2}| > \frac{1}{4}$, $\quad B_y = -1$ for $|x - \frac{1}{2}| < \frac{1}{4}$; $\quad B_x = B_z = 0$",
                           r"$\beta = 2p/B^2 = 0.2$: the field dominates. The sheets at $x = \frac{1}{4},\ \frac{3}{4}$ reconnect through numerical resistivity alone."],
    "test/gem": [r"Periodic unit box with two Harris sheets, at $y = \frac{1}{4}$ and $y = \frac{3}{4}$. Lengths are in units of $L_x = 25.6\,d_i$, so $\lambda = 0.5/25.6$ and $\psi_0 = 0.1/25.6$.",
                 r"$\gamma = 5/3$, run to $t = 40/L_x = 1.5625$, that is $\Omega_i t = 40$.",
                 r"$\rho = \mathrm{sech}^2\frac{y - 1/4}{\lambda} + \mathrm{sech}^2\frac{y - 3/4}{\lambda} + 0.2,\quad p = \rho/2,\quad \mathbf{v} = 0$",
                 r"$A_z = \lambda\,\ln\cosh\frac{y - 1/4}{\lambda} - \lambda\,\ln\cosh\frac{y - 3/4}{\lambda} - y - \psi_0\,\cos 2\pi x\,\cos 2\pi(y - \frac{1}{4})$",
                 r"$B_x = \partial_y A_z,\quad B_y = -\partial_x A_z$: away from the sheets $|B_x| = 1$, reversing across each one",
                 "No Hall term and no explicit resistivity: reconnection runs on each code's numerical resistivity."],
    "test/orszag_tang_3d": [BOX + r" (a cube), $\gamma = 5/3$, run to $t = 1$.", OT_STATE[0],
                            r"$\mathbf{v} = \left((1 + \epsilon\sin 2\pi z)\,(-\sin 2\pi y),\ (1 + \epsilon\sin 2\pi z)\,\sin 2\pi x,\ \epsilon\sin 2\pi z\right),\quad \epsilon = 0.2$",
                            OT_STATE[1], "The planar vortex with its flow modulated along z, so the third dimension takes part."],
    "test/field_loop_3d": [BOX + r" (a cube), $\gamma = 5/3$, run to $t = 1$.",
                           r"$\rho = 1,\quad p = 1,\quad \mathbf{v} = (1,\ 1,\ 2)$",
                           r"$\mathbf{A} = a_0\,\max(R - d,\ 0)\ \hat{\mathbf{a}},\quad \hat{\mathbf{a}} = (-1,\ 0,\ 1)/\sqrt{2},\quad a_0 = 10^{-3},\quad R = 0.3$",
                           r"with $d$ the distance from the axis through the centre along $\hat{\mathbf{a}}$: a cylinder of field, oblique to the grid (Gardiner & Stone 2008)."],
    "test/blast_3d_b0": blast(3, 0), "test/blast_3d_b50": blast(3, 50), "test/blast_3d_b100": blast(3, 100),
    "test/decay": [BOX + r" (a cube), $\gamma = 5/3$, run to $t = 0.5$; three seeds.",
                   r"$\rho = 1,\quad p = 1/\gamma$ (sound speed 1), $\quad\mathbf{B} = (0.1,\ 0,\ 0)$",
                   r"$\mathbf{v} = \sum_{\mathbf{k}}\mathbf{a}_{\mathbf{k}}\,\sin(2\pi\,\mathbf{k}\cdot\mathbf{x} + \phi_{\mathbf{k}})$ over integer $\mathbf{k}$ with $1 \leq |\mathbf{k}|^2 \leq 4$",
                   r"$\mathbf{a}_{\mathbf{k}}\perp\mathbf{k}$ (solenoidal), random with size $\propto 1/|\mathbf{k}|$, random phases $\phi_{\mathbf{k}}$",
                   "The mean velocity is removed and the field scaled to an rms Mach number of 2. Nothing drives it afterwards."],
    "test/decay_supersonic": [BOX + r" (a cube), $\gamma = 1.01$ (nearly isothermal), run to $t = 0.2$; two seeds.",
                              r"$\rho = 1,\quad p = 1/\gamma$ (sound speed 1), $\quad\mathbf{B} = (0,\ 0,\ 0.3)$, so $\beta = 2p/B^2 = 22$",
                              r"$\mathbf{v} = \sum_{\mathbf{k}}\mathbf{a}_{\mathbf{k}}\,\sin(2\pi\,\mathbf{k}\cdot\mathbf{x} + \phi_{\mathbf{k}})$ over integer $\mathbf{k}$ with $1 \leq |\mathbf{k}|^2 \leq 4$, $\ \mathbf{a}_{\mathbf{k}}\perp\mathbf{k}$",
                              "Scaled to an rms Mach number of 9, after Kritsuk et al. 2011 but from a smooth start, not their driven state."],
    "test/throughput_ot3d": [r"The 3D Orszag–Tang state, run to $t = 0.05$ on $64^3$ to $512^3$, including grids divisible by 6.",
                             r"Measured: cell updates per second $= N^3 \times \mathrm{steps}/\mathrm{seconds}$, and the memory held on the card."],
    "turb/mach-3-10": [BOX + r" (a cube), $\bar{\rho} = 1$, sound speed 1, $\gamma - 1 = 10^{-5}$ with no cooling, run to $10\,T$.",
                       r"A random acceleration $\mathbf{a}(\mathbf{x}, t)$ on the momentum equation: solenoidal, on $1 < |\mathbf{k}| < 3$ with the parabolic weight $1 - (|\mathbf{k}| - 2)^2$,",
                       TURB[0], r"$a_{\mathrm{rms}} = 15$ for Mach 3 and $165$ for Mach 10, found once with RAMSES at $64^3$.", *TURB[1:]],
    "turb/mach-30-50": [BOX + r" (a cube), $\bar{\rho} = 1$, sound speed 1, $\gamma - 1 = 10^{-6}$ with no cooling, run to $10\,T$.",
                        r"A random acceleration $\mathbf{a}(\mathbf{x}, t)$ on the momentum equation: solenoidal, on $1 < |\mathbf{k}| < 3$ with the parabolic weight $1 - (|\mathbf{k}| - 2)^2$,",
                        TURB[0], r"$a_{\mathrm{rms}} = 1450$ for Mach 30 and $4000$ for Mach 50. Density floor $10^{-6}$, which every code reaches.", *TURB[1:]],
}


def draw(tex, prop, parser):
    """One formula as an SVG; returns (file, height, depth) in points."""
    name = hashlib.sha1(tex.encode()).hexdigest()[:12] + ".svg"
    width, height, depth, _, _ = parser.parse(f"${tex}$", dpi=72, prop=prop)
    fig = Figure(figsize=(width/72, height/72))          # as mathtext.math_to_image, but with no background behind the formula
    fig.text(0, depth/height, f"${tex}$", fontproperties=prop, color=INK);fig.savefig(OUT/name, dpi=72, format="svg", transparent=True)
    return name, round(height, 2), round(depth, 2)


def main():
    matplotlib.rcParams.update({"mathtext.fontset": "cm", "svg.fonttype": "path"})
    OUT.mkdir(parents=True, exist_ok=True);prop = FontProperties(size=SIZE);parser = mathtext.MathTextParser("path");used = set();data = {}
    for key, lines in SETUPS.items():
        rows = []
        for line in lines:
            if line.count("$") % 2:sys.exit(f"{key}: unbalanced dollar signs in {line!r}")
            parts = []
            for i, piece in enumerate(line.split("$")):
                if not piece:continue
                if i % 2 == 0:parts.append(piece);continue
                try:
                    name, h, d = draw(piece, prop, parser)
                except Exception as err:        # a construct mathtext does not know
                    sys.exit(f"{key}: cannot typeset {piece!r}: {err}")
                used.add(name);parts.append([piece, name, h, d])
            rows.append({"src": line, "parts": parts})
        data[key] = rows
    for old in OUT.glob("*.svg"):
        if old.name not in used:old.unlink()
    (SITE/"setups.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    page = SITE/"notes.html";text = page.read_text();block = "/*SETUPS-BEGIN*/const SETUPS = " + json.dumps(data, ensure_ascii=False) + ";/*SETUPS-END*/"
    new, n = re.subn(r"/\*SETUPS-BEGIN\*/.*?/\*SETUPS-END\*/", lambda m: block, text, count=1, flags=re.S)
    if n != 1:sys.exit("notes.html has no SETUPS markers")
    page.write_text(new)
    size = sum(p.stat().st_size for p in OUT.glob("*.svg"))
    print(f"{len(data)} set-ups, {len(used)} formulas ({size/1e3:.0f} kB in {OUT.relative_to(ROOT)}); notes.html and setups.json updated")


if __name__ == "__main__":
    main()
