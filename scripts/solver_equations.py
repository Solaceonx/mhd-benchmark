"""Equations matched to the walkthrough's line annotations; render locally as SVG."""
import hashlib
import html
import os
from pathlib import Path

# Each entry: LaTeX lines, followed by the bridge from symbols to code variables.
EQUATIONS = {
    "state-0": ([r"\mathbf{U}=(\rho,\rho\mathbf{v},E,\mathbf{B},\psi)", r"\mathbf{q}=(\rho,\mathbf{v},p,\mathbf{B},\psi)"], "Bold v and B each stand for three components, giving nine slots in each tuple."),
    "state-1": ([r"\rho=\max(U_1,\rho_{\min}),\quad\mathbf{v}=\mathbf{m}/\rho", r"K=\frac{|\mathbf{m}|^2}{2\rho},\quad M=\frac{|\mathbf{B}|^2}{2}", r"p=\max\!\left[(\gamma-1)(E-K-M),p_{\min}\right]"], "m = (c[2], c[3], c[4]); smallr = ρmin; pfl = pmin. K and M correspond to ekin and emag."),
    "state-2": ([r"E=\frac{p}{\gamma-1}+\frac{\rho|\mathbf{v}|^2}{2}+\frac{|\mathbf{B}|^2}{2}", r"\mathbf{m}=\rho\mathbf{v}"], "Magnetic units absorb μ₀. This definition of E has no additional ψ-energy term."),
    "flux-0": ([r"p_{\rm tot}=p+\frac{|\mathbf{B}|^2}{2}", r"F_{m_x}=\rho v_x^2+p_{\rm tot}-B_x^2", r"F_{m_y}=\rho v_xv_y-B_xB_y", r"F_E=(E+p_{\rm tot})v_x-B_x(\mathbf{v}\cdot\mathbf{B})"], "ptot = total pressure; vdotb = v·B. The negative magnetic products are the tension terms."),
    "flux-1": ([r"F_\rho=\rho v_x", r"\mathbf{F}_{\mathbf{B}_\perp}=v_x\mathbf{B}_\perp-B_x\mathbf{v}_\perp", r"F_{B_x}=\psi^*,\quad F_\psi=c_h^2 B_x^*"], "⊥ means the y and z components at an x face. The last two fluxes are supplied by glm_pair, replacing the placeholders."),
    "slopes-0": ([r"d_L=q_i-q_{i-1},\quad d_R=q_{i+1}-q_i", r"d_C=\frac{d_L+d_R}{2}", r"\sigma={\rm sgn}(d_C)\min(2|d_L|,2|d_R|,|d_C|)", r"\sigma=0\quad\mathrm{if}\quad d_Ld_R\leq0"], "dl, dr, dc map to dL, dR, dC. The zero-slope rule overrides the preceding expression at an extremum."),
    "slopes-1": ([r"s_i=\frac{1}{2}\sigma_i", r"q_{i,L}=q_i-s_i,\quad q_{i,R}=q_i+s_i"], "prim_slope returns s, the half-slope. padd(q, s, ±1) builds the two spatial edges, component by component."),
    "predictor-0": ([r"h=\frac{\Delta t}{2\Delta x}", r"\Delta\mathbf{F}_d=\mathbf{F}_d(\mathbf{q}+\mathbf{s}_d)-\mathbf{F}_d(\mathbf{q}-\mathbf{s}_d)", r"\mathbf{U}^{n+1/2}=\mathbf{U}^n-h\sum_{d=x,y,z}\Delta\mathbf{F}_d"], "h is the code variable h. fxp−fxm, fyp−fym, and fzp−fzm are the three directional differences."),
    "predictor-1": ([r"\mathbf{U}^{n+1/2}=\mathbf{U}^n-\frac{\Delta t}{2\Delta x}\sum_d\Delta\mathbf{F}_d", r"\mathbf{q}_{d,\pm}^{n+1/2}=\mathbf{q}^{n+1/2}\pm\mathbf{s}_d"], "The predictor sums all directions at once. The second line shows the later face construction; floors or fallback may modify those states."),
    "cleaning-0": ([r"B_n^*=\frac{B_{n,L}+B_{n,R}}{2}-\frac{\psi_R-\psi_L}{2c_h}", r"\psi^*=\frac{\psi_L+\psi_R}{2}-\frac{c_h}{2}(B_{n,R}-B_{n,L})"], "bns = Bn*; psis = ψ*. A star marks an interface solution, not a time derivative."),
    "cleaning-1": ([r"\partial_t B_n+\partial_n\psi=0", r"\partial_t\psi+c_h^2\partial_n B_n=0", r"\lambda_{\rm GLM}=\pm c_h"], "These are the linear cleaning equations before damping, with ch held constant during the face solve."),
    "speeds-0": ([r"c_s^2=\frac{\gamma p}{\rho},\quad v_A^2=\frac{|\mathbf{B}|^2}{\rho}", r"v_{A,n}^2=\frac{B_n^2}{\rho},\quad D=\frac{c_s^2+v_A^2}{2}"], "c2 = cs², b2 = vA², and d2 = D. bn is the field component normal to this face."),
    "speeds-1": ([r"D=\frac{c_s^2+v_A^2}{2}", r"c_f=\sqrt{D+\sqrt{\max(D^2-c_s^2v_{A,n}^2,0)}}", r"\lambda_{f,\pm}=v_n\pm c_f"], "D is d2 in the source. cf is measured relative to the gas; λf,± are physical fast-wave speeds in the fixed-grid frame."),
    "llf-0": ([r"a=\max(|v_{n,L}|+c_{f,L},\ |v_{n,R}|+c_{f,R},\ c_h)"], "a is smax. Both physical fast signals and the cleaning speed enter this implementation's LLF bound."),
    "llf-1": ([r"\widehat{\mathbf{F}}_{\rm LLF}=\frac{\mathbf{F}_L+\mathbf{F}_R}{2}-\frac{a}{2}(\mathbf{U}_R-\mathbf{U}_L)"], "fL/fR are physical fluxes; uL/uR are conserved states; a = smax. A hat denotes a numerical face flux. GLM entries are overwritten next."),
    "llf-2": ([r"\widehat{F}_{B_n}=\psi^*,\quad\widehat{F}_\psi=c_h^2 B_n^*"], "fbn = ψ* and fpsi = ch²Bn*. These replace entries 6 and 9 in the x-normal tuple."),
    "hll-0": ([r"S_L=\min(v_{n,L}-c_{f,L},v_{n,R}-c_{f,R},0)", r"S_R=\max(v_{n,L}+c_{f,L},v_{n,R}+c_{f,R},0)"], "SL and SR bound the physical MHD fan after GLM has supplied a shared Bn*. These are this function's actual speed estimates."),
    "hll-1": ([r"\widehat{\mathbf{F}}_{\rm HLL}=\frac{S_R\mathbf{F}_L-S_L\mathbf{F}_R+S_LS_R(\mathbf{U}_R-\mathbf{U}_L)}{S_R-S_L}", r"S_L=-a,\ S_R=a\ \Longrightarrow\ \widehat{\mathbf{F}}_{\rm HLL}=\widehat{\mathbf{F}}_{\rm LLF}"], "ihd = 1/(SR−SL). The LLF identity assumes the same a and the same subsequent GLM flux replacement."),
    "hlld-0": ([r"S_L=\min[\min(v_L,v_R)-\max(c_{f,L},c_{f,R}),0]", r"S_R=\max[\max(v_L,v_R)+\max(c_{f,L},c_{f,R}),0]"], "vL and vR denote normal velocities (uL and uR inside hlld_x), rather than the full vectors."),
    "hlld-1": ([r"A_L=\rho_L(S_L-v_L),\quad A_R=\rho_R(S_R-v_R)", r"S_M=\frac{A_Rv_R-A_Lv_L-p_{{\rm tot},R}+p_{{\rm tot},L}}{A_R-A_L}", r"\rho_L^*=\frac{A_L}{S_L-S_M},\quad\rho_R^*=\frac{A_R}{S_R-S_M}"], "den = AR−AL; SM = contact speed. The source additionally floors each star density at 10⁻¹²."),
    "hlld-2": ([r"S_L^*=S_M-\frac{|B_n|}{\sqrt{\rho_L^*}}", r"S_R^*=S_M+\frac{|B_n|}{\sqrt{\rho_R^*}}", r"S_L\leq S_L^*\leq S_M\leq S_R^*\leq S_R"], "SLs and SRs are the rotational Alfvén speeds. The last line is the intended five-wave ordering for physical intermediate states."),
    "dispatch-0": ([r"\mathbf{q}'=P_n\mathbf{q}", r"\widehat{\mathbf{F}}_n=P_n^{-1}\widehat{\mathbf{F}}_x(\mathbf{q}'_L,\mathbf{q}'_R)"], "Pn is the component permutation implemented by rot_to. rot_flux_from undoes it; the grid itself does not rotate."),
    "dispatch-1": ([r"\rho_{L,R}\leftarrow\max(\rho_{L,R},\rho_{\min})", r"p_{L,R}\leftarrow\max(p_{L,R},p_{\min})", r"B_{n,L}^{\rm clean}=B_{n,R}^{\rm clean}=B_n^*"], "L and R receive floors; Lc and Rc then receive the common normal magnetic field returned by glm_pair."),
    "dispatch-2": ([r"\min(\rho_L,\rho_R)<\rho_{\rm switch}\ \Longrightarrow\ {\rm LLF}", r"\min(p_L,p_R)<p_{\rm switch}\ \Longrightarrow\ {\rm LLF}"], "Each threshold rule is enabled only if its configured threshold is positive. use_hlld=false also forces LLF; otherwise the dispatcher selects HLLD."),
    "update-0": ([r"\mathbf{U}_i^{n+1}=\mathbf{U}_i^n-\frac{\Delta t}{\Delta x}\sum_d\Delta\widehat{\mathbf{F}}_{i,d}", r"\Delta\widehat{\mathbf{F}}_{i,d}=\widehat{\mathbf{F}}_{i+\frac{1}{2}\mathbf{e}_d}-\widehat{\mathbf{F}}_{i-\frac{1}{2}\mathbf{e}_d}"], "u0 = old cell state; r1 = flux-updated state. The minus sign here equals the code's plus sign multiplying incoming minus outgoing flux."),
    "update-1": ([r"\rho\leftarrow\rho_{\min},\quad\mathbf{m}\leftarrow\mathbf{0}", r"E\leftarrow\frac{p_{\min}}{\gamma-1}+\frac{|\mathbf{B}|^2}{2}"], "Applied only when the flux-updated density is below smallr. B and ψ are retained during this cell reset."),
    "update-2": ([r"\psi^{n+1}=g\,\psi^{\rm flux}", r"g=\exp\!\left[-\frac{c_h\Delta t}{\alpha L}\right]"], "g = glm_fac, α = glm_cp_coef, L = boxlen. For α ≤ 0, the launcher sets g = 1. ψflux is the value after the conservative update."),
    "timing-0": ([r"c_h=\frac{C\,\Delta x}{3\,\Delta t}\,s_h", r"\Delta t=\frac{C\,\Delta x}{\max_i\sum_d(|v_{i,d}|+c_{f,i,d})}"], "C = courant; sh = glm_ch_scale; NDIM = 3. The second equation is the benchmark driver's timestep rule, separate from the launcher."),
    "timing-1": ([r"g=\exp\!\left[-\frac{c_h\Delta t}{\alpha L}\right]", r"\psi\leftarrow g\psi"], "This is the source expression simplified for positive ch and α = glm_cp_coef > 0. α ≤ 0 disables damping."),
    "gpu-0": ([r"q_{\rm stored}={\rm round}_{16}(q_{32})", r"q_{\rm read}={\rm Float32}(q_{\rm stored})"], "Storage operations when HALF=true. Reading into Float32 preserves the already-rounded value; it cannot recover discarded precision."),
    "gpu-1": ([r"T_T=T_B+4", r"N_{\rm primitive}=9T_T^3", r"N_{\rm face}=9(T_B+1)T_B^2"], "TT is the tile width including halos; TB is its owned width. These count scalar elements, not bytes. Each listed face array has Nface elements."),
    "ppm-0": ([r"\Delta q=q_R-q_L", r"A=\left(q_0-\frac{q_L+q_R}{2}\right)\Delta q", r"q_L\leftarrow3q_0-2q_R\quad\mathrm{if}\ A>(\Delta q)^2/6", r"q_R\leftarrow3q_0-2q_L\quad\mathrm{if}\ A<-(\Delta q)^2/6"], "A corresponds to diff and Δq to dq. If (qR−q0)(q0−qL) ≤ 0, the earlier extremum rule instead sets both edges to q0."),
    "ppm-1": ([r"s=\frac{q_{i+1}-q_{i-1}}{4},\quad a=\frac{q_{i-1}-2q_i+q_{i+1}}{12}", r"q_L=q_i-s+a,\quad q_R=q_i+s+a", r"q_{\rm face}^{n+1/2}=q^{n+1/2}+q_{\rm edge}-q_i"], "s = slope, a = curve; ppm_mono limits the two candidate edges. In mhd_face9, mh = predicted state, ed = limited spatial edge, m0 = original center."),
}


def render_equations(root):
    """Return trusted HTML per annotation and write self-contained vector assets."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/abel-mhd-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import mathtext
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties

    matplotlib.rcParams.update({"mathtext.fontset": "stix", "svg.fonttype": "path", "svg.hashsalt": "solver-equations"})
    out = root / "progress/media/solver-equations"; out.mkdir(parents=True, exist_ok=True)
    prop = FontProperties(size=15); parser = mathtext.MathTextParser("path")
    rendered = {}; used = set()
    for key, (equations, bridge) in EQUATIONS.items():
        images = []
        for tex in equations:
            name = hashlib.sha256(tex.encode()).hexdigest()[:16] + ".svg"
            width, height, depth, _, _ = parser.parse(f"${tex}$", dpi=72, prop=prop)
            pad = 3
            if name not in used:
                fig = Figure(figsize=((width+2*pad)/72, (height+2*pad)/72))
                fig.text(pad/(width+2*pad), (depth+pad)/(height+2*pad), f"${tex}$", fontproperties=prop, color="#253b33")
                fig.savefig(out/name, format="svg", transparent=True, metadata={"Date": None, "Description": tex})
            used.add(name)
            images.append(f'<div class="equation-line"><img src="media/solver-equations/{name}" width="{(width+2*pad)*4/3:.2f}" height="{(height+2*pad)*4/3:.2f}" alt="{html.escape(tex)}"></div>')
        latex = "\n".join(f"\\[ {tex} \\]" for tex in equations)
        rendered[key] = ('<div class="equation-set"><div class="equation-label">Matching equations</div>' + "".join(images)
                         + f'<p class="equation-bridge">{html.escape(bridge)}</p><details class="equation-latex"><summary>LaTeX source</summary><pre>{html.escape(latex)}</pre></details></div>')
    print(f"Rendered {len(used)} equation SVGs for {len(rendered)} annotations.")
    return rendered
