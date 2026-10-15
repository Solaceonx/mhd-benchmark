"""Completion versus finite-output audit of the saved Mach 9 decay records."""
from collections import Counter
import html
import json

import numpy as np
import paper_common as pc

EVIDENCE = {
    "glm": "One fine-grid run flags a nonphysical/nonfinite state. The first failing operation has not been isolated.",
    "glm_hll16": "Half-precision storage can overflow and lose small differences; shortened output steps also raise the cleaning flux. CPU checks support these mechanisms, without tracing every 3D failure.",
    "glm_ppm": "Uses the same half-precision HLL storage. Its stops do not establish that parabolic reconstruction itself is the cause.",
    "glm_constch": "Constant cleaning speed avoids its output-step jump, but retains the half-precision storage and its range/rounding limitations.",
    "glm_hlld": "Nonphysical/nonfinite-state stops. A degenerate-state HLLD tolerance problem is established in planar CPU checks; its contribution to these 3D stops is not yet isolated.",
    "ct": "All representative runs finish with finite saved diagnostics. The prototype is strongly diffusive and first order in the wave checks; finishing alone does not establish accuracy.",
    "athenak_matched": "PLM/HLLE/RK2 finishes with finite saved diagnostics on every planned grid and seed.",
    "athenak_best": "PPM/HLLD/RK3 hits the configured cycle cap before the requested endpoint. The saved tails show insufficient time progress; the first local instability is not isolated.",
    "ramses_matched": "HLL returns without all requested outputs at 64³ and 128³; the project records identify timestep collapse. The four larger-grid configurations were explicitly skipped.",
    "ramses_best": "HLLD finishes with finite saved diagnostics on every planned grid and seed.",
    "athenapp_matched": "PLM/HLLE finishes with finite saved diagnostics on every planned grid and seed.",
    "athenapp_best": "Reports completion, but all eight runs have nonfinite saved diagnostics after the last finite output at t = 0.02. A successful exit does not make these valid solutions.",
}


def section(sweep):
    test = "decay_supersonic"; records = []; rows = []; total = Counter()
    for method in sweep.labels(test):
        counts = Counter()
        for n in sweep.grids(test, method):
            for seed in sweep.seeds(test):
                raw = sweep.state_of(test, method, n, seed)
                m = sweep.meta(test, method, n, seed) or {}; d = sweep.diag(test, method, n, seed)
                finite_count = sweep.finite_rows(test, method, n, seed)
                if raw == "skipped": state = "skipped"
                elif (m.get("status") == "complete" and d and finite_count == len(d["time"])
                      and np.isclose(d["time"][-1], sweep.configs[test]["t_end"], rtol=0, atol=1e-5)):
                    state = "finite_complete"
                elif raw == "done": state = "invalid_complete"
                elif raw == "stopped": state = "stopped"
                else: state = "unavailable"
                counts[state] += 1; total[state] += 1
                records.append(dict(method=method, N=n, seed=seed, repeat=1, reported_state=raw, audited_state=state,
                    finite_outputs=finite_count, saved_outputs=len(d["time"]) if d else 0,
                    last_finite_saved_time=float(d["time"][finite_count-1]) if d and finite_count else None,
                    error=str(m.get("error", "")).splitlines()[0] if m.get("error") else ""))
        label = pc.plain(pc.STYLE[method][1]).replace("$c_h$", "cₕ")
        rows.append(f'<tr><th scope="row">{html.escape(label)}</th><td>{counts["finite_complete"]}</td><td>{counts["stopped"]}</td>'
                    f'<td>{counts["invalid_complete"]}</td><td>{counts["skipped"]}</td><td>{html.escape(EVIDENCE[method])}</td></tr>')
    root = pc.SITE / "media/analysis/decay_supersonic"
    root.mkdir(parents=True, exist_ok=True)
    (root / "failure-audit.json").write_text(json.dumps(dict(scope="Main sweep, first repeat per method/grid/seed; not a count of all timing repeats",
        counts=dict(total), records=records), indent=2, allow_nan=False)+"\n")
    attempted = sum(total.values())-total["skipped"]-total["unavailable"]
    return f'''<article class="card" id="failure-audit"><h3>Why the Mach 9 runs failed</h3>
<p>Of {attempted} attempted method/grid/seed combinations, {total['finite_complete']} reached t = 0.2 with finite saved diagnostics,
{total['stopped']} stopped early, and {total['invalid_complete']} reported completion while their output became nonfinite.
Another {total['skipped']} were explicitly skipped. These counts use repeat 1, rather than counting repeated timing runs as independent experiments.</p>
<p>This is decaying <strong>MHD</strong> turbulence: initial Mach 9, γ = 1.01, mean B<sub>z</sub> = 0.3, two seeds, 64³–512³.
Strong shocks and rarefactions test pressure/density positivity; magnetic amplification and cleaning add further numerical constraints.
The <a href="test-driven-mach-30.html">Mach 30</a> and <a href="test-driven-mach-50.html">Mach 50</a> driven runs use zero magnetic field,
different kernels/settings and γ − 1 = 10⁻⁶. A larger Mach number alone does not make them a stricter version of this test.</p>
<div class="table-scroll"><table class="metrics-table sweep-text"><thead><tr><th>Set-up</th><th>Finite complete</th><th>Stopped</th><th>Complete but invalid</th><th>Skipped</th><th>Evidence and interpretation</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>Half precision has a largest finite value of 65,504. CPU checks reproduce flux overflow and show why
c<sub>h</sub> ∝ 1/Δt can worsen it on steps shortened for output. They also identify the HLLD tolerance issue.
The 3D records usually report the downstream invalid-state check, so those mechanisms should not be presented as a traced cause of every individual stop.</p>
<p><a href="media/analysis/decay_supersonic/failure-audit.json">Download the audit records (JSON)</a>.
“Finite complete” checks the saved history and endpoint; it is not a guarantee of correctness between saved outputs.</p></article>'''
