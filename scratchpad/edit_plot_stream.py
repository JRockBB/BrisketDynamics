from pathlib import Path
p = Path("C:/Users/jmurp/.julia/dev/BrisketDynamics/estimator/plot_stream.py")
s = p.read_text(encoding="utf-8")
pairs = [
('SHOW = ["L", "h_c", "k_m", "D_eff", "A_meat", "core_depth", "r_max", "a_myosin"]',
 'SHOW = ["L", "h_c", "k_m", "a0", "A_meat", "core_depth", "r_max", "a_myosin"]'),
('''    cols = [n for n in SHOW if n in idx]
''', '''    cols = [n for n in SHOW if n in idx]
    # one error scale for the whole row: the panels are read against each other
    lim = max(3.0, 1.1 * float(np.nanmax(np.abs(np.array(
        [b["err_sd"][idx[n]] for n in cols for s in sets for b in s["batches"]])))))
'''),
('''        lim = max(3.0, float(np.nanmax(np.abs(np.array(
            [b["err_sd"][j] for s in sets for b in s["batches"]])))) * 1.1)
        ax_err.set_ylim(-lim, lim)
''', '''        ax_err.set_ylim(-lim, lim)
'''),
]
for old, new in pairs:
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
print("plot_stream.py patched")
