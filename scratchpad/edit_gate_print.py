from pathlib import Path
p = Path("C:/Users/jmurp/.julia/dev/BrisketDynamics/estimator/gate_noise_real.jl")
s = p.read_text(encoding="utf-8")
old = '''    for (i, nm) in ((1, "pit"), (2, "core"))
        np = res.noise[i]
'''
new = '''    for (i, nm) in ((1, "pit"), (2, "core"))
        haskey(res.noise, i) || continue
        np = res.noise[i]
'''
assert s.count(old) == 1
p.write_text(s.replace(old, new), encoding="utf-8")
print("gate print fixed")
