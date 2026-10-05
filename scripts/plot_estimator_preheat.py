"""Plot saved estimator runs, retaining failures and the provenance of each run.

Run from the repo root: app/.venv/Scripts/python.exe scripts/plot_estimator_preheat.py
Optional positional JSON paths select new runs instead of the recorded baseline.
"""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/estimator"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

def rows_of(d):
    runs = [(r, "Completed requested horizon") for r in d.get("cooks", [])]
    runs += [(r, "FAILED") for r in d.get("failures", [])]
    if d.get("active_cook"):
        runs += [(d["active_cook"], "INCOMPLETE")]
    return [(r.get("seed"), r.get("temps", []), status) for r, status in runs if r.get("temps")]


def temperature_errors(rows):
    result = {}
    for key in ("pit", "centre", "surface"):
        a = np.array([r[key] for r in rows], dtype=float).reshape(-1, 3)
        if len(a):
            error = a[:, 0] - a[:, 2]
            result[key] = {
                "rmse_K": float(np.sqrt(np.mean(error**2))),
                "max_abs_K": float(np.max(np.abs(error))),
                "max_abs_standardized": float(np.max(np.abs(error / a[:, 1]))),
            }
    return result


def plot_run(path):
    d = json.loads(path.read_text())
    runs = rows_of(d)
    for seed, rows, status in runs:
        # A stopped run is labelled stopped, nothing more: there is no runtime pass/fail gate. Wall
        # times and latencies are reported as data in the summary and the timing panels.
        if status in ("INCOMPLETE", "FAILED") and d.get("termination"):
            status = "STOPPED early (incomplete)"
        stem = path.stem + (f"_seed{seed}" if len(runs) > 1 else "")
        pace = d.get("pace", 0)
        policy = d.get("update_policy", "node_batches" if pace < 0 else "every_arrival" if pace == 0 else "processing_backlog")
        policy_label = {"node_batches": "NODE-BATCHED diagnostic", "every_arrival": "Every measurement arrival (unpaced)",
                        "processing_backlog": "Causal replay with measured processing backlog"}[policy]
        timing_context = d.get("timing_context") or {}
        first_contended = timing_context.get("last_completed_update_before_diagnostic")
        timing_caveat = (f"\nTiming from update {first_contended + 1} may include concurrent diagnostic work."
                         if first_contended is not None else "")
        if timing_context.get("caption"):
            timing_caveat = "\n" + timing_context["caption"]
        good = [r for r in rows if r.get("committed", False)]
        fail = [r for r in rows if not r.get("committed", False)]
        summary = {
            "source": str(path), "seed": seed, "status": status,
            "backend": d.get("backend_fingerprint"),
            "update_policy": policy,
            "last_hour": rows[-1]["t_h"], "accepted": len(good), "rejected": len(fail),
            "readings": sum(r["n_readings"] for r in rows),
            "update_seconds": {
                "median": float(np.median([r["wall_s"] for r in rows])),
                "p95": float(np.percentile([r["wall_s"] for r in rows], 95)),
                "max": max(r["wall_s"] for r in rows),
            },
            "temperature_errors": temperature_errors(good),
            "temperature_errors_by_phase": {},
            "timing_context": d.get("timing_context"),
            "runtime_validation": d.get("runtime_validation"),
        }
        load_h = d.get("load_h", 1.5)
        for phase, selected in (("before_loading", [r for r in good if r["t_h"] < load_h]),
                                ("after_loading", [r for r in good if r["t_h"] >= load_h])):
            summary["temperature_errors_by_phase"][phase] = {
                "updates": len(selected), "errors": temperature_errors(selected),
                "time_range_h": [selected[0]["t_h"], selected[-1]["t_h"]] if selected else None,
            }
        if all("processing_finish_s" in r for r in rows):
            summary["availability"] = {
                "last_available_h": rows[-1]["processing_finish_s"] / 3600,
                "updates_with_multiple_readings": sum(r["n_readings"] > 1 for r in rows),
                "updates_with_queued_readings": sum(r["processing_start_s"] > r["arrivals"][0]["t_s"] + 1e-6 for r in rows),
                "nodes_added": sum(r["nodes_added"] for r in rows),
                "max_retained_nodes": max(r["n_nodes"] for r in rows),
                "oldest_reading_latency_s": {
                    "median": float(np.median([r["oldest_reading_latency_s"] for r in rows])),
                    "p95": float(np.percentile([r["oldest_reading_latency_s"] for r in rows], 95)),
                    "max": max(r["oldest_reading_latency_s"] for r in rows),
                },
            }
        if good and all("z" in r for r in good):
            errors = np.array([(np.array(r["z"]) - r["z_true"]) / r["z_sd"] for r in good])
            peak = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
            summary["latent_errors"] = {
                "max_abs_standardized": float(abs(errors[peak])),
                "worst_parameter": d["names"][peak[1]],
                "worst_time_h": good[peak[0]]["t_h"],
                "final_standardized": dict(zip(d["names"], errors[-1].tolist())),
            }
        (OUT / (stem + "_summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        fig, axs = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
        t = np.array([r["t_h"] for r in good])
        for ax, key, name in zip(axs[:, 0], ["pit", "centre", "surface"], ["Pit", "Meat centre", "Meat surface"]):
            a = np.array([r[key] for r in good], dtype=float).reshape(-1,3)
            ax.plot(t, a[:, 2]-273.15, "k.--", label="Simulated truth")
            if len(good) > 40:
                ax.fill_between(t, a[:,0]-273.15-1.96*a[:,1], a[:,0]-273.15+1.96*a[:,1], color="#176b9a", alpha=.18)
                ax.plot(t, a[:,0]-273.15, color="#176b9a", label="Estimate +/− 1.96 marginal SD")
            else:
                ax.errorbar(t, a[:, 0]-273.15, yerr=1.96*a[:, 1], fmt="o-", ms=4, capsize=4,
                            color="#176b9a", label="Estimate +/− 1.96 marginal SD")
            ax.set_ylabel(f"{name} (°C)")
            ax.grid(alpha=.2)
        axs[0, 0].legend(fontsize=9)
        ax=axs[0,1]
        for key,name in [("pit","Pit"),("centre","Centre"),("surface","Surface")]:
            a=np.array([r[key] for r in good], dtype=float).reshape(-1,3)
            ax.plot(t,(a[:,0]-a[:,2])/a[:,1],"o-",label=name)
        ax.axhspan(-1.96,1.96,color="grey",alpha=.13)
        ax.axhline(0,color="grey",lw=.8)
        ax.set_ylabel("Temperature error / marginal SD"); ax.legend(fontsize=9)
        ax=axs[1,1]
        tr=np.array([r["t_h"] for r in rows])
        ax.plot(tr,[r["wall_s"] for r in rows],"o-",color="#176b9a",label="Update wall time")
        ax.set_ylabel("Update wall time (seconds)"); ax.legend(fontsize=9)
        ax=axs[2,1]
        ax.semilogy(tr,[max(float(r["newton_sd"]),1e-12) for r in rows],"o-",color="#176b9a",
                    label="Largest marginal-SD step")
        ax.plot(tr, [r.get("newton_tol", .01) for r in rows], color="#ad3333", ls="--",
                label="Stalled-solve posterior-step limit")
        if all("newton_joint_sd" in r for r in rows):
            ax.semilogy(tr, [max(float(r["newton_joint_sd"]),1e-12) for r in rows], ".-",
                        color="#6d4296", label="Joint posterior distance")
            ax.plot(tr, [r["joint_tol"] for r in rows], color="#6d4296", ls="--", label="Fresh joint target")
        ax.set_ylabel("Remaining Newton step / posterior SD"); ax.legend(fontsize=8)
        for ax in axs.flat:
            ax.axvline(d.get("load_h",1.5),color="#666",ls=":",lw=1)
            for r in fail: ax.axvline(r["t_h"],color="#ad3333",ls="--",alpha=.6)
            ax.set_xlim(0, max(r["t_h"] for r in rows)+.1)
            ax.grid(alpha=.15)
        for ax in axs[-1]: ax.set_xlabel("Hours since lighting (dotted line: load-transition midpoint)")
        status = "FAILED: rejected arrival" if fail else status
        experimental = "EXPERIMENTAL / " if "EXPERIMENTAL" in d.get("backend_fingerprint", "") else ""
        fig.suptitle(f"{experimental}{status} · seed {seed}\n{policy_label} · {path.name}",fontsize=12)
        fig.supxlabel("One cook is diagnostic evidence, not coverage validation. Failed arrivals excluded from belief panels.\n"
                       f"{len(good)} accepted / {len(fail)} rejected updates; backend {d.get('backend_fingerprint','unknown')[:16]}."
                       + timing_caveat, fontsize=9)
        target=OUT/(stem+".png")
        fig.savefig(target,dpi=160); plt.close(fig); print(target)
        if good and all("z" in r for r in good):
            n=len(d["names"])
            fig,axs=plt.subplots((n+2)//3,3,figsize=(13,2.05*((n+2)//3)),sharex=True,layout="constrained")
            values=np.array([r["z"] for r in good])
            truth=np.array([r["z_true"] for r in good])
            widths=np.array([r["z_sd"] for r in good])
            errors=(values-truth)/widths
            for j,ax in enumerate(axs.flat):
                if j>=n:
                    ax.set_visible(False)
                    continue
                ax.axhspan(-1.96,1.96,color="grey",alpha=.15)
                ax.axhline(0,color="grey",lw=.8)
                ax.plot(t,errors[:,j],"o-",ms=3,color="#176b9a")
                ax.axvline(d.get("load_h",1.5),color="grey",ls=":",lw=1)
                ax.set_xlim(0, t[-1]+.1)
                ax.set_title(d["names"][j],fontsize=10)
                ax.set_ylim(min(-3,np.min(errors[:,j])-.2),max(3,np.max(errors[:,j])+.2))
                ax.grid(alpha=.15)
            fig.suptitle(f"Latent estimation errors / reported SD · seed {seed}\n{policy_label} · {experimental}{status}",fontsize=12)
            fig.supxlabel("Hours since lighting. Shading is +/− 1.96 SD; one cook is not a coverage study.",fontsize=10)
            target=OUT/(stem+"_parameters.png")
            fig.savefig(target,dpi=160);plt.close(fig);print(target)
            fig,axs=plt.subplots((n+2)//3,3,figsize=(13,2.05*((n+2)//3)),sharex=True,layout="constrained")
            for j,ax in enumerate(axs.flat):
                if j>=n:
                    ax.set_visible(False)
                    continue
                ax.fill_between(t,values[:,j]-1.96*widths[:,j],values[:,j]+1.96*widths[:,j],color="#176b9a",alpha=.18,label="±1.96 SD")
                ax.plot(t,values[:,j],"o-",ms=3,color="#176b9a",label="Estimate")
                ax.plot(t,truth[:,j],"k--",lw=1.1,label="Truth")
                ax.axvline(d.get("load_h",1.5),color="grey",ls=":",lw=1)
                ax.set_xlim(0, t[-1]+.1)
                ax.set_title(d["names"][j],fontsize=10)
                ax.grid(alpha=.15)
            axs.flat[0].legend(fontsize=8)
            fig.suptitle(f"Parameter estimates and truth in estimator coordinates · seed {seed}\n{policy_label} · {experimental}{status}",fontsize=12)
            fig.supxlabel("Hours since lighting. Coordinates are logarithms for lognormal sites and Gaussian scores for score sites.",fontsize=10)
            target=OUT/(stem+"_parameter_estimates.png")
            fig.savefig(target,dpi=160);plt.close(fig);print(target)
        if all("processing_finish_s" in r for r in rows):
            fig, axs = plt.subplots(3, 1, figsize=(13, 9), layout="constrained")
            colors = dict(pit="#176b9a", core="#bc5317", humidity="#269477", mass="#7957a8")
            for k, (sensor, color) in enumerate(colors.items()):
                arrivals = [a["t_s"]/3600 for r in rows for a in r["arrivals"] if a["sensor"] == sensor]
                axs[0].scatter(arrivals, np.full(len(arrivals), k), marker="|", color=color, s=40)
            finish = np.array([r["processing_finish_s"]/3600 for r in rows])
            start = np.array([r["processing_start_s"]/3600 for r in rows])
            for a, b, r in zip(start, finish, rows):
                axs[0].plot([a,b], [4,4], color="#176b9a" if r["committed"] else "#ad3333", lw=3)
            axs[0].scatter(finish, np.full(len(rows), 4), s=10, color="black", label="Estimate available")
            new_nodes = [r["processing_start_s"]/3600 for r in rows if r["nodes_added"] > 0]
            axs[0].scatter(new_nodes, np.full(len(new_nodes), 5), marker="|", color="black", s=60)
            axs[0].set_yticks(range(6), [*colors, "Processing / available", "Graph node added"])
            axs[0].set_ylim(-.5,5.5)
            axs[0].legend(loc="lower right", bbox_to_anchor=(1, 1.01), frameon=False)
            for key, label in [("oldest_reading_latency_s", "Oldest reading"), ("newest_reading_latency_s", "Newest reading")]:
                axs[1].plot(finish, [r[key] for r in rows], ".-", label=label)
            axs[1].set_ylabel("Reading to estimate latency (s)"); axs[1].legend()
            axs[2].step(start, [r["n_readings"] for r in rows], where="post", label="Readings per update")
            axs[2].set_ylabel("Readings per update"); axs[2].set_xlabel("Hours since lighting")
            for ax in axs:
                ax.axvline(d.get("load_h",1.5), color="grey", ls=":", label="Load-transition midpoint")
                ax.set_xlim(0, max(finish.max(), tr.max())+.02)
                ax.grid(alpha=.15)
            fig.suptitle(f"{policy_label} · seed {seed}\n{status}", fontsize=13)
            fig.supxlabel("Each reading keeps its measurement time. Computation finish determines when an estimate is usable."
                           + timing_caveat, fontsize=10)
            target=OUT/(stem+"_timing.png")
            fig.savefig(target,dpi=160);plt.close(fig);print(target)

def plot_agreement():
    p=ROOT/"data/mock/truth_lighting_41002.json"
    d=json.loads(p.read_text()); rows=d["dynamics"]
    t=[r["time_h"] for r in rows]
    fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True,layout="constrained")
    for key,name in [("T_pit","Pit"),("T","Meat field"),("T_fb","Firebox")]:
        axs[0].semilogy(t,[max(r["group_max_physical_error"][key],1e-16) for r in rows],label=name)
    for key in rows[0]["group_whitened_mean_square"]:
        axs[1].semilogy(t,[max(r["group_whitened_mean_square"][key],1e-20) for r in rows],label=key)
    axs[0].set_ylabel("Max physical propagation error (K)"); axs[0].legend(ncol=3)
    axs[1].set_ylabel("Whitened residual mean square"); axs[1].legend(ncol=6,fontsize=8)
    axs[1].axhline(1,color="k",ls="--",lw=1)
    for ax in axs:
        ax.grid(alpha=.2)
        for h,label in [(1.5,"Load"),(12.5,"Wrap"),(13.5,"Fuel")]:
            ax.axvline(h,color="grey",ls=":",lw=1)
            if ax is axs[0]: ax.text(h,.97,label,transform=ax.get_xaxis_transform(),ha="right",va="top",fontsize=9)
    axs[1].set_xlabel("Hours since lighting")
    fig.suptitle("Plant / estimator propagation agreement over 18 hours · seed 41002")
    fig.supxlabel("Numerical model agreement only — this does not test posterior uncertainty or estimator convergence.",fontsize=10)
    target=OUT/"preheat_model_agreement.png"; fig.savefig(target,dpi=160); plt.close(fig); print(target)

if __name__ == "__main__":
    paths=[Path(p) for p in sys.argv[1:]] or [ROOT/"data/mock/stream_lighting_tune_cache_41002.json",ROOT/"data/mock/stream_lighting_censor_fixed_41002.json"]
    for p in paths: plot_run(p)
    plot_agreement()
