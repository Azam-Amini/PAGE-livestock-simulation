#!/usr/bin/env python3
"""
make_figures.py
Regenerate all main-text figures (1-12) and summary tables from the
archived result arrays (results/main.npz, results/sens.npz).

All quantities are expressed in the units used in the manuscript:
- cumulative gain: mean_tbv(t) - mean_tbv(0)
- total genetic load: load + pleio
- fitness: mean_w(t) / mean_w(0)   (relative to founders)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "results"
FIGDIR = os.path.join(RESULTS, "figures")
TABDIR = os.path.join(RESULTS, "tables")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

# ----------------------------------------------------------------------
# Scenario definitions
# ----------------------------------------------------------------------
SCEN_LABEL = {
    "S1": "GS",
    "S2": "PAGE-mod",
    "S3": "PAGE-agg",
    "S4": "PAGE+OT",
    "S5": "PAGE+PL",
    "S6": "PAGE+OT+PL",
    "S7": "PAGE-mit",
    "S8": "PAGE-delay",
    "S9": "PAGE-clone",
}
SCEN_ORDER = list(SCEN_LABEL.keys())
COLORS = {
    "S1": "#000000",
    "S2": "#1f77b4",
    "S3": "#2ca02c",
    "S4": "#ff7f0e",
    "S5": "#9467bd",
    "S6": "#d62728",
    "S7": "#17becf",
    "S8": "#8c564b",
    "S9": "#e377c2",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def mean_ci(arr, axis=0, z=1.96):
    m = arr.mean(axis=axis)
    sd = arr.std(axis=axis, ddof=1)
    n = arr.shape[axis]
    return m, z * sd / np.sqrt(n)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png/.pdf")


def gain_array(d, scen):
    tbv = d[f"{scen}__mean_tbv"]
    return tbv - tbv[:, [0]]


def fitness_relative(d, scen):
    w = d[f"{scen}__mean_w"]
    return w / w[:, [0]]


def total_load(d, scen):
    return d[f"{scen}__load"] + d[f"{scen}__pleio"]


def _plot_traj(ax, arr, scen):
    m, ci = mean_ci(arr, axis=0)
    c = COLORS[scen]
    ax.plot(X, m, color=c, lw=1.6, label=SCEN_LABEL[scen])
    ax.fill_between(X, m - ci, m + ci, color=c, alpha=0.13, lw=0)


# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
print("Loading result arrays ...")
main = np.load(os.path.join(RESULTS, "main.npz"))
sens = np.load(os.path.join(RESULTS, "sens.npz"))
NGEN = main["S1__mean_tbv"].shape[1]
X = np.arange(NGEN)
NREP = main["S1__mean_tbv"].shape[0]
print(f"  main: {NREP} replicates x {NGEN} generations, {len(SCEN_ORDER)} scenarios")

# ----------------------------------------------------------------------
# Figure 1 - cumulative genetic gain
# ----------------------------------------------------------------------
print("Figure 1: cumulative gain")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for s in SCEN_ORDER:
    _plot_traj(ax, gain_array(main, s), s)
ax.set_xlabel("Generation")
ax.set_ylabel(r"Cumulative genetic gain ($\sigma_{A0}$ units)")
ax.set_xlim(0, NGEN - 1)
ax.legend(ncol=2, frameon=False, loc="upper left")
save(fig, "fig01_gain")

# ----------------------------------------------------------------------
# Figure 2 - favourable-allele frequency
# ----------------------------------------------------------------------
print("Figure 2: favourable-allele frequency")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for s in ["S1", "S2", "S3", "S6", "S7", "S9"]:
    _plot_traj(ax, main[f"{s}__kfav"], s)
ax.set_xlabel("Generation")
ax.set_ylabel("Favourable-allele frequency at known QTL")
ax.set_xlim(0, NGEN - 1)
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, loc="lower right")
save(fig, "fig02_kfav")

# ----------------------------------------------------------------------
# Figure 3 - off-target alleles per individual
# ----------------------------------------------------------------------
print("Figure 3: off-target load")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for s in ["S4", "S6", "S7", "S8", "S9"]:
    _plot_traj(ax, main[f"{s}__ot_n"], s)
ax.set_xlabel("Generation")
ax.set_ylabel("Off-target alleles per individual")
ax.set_xlim(0, NGEN - 1)
ax.legend(ncol=2, frameon=False, loc="upper left")
save(fig, "fig03_ot")

# ----------------------------------------------------------------------
# Figure 4 - total genetic load (segregating + off-target + pleiotropic)
# ----------------------------------------------------------------------
print("Figure 4: genetic load")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for s in SCEN_ORDER:
    _plot_traj(ax, total_load(main, s), s)
ax.set_xlabel("Generation")
ax.set_ylabel("Total genetic load (lethal equivalents)")
ax.set_xlim(0, NGEN - 1)
ax.legend(ncol=2, frameon=False, loc="upper right")
save(fig, "fig04_load")

# ----------------------------------------------------------------------
# Figure 5 - linked deleterious alleles
# ----------------------------------------------------------------------
print("Figure 5: linked deleterious alleles")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for s in ["S1", "S3", "S6", "S7", "S9"]:
    _plot_traj(ax, main[f"{s}__ldel"], s)
ax.set_xlabel("Generation")
ax.set_ylabel("Frequency of linked deleterious alleles")
ax.set_xlim(0, NGEN - 1)
ax.legend(frameon=False, loc="lower left")
save(fig, "fig05_ldel")

# ----------------------------------------------------------------------
# Figure 6 - neutral-marker heterozygosity
# ----------------------------------------------------------------------
print("Figure 6: neutral diversity")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for s in SCEN_ORDER:
    _plot_traj(ax, main[f"{s}__h_neu"], s)
ax.set_xlabel("Generation")
ax.set_ylabel("Neutral-locus heterozygosity")
ax.set_xlim(0, NGEN - 1)
ax.legend(ncol=2, frameon=False, loc="lower left")
save(fig, "fig06_hneu")

# ----------------------------------------------------------------------
# Figure 7 - mean fitness relative to founders
# ----------------------------------------------------------------------
print("Figure 7: mean fitness")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for s in SCEN_ORDER:
    _plot_traj(ax, fitness_relative(main, s), s)
ax.axhline(1.0, color="grey", ls=":", lw=1.0)
ax.set_xlabel("Generation")
ax.set_ylabel("Mean fitness relative to founders")
ax.set_xlim(0, NGEN - 1)
ax.legend(ncol=2, frameon=False, loc="lower right")
save(fig, "fig07_fitness")

# ----------------------------------------------------------------------
# Figure 8 - fitness relative to the GS control
# ----------------------------------------------------------------------
print("Figure 8: fitness relative to control")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ref = fitness_relative(main, "S1")
for s in SCEN_ORDER[1:]:
    ratio = fitness_relative(main, s) / ref
    m, ci = mean_ci(ratio, axis=0)
    c = COLORS[s]
    ax.plot(X, m, color=c, lw=1.6, label=SCEN_LABEL[s])
    ax.fill_between(X, m - ci, m + ci, color=c, alpha=0.13, lw=0)
ax.axhline(1.0, color="grey", ls=":", lw=1.0)
ax.set_xlabel("Generation")
ax.set_ylabel("Mean fitness relative to GS control")
ax.set_xlim(0, NGEN - 1)
ax.legend(ncol=2, frameon=False, loc="lower left")
save(fig, "fig08_fitness_ratio")

# ----------------------------------------------------------------------
# Figure 9 - replicate-level gain-fitness plane at final generation
# ----------------------------------------------------------------------
print("Figure 9: gain-fitness plane")
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for s in SCEN_ORDER:
    g = gain_array(main, s)[:, -1]
    w = fitness_relative(main, s)[:, -1]
    ax.scatter(g, w, s=14, color=COLORS[s], alpha=0.65,
               edgecolors="none", label=SCEN_LABEL[s])
ax.set_xlabel(r"Cumulative genetic gain at generation 24 ($\sigma_{A0}$ units)")
ax.set_ylabel("Mean fitness relative to founders")
ax.legend(ncol=2, frameon=False, loc="lower right", markerscale=1.4)
save(fig, "fig09_gain_fitness")

# ----------------------------------------------------------------------
# Figure 10 - tail probabilities
# ----------------------------------------------------------------------
print("Figure 10: tail probabilities")
ref_w = fitness_relative(main, "S1")[:, -1]
ref_gain_median = np.median(gain_array(main, "S1")[:, -1])
defs = ["P(W<0.95)", "P(W<0.90)", "P(W<W_GS)", "P(gain<GS median)"]
P = np.zeros((len(SCEN_ORDER), len(defs)))
for i, s in enumerate(SCEN_ORDER):
    w = fitness_relative(main, s)[:, -1]
    g = gain_array(main, s)[:, -1]
    P[i, 0] = np.mean(w < 0.95)
    P[i, 1] = np.mean(w < 0.90)
    P[i, 2] = np.mean(w < ref_w)
    P[i, 3] = np.mean(g < ref_gain_median)

fig, ax = plt.subplots(figsize=(7.0, 4.0))
w_bar = 0.09
idx = np.arange(len(defs))
for i, s in enumerate(SCEN_ORDER):
    ax.bar(idx + (i - 4) * w_bar, P[i], width=w_bar,
           color=COLORS[s], label=SCEN_LABEL[s], edgecolor="none")
ax.set_xticks(idx)
ax.set_xticklabels(defs, rotation=12, ha="right")
ax.set_ylabel("Probability across replicates")
ax.set_ylim(0, 1.05)
ax.legend(ncol=3, frameon=False, loc="upper left", fontsize=7)
save(fig, "fig10_tail")

# ----------------------------------------------------------------------
# Figures 11 & 12 - tornado sensitivity
# ----------------------------------------------------------------------
print("Figures 11-12: sensitivity tornado charts")
PARAMS = [
    ("r_p",   ["r_p=0.50",   "r_p=0.80"]),
    ("p_edit",["p_edit=0.30","p_edit=0.60"]),
    ("lam",   ["lam=0.05",   "lam=0.30"]),
    ("s_p",   ["s_p=0.02",   "s_p=0.08"]),
    ("Ns",    ["Ns=10",      "Ns=40"]),
    ("K",     ["K=3",        "K=9"]),
]
PRETTY = {"r_p": "r_p", "p_edit": "p_edit", "lam": r"$\lambda$",
          "s_p": "s_p", "Ns": "Ns", "K": "K"}

base_gain = gain_array(main, "S6")[:, -1].mean()
base_fit  = fitness_relative(main, "S6")[:, -1].mean()

rows = []
for key, variants in PARAMS:
    for v in variants:
        g = gain_array(sens, v)[:, -1].mean()
        w = fitness_relative(sens, v)[:, -1].mean()
        rows.append({"parameter": key, "variant": v,
                     "gain": g, "fitness": w,
                     "d_gain": g - base_gain,
                     "d_fitness": w - base_fit})
    print(f"  {key}: done")

with open(os.path.join(TABDIR, "sensitivity.csv"), "w") as fh:
    fh.write("parameter,variant,gain_gen24,fitness_gen24,d_gain,d_fitness\n")
    for r in rows:
        fh.write(f"{r['parameter']},{r['variant']},{r['gain']:.4f},"
                 f"{r['fitness']:.4f},{r['d_gain']:.4f},{r['d_fitness']:.4f}\n")

def tornado(ax, values_by_param, xlabel):
    for i, (key, _) in enumerate(PARAMS):
        lo, hi = values_by_param[key]
        ax.barh(i, lo, color="#4c72b0", alpha=0.85, height=0.55)
        ax.barh(i, hi, color="#4c72b0", alpha=0.85, height=0.55)
        ax.text(lo, i, f"{lo:+.2f}", va="center", ha="right", fontsize=8)
        ax.text(hi, i, f"{hi:+.2f}", va="center", ha="left", fontsize=8)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(np.arange(len(PARAMS)))
    ax.set_yticklabels([PRETTY[p[0]] for p in PARAMS])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)

gv = {}
for key, _ in PARAMS:
    vals = [r["d_gain"] for r in rows if r["parameter"] == key]
    gv[key] = (min(vals), max(vals))
fig, ax = plt.subplots(figsize=(6.4, 3.8))
tornado(ax, gv, r"Change in gain at generation 24 vs. PAGE+OT+PL ($\sigma_{A0}$)")
save(fig, "fig11_tornado_gain")

fv = {}
for key, _ in PARAMS:
    vals = [r["d_fitness"] for r in rows if r["parameter"] == key]
    fv[key] = (min(vals), max(vals))
fig, ax = plt.subplots(figsize=(6.4, 3.8))
tornado(ax, fv, "Change in relative fitness at generation 24 vs. PAGE+OT+PL")
save(fig, "fig12_tornado_fit")

# ----------------------------------------------------------------------
# Control tables
# ----------------------------------------------------------------------
print("Summary tables")

with open(os.path.join(TABDIR, "table3_recomputed.csv"), "w") as fh:
    fh.write("scenario,gain_gen24_mean,gain_sd,fitness_mean,fitness_sd,"
             "load_mean,load_sd,h_neu_mean,h_neu_sd,kfav_mean,kfav_sd,"
             "ot_mean,ot_sd\n")
    for s in SCEN_ORDER:
        g = gain_array(main, s)[:, -1]
        w = fitness_relative(main, s)[:, -1]
        L = total_load(main, s)[:, -1]
        h = main[f"{s}__h_neu"][:, -1]
        k = main[f"{s}__kfav"][:, -1]
        o = main[f"{s}__ot_n"][:, -1]
        fh.write(f"{SCEN_LABEL[s]},{g.mean():.3f},{g.std(ddof=1):.3f},"
                 f"{w.mean():.3f},{w.std(ddof=1):.3f},"
                 f"{L.mean():.3f},{L.std(ddof=1):.3f},"
                 f"{h.mean():.4f},{h.std(ddof=1):.4f},"
                 f"{k.mean():.3f},{k.std(ddof=1):.3f},"
                 f"{o.mean():.3f},{o.std(ddof=1):.3f}\n")

with open(os.path.join(TABDIR, "tail_probabilities.csv"), "w") as fh:
    fh.write("scenario," + ",".join(defs) + "\n")
    for i, s in enumerate(SCEN_ORDER):
        fh.write(SCEN_LABEL[s] + "," + ",".join(f"{v:.3f}" for v in P[i]) + "\n")

print("\nAll outputs written to:")
print(f"  {os.path.abspath(FIGDIR)}")
print(f"  {os.path.abspath(TABDIR)}")
print("Done.")