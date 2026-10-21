#!/usr/bin/env python3
"""
Stochastic simulation of gene editing (Promotion of Alleles by Genome Editing,
PAGE) embedded in a genomic-selection livestock breeding program.

Model overview
--------------
- Discrete generations, N=800 offspring per generation (400 male / 400 female).
- Genome: 5 chromosomes x 300 loci (150 additive production QTL, 90 fitness
  loci carrying segregating deleterious alleles, 60 neutral markers), plus a
  pool of 600 slots for de-novo off-target (OT) deleterious alleles.
- One crossover per chromosome per meiosis (approximation for ~1 Morgan
  chromosomes); OT loci inherit by free assortment (documented approximation).
- Selection: index I = w_p*GEBV_prod + w_f*GEBV_fit; GEBVs are true values
  plus noise calibrated to accuracy r (standard selection-index construction).
- Viability (natural selection): eligibility probability proportional to W.
- Editing: a fraction p_edit of selected sires is edited; each known QTL is
  converted to the favourable homozygote with probability q_edit; each edited
  sire accrues Poisson(lambda) off-target deleterious alleles.
- Pleiotropy: each copy of a favourable known-QTL allele costs s_p in
  log-fitness when enabled.

All scenarios within a replicate share the same founder population (common
random numbers) for fair comparison.
"""
import json
import os
import sys
import time
from dataclasses import dataclass, replace, asdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# ----------------------------------------------------------------------------
# Genome layout
# ----------------------------------------------------------------------------
NCHR = 5
GROUPS = 6          # interleaved prod/fit groups per chromosome
PQ_G = 25           # production QTL per group  -> 150 per chromosome
FQ_G = 15           # fitness loci per group    ->  90 per chromosome
NEU = 60            # neutral loci per chromosome
L_CHR = GROUPS * (PQ_G + FQ_G) + NEU          # 300
L0 = NCHR * L_CHR                             # 1500
OT_MAX = 600
L = L0 + OT_MAX                               # 2100

CLS_PROD, CLS_FIT, CLS_NEU, CLS_OT = 0, 1, 2, 3


def build_layout():
    cls = np.full(L, CLS_OT, dtype=np.int8)
    for c in range(NCHR):
        base = c * L_CHR
        for g in range(GROUPS):
            s = base + g * (PQ_G + FQ_G)
            cls[s:s + PQ_G] = CLS_PROD
            cls[s + PQ_G:s + PQ_G + FQ_G] = CLS_FIT
        cls[base + GROUPS * (PQ_G + FQ_G): base + L_CHR] = CLS_NEU
    return cls


CLS = build_layout()
PROD_IDX = np.where(CLS == CLS_PROD)[0]
FIT_IDX = np.where(CLS == CLS_FIT)[0]
NEU_IDX = np.where(CLS == CLS_NEU)[0]


# ----------------------------------------------------------------------------
# Scenario definition
# ----------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str = "GS"
    N: int = 800
    Ns: int = 20            # sires selected per generation
    Nd: int = 320           # dams selected per generation
    gens: int = 25
    mu: float = 2e-4        # deleterious mutation rate / fitness locus / gamete
    p_edit: float = 0.0     # fraction of selected sires edited
    q_edit: float = 0.9     # per-known-QTL editing success probability
    K: int = 6              # number of known (editable) QTL
    lam: float = 0.0        # Poisson mean of off-target events per edited sire
    s_p: float = 0.0        # pleiotropic log-fitness cost per favourable copy
    r_p: float = 0.65       # accuracy of genomic EBV for production
    r_f: float = 0.35       # accuracy of EBV for fitness
    w_p: float = 1.0        # index weight on production
    w_f: float = 0.0        # index weight on fitness
    edit_start: int = 1     # first generation in which sires are edited
    clone_donors: int = 0   # >0: edited sires are clones of this many donors


# ----------------------------------------------------------------------------
# Founder population
# ----------------------------------------------------------------------------
def make_founder(seed, K_max=9):
    rng = np.random.default_rng(seed)
    sc = Scenario()
    N = sc.N
    a = np.zeros(L)
    d = np.zeros(L)
    fav_sign = np.ones(L, dtype=np.int8)

    # production QTL effects: gamma-distributed |a| with random sign
    a[PROD_IDX] = (rng.gamma(0.4, 1.66, len(PROD_IDX))
                   * rng.choice([-1.0, 1.0], len(PROD_IDX)))
    fav_sign[PROD_IDX] = np.where(a[PROD_IDX] > 0, 1, -1)

    # fitness-locus deleterious effects (founder load ~1.6 lethal equivalents)
    d[FIT_IDX] = -np.minimum(rng.exponential(0.06, len(FIT_IDX)), 0.4)

    # initial allele frequencies
    p = np.zeros(L)
    p[PROD_IDX] = rng.uniform(0.10, 0.90, len(PROD_IDX))
    p[FIT_IDX] = rng.uniform(0.01, 0.05, len(FIT_IDX))
    p[NEU_IDX] = rng.uniform(0.10, 0.90, len(NEU_IDX))

    hap = rng.binomial(1, p[None, None, :], (N, 2, L)).astype(np.int8)

    # scale production effects to unit founder additive SD
    g = hap[:, 0, :].astype(np.int16) + hap[:, 1, :]
    sd0 = float(np.std(g @ a))
    a /= sd0

    # known (editable) QTL: largest |a| whose favourable allele is rare (<0.3)
    fav_af = np.where(a[PROD_IDX] > 0, p[PROD_IDX], 1 - p[PROD_IDX])
    order = np.argsort(-np.abs(a[PROD_IDX]))
    cand = [PROD_IDX[i] for i in order if fav_af[i] < 0.30][:K_max]
    known = np.array(cand, dtype=int)

    # linked deleterious locus: nearest fitness locus (same chromosome) to each
    # known QTL; forced to a moderate deleterious effect at 5% frequency
    linked = np.zeros(len(known), dtype=int)
    for j, k in enumerate(known):
        same_chr_fit = FIT_IDX[(FIT_IDX // L_CHR) == (k // L_CHR)]
        near = same_chr_fit[np.argmin(np.abs(same_chr_fit - k))]
        linked[j] = near
        d[near] = -0.06
        hap[:, :, near] = rng.binomial(1, 0.05, (N, 2)).astype(np.int8)

    return dict(hap=hap, a=a, d=d, fav_sign=fav_sign, known=known,
                linked=linked)


# ----------------------------------------------------------------------------
# One simulation run
# ----------------------------------------------------------------------------
METRICS = ["mean_tbv", "sd_tbv", "mean_w", "h_neu", "load", "pleio",
           "ot_n", "kfav", "ldel"]


def run(sc, founder_seed):
    f = make_founder(founder_seed)
    rng = np.random.default_rng(founder_seed * 7919 + 17)
    N = sc.N
    hap = f["hap"].copy()
    a, d = f["a"].copy(), f["d"].copy()
    fav_sign = f["fav_sign"]
    known = f["known"][:sc.K]
    linked = f["linked"][:sc.K]
    fitmask_l0 = (CLS[:L0] == CLS_FIT)
    ot_ptr = 0

    out = {k: np.zeros(sc.gens) for k in METRICS}
    nm = N // 2  # number of males

    for t in range(sc.gens):
        g = hap[:, 0, :].astype(np.int16) + hap[:, 1, :]
        tbv = g @ a
        logw = g @ d
        fav_copies_l = np.where(fav_sign[known] > 0, g[:, known],
                                2 - g[:, known]) if len(known) else np.zeros((N, 0))
        fav_copies = fav_copies_l.sum(axis=1)
        logw = logw - sc.s_p * fav_copies
        W = np.exp(logw)

        # ---- record metrics (pre-selection population) ----
        out["mean_tbv"][t] = tbv.mean()
        out["sd_tbv"][t] = tbv.std()
        out["mean_w"][t] = W.mean()
        out["h_neu"][t] = (g[:, NEU_IDX] == 1).mean()
        out["load"][t] = -(g @ d).mean()
        out["pleio"][t] = sc.s_p * fav_copies.mean()
        if ot_ptr > 0:
            out["ot_n"][t] = g[:, L0:L0 + ot_ptr].sum(axis=1).mean()
        out["kfav"][t] = fav_copies.mean() / (2.0 * max(len(known), 1))
        out["ldel"][t] = g[:, linked].mean() / 2.0

        # ---- breeding values and selection index ----
        zt = (tbv - tbv.mean()) / (tbv.std() + 1e-12)
        zw = (logw - logw.mean()) / (logw.std() + 1e-12)
        gebv_p = sc.r_p * zt + np.sqrt(1 - sc.r_p ** 2) * rng.standard_normal(N)
        gebv_f = sc.r_f * zw + np.sqrt(1 - sc.r_f ** 2) * rng.standard_normal(N)
        idx = sc.w_p * gebv_p + sc.w_f * gebv_f

        # ---- natural + artificial selection: among the top-index candidates,
        # parents are sampled with probability proportional to fitness W ----
        males = np.arange(nm)
        females = np.arange(nm, N)
        msort = males[np.argsort(-idx[males])]
        fsort = females[np.argsort(-idx[females])]
        mc = msort[:min(4 * sc.Ns, len(msort))]
        fc = fsort[:min(sc.Nd + 60, len(fsort))]
        pw = W[mc] / W[mc].sum()
        sires = rng.choice(mc, sc.Ns, replace=False, p=pw)
        pw = W[fc] / W[fc].sum()
        dams = rng.choice(fc, sc.Nd, replace=False, p=pw)

        # ---- gene editing of sires ----
        if sc.p_edit > 0 and t >= sc.edit_start and sc.clone_donors > 0:
            order = sires[np.argsort(-idx[sires])]
            donors = order[:sc.clone_donors]
            for dnr in donors:
                if rng.random() < sc.p_edit:
                    for k in known:
                        if rng.random() < sc.q_edit:
                            hap[dnr, :, k] = 1 if fav_sign[k] > 0 else 0
                    for _ in range(rng.poisson(sc.lam)):
                        if ot_ptr >= OT_MAX:
                            break
                        slot = L0 + ot_ptr
                        d[slot] = -min(rng.exponential(0.08), 0.5)
                        hap[dnr, rng.integers(0, 2), slot] = 1
                        ot_ptr += 1
            for c in order[sc.clone_donors:]:
                dnr = donors[rng.integers(0, len(donors))]
                hap[c] = hap[dnr]
        elif sc.p_edit > 0 and t >= sc.edit_start:
            for s in sires:
                if rng.random() < sc.p_edit:
                    for k in known:
                        if rng.random() < sc.q_edit:
                            hap[s, :, k] = 1 if fav_sign[k] > 0 else 0
                    n_ot = rng.poisson(sc.lam)
                    for _ in range(n_ot):
                        if ot_ptr >= OT_MAX:
                            break
                        slot = L0 + ot_ptr
                        d[slot] = -min(rng.exponential(0.08), 0.5)
                        hap[s, rng.integers(0, 2), slot] = 1
                        ot_ptr += 1

        # ---- reproduction ----
        sp = sires[rng.integers(0, len(sires), N)]
        dp = dams[rng.integers(0, len(dams), N)]
        nh = np.empty((N, 2, L), dtype=np.int8)
        for par, slot in ((sp, 0), (dp, 1)):
            for c in range(NCHR):
                lo, hi = c * L_CHR, (c + 1) * L_CHR
                s0 = rng.integers(0, 2, N)
                b = rng.integers(1, L_CHR, N)
                mask = (np.arange(L_CHR)[None, :] >= b[:, None])
                h0 = hap[par, s0, lo:hi]
                h1 = hap[par, 1 - s0, lo:hi]
                nh[:, slot, lo:hi] = np.where(mask, h1, h0)
            choose = rng.random((N, L - L0)) < 0.5
            nh[:, slot, L0:] = np.where(choose, hap[par, 1, L0:],
                                        hap[par, 0, L0:])
        # deleterious mutation at fitness loci
        m = (rng.random((N, 2, L0)) < sc.mu) & fitmask_l0[None, None, :]
        nh_view = nh[:, :, :L0]
        nh_view[m] = 1
        hap = nh

    # final-generation extras
    g = hap[:, 0, :].astype(np.int16) + hap[:, 1, :]
    fav = np.where(fav_sign[known] > 0, g[:, known], 2 - g[:, known])
    fix_known = float((fav.mean(axis=0) >= 1.98).mean()) if len(known) else 0.0
    out["_final"] = dict(fix_known=fix_known, ot_slots=int(ot_ptr))
    return out


# ----------------------------------------------------------------------------
# Scenarios and driver
# ----------------------------------------------------------------------------
SCEN = {
    "S1": Scenario(name="GS", p_edit=0.0),
    "S2": Scenario(name="PAGE-mod", p_edit=0.25, K=3),
    "S3": Scenario(name="PAGE-agg", p_edit=0.90, K=6),
    "S4": Scenario(name="PAGE+OT", p_edit=0.90, K=6, lam=0.15),
    "S5": Scenario(name="PAGE+PL", p_edit=0.90, K=6, s_p=0.04),
    "S6": Scenario(name="PAGE+OT+PL", p_edit=0.90, K=6, lam=0.15, s_p=0.04),
    "S7": Scenario(name="PAGE-mit", p_edit=0.90, K=6, lam=0.15, s_p=0.04,
                   w_p=0.7, w_f=0.3, r_f=0.50),
    "S8": Scenario(name="PAGE-delay", p_edit=0.90, K=6, lam=0.15, s_p=0.04,
                   edit_start=10),
    "S9": Scenario(name="PAGE-clone", p_edit=0.90, K=6, lam=0.15, s_p=0.04,
                   clone_donors=3),
}

SENS = {  # one-at-a-time perturbations of S6
    "r_p=0.50": replace(SCEN["S6"], r_p=0.50),
    "r_p=0.80": replace(SCEN["S6"], r_p=0.80),
    "p_edit=0.30": replace(SCEN["S6"], p_edit=0.30),
    "p_edit=0.60": replace(SCEN["S6"], p_edit=0.60),
    "lam=0.05": replace(SCEN["S6"], lam=0.05),
    "lam=0.30": replace(SCEN["S6"], lam=0.30),
    "s_p=0.02": replace(SCEN["S6"], s_p=0.02),
    "s_p=0.08": replace(SCEN["S6"], s_p=0.08),
    "Ns=10": replace(SCEN["S6"], Ns=10),
    "Ns=40": replace(SCEN["S6"], Ns=40),
    "K=3": replace(SCEN["S6"], K=3),
    "K=9": replace(SCEN["S6"], K=9),
}

FOUNDER_SEED = 7000
N_REP_MAIN = 50
N_REP_SENS = 24


def _task_main(args):
    key, rep = args
    return key, rep, run(SCEN[key], FOUNDER_SEED + rep)


def _task_sens(args):
    key, rep = args
    return key, rep, run(SENS[key], FOUNDER_SEED + rep)


def main(quick=False):
    t0 = time.time()
    n_rep_main = 3 if quick else N_REP_MAIN
    n_rep_sens = 2 if quick else N_REP_SENS
    gens = 5 if quick else 25
    for k in SCEN:
        SCEN[k] = replace(SCEN[k], gens=gens)
    for k in SENS:
        SENS[k] = replace(SENS[k], gens=gens)

    tasks = [(k, r) for k in SCEN for r in range(n_rep_main)]
    tasks_s = [(k, r) for k in SENS for r in range(n_rep_sens)]
    results, results_s = {}, {}
    workers = 1 if quick else min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for key, rep, out in ex.map(_task_main, tasks):
            results.setdefault(key, {})[rep] = out
        for key, rep, out in ex.map(_task_sens, tasks_s):
            results_s.setdefault(key, {})[rep] = out

    os.makedirs("results", exist_ok=True)
    # main results -> npz  (arrays shaped [rep, gen])
    pack = {}
    finals = {}
    for key, reps in results.items():
        rids = sorted(reps)
        for m in METRICS:
            pack[f"{key}__{m}"] = np.stack([reps[r][m] for r in rids])
        finals[key] = [reps[r]["_final"] for r in rids]
    np.savez("results/main.npz", **pack)

    spack = {}
    for key, reps in results_s.items():
        rids = sorted(reps)
        for m in ["mean_tbv", "mean_w", "h_neu", "load"]:
            spack[f"{key}__{m}"] = np.stack([reps[r][m] for r in rids])
    np.savez("results/sens.npz", **spack)

    meta = dict(scenarios={k: asdict(v) for k, v in SCEN.items()},
                sens={k: asdict(v) for k, v in SENS.items()},
                n_rep_main=n_rep_main, n_rep_sens=n_rep_sens,
                founder_seed=FOUNDER_SEED, finals=finals,
                runtime_sec=round(time.time() - t0, 1))
    with open("results/meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"done in {time.time()-t0:.1f}s -> results/main.npz, results/sens.npz")


if __name__ == "__main__":
    main(quick=("--quick" in sys.argv))
