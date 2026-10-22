"""Regenerate camera-ready tables and figures from archived experiment records.

Run from any directory. No model training, network access, or raw tissue data needed.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
OUT = ROOT / "generated"
OUT.mkdir(exist_ok=True)


def read(relative):
    return json.loads((EVIDENCE / relative).read_text())


def rows(folder, pattern):
    return [r for p in sorted((EVIDENCE / folder).glob(pattern))
            for r in json.loads(p.read_text())]


def select(records, protocol, selector, budget=None):
    return [r for r in records if r["protocol"] == protocol
            and r["selector"] == selector
            and (budget is None or r.get("budget") == budget)]


def mean(records, key):
    return float(np.mean([r[key] for r in records]))


def paired(records, a, b, budget):
    groups = []
    for selector in (a, b):
        by_section = defaultdict(list)
        for r in select(records, "transfer_gcn", selector, budget):
            by_section[r["section"]].append(r["macro_f1"])
        assert len(by_section) == 12
        assert all(len(v) == 3 for v in by_section.values())
        groups.append({k: np.mean(v) for k, v in by_section.items()})
    keys = sorted(groups[0])
    assert keys == sorted(groups[1])
    difference = np.array([groups[0][k] - groups[1][k] for k in keys])
    return float(wilcoxon(difference).pvalue), int((difference > 0).sum())


dlpfc = rows("dlpfc_refit", "unified_*.json")
mouse = rows("mouse_refit", "records_seed*.json")
assert len({r["section"] for r in dlpfc}) == 12
assert len({r["seed"] for r in mouse}) == 10
assert len({(r["section"], r["seed"], r["budget"], r["protocol"], r["selector"])
            for r in dlpfc}) == len(dlpfc)
assert len({(r["seed"], r["protocol"], r["selector"]) for r in mouse}) == len(mouse)

# TSB-dual was refit later with dual_refit.py, reusing each archived run's split, seed
# and k. Its re-run random selector must reproduce the archived random score exactly.
dual_runs = rows("dual_refit", "dlpfc_*.json") + rows("dual_refit", "mouse_seed*.json")
checks = [r for r in dual_runs if r["selector"] == "random_check"]
assert len(checks) == 12 * 3 * 3 + 10
max_check_diff = max(abs(r["macro_f1"] - r["archived_macro_f1"]) for r in checks)
assert max_check_diff < 1e-6, max_check_diff
holm_comparators = ["graphmask", "gnnexplainer", "degree", "gradient", "random",
                    "tsb_submitted", "attention"]
dlpfc += [r for r in dual_runs if r["task"] == "dlpfc" and r["selector"] == "tsb_dual"]
mouse += [r for r in dual_runs if r["task"] == "mouse" and r["selector"] == "tsb_dual"]

names = {
    "tsb_final": "TSB-cov", "tsb_dual": "TSB-dual", "graphmask": "GraphMask adaptation",
    "gnnexplainer": "GNNExplainer", "degree": "Degree", "gradient": "Gradient",
    "random": "Random", "tsb_submitted": "TSB-fixed", "attention": "Attention",
}
table = []
for selector, name in sorted(names.items(),
                             key=lambda kv: -mean(select(dlpfc, "transfer_gcn", kv[0], .2), "macro_f1")):
    rr = select(dlpfc, "transfer_gcn", selector, .2)
    p = "---" if selector == "random" else f"{paired(dlpfc, selector, 'random', .2)[0]:.4f}"
    table.append(f"{name} & {mean(rr, 'macro_f1'):.4f} & "
                 f"{np.std([r['macro_f1'] for r in rr], ddof=1):.4f} & "
                 f"{mean(rr, 'frac_isolated'):.3f} & {p} \\")
(OUT / "transfer_rows.tex").write_text("\n".join(s + "\\" for s in table) + "\n")

# Holm adjustment over final-versus-seven-other-selectors at three budgets.
tests = []
for budget in (.1, .2, .3):
    for comparator in holm_comparators:
        p, wins = paired(dlpfc, "tsb_final", comparator, budget)
        tests.append(dict(budget=budget, comparator=comparator, p=p, wins=wins))
order = np.argsort([t["p"] for t in tests])
running = 0.0
for rank, idx in enumerate(order):
    running = max(running, min(1.0, (len(tests) - rank) * tests[idx]["p"]))
    tests[idx]["p_holm"] = running

budget_rows = []
for budget in (.1, .2, .3):
    final = select(dlpfc, "transfer_gcn", "tsb_final", budget)
    gm = select(dlpfc, "transfer_gcn", "graphmask", budget)
    rnd = select(dlpfc, "transfer_gcn", "random", budget)
    dual = select(dlpfc, "transfer_gcn", "tsb_dual", budget)
    own = select(dlpfc, "end_to_end", "tsb_final", budget)
    p = paired(dlpfc, "tsb_final", "random", budget)[0]
    budget_rows.append(f"{budget:.2f} & {mean(final,'edge_rate'):.3f} & "
                       f"{mean(own,'macro_f1'):.3f} & {mean(final,'macro_f1'):.3f} & "
                       f"{mean(dual,'macro_f1'):.3f} & {mean(gm,'macro_f1'):.3f} & "
                       f"{mean(rnd,'macro_f1'):.3f} & {p:.4f} \\")
(OUT / "budget_rows.tex").write_text("\n".join(s + "\\" for s in budget_rows) + "\n")

analyses = [read(p) for p in ("consensus_fixed_penalty/analysis.json",
                            "consensus_dual_only/analysis.json", "consensus_final/analysis.json")]
final_analysis = analyses[-1]
targets = final_analysis["target_order"]
labels = ["Layer label", "Excitatory", "Oligodendrocyte", "Astrocyte",
          "Microglia", "L1 markers", "Deep layer"]
stability_rows = []
dual_summary = analyses[1]["summary"]
for t, label in zip(targets, labels):
    c, d = final_analysis["summary"][t], dual_summary[t]
    stability_rows.append(f"{label} & {c['performance']:.2f} & {c['consensus_reliability']:.2f} & "
                          f"{c['consensus_reliability_residual']:.2f} & {c['corr_with_default']:.2f} & "
                          f"{d['performance']:.2f} & {d['consensus_reliability']:.2f} & "
                          f"{d['corr_with_default']:.2f} \\")
(OUT / "stability_rows.tex").write_text("\n".join(s + "\\" for s in stability_rows) + "\n")

# One visual system for every figure: typography, variant colors, and colormaps.
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, RULE = "#262626", "#6B6B6B", "#BDBDBD"
COV, DUAL, FIXED = "#1F5A8B", "#D9822B", "#8E7BA6"      # TSB-cov, TSB-dual, TSB-fixed
GRAPHMASK, RANDOM = "#4E9A6F", "#8C8C8C"
NEG, POS = "#2B6CA3", "#C2493D"
DIVERGING = LinearSegmentedColormap.from_list(
    "signed", ["#173F66", NEG, "#9EC3E0", "#F7F7F7", "#F0B3A6", POS, "#7A221B"])
DIVERGING.set_bad("#E9E9E9")
SEQUENTIAL = LinearSegmentedColormap.from_list("magnitude", ["#F5F5F5", "#BDBDBD", "#6B6B6B", "#1F1F1F"])
LAYER_COLORS = ["#FBE3A6", "#F4BE5F", "#E5913A", "#C8642A", "#9A4222", "#5E2716", "#C9CDD2"]
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.1,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})


def panel_title(ax, letter, text, color=INK):
    ax.set_title(f"$\\bf{{{letter}}}$   {text}", loc="left", color=color, pad=3)


def colorbar(fig, mappable, cax, label, ticks):
    bar = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    bar.set_label(label, fontsize=7, labelpad=1.5)
    bar.outline.set_linewidth(.4)
    bar.ax.tick_params(labelsize=6.5, length=1.5, pad=1)
    return bar


def sem(values):
    values = np.asarray(values, float)
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def section_means(records, key):
    """Seed-averaged value per section, the unit used for paired tests."""
    by_section = defaultdict(list)
    for r in records:
        by_section[r.get("section", r.get("sample_id"))].append(r[key])
    return [np.mean(v) for v in by_section.values()]


# Figure 1: pipeline schematic and consensus edge maps on one section.
maps = np.load(EVIDENCE / "edge_maps/consensus_151673.npz")
coords, layer, edges = maps["coords"], maps["layer"], maps["edge_index"]
fig = plt.figure(figsize=(5.5, 3.3))
top = fig.add_axes([0, .77, 1, .23])
top.set_xlim(0, 100)
top.set_ylim(0, 20)
top.axis("off")
top.text(.4, 19.8, "$\\bf{A}$   Pipeline", fontsize=8, va="top")
steps = ["Spatial kNN graph\n+ PCA features", "Gated attention:\nedge gates $g_{ij}$",
         "Train on target $Y$\nunder budget $r$", "Export top-$k$\nedges",
         "Refit GCN; compare\nwith permuted default"]
width, gap = 17.2, 3.3
for i, text in enumerate(steps):
    x = .6 + i * (width + gap)
    top.add_patch(FancyBboxPatch((x, 1.5), width, 13.5, boxstyle="round,pad=0,rounding_size=1.4",
                                 fc="#F2F4F7", ec=MUTED, lw=.6))
    top.text(x + width / 2, 8.25, text, ha="center", va="center", fontsize=6.8, linespacing=1.3)
    if i < 4:
        top.add_patch(FancyArrowPatch((x + width + .4, 8.25), (x + width + gap - .4, 8.25),
                                      arrowstyle="-|>", mutation_scale=7, color=MUTED, lw=.7))

layer_names = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]
spot_colors = np.array([to_rgba(LAYER_COLORS[v]) if v >= 0 else to_rgba("#EDEDED") for v in layer])


def node_score(score):
    """Mean consensus score over each spot's incoming candidate edges."""
    total, degree = np.zeros(len(coords)), np.zeros(len(coords))
    np.add.at(total, edges[1], score)
    np.add.at(degree, edges[1], 1)
    return total / np.maximum(degree, 1)


def rank(values):
    return (np.argsort(np.argsort(values)) + .5) / len(values)


def departure(target, suffix):
    return (rank(node_score(maps[f"real__{target}{suffix}"]))
            - rank(node_score(maps[f"permuted_default{suffix}"])))


oligo_score = maps["oligodendrocyte_score"]
low, high = np.percentile(oligo_score, [2, 98])
panels = [("B", "Layer annotation", INK, spot_colors, None),
          ("C", "Oligodendrocyte", INK, oligo_score, (SEQUENTIAL, low, high)),
          ("D", "TSB-dual $-$ default", DUAL, departure("oligodendrocyte", "_dual"), (DIVERGING, -.8, .8)),
          ("E", "TSB-cov $-$ default", COV, departure("oligodendrocyte", ""), (DIVERGING, -.8, .8))]
tissue = []
for idx, (letter, title, color, values, scale) in enumerate(panels):
    ax = fig.add_axes([idx * .25 + .005, .15, .24, .54])
    kwargs = dict(c=values) if scale is None else dict(c=values, cmap=scale[0], vmin=scale[1], vmax=scale[2])
    tissue.append(ax.scatter(coords[:, 0], coords[:, 1], s=3.0, linewidths=0, rasterized=True, **kwargs))
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    panel_title(ax, letter, title, color)
handles = [plt.Line2D([], [], ls="", marker="o", ms=4, color=c) for c in LAYER_COLORS]
fig.legend(handles, layer_names, loc="lower left", bbox_to_anchor=(.0, .0), ncol=4,
           handletextpad=.05, columnspacing=.5, labelspacing=.25, fontsize=6.8)
colorbar(fig, tissue[1], fig.add_axes([.315, .095, .17, .022]), "program score (z)", [-10, 0, 10])
colorbar(fig, tissue[3], fig.add_axes([.56, .095, .4, .022]),
         "spot rank: target map $-$ default", [-.8, -.4, 0, .4, .8])
fig.savefig(OUT / "overview.pdf", dpi=300, bbox_inches="tight", pad_inches=.01)
plt.close(fig)

# Figure 2: budget control, matched-mask refit, and the cost of coverage in specificity.
sweep = read("lambda_sweep_dlpfc/records.json")
assert len(sweep) == 540 and {r["budget_mode"] for r in sweep} == {"squared"}
lambdas = sorted({r["lambda_l0"] for r in sweep})
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.3), layout="constrained",
                         gridspec_kw=dict(width_ratios=[1.1, 1.1, .8]))
ax = axes[0]
for budget, marker in zip((.1, .2, .3), ("o", "s", "^")):
    ax.axhline(budget, color=RULE, lw=.6, ls=(0, (3, 2)), zorder=0)
    cells = [section_means([r for r in sweep if r["lambda_l0"] == lam
                            and r["target_edge_rate"] == budget], "edge_prob_mean")
             for lam in lambdas]
    ax.errorbar(range(len(lambdas)), [np.mean(c) for c in cells], [sem(c) for c in cells],
                fmt=marker + "-", color=FIXED, ms=3.2, capsize=1.2, elinewidth=.6)
    cov = section_means(select(dlpfc, "end_to_end", "tsb_final", budget), "edge_rate")
    ax.errorbar(len(lambdas) + .5, np.mean(cov), sem(cov), fmt=marker, color=COV, ms=3.8,
                capsize=1.2, elinewidth=.6)
ax.axvline(len(lambdas) - .25, ymax=.82, color=RULE, lw=.6)
ax.set_xticks(list(range(len(lambdas))) + [len(lambdas) + .5], [f"{lam:g}" for lam in lambdas] + ["TSB-\ncov"])
ax.get_xticklabels()[-1].set_color(COV)
ax.set(xlabel=r"TSB-fixed penalty $\lambda_0$", ylabel="Achieved expected retention",
       ylim=(0, .4), yticks=[0, .1, .2, .3])
ax.legend([plt.Line2D([], [], ls="", marker=m, color=MUTED, ms=3.2) for m in "os^"],
          ["$r$ = 0.1", "$r$ = 0.2", "$r$ = 0.3"], ncol=3, loc="upper center",
          handletextpad=.05, columnspacing=.6, borderaxespad=0, fontsize=6.5)
panel_title(ax, "A", "Budget control")
ax = axes[1]
for selector, color, label, style in [("tsb_final", COV, "TSB-cov", "-"),
                                      ("tsb_dual", DUAL, "TSB-dual", "-"),
                                      ("graphmask", GRAPHMASK, "GraphMask", "-"),
                                      ("random", RANDOM, "Random", (0, (3, 2))),
                                      ("tsb_submitted", FIXED, "TSB-fixed", "-")]:
    cells = [section_means(select(dlpfc, "transfer_gcn", selector, b), "macro_f1") for b in (.1, .2, .3)]
    rates = [mean(select(dlpfc, "transfer_gcn", selector, b), "edge_rate") for b in (.1, .2, .3)]
    vals, err = np.array([np.mean(c) for c in cells]), np.array([sem(c) for c in cells])
    ax.fill_between(rates, vals - err, vals + err, color=color, alpha=.14, lw=0)
    ax.plot(rates, vals, marker="o", ls=style, color=color, ms=2.8, label=label)
ax.set(xlabel="Exported edge fraction", ylabel="Refit GCN macro F$_1$", xticks=[.1, .2, .3],
       yticks=[.75, .8, .85, .9])
ax.legend(loc="lower right", handlelength=1.6, labelspacing=.25)
panel_title(ax, "B", "Matched-mask refit")
ax = axes[2]
gaps = []
for analysis in analyses:
    per_section = []
    for section in analysis["per_section"]:
        real_s, perm_s = np.array(section["gate_matrix"]), np.array(section["permuted_matrix"])
        pairs = [(0, 1), (0, 2), (1, 2)]
        per_section.append(np.mean([perm_s[p] for p in pairs]) - np.mean([real_s[p] for p in pairs]))
    gaps.append(per_section)
assert abs(np.mean(gaps[-1]) - .2709) < 1e-3
ax.bar(range(3), [np.mean(g) for g in gaps], yerr=[sem(g) for g in gaps], width=.62,
       color=[FIXED, DUAL, COV], error_kw=dict(lw=.6, capsize=1.8, ecolor=INK))
ax.set_xticks(range(3), ["TSB-\nfixed", "TSB-\ndual", "TSB-\ncov"])
for tick, color in zip(ax.get_xticklabels(), [FIXED, DUAL, COV]):
    tick.set_color(color)
ax.set(ylabel="Permuted $-$ real correlation", ylim=(0, .55), yticks=[0, .1, .2, .3, .4, .5])
panel_title(ax, "C", "Target separation")
fig.savefig(OUT / "tradeoff.pdf")
plt.close(fig)

# Figure 3: conditioning with and without coverage, and reliability versus learnability.
dual_analysis = analyses[1]
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.55), layout="constrained",
                         gridspec_kw=dict(width_ratios=[1, 1, .74]))
for ax, analysis, letter, title, color in [(axes[0], dual_analysis, "A", "TSB-dual", DUAL),
                                           (axes[1], final_analysis, "B", "TSB-cov", COV)]:
    matrix = np.array(analysis["mean_permuted_matrix"]) - np.array(analysis["mean_gate_matrix"])
    im = ax.imshow(np.ma.masked_where(np.eye(7, dtype=bool), matrix), vmin=-.8, vmax=.8,
                   cmap=DIVERGING)
    for i in range(7):
        for j in range(7):
            if i != j:
                ax.text(j, i, f"{matrix[i, j]:.2f}".replace("0.", "."), ha="center", va="center",
                        fontsize=5.2, color="white" if abs(matrix[i, j]) > .35 else INK)
    for edge in (.5, 2.5):
        ax.axhline(edge, color="white", lw=1.4)
        ax.axvline(edge, color="white", lw=1.4)
    ax.set_xticks(range(7), labels, rotation=50, ha="right", fontsize=6.5)
    ax.set_yticks(range(7), labels if ax is axes[0] else [""] * 7, fontsize=6.5)
    ax.tick_params(length=0, pad=1.5)
    ax.spines[:].set_visible(False)
    panel_title(ax, letter, title, color)
bar = fig.colorbar(im, ax=axes[:2], shrink=.72, pad=.015, aspect=22, ticks=[-.8, -.4, 0, .4, .8])
bar.set_label(r"$\Delta\rho$ (permuted $-$ real)", fontsize=7)
bar.outline.set_linewidth(.4)
bar.ax.tick_params(labelsize=6.5, length=1.5)
ax = axes[2]
perf = [final_analysis["summary"][t]["performance"] for t in targets]
raw = [final_analysis["summary"][t]["consensus_reliability"] for t in targets]
residual = [final_analysis["summary"][t]["consensus_reliability_residual"] for t in targets]
for x, a, b in zip(perf, raw, residual):
    ax.plot([x, x], [b, a], color=RULE, lw=.6, zorder=1)
ax.scatter(perf, raw, s=14, facecolors="white", edgecolors=COV, linewidths=.8, zorder=3,
           label="Reliability")
ax.scatter(perf, residual, s=14, color=COV, edgecolors="white", linewidths=.4, zorder=3,
           label="Residual reliability")
for name, i, dx, dy, ha in [("Microglia", 4, .05, 0, "left"), ("Astro.", 3, .05, 0, "left"),
                            ("L1, Deep", 5, -.02, -.07, "center"),
                            ("Layer", 0, -.1, .02, "right"), ("Exc., Oligo.", 2, -.11, -.01, "right")]:
    ax.annotate(name, (perf[i], residual[i]), (perf[i] + dx, residual[i] + dy), fontsize=6,
                ha=ha, va="center", color=MUTED)
ax.set(xlabel="Performance", ylabel="Split-half correlation", ylim=(.15, 1.08),
       yticks=[.2, .4, .6, .8], xlim=(-.3, .92), xticks=[-.2, .2, .6])
ax.legend(loc="upper center", ncol=1, handletextpad=.05, borderaxespad=0, fontsize=6.3, labelspacing=.2)
panel_title(ax, "C", "Reliability")
fig.canvas.draw()
heat, side = axes[0].get_position(), ax.get_position()
fig.set_layout_engine("none")
ax.set_position([side.x0 + .015, heat.y0, side.width - .015, heat.height])
fig.savefig(OUT / "conditioning.pdf", bbox_inches="tight", pad_inches=.02)
plt.close(fig)

lambda_rows = []
for lam in lambdas:
    cells = []
    for budget in (.1, .2, .3):
        rr = [r for r in sweep if r["lambda_l0"] == lam and r["target_edge_rate"] == budget]
        assert len(rr) == 36
        cells.append(f"{np.mean([r['edge_prob_mean'] for r in rr]):.3f} & "
                     f"{np.mean([r['macro_f1'] for r in rr]):.3f}")
    lambda_rows.append(f"{lam:g} & " + " & ".join(cells) + " \\")
cells = []
for budget in (.1, .2, .3):
    rr = select(dlpfc, "end_to_end", "tsb_final", budget)
    cells.append(f"{mean(rr, 'edge_rate'):.3f} & {mean(rr, 'macro_f1'):.3f}")
lambda_rows.append("\\midrule TSB-cov & " + " & ".join(cells) + " \\")
(OUT / "lambda_rows.tex").write_text("\n".join(s + "\\" for s in lambda_rows) + "\n")

bio = read("consensus_fixed_penalty/bio_validation_v2.json")
bio_rows = []
for key, v in bio["heldout_marker_morans"].items():
    a,b = key.split("|")
    if a == b or (a == "astrocyte" and b == "oligodendrocyte"):
        bio_rows.append(f"{a.replace('_',' ')} $\\to$ {b.replace('_',' ')} & "
                        f"{v['tsb_enhancement']:.2f} & {v['random_enhancement']:.2f} & "
                        f"{v['density_matched_enhancement']:.2f} & {v['n_sections']} \\")
(OUT / "bio_rows.tex").write_text("\n".join(s + "\\" for s in bio_rows) + "\n")

summary = {"dlpfc_tests": tests, "mouse": {}, "dlpfc": {}}
for data, key in [(dlpfc,"dlpfc"),(mouse,"mouse")]:
    for protocol, selector in sorted({(r['protocol'],r['selector']) for r in data}):
        rr = select(data, protocol, selector, .2 if key == "dlpfc" and protocol != "full_graph" else None)
        if rr:
            summary[key][protocol + "/" + selector] = dict(n=len(rr), mean=mean(rr,"macro_f1"),
                sd=float(np.std([r['macro_f1'] for r in rr],ddof=1)), edge_rate=mean(rr,"edge_rate"))
final_mouse = select(mouse, "transfer_gcn", "tsb_final")
random_mouse = select(mouse, "transfer_gcn", "random")
summary["mouse_p"] = float(wilcoxon([r['macro_f1'] for r in final_mouse],
                                    [r['macro_f1'] for r in random_mouse]).pvalue)
assert abs(summary['dlpfc']['transfer_gcn/tsb_final']['mean'] - .868276) < 1e-5
assert abs(summary['mouse']['transfer_gcn/tsb_final']['mean'] - .884918) < 1e-5
summary["dual_refit_random_check_max_abs_diff"] = max_check_diff
summary["dual_vs_random_dlpfc"] = {str(b): dict(zip(("p", "wins"), paired(dlpfc, "tsb_dual", "random", b)))
                                   for b in (.1, .2, .3)}
summary["dual_vs_cov_dlpfc"] = {str(b): dict(zip(("p", "wins"), paired(dlpfc, "tsb_dual", "tsb_final", b)))
                                for b in (.1, .2, .3)}
summary["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(EVIDENCE.rglob('*')) if p.suffix in {'.json', '.npz'}}
(OUT / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"Verified 12 sections x 3 seeds and 10 mouse seeds; wrote {OUT}")
