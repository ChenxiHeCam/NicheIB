'The single DLPFC table: every selector, one protocol, matched retention, coverage reported.'

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from nicheib.baselines import PlainGAT, PlainGCN
from nicheib.external_baselines import (
    degree_edge_scores,
    gnnexplainer_edge_scores,
    gradient_edge_scores,
)
from nicheib.gates import HardConcreteSampler
from nicheib.tsb import TargetedSpatialBottleneck


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #

class _GraphMaskGate(nn.Module):
    def __init__(self, dim, hidden=64, init_bias=3.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, h, edge_index):
        src, dst = edge_index
        return self.net(torch.cat([h[src], h[dst], h[src] * h[dst]], -1)).squeeze(-1)


def graphmask_scores(model, x, edge_index, epochs=300, lr=1e-3, tol=0.05, dual_lr=0.1):
    """GraphMask on the frozen network: amortised gates, Lagrangian on output divergence."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        ref = F.softmax(model(x, edge_index)["logits"], dim=-1)
    dim = F.relu(model.encoder(x)).shape[1]
    g1, g2 = _GraphMaskGate(dim), _GraphMaskGate(dim)
    smp = HardConcreteSampler(2.0 / 3.0, -0.1, 1.1)
    lam = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam(list(g1.parameters()) + list(g2.parameters()), lr=lr)
    opt_l = torch.optim.Adam([lam], lr=dual_lr)

    def fwd(sample):
        h = F.relu(model.encoder(x))
        l1 = g1(h, edge_index)
        m1 = smp.sample(l1) if sample else smp.deterministic(l1)
        h = F.relu(model.conv1(h, edge_index, m1))
        l2 = g2(h, edge_index)
        m2 = smp.sample(l2) if sample else smp.deterministic(l2)
        return model.head(model.conv2(h, edge_index, m2)), l1, l2, m1, m2

    for _ in range(epochs):
        opt.zero_grad(); opt_l.zero_grad()
        logits, l1, l2, _, _ = fwd(True)
        div = F.kl_div(F.log_softmax(logits, -1), ref, reduction="batchmean")
        spars = 0.5 * (smp.expected_l0(l1).mean() + smp.expected_l0(l2).mean())
        (spars + F.softplus(lam) * (div - tol)).backward()
        opt.step()
        lam.grad = -lam.grad
        opt_l.step()

    with torch.no_grad():
        _, _, _, m1, m2 = fwd(False)
    for p in model.parameters():
        p.requires_grad_(True)
    return ((m1 + m2) / 2).cpu().numpy()


@torch.no_grad()
def attention_scores(gat, x, edge_index):
    from torch_geometric.utils import softmax

    def alpha(conv, h):
        proj = conv.lin(h).view(-1, conv.heads, conv.head_dim)
        a = F.leaky_relu((proj * conv.att_src).sum(-1)[edge_index[0]]
                         + (proj * conv.att_dst).sum(-1)[edge_index[1]], 0.2)
        return softmax(a, edge_index[0], None, h.shape[0]).mean(-1)

    ones = x.new_ones(edge_index.shape[1])
    h0 = F.relu(gat.encoder(x))
    a1 = alpha(gat.conv1, h0)
    a2 = alpha(gat.conv2, F.relu(gat.conv1(h0, edge_index, ones)))
    return ((a1 + a2) / 2).cpu().numpy()


# --------------------------------------------------------------------------- #

def coverage_penalty(el0, edge_index, n_nodes, d_min):
    deg = torch.zeros(n_nodes).scatter_add(0, edge_index[1], el0)
    return F.relu(d_min - deg).pow(2).mean()


def train_tsb(data, train_mask, n_classes, seed, epochs, budget, variant,
              lambda_l0=5.0, rho=20.0, dual_lr=0.1, lambda_cov=5.0,
              coverage_frac=0.75, lr=1e-3):
    """variant: 'submitted' (fixed penalty), 'dual' (dual ascent only), or 'final'
    (dual ascent + coverage)."""
    seed_everything(seed)
    n_nodes, n_edges = data.x.shape[0], data.edge_index.shape[1]
    # d_min must be reachable at this budget, else the coverage and budget terms fight.
    d_min = coverage_frac * budget * (n_edges / n_nodes)
    model = TargetedSpatialBottleneck(
        input_dim=data.x.shape[1], hidden_dim=64, latent_dim=32, output_dim=n_classes,
        task_type="classification", use_xib=False, gate_init_bias=3.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = max(1, int(0.2 * epochs))
    mu = 0.0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        ws = min(1.0, (epoch + 1) / warmup)
        pred = F.cross_entropy(out["logits"][train_mask], data.y[train_mask])
        el0 = out["expected_l0"]
        gap = el0.mean() - budget
        if variant == "submitted":
            total = pred + lambda_l0 * ws * gap.pow(2)
        else:
            cov = 0.0 if variant == "dual" else lambda_cov
            total = (pred + ws * (mu * gap + 0.5 * rho * gap.pow(2))
                     + cov * ws * coverage_penalty(
                         el0, data.edge_index, n_nodes, d_min))
        total.backward()
        if variant != "submitted":
            torch.nn.utils.clip_grad_norm_(
                list(model.gate1.parameters()) + list(model.gate2.parameters()), 1.0)
        opt.step()
        if variant != "submitted" and epoch >= warmup:
            mu = float(np.clip(mu + dual_lr * float(gap.detach()), 0.0, 50.0))
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    return model, out["logits"], out["edge_prob"].cpu().numpy(), float(out["edge_rate"])


def fit(model, x, y, train_mask, epochs, edge_index=None, lr=1e-3, wd=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x) if edge_index is None else model(x, edge_index)
        F.cross_entropy(out["logits"][train_mask], y[train_mask]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return (model(x) if edge_index is None else model(x, edge_index))["logits"]


def macro_f1(y, test_idx, logits):
    from sklearn.metrics import f1_score

    return float(f1_score(y[test_idx].numpy(),
                          logits.argmax(-1)[test_idx].numpy(), average="macro"))


def make_split(y, seed, test_size=0.3):
    from sklearn.model_selection import train_test_split

    valid = (y >= 0).nonzero(as_tuple=True)[0].numpy()
    tr, te = train_test_split(valid, test_size=test_size, random_state=seed,
                              stratify=y.numpy()[valid])
    m = torch.zeros(y.shape[0], dtype=torch.bool)
    m[tr] = True
    return m, torch.tensor(te, dtype=torch.long)


def topk(scores, k):
    keep = np.zeros(len(scores), dtype=bool)
    keep[np.argsort(np.asarray(scores).ravel())[-k:]] = True
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",
                        default="data/dlpfc")
    parser.add_argument("--sections", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from nicheib.data import load_dlpfc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for section in args.sections:
        data = load_dlpfc(section, data_dir=args.data_dir,
                          n_hvgs=1000, n_pcs=64, n_neighbors=6)
        n_nodes, n_edges = data.x.shape[0], data.edge_index.shape[1]
        n_classes = len(data.label_names)
        print(f"\n=== {section}: {n_nodes} spots, {n_edges} edges ===", flush=True)

        for seed in args.seeds:
            train_mask, test_idx = make_split(data.y, seed)

            def add(**kw):
                records.append(dict(section=section, seed=seed, **kw))

            # Full-graph references, once per seed.
            for name, mdl, ei in (
                    ("plain_gcn_full", PlainGCN(data.x.shape[1], 64, n_classes,
                                                task_level="node"), data.edge_index),
                    ("plain_gat_full", PlainGAT(data.x.shape[1], 64, 32, n_classes,
                                                task_level="node"), data.edge_index)):
                seed_everything(seed)
                add(budget=None, selector=name, protocol="full_graph",
                    macro_f1=macro_f1(data.y, test_idx,
                                      fit(mdl, data.x, data.y, train_mask,
                                          args.epochs, ei)),
                    edge_rate=1.0, frac_isolated=0.0)

            for budget in args.budgets:
                # TSB-final defines k; TSB-submitted is reported for reference.
                gated = {}
                for variant in ("final", "submitted"):
                    mdl, logits, scores, er = train_tsb(
                        data, train_mask, n_classes, seed, args.epochs, budget, variant)
                    gated[variant] = (mdl, scores, er)
                    add(budget=budget, selector=f"tsb_{variant}", protocol="end_to_end",
                        macro_f1=macro_f1(data.y, test_idx, logits),
                        edge_rate=er, budget_error=abs(er - budget), frac_isolated=None)

                k = max(1, int(round(gated["final"][2] * n_edges)))
                tsb_model = gated["final"][0]

                # A gate-free GAT for the attention selector and as a random control.
                seed_everything(seed)
                gat = PlainGAT(data.x.shape[1], 64, 32, n_classes, task_level="node")
                fit(gat, data.x, data.y, train_mask, args.epochs, data.edge_index)

                class _D:
                    pass

                d = _D()
                d.x, d.edge_index, d.y = data.x, data.edge_index, data.y
                d.valid_mask = data.y >= 0

                rng = np.random.default_rng(7919 + seed)
                seed_everything(seed)
                scores = {
                    "tsb_final": gated["final"][1],
                    "tsb_submitted": gated["submitted"][1],
                    "graphmask": graphmask_scores(tsb_model, data.x, data.edge_index,
                                                  epochs=args.epochs),
                    "gnnexplainer": gnnexplainer_edge_scores(tsb_model, d, epochs=200),
                    "attention": attention_scores(gat, data.x, data.edge_index),
                    "gradient": gradient_edge_scores(tsb_model, d),
                    "degree": degree_edge_scores(data.edge_index),
                    "random": rng.random(n_edges),
                }

                # Transfer protocol: every selector keeps exactly k edges, fresh GCN.
                for name, sc in scores.items():
                    ei = data.edge_index[:, torch.tensor(topk(sc, k))]
                    deg = np.bincount(ei.reshape(-1).numpy(), minlength=n_nodes)
                    seed_everything(seed)
                    add(budget=budget, selector=name, protocol="transfer_gcn",
                        macro_f1=macro_f1(data.y, test_idx,
                                          fit(PlainGCN(data.x.shape[1], 64, n_classes,
                                                       task_level="node"),
                                              data.x, data.y, train_mask,
                                              args.epochs, ei)),
                        edge_rate=k / n_edges,
                        frac_isolated=float((deg == 0).mean()))

                # Both random controls under the end-to-end protocol, same k.
                ei_r = data.edge_index[:, torch.tensor(topk(rng.random(n_edges), k))]
                deg_r = np.bincount(ei_r.reshape(-1).numpy(), minlength=n_nodes)
                for name, mdl in (("gcn_on_random", PlainGCN(data.x.shape[1], 64,
                                                             n_classes,
                                                             task_level="node")),
                                  ("gat_on_random", PlainGAT(data.x.shape[1], 64, 32,
                                                             n_classes,
                                                             task_level="node"))):
                    seed_everything(seed)
                    add(budget=budget, selector=name, protocol="end_to_end",
                        macro_f1=macro_f1(data.y, test_idx,
                                          fit(mdl, data.x, data.y, train_mask,
                                              args.epochs, ei_r)),
                        edge_rate=k / n_edges,
                        frac_isolated=float((deg_r == 0).mean()))

                fin = [r for r in records if r["selector"] == "tsb_final"
                       and r.get("budget") == budget and r["seed"] == seed]
                print(f"  seed {seed} budget {budget}: TSB-final ER "
                      f"{gated['final'][2]:.3f} (err {abs(gated['final'][2]-budget):.3f}), "
                      f"k={k} | " + " ".join(
                          "%s=%.4f" % (r["selector"], r["macro_f1"])
                          for r in records
                          if r.get("budget") == budget and r["seed"] == seed
                          and r["protocol"] == "transfer_gcn"), flush=True)

        with (out_dir / f"unified_{section}.json").open("w") as f:
            json.dump([r for r in records if r["section"] == section], f, indent=2)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
