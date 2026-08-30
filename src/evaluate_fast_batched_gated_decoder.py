import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import stim
import pymatching
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from utils.metrics import wilson_score_interval
from audit_phase_a_d_opportunity import (
    build_parity_expanded_graph,
    find_exact_logical_reference_chain,
    standardize_edge,
    compute_chain_observable
)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

class AnisotropicRelationalLayer(nn.Module):
    def __init__(self, hidden_dim=64, in_edge_dim=4):
        super().__init__()
        # Input: [h_src (64) + h_dst (64) + edge_attr (4) + is_parallel (1)] = 133
        msg_dim = hidden_dim * 2 + in_edge_dim + 1
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_attr, is_parallel):
        src, dst = edge_index[0], edge_index[1]
        msg_input = torch.cat([h[src], h[dst], edge_attr, is_parallel], dim=-1)
        messages = self.msg_mlp(msg_input)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)
        return self.norm(h + self.node_update(torch.cat([h, agg], dim=-1)))

class TopoDephaseGNN(nn.Module):
    def __init__(self, in_node_dim=6, in_edge_dim=4, hidden_dim=64, num_layers=4):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(in_node_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList([
            AnisotropicRelationalLayer(hidden_dim=hidden_dim, in_edge_dim=in_edge_dim)
            for _ in range(num_layers)
        ])
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + in_edge_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x6, edge_index, edge_attr, is_parallel, batch=None):
        h = self.node_embed(x6)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, is_parallel)

        src, dst = edge_index[0], edge_index[1]
        edge_feat = torch.cat([h[src], h[dst], edge_attr, is_parallel], dim=-1)
        edge_logits = self.edge_head(edge_feat).squeeze(-1)

        if batch is None:
            pooled = h.mean(dim=0, keepdim=True)
        else:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            pooled = torch.zeros((num_graphs, h.shape[-1]), device=h.device)
            pooled.index_add_(0, batch, h)
            counts = torch.bincount(batch, minlength=num_graphs).unsqueeze(-1).clamp(min=1).float()
            pooled = pooled / counts

        prob = torch.sigmoid(self.global_mlp(pooled))
        return prob, edge_logits

def collate_subgraph_batch(samples, device):
    x6_list, e_idx_list, e_attr_list, e_par_list = [], [], [], []
    batch_map = []
    y_list = []
    node_offset = 0

    for i, item in enumerate(samples):
        if len(item) == 5:
            x6, e_idx, e_attr, e_par, y_val = item
            y_list.append(float(y_val))
        else:
            x6, e_idx, e_attr, e_par = item

        num_nodes = x6.shape[0]
        x6_list.append(x6)
        e_idx_list.append(e_idx + node_offset)
        e_attr_list.append(e_attr)
        e_par_2d = e_par.view(-1, 1) if e_par.dim() == 1 else e_par
        e_par_list.append(e_par_2d)
        batch_map.append(torch.full((num_nodes,), i, dtype=torch.long))
        node_offset += num_nodes

    batched_x6 = torch.cat(x6_list, dim=0).to(device)
    batched_e_idx = torch.cat(e_idx_list, dim=1).to(device)
    batched_e_attr = torch.cat(e_attr_list, dim=0).to(device)
    batched_e_par = torch.cat(e_par_list, dim=0).to(device)
    batched_map = torch.cat(batch_map, dim=0).to(device)

    targets = torch.tensor(y_list, dtype=torch.float32, device=device).unsqueeze(-1) if y_list else None
    return batched_x6, batched_e_idx, batched_e_attr, batched_e_par, batched_map, targets

def collect_fast_dataset(distances, p_val, eta, total_shots):
    dataset = {}
    for d in distances:
        t0 = time.time()
        circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
        dem = circuit.detector_error_model(decompose_errors=True)
        coords = circuit.get_detector_coordinates()
        num_dets = circuit.num_detectors

        edge_dict, adj, bnd_z_idx, bnd_x_idx = build_parity_expanded_graph(dem, num_dets, coords, d)
        raw_edge_dict, _, _, _ = extract_complete_dem_graph(dem, num_dets, coords, d)
        R_L, _ = find_exact_logical_reference_chain(adj, num_dets)

        matcher = pymatching.Matching.from_detector_error_model(dem)
        sampler = circuit.compile_detector_sampler()
        syn, flips = sampler.sample(shots=total_shots, separate_observables=True)
        flips = flips.flatten().astype(np.int64)

        preds_mwpm = matcher.decode_batch(syn).flatten().astype(np.int64)
        mwpm_wrong = (preds_mwpm != flips).astype(np.int64)

        tensors_by_shot = {}
        active_indices = np.where(np.sum(syn, axis=1) >= 2)[0]

        for idx in active_indices:
            s = syn[idx].astype(np.uint8)
            x4, x6, e_idx, e_attr, e_par, _, _, _ = extract_active_subgraph_tensors(
                s, coords, raw_edge_dict, bnd_z_idx, bnd_x_idx, d, torch.device("cpu")
            )
            if e_idx.numel() > 0:
                tensors_by_shot[idx] = (x6, e_idx, e_attr, e_par)

        dataset[d] = {
            "circuit": circuit, "dem": dem, "coords": coords, "num_dets": num_dets,
            "edge_dict": edge_dict, "R_L": R_L, "bnd_z_idx": bnd_z_idx, "bnd_x_idx": bnd_x_idx,
            "matcher": matcher, "syn": syn, "flips": flips, "preds_mwpm": preds_mwpm,
            "mwpm_wrong": mwpm_wrong, "tensors_by_shot": tensors_by_shot, "total_shots": total_shots,
            "total_fails": int(np.sum(mwpm_wrong))
        }
        print(f"  [+] Collected d={d:2d} ({total_shots:,} shots, {np.sum(mwpm_wrong):>4d} failures, {len(tensors_by_shot):,d} active subgraphs) in {time.time()-t0:.2f}s")
    return dataset

def run_fast_batched_gated_decoder(
    train_distances=[3, 5, 7],
    test_distance=9,
    p_val=0.002,
    eta=100.0,
    train_shots_per_dist=20000,
    test_shots=30000,
    train_steps=120,
    batch_size=128
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 130)
    print("PHASE F: BATCHED FAILURE GATE & ZERO-SHOT HELD-OUT AUDIT")
    print(f"Setup: p={p_val}, eta={eta}, Train d={train_distances} ({train_shots_per_dist:,} shots/d), Test d={train_distances+[test_distance]} ({test_shots:,} shots/d)")
    print(f"Training Budget: 1 epoch, {train_steps} vectorized steps of batch_size={batch_size}")
    print("=" * 130 + "\n")

    print("[1/3] Collecting train and test datasets...")
    train_data = collect_fast_dataset(train_distances, p_val, eta, train_shots_per_dist)
    eval_data = collect_fast_dataset(train_distances + [test_distance], p_val, eta, test_shots)

    pos_pool, neg_pool = [], []
    for d in train_distances:
        dat = train_data[d]
        for idx, tensors in dat["tensors_by_shot"].items():
            y_fail = dat["mwpm_wrong"][idx]
            if y_fail == 1:
                pos_pool.append((*tensors, 1.0))
            else:
                neg_pool.append((*tensors, 0.0))

    print(f"\n  [Pools] Failures: {len(pos_pool):,d} | Correct: {len(neg_pool):,d}")

    print("\n[2/3] Vectorized Training TopoDephaseGNN Failure Gate...")
    model = TopoDephaseGNN(in_node_dim=6, in_edge_dim=4, hidden_dim=64, num_layers=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.BCELoss()

    model.train()
    t_train = time.time()
    half_b = batch_size // 2

    for step in range(1, train_steps + 1):
        p_idx = np.random.choice(len(pos_pool), half_b, replace=True)
        n_idx = np.random.choice(len(neg_pool), half_b, replace=False)
        batch_samples = [pos_pool[i] for i in p_idx] + [neg_pool[j] for j in n_idx]

        bx6, be_idx, be_attr, be_par, bmap, targets = collate_subgraph_batch(batch_samples, device)

        optimizer.zero_grad()
        probs, _ = model(bx6, be_idx, be_attr, be_par, batch=bmap)
        loss = criterion(probs, targets)
        loss.backward()
        optimizer.step()

        if step % 30 == 0 or step == train_steps:
            print(f"  Step {step:>3d}/{train_steps} | Vectorized BCE Loss: {loss.item():.4f} | Elapsed: {time.time()-t_train:.2f}s")

    torch.save(model.state_dict(), "checkpoints/topo_failure_gate_fast.pt")

    print("\n[3/3] Evaluating Gated Decoder (with Held-Out Zero-Shot d=9)...")
    model.eval()

    tau_sweep = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]
    optimal_results = []

    for d in train_distances + [test_distance]:
        dat = eval_data[d]
        shots = dat["total_shots"]
        flips = dat["flips"]
        preds_mwpm = dat["preds_mwpm"]
        mwpm_wrong = dat["mwpm_wrong"]
        R_L = dat["R_L"]
        edge_dict = dat["edge_dict"]
        bnd_z = dat["bnd_z_idx"]
        bnd_x = dat["bnd_x_idx"]
        matcher = dat["matcher"]
        syn = dat["syn"]
        tensors_by_shot = dat["tensors_by_shot"]

        pred_probs = np.zeros(shots, dtype=np.float64)
        active_keys = list(tensors_by_shot.keys())
        eval_batch_size = 256

        with torch.no_grad():
            for i in range(0, len(active_keys), eval_batch_size):
                chunk_keys = active_keys[i:i + eval_batch_size]
                chunk_samples = [tensors_by_shot[k] for k in chunk_keys]
                bx6, be_idx, be_attr, be_par, bmap, _ = collate_subgraph_batch(chunk_samples, device)
                probs, _ = model(bx6, be_idx, be_attr, be_par, batch=bmap)
                probs_np = probs.cpu().numpy().flatten()
                for k, p_val in zip(chunk_keys, probs_np):
                    pred_probs[k] = p_val

        auroc = roc_auc_score(mwpm_wrong, pred_probs) if np.sum(mwpm_wrong) > 0 else 0.0
        precisions, recalls, _ = precision_recall_curve(mwpm_wrong, pred_probs)
        auprc = auc(recalls, precisions) if np.sum(mwpm_wrong) > 0 else 0.0

        obs_B_cache = {}
        for k in active_keys:
            s = syn[k].astype(np.uint8)
            edges_a_raw = matcher.decode_to_edges_array(s)
            C_A = set(standardize_edge(int(e[0]), int(e[1]), bnd_z, bnd_x) for e in edges_a_raw)
            C_B = C_A.symmetric_difference(R_L)
            obs_B_cache[k] = compute_chain_observable(C_B, edge_dict)

        print(f"\n>>> DISTANCE d = {d:2d} {'(HELD-OUT ZERO-SHOT)' if d==test_distance else ''} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} <<<")
        print(f"{'Tau':<6} | {'Gated P_L':<10} | {'MWPM P_L':<10} | {'Recoveries':<11} | {'Regressions':<12} | {'Net Gain':<10} | {'Recall':<8} | {'Precision':<10} | {'FPR':<8}")
        print("-" * 115)

        best_row = None
        best_net_gain = -float("inf")

        for tau in tau_sweep:
            gated_preds = preds_mwpm.copy()
            altered_shots = np.where(pred_probs >= tau)[0]

            for idx in altered_shots:
                if idx in obs_B_cache:
                    gated_preds[idx] = obs_B_cache[idx]

            rec = int(np.sum((preds_mwpm != flips) & (gated_preds == flips)))
            reg = int(np.sum((preds_mwpm == flips) & (gated_preds != flips)))
            net_gain = rec - reg

            pred_fail_binary = (pred_probs >= tau).astype(np.int64)
            tp = int(np.sum((pred_fail_binary == 1) & (mwpm_wrong == 1)))
            fp = int(np.sum((pred_fail_binary == 1) & (mwpm_wrong == 0)))
            tn = int(np.sum((pred_fail_binary == 0) & (mwpm_wrong == 0)))
            fn = int(np.sum((pred_fail_binary == 0) & (mwpm_wrong == 1)))

            rec_pct = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
            fpr_pct = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
            prec_pct = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0

            gated_pl = (np.sum(gated_preds != flips) / shots) * 100.0
            mwpm_pl = (np.sum(preds_mwpm != flips) / shots) * 100.0

            print(f"{tau:<6.2f} | {gated_pl:6.3f}%   | {mwpm_pl:6.3f}%   | {rec:>5d}/{int(np.sum(mwpm_wrong)):<4d}  | {reg:>5d}/{shots-int(np.sum(mwpm_wrong)):<6d} | {net_gain:>+5d}     | {rec_pct:6.2f}% | {prec_pct:6.2f}%    | {fpr_pct:5.2f}%")

            if net_gain > best_net_gain:
                best_net_gain = net_gain
                best_row = {
                    "d": d, "tau": tau, "gated_pl": gated_pl, "mwpm_pl": mwpm_pl,
                    "rec": rec, "reg": reg, "net_gain": net_gain,
                    "mwpm_errs": int(np.sum(mwpm_wrong)), "gated_errs": int(np.sum(gated_preds != flips)),
                    "auroc": auroc, "auprc": auprc
                }

        optimal_results.append(best_row)
        print(f"  --> Best Operating Point at d={d}: tau = {best_row['tau']:.2f} | Net Gain = {best_row['net_gain']:+d} | Gated P_L = {best_row['gated_pl']:.3f}% (MWPM: {best_row['mwpm_pl']:.3f}%)")

    print("\n" + "=" * 135)
    print("PHASE F FAST BATCHED SUMMARY TABLE")
    print("=" * 135)
    print(f"{'Distance':<14} | {'MWPM P_L (95% CI)':<26} | {'Gated P_L (95% CI)':<26} | {'Optimal Tau':<12} | {'Rec / Reg':<12} | {'Net Gain':<10} | {'AUROC / AUPRC'}")
    print("-" * 135)

    for r in optimal_results:
        d = r["d"]
        _, m_l, m_u = wilson_score_interval(r["mwpm_errs"], test_shots)
        _, g_l, g_u = wilson_score_interval(r["gated_errs"], test_shots)
        mwpm_str = f"{r['mwpm_pl']:5.3f}% [{m_l*100:.3f}%, {m_u*100:.3f}%]"
        gated_str = f"{r['gated_pl']:5.3f}% [{g_l*100:.3f}%, {g_u*100:.3f}%]"
        rec_reg_str = f"{r['rec']:>4d} / {r['reg']:<4d}"
        held_str = " (HELD-OUT)" if d == test_distance else ""
        print(f"d = {d:<2d}{held_str:<9} | {mwpm_str:<26} | {gated_str:<26} | tau = {r['tau']:<6.2f} | {rec_reg_str:<12} | {r['net_gain']:>+5d}      | {r['auroc']:.3f} / {r['auprc']:.3f}")
    print("=" * 135 + "\n")

if __name__ == "__main__":
    run_fast_batched_gated_decoder()
