import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from audit_phase_a_d_opportunity import (
    build_parity_expanded_graph,
    find_exact_logical_reference_chain,
    standardize_edge,
    compute_chain_observable,
    compute_chain_weight
)

def wilson_score_interval(errors, total_shots, z=1.96):
    if total_shots == 0:
        return 0.0, 0.0, 0.0
    p = errors / total_shots
    denom = 1.0 + (z**2) / total_shots
    centre = (p + (z**2) / (2 * total_shots)) / denom
    margin = z * np.sqrt((p * (1 - p) + (z**2) / (4 * total_shots)) / total_shots) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)

class RelationalMessageLayer(nn.Module):
    def __init__(self, hidden_dim=64, in_edge_dim=4):
        super().__init__()
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

    def forward(self, h, edge_index, edge_attr, is_par):
        src = edge_index[0].long()
        dst = edge_index[1].long()
        msg_input = torch.cat([h[src], h[dst], edge_attr, is_par], dim=-1)
        messages = self.msg_mlp(msg_input)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)
        return self.norm(h + self.node_update(torch.cat([h, agg], dim=-1)))

class PrecisionCycleRanker(nn.Module):
    def __init__(self, in_node_dim=6, in_edge_dim=4, hidden_dim=64, num_layers=4):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(in_node_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList([
            RelationalMessageLayer(hidden_dim=hidden_dim, in_edge_dim=in_edge_dim)
            for _ in range(num_layers)
        ])
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + in_edge_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.global_arbitrator = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x6, edge_index, edge_attr, is_par, mask_diff, delta_w, num_defects, batch_map=None, num_graphs=None):
        h = self.node_embed(x6)
        src = edge_index[0].long()
        dst = edge_index[1].long()

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, is_par)

        edge_feat = torch.cat([h[src], h[dst], edge_attr, is_par], dim=-1)
        edge_bias = self.edge_scorer(edge_feat)

        cycle_edges = (mask_diff.abs() > 0).float()
        pooled_cycle = torch.zeros((num_graphs, h.shape[-1]), device=h.device)
        pooled_cycle.index_add_(0, batch_map[src], edge_feat[:, :h.shape[-1]] * cycle_edges)

        cycle_counts = torch.bincount(batch_map[src], weights=cycle_edges.squeeze(-1), minlength=num_graphs).unsqueeze(-1).clamp(min=1)
        pooled_cycle = pooled_cycle / cycle_counts

        diff_energy = torch.zeros((num_graphs, 1), device=h.device)
        diff_energy.index_add_(0, batch_map[src], edge_bias * mask_diff)

        global_in = torch.cat([pooled_cycle, delta_w, num_defects], dim=-1)
        confidence = torch.tanh(self.global_arbitrator(global_in))

        return diff_energy + confidence * 3.0

def collate_cycle_batch(samples, device):
    x6_list, e_idx_list, e_attr_list, e_par_list, mask_diff_list = [], [], [], [], []
    batch_map, y_list, delta_w_list, n_def_list = [], [], [], []
    node_offset = 0

    for i, item in enumerate(samples):
        x6, e_idx, e_attr, e_par, mask_diff, y_val, delta_w, n_def = item
        num_nodes = x6.shape[0]

        x6_list.append(x6)
        e_idx_list.append((e_idx + node_offset).long())
        e_attr_list.append(e_attr)
        e_par_list.append(e_par.view(-1, 1) if e_par.dim() == 1 else e_par)
        mask_diff_list.append(mask_diff)
        batch_map.append(torch.full((num_nodes,), i, dtype=torch.long))
        y_list.append(y_val)
        delta_w_list.append(delta_w)
        n_def_list.append(n_def)
        node_offset += num_nodes

    bx6 = torch.cat(x6_list, dim=0).to(device)
    be_idx = torch.cat(e_idx_list, dim=1).long().to(device)
    be_attr = torch.cat(e_attr_list, dim=0).to(device)
    be_par = torch.cat(e_par_list, dim=0).to(device)
    bmask = torch.cat(mask_diff_list, dim=0).to(device)
    bmap = torch.cat(batch_map, dim=0).long().to(device)

    targets = torch.tensor(y_list, dtype=torch.float32, device=device).unsqueeze(-1)
    delta_w = torch.tensor(delta_w_list, dtype=torch.float32, device=device).unsqueeze(-1)
    num_defs = torch.tensor(n_def_list, dtype=torch.float32, device=device).unsqueeze(-1)
    return bx6, be_idx, be_attr, be_par, bmask, delta_w, num_defs, bmap, targets, len(samples)

def collect_dataset(distances, p_val, eta, shots):
    dataset = {}
    for d in distances:
        t0 = time.time()
        circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
        dem = circuit.detector_error_model(decompose_errors=True)
        coords = circuit.get_detector_coordinates()
        num_dets = circuit.num_detectors

        edge_dict, adj, bnd_z, bnd_x = build_parity_expanded_graph(dem, num_dets, coords, d)
        raw_edge_dict, _, _, _ = extract_complete_dem_graph(dem, num_dets, coords, d)
        R_L, _ = find_exact_logical_reference_chain(adj, num_dets)

        matcher = pymatching.Matching.from_detector_error_model(dem)
        sampler = circuit.compile_detector_sampler()
        syn, flips = sampler.sample(shots=shots, separate_observables=True)
        flips = flips.flatten().astype(np.int64)

        preds_mwpm = matcher.decode_batch(syn).flatten().astype(np.int64)
        active_indices = np.where(np.sum(syn, axis=1) >= 2)[0]

        samples = []
        for idx in active_indices:
            s = syn[idx].astype(np.uint8)
            y_true = flips[idx]

            edges_a_raw = matcher.decode_to_edges_array(s)
            C_A = set(standardize_edge(int(e[0]), int(e[1]), bnd_z, bnd_x) for e in edges_a_raw)
            obs_A = compute_chain_observable(C_A, edge_dict)
            C_B = C_A.symmetric_difference(R_L)
            obs_B = compute_chain_observable(C_B, edge_dict)

            C_0 = C_A if obs_A == 0 else C_B
            C_1 = C_B if obs_A == 0 else C_A

            w_0 = compute_chain_weight(C_0, edge_dict)
            w_1 = compute_chain_weight(C_1, edge_dict)
            delta_w = w_1 - w_0

            y_target = 1.0 if y_true == 0 else -1.0
            n_def = float(np.sum(s))

            x4, x6, e_idx, e_attr, e_par, _, _, global_pairs = extract_active_subgraph_tensors(
                s, coords, raw_edge_dict, bnd_z, bnd_x, d, torch.device("cpu")
            )
            if e_idx.numel() == 0:
                continue

            mask_diff = torch.zeros((e_idx.shape[1], 1), dtype=torch.float32)
            for i, gp in enumerate(global_pairs):
                canon = standardize_edge(int(gp[0]), int(gp[1]), bnd_z, bnd_x)
                in_0 = canon in C_0
                in_1 = canon in C_1
                if in_1 and not in_0: mask_diff[i, 0] = 1.0
                elif in_0 and not in_1: mask_diff[i, 0] = -1.0

            samples.append((x6, e_idx.long(), e_attr, (e_par.view(-1) > 0).float(), mask_diff, y_target, delta_w, n_def, obs_A, y_true, idx))

        dataset[d] = {"samples": samples, "total_shots": shots, "preds_mwpm": preds_mwpm, "flips": flips}
        print(f"  [+] Prepared d={d:2d} ({shots:,} shots, {len(samples):,d} active cycles) in {time.time()-t0:.2f}s")
    return dataset

def run_precision_gate():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_val, eta = 0.002, 100.0
    train_distances = [3, 5, 7]
    test_distances = [3, 5, 7, 9]
    shots_per_dist = 10000
    epochs = 12
    batch_size = 128
    lr = 5e-4

    print("=" * 95)
    print("PHASE G.5: DISTANCE-NORMALIZED PRECISION GATE")
    print("=" * 95 + "\n")

    train_data = collect_dataset(train_distances, p_val, eta, shots_per_dist)
    test_data = collect_dataset(test_distances, p_val, eta, shots_per_dist)

    train_pool = []
    for d in train_distances:
        train_pool.extend(train_data[d]["samples"])

    # High-signal training pairs: failures vs borderline correct shots (Delta W <= 6.0)
    pool_fail = [s[:8] for s in train_pool if s[8] != s[9]]
    pool_corr = [s[:8] for s in train_pool if (s[8] == s[9] and abs(s[6]) <= 6.0)]

    print(f"  Training Pool: {len(pool_fail):,d} Failures | {len(pool_corr):,d} Borderline Correct Shots")

    model = PrecisionCycleRanker(in_node_dim=6, in_edge_dim=4, hidden_dim=64, num_layers=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    steps_per_epoch = 150
    half_b = batch_size // 2

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct_preds, total_items = 0.0, 0, 0

        for _ in range(steps_per_epoch):
            idx_f = np.random.choice(len(pool_fail), half_b, replace=True)
            idx_c = np.random.choice(len(pool_corr), half_b, replace=True)
            batch = [pool_fail[i] for i in idx_f] + [pool_corr[j] for j in idx_c]

            bx6, be_idx, be_attr, be_par, bmask, d_w, n_def, bmap, targets, n_g = collate_cycle_batch(batch, device)

            optimizer.zero_grad()
            energy = model(bx6, be_idx, be_attr, be_par, bmask, d_w, n_def, batch_map=bmap, num_graphs=n_g)

            loss = F.softplus(-targets * energy).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * n_g
            total_items += n_g
            preds = (energy > 0).float() * 2.0 - 1.0
            correct_preds += (preds == targets).sum().item()

        acc = (correct_preds / total_items) * 100.0
        print(f"  Epoch {epoch:2d}/{epochs:2d} | Loss: {total_loss/total_items:.4f} | Accuracy: {acc:6.2f}%")

    print("\n[+] Evaluating Distance-Normalized Thresholds across Test Sets...\n")
    model.eval()

    # Distance-normalized threshold sweep: tau_scaled = tau * sqrt(d)
    base_tau_sweep = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    for d in test_distances:
        dat = test_data[d]
        active_samples = dat["samples"]
        preds_mwpm = dat["preds_mwpm"]
        flips = dat["flips"]
        mwpm_errs = int(np.sum(preds_mwpm != flips))

        eval_batch_size = 256
        energies, d_w_list, obs_A_list, shot_indices = [], [], [], []

        for i in range(0, len(active_samples), eval_batch_size):
            batch_raw = active_samples[i:i + eval_batch_size]
            batch_inputs = [s[:8] for s in batch_raw]
            bx6, be_idx, be_attr, be_par, bmask, d_w, n_def, bmap, _, n_g = collate_cycle_batch(batch_inputs, device)

            with torch.no_grad():
                e_out = model(bx6, be_idx, be_attr, be_par, bmask, d_w, n_def, batch_map=bmap, num_graphs=n_g)
                energies.extend(e_out.cpu().numpy().flatten())
                d_w_list.extend(d_w.cpu().numpy().flatten())

            for item in batch_raw:
                obs_A_list.append(item[8])
                shot_indices.append(item[10])

        e_arr = np.array(energies)
        d_w_arr = np.array(d_w_list)

        print(f">>> DISTANCE d = {d:2d} {'(ZERO-SHOT)' if d==9 else ''} (MWPM P_L: {mwpm_errs/shots_per_dist*100:.3f}%) <<<")
        print(f"{'Base Tau':<10} | {'Effective Tau':<14} | {'Topo P_L':<10} | {'Recoveries':<11} | {'Regressions':<12} | {'Net Gain'}")
        print("-" * 80)

        for base_tau in base_tau_sweep:
            eff_tau = base_tau * np.sqrt(d)
            topo_preds = preds_mwpm.copy()

            for idx_k, (e_val, dw_val, obs_a, s_idx) in enumerate(zip(e_arr, d_w_arr, obs_A_list, shot_indices)):
                # Strictly evaluate the ambiguous boundary window (Delta W <= 4.5)
                if abs(dw_val) <= 4.5:
                    if obs_a == 0 and e_val < -eff_tau:
                        topo_preds[s_idx] = 1
                    elif obs_a == 1 and e_val > eff_tau:
                        topo_preds[s_idx] = 0

            rec = int(np.sum((preds_mwpm != flips) & (topo_preds == flips)))
            reg = int(np.sum((preds_mwpm == flips) & (topo_preds != flips)))
            net = rec - reg
            topo_pl = (np.sum(topo_preds != flips) / shots_per_dist) * 100.0
            print(f"{base_tau:<10.1f} | {eff_tau:<14.2f} | {topo_pl:6.3f}%   | {rec:>5d}/{mwpm_errs:<4d}  | {reg:>5d}/{shots_per_dist-mwpm_errs:<6d} | {net:>+5d}")
        print()

if __name__ == "__main__":
    run_precision_gate()
