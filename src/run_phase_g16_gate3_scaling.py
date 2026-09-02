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

class TopoOracle(nn.Module):
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
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, x6, edge_index, edge_attr, is_par, mask_diff, batch_map=None, num_graphs=None):
        h = self.node_embed(x6)
        src = edge_index[0].long()
        dst = edge_index[1].long()

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, is_par)

        edge_feat = torch.cat([h[src], h[dst], edge_attr, is_par], dim=-1)
        edge_bias = self.edge_scorer(edge_feat)

        cycle_edges = (mask_diff.abs() > 0).float()
        cycle_counts = torch.bincount(batch_map[src], weights=cycle_edges.squeeze(-1), minlength=num_graphs).unsqueeze(-1).clamp(min=1)

        diff_energy_sum = torch.zeros((num_graphs, 1), device=h.device)
        diff_energy_sum.index_add_(0, batch_map[src], edge_bias * mask_diff)
        return diff_energy_sum / cycle_counts

def collate_cycle_batch(samples, device):
    x6_list, e_idx_list, e_attr_list, e_par_list, mask_diff_list = [], [], [], [], []
    batch_map, y_list, delta_w_list = [], [], []
    node_offset = 0

    for i, s in enumerate(samples):
        num_nodes = s["x6"].shape[0]
        x6_list.append(s["x6"])
        e_idx_list.append(s["e_idx"].long() + node_offset)
        e_attr_list.append(s["e_attr"])
        e_par_list.append(s["e_par"].view(-1, 1) if s["e_par"].dim() == 1 else s["e_par"])
        mask_diff_list.append(s["mask_diff"])
        batch_map.append(torch.full((num_nodes,), i, dtype=torch.long))
        y_list.append(s["y_target"])
        delta_w_list.append(s["delta_w"])
        node_offset += num_nodes

    bx6 = torch.cat(x6_list, dim=0).to(device)
    be_idx = torch.cat(e_idx_list, dim=1).long().to(device)
    be_attr = torch.cat(e_attr_list, dim=0).to(device)
    be_par = torch.cat(e_par_list, dim=0).to(device)
    bmask = torch.cat(mask_diff_list, dim=0).to(device)
    bmap = torch.cat(batch_map, dim=0).long().to(device)

    targets = torch.tensor(y_list, dtype=torch.float32, device=device).unsqueeze(-1)
    delta_w = torch.tensor(delta_w_list, dtype=torch.float32, device=device).unsqueeze(-1)
    return bx6, be_idx, be_attr, be_par, bmask, delta_w, bmap, targets, len(samples)

def collect_single_distance(d, p_val, eta, shots):
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
        
        C_0 = C_A if obs_A == 0 else C_B
        C_1 = C_B if obs_A == 0 else C_A

        w_0 = compute_chain_weight(C_0, edge_dict)
        w_1 = compute_chain_weight(C_1, edge_dict)
        delta_w = w_1 - w_0
        y_target = 1.0 if y_true == 0 else -1.0

        x4, x6, e_idx, e_attr, e_par, _, _, global_pairs = extract_active_subgraph_tensors(
            s, coords, raw_edge_dict, bnd_z, bnd_x, d, torch.device("cpu")
        )
        if e_idx.numel() == 0: continue

        mask_diff = torch.zeros((e_idx.shape[1], 1), dtype=torch.float32)
        for i, gp in enumerate(global_pairs):
            canon = standardize_edge(int(gp[0]), int(gp[1]), bnd_z, bnd_x)
            in_0 = canon in C_0
            in_1 = canon in C_1
            if in_1 and not in_0: mask_diff[i, 0] = 1.0
            elif in_0 and not in_1: mask_diff[i, 0] = -1.0

        samples.append({
            "x6": x6, "e_idx": e_idx, "e_attr": e_attr, "e_par": (e_par.view(-1) > 0).float(),
            "mask_diff": mask_diff, "y_target": y_target, "delta_w": delta_w,
            "d_val": float(d), "obs_A": obs_A, "idx": idx
        })

    print(f"  [+] Prepared d={d} ({shots:,} shots, {len(samples):,d} active cycles) in {time.time()-t0:.2f}s")
    return {"samples": samples, "total_shots": shots, "preds_mwpm": preds_mwpm, "flips": flips}

def run_gate3_scaling():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_val, eta = 0.002, 100.0
    shots = 10000  # Scaled slightly for higher distance memory constraints

    print("=" * 95)
    print("PHASE G.16: GATE 3 SIZE-TRANSFERABILITY SCALING (d=11, 13)")
    print("=" * 95 + "\n")

    train_samples = []
    for d in [3, 5, 7]:
        dat = collect_single_distance(d, p_val, eta, shots)
        for s in dat["samples"]:
            y_true_obs = 0 if s["y_target"] == 1.0 else 1
            if y_true_obs != s["obs_A"] or abs(s["delta_w"]) <= (2.5 * d):
                train_samples.append(s)

    pool_fail = [s for s in train_samples if (0 if s["y_target"] == 1.0 else 1) != s["obs_A"]]
    pool_corr = [s for s in train_samples if (0 if s["y_target"] == 1.0 else 1) == s["obs_A"]]

    print(f"\n  Training Pool: {len(pool_fail):,d} Failures | {len(pool_corr):,d} Ambiguous Correct Shots")

    model = TopoOracle(in_node_dim=6, in_edge_dim=4, hidden_dim=64, num_layers=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(1, 10):
        model.train()
        total_loss, correct_preds, total_items = 0.0, 0, 0
        for _ in range(120):
            idx_f = np.random.choice(len(pool_fail), 64, replace=True)
            idx_c = np.random.choice(len(pool_corr), 64, replace=True)
            batch = [pool_fail[i] for i in idx_f] + [pool_corr[j] for j in idx_c]

            bx6, be_idx, be_attr, be_par, bmask, _, bmap, targets, n_g = collate_cycle_batch(batch, device)
            optimizer.zero_grad()
            phi_out = model(bx6, be_idx, be_attr, be_par, bmask, batch_map=bmap, num_graphs=n_g)
            loss = criterion(phi_out, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * n_g
            total_items += n_g
            preds = torch.where(phi_out > 0, 1.0, -1.0)
            correct_preds += (preds == targets).sum().item()
        print(f"  Epoch {epoch:2d}/9 | MSE Loss: {total_loss/total_items:.4f} | Oracle Acc: {(correct_preds/total_items)*100:6.2f}%")

    model.eval()
    for d_target in [11, 13]:
        print(f"\n[+] Evaluating Zero-Shot Transfer at Distance d = {d_target}...\n")
        test_d = collect_single_distance(d_target, p_val, eta, shots)
        
        active_samples = test_d["samples"]
        preds_mwpm = test_d["preds_mwpm"]
        flips = test_d["flips"]
        mwpm_errs = int(np.sum(preds_mwpm != flips))

        phi_list, obs_A_list, d_w_list, shot_indices = [], [], [], []
        for i in range(0, len(active_samples), 128):
            batch_raw = active_samples[i:i + 128]
            bx6, be_idx, be_attr, be_par, bmask, d_w, bmap, _, n_g = collate_cycle_batch(batch_raw, device)
            with torch.no_grad():
                phi_out = model(bx6, be_idx, be_attr, be_par, bmask, batch_map=bmap, num_graphs=n_g)
                phi_list.extend(phi_out.cpu().numpy().flatten())
            for s_dict in batch_raw:
                obs_A_list.append(s_dict["obs_A"])
                d_w_list.append(s_dict["delta_w"])
                shot_indices.append(s_dict["idx"])

        phi_arr, obs_A_arr, d_w_arr = np.array(phi_list), np.array(obs_A_list), np.array(d_w_list)

        print(f">>> DISTANCE d = {d_target} (MWPM P_L: {mwpm_errs/shots*100:.3f}%) <<<")
        print(f"{'Max dW (k*d)':<14} | {'Min Topo (tau)':<14} | {'Topo P_L':<10} | {'Recoveries':<11} | {'Regressions':<12} | {'Net Gain'}")
        print("-" * 90)

        for k in [1.5, 2.0, 2.5]:
            for tau in [0.75, 0.90]:
                topo_preds = preds_mwpm.copy()
                max_dw = k * float(d_target)

                for phi, obs_a, dw, s_idx in zip(phi_arr, obs_A_arr, d_w_arr, shot_indices):
                    if abs(dw) <= max_dw:
                        if obs_a == 0 and phi < -tau:
                            topo_preds[s_idx] = 1
                        elif obs_a == 1 and phi > tau:
                            topo_preds[s_idx] = 0

                rec = int(np.sum((preds_mwpm != flips) & (topo_preds == flips)))
                reg = int(np.sum((preds_mwpm == flips) & (topo_preds != flips)))
                net = rec - reg
                topo_pl = (np.sum(topo_preds != flips) / shots) * 100.0
                print(f"<= {max_dw:<11.1f} | > {tau:<12.2f} | {topo_pl:6.3f}%   | {rec:>5d}/{mwpm_errs:<4d}  | {reg:>5d}/{shots-mwpm_errs:<6d} | {net:>+5d}")

if __name__ == "__main__":
    run_gate3_scaling()
