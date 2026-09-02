import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import time
import torch
import torch.nn as nn
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
        self.msg_mlp = nn.Sequential(nn.Linear(msg_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.node_update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_attr, is_par):
        src, dst = edge_index[0].long(), edge_index[1].long()
        msg_input = torch.cat([h[src], h[dst], edge_attr, is_par], dim=-1)
        messages = self.msg_mlp(msg_input)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)
        return self.norm(h + self.node_update(torch.cat([h, agg], dim=-1)))

class TopoOracle(nn.Module):
    def __init__(self, in_node_dim=6, in_edge_dim=4, hidden_dim=64, num_layers=6):
        super().__init__()
        self.node_embed = nn.Sequential(nn.Linear(in_node_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList([RelationalMessageLayer(hidden_dim, in_edge_dim) for _ in range(num_layers)])
        self.edge_scorer = nn.Sequential(nn.Linear(hidden_dim * 2 + in_edge_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Tanh())

    def forward(self, x6, edge_index, edge_attr, is_par, mask_diff, batch_map=None, num_graphs=None):
        h = self.node_embed(x6)
        src, dst = edge_index[0].long(), edge_index[1].long()
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, is_par)
        edge_feat = torch.cat([h[src], h[dst], edge_attr, is_par], dim=-1)
        edge_bias = self.edge_scorer(edge_feat)
        cycle_edges = (mask_diff.abs() > 0).float()
        cycle_lens = torch.bincount(batch_map[src], weights=cycle_edges.squeeze(-1), minlength=num_graphs).unsqueeze(-1).clamp(min=1)
        scale_factor = torch.sqrt(cycle_lens)
        diff_energy_sum = torch.zeros((num_graphs, 1), device=h.device)
        diff_energy_sum.index_add_(0, batch_map[src], edge_bias * mask_diff)
        return diff_energy_sum / scale_factor

def collate_cycle_batch(samples, device):
    x6_list, e_idx_list, e_attr_list, e_par_list, mask_diff_list, batch_map, y_list, delta_w_list = [], [], [], [], [], [], [], []
    node_offset = 0
    for i, s in enumerate(samples):
        n = s["x6"].shape[0]
        x6_list.append(s["x6"]); e_idx_list.append(s["e_idx"].long() + node_offset)
        e_attr_list.append(s["e_attr"]); e_par_list.append(s["e_par"].view(-1, 1) if s["e_par"].dim() == 1 else s["e_par"])
        mask_diff_list.append(s["mask_diff"]); batch_map.append(torch.full((n,), i, dtype=torch.long))
        y_list.append(s["y_target"]); delta_w_list.append(s["delta_w"])
        node_offset += n
    return (torch.cat(x6_list, dim=0).to(device), torch.cat(e_idx_list, dim=1).long().to(device),
            torch.cat(e_attr_list, dim=0).to(device), torch.cat(e_par_list, dim=0).to(device),
            torch.cat(mask_diff_list, dim=0).to(device), torch.tensor(delta_w_list, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.cat(batch_map, dim=0).long().to(device), torch.tensor(y_list, dtype=torch.float32, device=device).unsqueeze(-1), len(samples))

def collect_single_distance(d, p_val, eta, shots):
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
        edges_a_raw = matcher.decode_to_edges_array(s)
        C_A = set(standardize_edge(int(e[0]), int(e[1]), bnd_z, bnd_x) for e in edges_a_raw)
        obs_A = compute_chain_observable(C_A, edge_dict)
        C_B = C_A.symmetric_difference(R_L)
        C_0 = C_A if obs_A == 0 else C_B
        C_1 = C_B if obs_A == 0 else C_A
        w_0, w_1 = compute_chain_weight(C_0, edge_dict), compute_chain_weight(C_1, edge_dict)
        x4, x6, e_idx, e_attr, e_par, _, _, global_pairs = extract_active_subgraph_tensors(s, coords, raw_edge_dict, bnd_z, bnd_x, d, torch.device("cpu"))
        if e_idx.numel() == 0: continue
        mask_diff = torch.zeros((e_idx.shape[1], 1), dtype=torch.float32)
        for i, gp in enumerate(global_pairs):
            canon = standardize_edge(int(gp[0]), int(gp[1]), bnd_z, bnd_x)
            in_0, in_1 = canon in C_0, canon in C_1
            if in_1 and not in_0: mask_diff[i, 0] = 1.0
            elif in_0 and not in_1: mask_diff[i, 0] = -1.0
        samples.append({"x6": x6, "e_idx": e_idx, "e_attr": e_attr, "e_par": (e_par.view(-1) > 0).float(),
                        "mask_diff": mask_diff, "y_target": 1.0 if flips[idx] == 0 else -1.0, 
                        "delta_w": w_1 - w_0, "d_val": float(d), "obs_A": obs_A, "idx": idx})
    return {"samples": samples, "total_shots": shots, "preds_mwpm": preds_mwpm, "flips": flips}

def run_diagnostics():
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_val, eta, shots = 0.0035, 100.0, 10000

    train_samples = []
    for d in [3, 5, 7, 9]:
        dat = collect_single_distance(d, p_val, eta, shots)
        for s in dat["samples"]:
            y_true_obs = 0 if s["y_target"] == 1.0 else 1
            if y_true_obs != s["obs_A"] or abs(s["delta_w"]) <= (2.0 * d):
                train_samples.append(s)

    pool_fail = [s for s in train_samples if (0 if s["y_target"] == 1.0 else 1) != s["obs_A"]]
    pool_corr = [s for s in train_samples if (0 if s["y_target"] == 1.0 else 1) == s["obs_A"]]

    model = TopoOracle(in_node_dim=6, in_edge_dim=4, hidden_dim=64, num_layers=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("[+] Training model...")
    for epoch in range(1, 10):
        model.train()
        for _ in range(100):
            batch = [pool_fail[i] for i in np.random.choice(len(pool_fail), 64, replace=True)] + \
                    [pool_corr[j] for j in np.random.choice(len(pool_corr), 64, replace=True)]
            bx6, be_idx, be_attr, be_par, bmask, _, bmap, targets, n_g = collate_cycle_batch(batch, device)
            optimizer.zero_grad()
            loss = criterion(model(bx6, be_idx, be_attr, be_par, bmask, batch_map=bmap, num_graphs=n_g), targets)
            loss.backward(); optimizer.step()

    model.eval()
    print("[+] Evaluating Zero-Shot Transfer at Distance d = 11...\n")
    test_d11 = collect_single_distance(11, p_val, eta, shots)
    
    active_samples, preds_mwpm, flips = test_d11["samples"], test_d11["preds_mwpm"], test_d11["flips"]
    
    phi_list, obs_A_list, d_w_list, shot_indices = [], [], [], []
    for i in range(0, len(active_samples), 128):
        batch_raw = active_samples[i:i + 128]
        bx6, be_idx, be_attr, be_par, bmask, d_w, bmap, _, n_g = collate_cycle_batch(batch_raw, device)
        with torch.no_grad():
            phi_out = model(bx6, be_idx, be_attr, be_par, bmask, batch_map=bmap, num_graphs=n_g)
            phi_list.extend(phi_out.cpu().numpy().flatten())
        for s_dict in batch_raw:
            obs_A_list.append(s_dict["obs_A"]); d_w_list.append(s_dict["delta_w"]); shot_indices.append(s_dict["idx"])

    phi_arr, obs_A_arr, d_w_arr = np.array(phi_list), np.array(obs_A_list), np.array(d_w_list)

    # --- DIAGNOSTICS SECTION ---
    fail_sub_indices = [i for i, s in enumerate(active_samples) if (preds_mwpm[s["idx"]] != flips[s["idx"]])]
    print("=" * 80)
    print("GATE 3 DIAGNOSTICS FOR d=11 FAILURES")
    print("=" * 80)
    print(f"Total MWPM failures analyzed: {len(fail_sub_indices)}")
    
    if len(fail_sub_indices) > 0:
        d_w_fails = np.abs(d_w_arr[fail_sub_indices])
        phi_fails = np.abs(phi_arr[fail_sub_indices])
        
        print(f"\nDelta W (|ΔW|) distribution:")
        print(f"  Min: {np.min(d_w_fails):.2f} | Median: {np.median(d_w_fails):.2f} | Max: {np.max(d_w_fails):.2f}")
        
        print(f"\nOracle Confidence (|Phi|) distribution:")
        print(f"  Min: {np.min(phi_fails):.3f} | Median: {np.median(phi_fails):.3f} | Max: {np.max(phi_fails):.3f}")
        
        print(f"\nPotential flips if thresholds are relaxed:")
        for test_tau in [0.0, 0.1, 0.2, 0.3, 0.4]:
            correct_dir = 0
            for i in fail_sub_indices:
                phi, obs_a = phi_arr[i], obs_A_arr[i]
                if (obs_a == 0 and phi < -test_tau) or (obs_a == 1 and phi > test_tau):
                    correct_dir += 1
            print(f"  |Phi| > {test_tau:<3.1f} and correct sign: {correct_dir} / {len(fail_sub_indices)}")

if __name__ == "__main__":
    run_diagnostics()
