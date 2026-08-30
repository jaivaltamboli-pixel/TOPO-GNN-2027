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

class RelationalCandidateLayer(nn.Module):
    def __init__(self, hidden_dim=64, edge_dim=5):
        super().__init__()
        msg_dim = hidden_dim * 2 + edge_dim + 1
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

    def forward(self, h, edge_index, edge_attr_5d, is_par):
        src = edge_index[0].long()
        dst = edge_index[1].long()
        msg_input = torch.cat([h[src], h[dst], edge_attr_5d, is_par], dim=-1)
        messages = self.msg_mlp(msg_input)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)
        return self.norm(h + self.node_update(torch.cat([h, agg], dim=-1)))

class BatchedTopoCandidateRanker(nn.Module):
    def __init__(self, in_node_dim=6, in_edge_dim=5, hidden_dim=64, num_layers=4):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(in_node_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList([
            RelationalCandidateLayer(hidden_dim=hidden_dim, edge_dim=in_edge_dim) 
            for _ in range(num_layers)
        ])
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x6, edge_index, edge_attr_5d, is_par, batch_map=None, num_graphs=None):
        h = self.node_embed(x6)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr_5d, is_par)

        if batch_map is None:
            pooled = h.mean(dim=0, keepdim=True)
        else:
            pooled = torch.zeros((num_graphs, h.shape[-1]), device=h.device)
            pooled.index_add_(0, batch_map.long(), h)
            counts = torch.bincount(batch_map.long(), minlength=num_graphs).unsqueeze(-1).clamp(min=1).float()
            pooled = pooled / counts

        return self.scorer(pooled)

def collate_candidate_pairs(samples, device):
    x6_list, e_idx_list, e_attr_A_list, e_attr_B_list, e_par_list = [], [], [], [], []
    batch_map, y_list, w_A_list, w_B_list = [], [], [], []
    node_offset = 0

    for i, item in enumerate(samples):
        x6, e_idx, e_attr, e_par, mask_A, mask_B, y_val, w_A, w_B = item
        num_nodes = x6.shape[0]

        x6_list.append(x6)
        e_idx_list.append((e_idx + node_offset).long())
        e_attr_A_list.append(torch.cat([e_attr, mask_A], dim=-1))
        e_attr_B_list.append(torch.cat([e_attr, mask_B], dim=-1))
        e_par_list.append(e_par.view(-1, 1) if e_par.dim() == 1 else e_par)
        batch_map.append(torch.full((num_nodes,), i, dtype=torch.long))
        
        y_list.append(y_val)
        w_A_list.append(w_A)
        w_B_list.append(w_B)
        node_offset += num_nodes

    bx6 = torch.cat(x6_list, dim=0).to(device)
    be_idx = torch.cat(e_idx_list, dim=1).long().to(device)
    be_attr_A = torch.cat(e_attr_A_list, dim=0).to(device)
    be_attr_B = torch.cat(e_attr_B_list, dim=0).to(device)
    be_par = torch.cat(e_par_list, dim=0).to(device)
    bmap = torch.cat(batch_map, dim=0).long().to(device)
    
    targets = torch.tensor(y_list, dtype=torch.float32, device=device).unsqueeze(-1)
    weights_A = torch.tensor(w_A_list, dtype=torch.float32, device=device).unsqueeze(-1)
    weights_B = torch.tensor(w_B_list, dtype=torch.float32, device=device).unsqueeze(-1)
    
    return bx6, be_idx, be_attr_A, be_attr_B, be_par, bmap, targets, weights_A, weights_B, len(samples)

def collect_pairwise_dataset(distances, p_val, eta, shots):
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
            C_B = C_A.symmetric_difference(R_L)

            obs_A = compute_chain_observable(C_A, edge_dict)
            obs_B = compute_chain_observable(C_B, edge_dict)
            w_A = compute_chain_weight(C_A, edge_dict)
            w_B = compute_chain_weight(C_B, edge_dict)

            y_target = 1.0 if obs_A == y_true else -1.0

            x4, x6, e_idx, e_attr, e_par, _, _, global_pairs = extract_active_subgraph_tensors(
                s, coords, raw_edge_dict, bnd_z, bnd_x, d, torch.device("cpu")
            )
            if e_idx.numel() == 0:
                continue

            mask_A = torch.zeros((e_idx.shape[1], 1), dtype=torch.float32)
            mask_B = torch.zeros((e_idx.shape[1], 1), dtype=torch.float32)

            for i, gp in enumerate(global_pairs):
                canon = standardize_edge(int(gp[0]), int(gp[1]), bnd_z, bnd_x)
                if canon in C_A: mask_A[i, 0] = 1.0
                if canon in C_B: mask_B[i, 0] = 1.0

            samples.append((x6, e_idx.long(), e_attr, (e_par.view(-1) > 0).float(), mask_A, mask_B, y_target, w_A, w_B, obs_A, obs_B, idx))

        dataset[d] = {
            "samples": samples,
            "total_shots": shots,
            "preds_mwpm": preds_mwpm,
            "flips": flips
        }
        print(f"  [+] Prepared d={d:2d} ({shots:,} shots, {len(samples):,d} active pairs) in {time.time()-t0:.2f}s")
    return dataset

def run_calibrated_evaluation():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_val, eta = 0.002, 100.0
    train_distances = [3, 5, 7]
    test_distances = [3, 5, 7, 9]
    shots_per_dist = 10000
    epochs = 8
    batch_size = 128
    lr = 1e-3

    print("=" * 90)
    print("PHASE G.1: ENERGY-CALIBRATED CANDIDATE RANKER EVALUATION")
    print(f"Decision Rule: Select C_B iff (Phi_B - Phi_A) > (W_B - W_A) / alpha + tau")
    print("=" * 90 + "\n")

    train_data = collect_pairwise_dataset(train_distances, p_val, eta, shots_per_dist)
    test_data = collect_pairwise_dataset(test_distances, p_val, eta, shots_per_dist)

    train_pool = []
    for d in train_distances:
        train_pool.extend(train_data[d]["samples"])
    
    pos_samples = [s[:9] for s in train_pool if s[6] == 1.0]
    neg_samples = [s[:9] for s in train_pool if s[6] == -1.0]

    model = BatchedTopoCandidateRanker(in_node_dim=6, in_edge_dim=5, hidden_dim=64, num_layers=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    steps_per_epoch = 200
    half_b = batch_size // 2

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            p_idx = np.random.choice(len(pos_samples), half_b, replace=True)
            n_idx = np.random.choice(len(neg_samples), half_b, replace=True)
            batch = [pos_samples[i] for i in p_idx] + [neg_samples[j] for j in n_idx]

            bx6, be_idx, be_attr_A, be_attr_B, be_par, bmap, targets, _, _, n_g = collate_candidate_pairs(batch, device)
            optimizer.zero_grad()
            phi_A = model(bx6, be_idx, be_attr_A, be_par, batch_map=bmap, num_graphs=n_g)
            phi_B = model(bx6, be_idx, be_attr_B, be_par, batch_map=bmap, num_graphs=n_g)

            delta_phi = phi_A - phi_B
            loss = F.softplus(-targets * delta_phi).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    print("[+] Training completed. Sweeping alpha calibration across test sets...\n")
    model.eval()

    # Alpha represents the effective neural temperature: how much neural evidence balances 1 unit of Delta W
    alpha_sweep = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]

    for d in test_distances:
        dat = test_data[d]
        active_samples = dat["samples"]
        preds_mwpm = dat["preds_mwpm"]
        flips = dat["flips"]
        mwpm_errs = int(np.sum(preds_mwpm != flips))

        # Precompute model outputs for all active shots
        eval_batch_size = 256
        phi_diffs = []
        w_diffs = []
        obs_A_list = []
        obs_B_list = []
        shot_indices = []

        for i in range(0, len(active_samples), eval_batch_size):
            batch_raw = active_samples[i:i + eval_batch_size]
            batch_inputs = [s[:9] for s in batch_raw]
            bx6, be_idx, be_attr_A, be_attr_B, be_par, bmap, _, w_A, w_B, n_g = collate_candidate_pairs(batch_inputs, device)

            with torch.no_grad():
                phi_A = model(bx6, be_idx, be_attr_A, be_par, batch_map=bmap, num_graphs=n_g)
                phi_B = model(bx6, be_idx, be_attr_B, be_par, batch_map=bmap, num_graphs=n_g)
                d_phi = (phi_B - phi_A).cpu().numpy().flatten()
                d_w = (w_B - w_A).cpu().numpy().flatten()

            phi_diffs.extend(d_phi)
            w_diffs.extend(d_w)
            for item in batch_raw:
                obs_A_list.append(item[9])
                obs_B_list.append(item[10])
                shot_indices.append(item[11])

        d_phi_arr = np.array(phi_diffs)
        d_w_arr = np.array(w_diffs)

        print(f">>> DISTANCE d = {d:2d} {'(ZERO-SHOT)' if d==9 else ''} (MWPM P_L: {mwpm_errs/shots_per_dist*100:.3f}%) <<<")
        print(f"{'Alpha':<8} | {'Topo P_L':<10} | {'Recoveries':<11} | {'Regressions':<12} | {'Net Gain':<10} | {'Flips Altered'}")
        print("-" * 75)

        for alpha in alpha_sweep:
            topo_preds = preds_mwpm.copy()
            # Select Candidate B iff neural support overcomes the classical weight barrier:
            # (Phi_B - Phi_A) > (W_B - W_A) / alpha
            flip_mask = d_phi_arr > (d_w_arr / alpha)
            altered_indices = np.where(flip_mask)[0]

            for idx_in_list in altered_indices:
                shot_idx = shot_indices[idx_in_list]
                topo_preds[shot_idx] = obs_B_list[idx_in_list]

            rec = int(np.sum((preds_mwpm != flips) & (topo_preds == flips)))
            reg = int(np.sum((preds_mwpm == flips) & (topo_preds != flips)))
            net = rec - reg
            topo_pl = (np.sum(topo_preds != flips) / shots_per_dist) * 100.0
            print(f"{alpha:<8.1f} | {topo_pl:6.3f}%   | {rec:>5d}/{mwpm_errs:<4d}  | {reg:>5d}/{shots_per_dist-mwpm_errs:<6d} | {net:>+5d}     | {len(altered_indices):>5d}")
        print()

if __name__ == "__main__":
    run_calibrated_evaluation()
