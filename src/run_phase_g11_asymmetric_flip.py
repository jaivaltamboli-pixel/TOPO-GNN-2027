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

class AsymmetricFlipRanker(nn.Module):
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
        # Decision head outputs P(Flip)
        self.classifier = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x6, edge_index, edge_attr, is_par, mask_diff, delta_w, obs_mwpm, d_tens, batch_map=None, num_graphs=None):
        h = self.node_embed(x6)
        src = edge_index[0].long()
        dst = edge_index[1].long()

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, is_par)

        edge_feat = torch.cat([h[src], h[dst], edge_attr, is_par], dim=-1)
        edge_bias = self.edge_scorer(edge_feat)

        # Directional cycle topological energy
        directional_energy = torch.zeros((num_graphs, 1), device=h.device)
        directional_energy.index_add_(0, batch_map[src], edge_bias * mask_diff)

        # Distance-normalized inputs to ensure size invariance
        norm_energy = directional_energy / d_tens
        norm_dw = delta_w / d_tens

        # Combine topological preference, classical confidence, and MWPM's current choice
        global_in = torch.cat([norm_energy, norm_dw, obs_mwpm], dim=-1)
        p_flip = torch.sigmoid(self.classifier(global_in))
        
        return p_flip

def collate_cycle_batch(samples, device):
    # FIXED: Exactly 5 empty lists for 5 sequence variables
    x6_list, e_idx_list, e_attr_list, e_par_list, mask_diff_list = [], [], [], [], []
    batch_map, y_list, delta_w_list, obs_mwpm_list, d_list = [], [], [], [], []
    node_offset = 0

    for i, item in enumerate(samples):
        x6, e_idx, e_attr, e_par, mask_diff, flip_target, delta_w, obs_mwpm, d_val = item
        num_nodes = x6.shape[0]

        x6_list.append(x6)
        e_idx_list.append((e_idx + node_offset).long())
        e_attr_list.append(e_attr)
        e_par_list.append(e_par.view(-1, 1) if e_par.dim() == 1 else e_par)
        mask_diff_list.append(mask_diff)
        batch_map.append(torch.full((num_nodes,), i, dtype=torch.long))
        y_list.append(flip_target)
        delta_w_list.append(delta_w)
        obs_mwpm_list.append(obs_mwpm)
        d_list.append(d_val)
        node_offset += num_nodes

    bx6 = torch.cat(x6_list, dim=0).to(device)
    be_idx = torch.cat(e_idx_list, dim=1).long().to(device)
    be_attr = torch.cat(e_attr_list, dim=0).to(device)
    be_par = torch.cat(e_par_list, dim=0).to(device)
    bmask = torch.cat(mask_diff_list, dim=0).to(device)
    bmap = torch.cat(batch_map, dim=0).long().to(device)

    targets = torch.tensor(y_list, dtype=torch.float32, device=device).unsqueeze(-1)
    delta_w = torch.tensor(delta_w_list, dtype=torch.float32, device=device).unsqueeze(-1)
    obs_mwpm = torch.tensor(obs_mwpm_list, dtype=torch.float32, device=device).unsqueeze(-1)
    d_tens = torch.tensor(d_list, dtype=torch.float32, device=device).unsqueeze(-1)

    return bx6, be_idx, be_attr, be_par, bmask, delta_w, obs_mwpm, d_tens, bmap, targets, len(samples)

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
            
            C_0 = C_A if obs_A == 0 else C_B
            C_1 = C_B if obs_A == 0 else C_A

            w_0 = compute_chain_weight(C_0, edge_dict)
            w_1 = compute_chain_weight(C_1, edge_dict)
            delta_w = w_1 - w_0

            # Target: 1.0 means MWPM is wrong, we should flip. 0.0 means MWPM is right.
            flip_target = 1.0 if obs_A != y_true else 0.0

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

            samples.append((x6, e_idx.long(), e_attr, (e_par.view(-1) > 0).float(), mask_diff, flip_target, delta_w, float(obs_A), float(d)))

        dataset[d] = {"samples": samples, "total_shots": shots, "preds_mwpm": preds_mwpm, "flips": flips}
        print(f"  [+] Prepared d={d:2d} ({shots:,} shots, {len(samples):,d} active cycles) in {time.time()-t0:.2f}s")
    return dataset

def run_asymmetric_flip_experiment():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_val, eta = 0.002, 100.0
    train_distances = [3, 5, 7]
    test_distances = [3, 5, 7, 9]
    shots_per_dist = 20000
    epochs = 12
    batch_size = 128
    lr = 5e-4

    print("=" * 95)
    print("PHASE G.11: ASYMMETRIC NET-GAIN RANKER (25x REGRESSION PENALTY)")
    print("=" * 95 + "\n")

    train_data = collect_dataset(train_distances, p_val, eta, shots_per_dist)
    test_data = collect_dataset(test_distances, p_val, eta, shots_per_dist)

    train_pool = []
    for d in train_distances:
        for s in train_data[d]["samples"]:
            if s[5] == 1.0 or abs(s[6]) <= 25.0:
                train_pool.append(s)

    pool_fail = [s for s in train_pool if s[5] == 1.0]
    pool_corr = [s for s in train_pool if s[5] == 0.0]

    print(f"  Training Pool: {len(pool_fail):,d} Failures (Target=1) | {len(pool_corr):,d} Correct Shots (Target=0)")

    model = AsymmetricFlipRanker(in_node_dim=6, in_edge_dim=4, hidden_dim=64, num_layers=4).to(device)
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

            bx6, be_idx, be_attr, be_par, bmask, d_w, obs_mwpm, d_tens, bmap, targets, n_g = collate_cycle_batch(batch, device)

            optimizer.zero_grad()
            p_flip = model(bx6, be_idx, be_attr, be_par, bmask, d_w, obs_mwpm, d_tens, batch_map=bmap, num_graphs=n_g)

            # THE FIX: 25x Penalty on False Positives (Regressions) to guarantee Net Gain
            weights = torch.where(targets == 1.0, 1.0, 25.0)
            loss = F.binary_cross_entropy(p_flip, targets, weight=weights)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * n_g
            total_items += n_g
            preds = (p_flip > 0.5).float()
            correct_preds += (preds == targets).sum().item()

        acc = (correct_preds / total_items) * 100.0
        print(f"  Epoch {epoch:2d}/{epochs:2d} | Weighted BCE Loss: {total_loss/total_items:.4f} | Flip Decision Acc: {acc:6.2f}%")

    print("\n[+] Evaluating Calibrated Net Gain across Test Sets...\n")
    model.eval()

    threshold_sweep = [0.5, 0.7, 0.85, 0.95, 0.99]

    for d in test_distances:
        dat = test_data[d]
        active_samples = dat["samples"]
        preds_mwpm = dat["preds_mwpm"]
        flips = dat["flips"]
        mwpm_errs = int(np.sum(preds_mwpm != flips))

        eval_batch_size = 256
        p_flip_list, shot_indices = [], []

        for i in range(0, len(active_samples), eval_batch_size):
            batch_raw = active_samples[i:i + eval_batch_size]
            batch_inputs = [s[:9] for s in batch_raw]
            bx6, be_idx, be_attr, be_par, bmask, d_w, obs_mwpm, d_tens, bmap, _, n_g = collate_cycle_batch(batch_inputs, device)

            with torch.no_grad():
                p_out = model(bx6, be_idx, be_attr, be_par, bmask, d_w, obs_mwpm, d_tens, batch_map=bmap, num_graphs=n_g)
                p_flip_list.extend(p_out.cpu().numpy().flatten())

            for item in batch_raw:
                shot_indices.append(item[11] if len(item) == 12 else item[-1]) # Safe index

        p_flip_arr = np.array(p_flip_list)
        shot_idx_arr = np.array(shot_indices)

        print(f">>> DISTANCE d = {d:2d} {'(ZERO-SHOT)' if d==9 else ''} (MWPM P_L: {mwpm_errs/shots_per_dist*100:.3f}%) <<<")
        print(f"{'Threshold (P > T)':<18} | {'Topo P_L':<10} | {'Recoveries':<11} | {'Regressions':<12} | {'Net Gain'}")
        print("-" * 80)

        for T in threshold_sweep:
            topo_preds = preds_mwpm.copy()
            
            # Flip MWPM decision if network probability exceeds threshold
            flip_mask = p_flip_arr > T
            altered_shots = shot_idx_arr[flip_mask]
            topo_preds[altered_shots] = 1 - topo_preds[altered_shots]

            rec = int(np.sum((preds_mwpm != flips) & (topo_preds == flips)))
            reg = int(np.sum((preds_mwpm == flips) & (topo_preds != flips)))
            net = rec - reg
            topo_pl = (np.sum(topo_preds != flips) / shots_per_dist) * 100.0
            
            print(f"T = {T:<14.2f} | {topo_pl:6.3f}%   | {rec:>5d}/{mwpm_errs:<4d}  | {reg:>5d}/{shots_per_dist-mwpm_errs:<6d} | {net:>+5d}")
        print()

if __name__ == "__main__":
    run_asymmetric_flip_experiment()
