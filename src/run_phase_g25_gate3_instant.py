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
from utils.graph_builder import extract_complete_dem_graph
from audit_phase_a_d_opportunity import (
    build_parity_expanded_graph,
    find_exact_logical_reference_chain,
    standardize_edge,
    compute_chain_observable,
    compute_chain_weight
)

# --- ARCHITECTURE ---
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
        diff_energy_sum = torch.zeros((num_graphs, 1), device=h.device)
        diff_energy_sum.index_add_(0, batch_map[src], edge_bias * mask_diff)
        return diff_energy_sum / torch.sqrt(cycle_lens)

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

def collect_single_distance_fast(d, p_val, eta, shots):
    t0 = time.time()
    circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
    dem = circuit.detector_error_model(decompose_errors=True)
    coords = circuit.get_detector_coordinates()
    num_dets = circuit.num_detectors
    edge_dict, adj, bnd_z, bnd_x = build_parity_expanded_graph(dem, num_dets, coords, d)
    raw_edge_dict, _, _, _ = extract_complete_dem_graph(dem, num_dets, coords, d)
    R_L, _ = find_exact_logical_reference_chain(adj, num_dets)
    
    # --- INSTANT GRAPH CACHE ---
    max_idx = max(max(u, v) for u, v in raw_edge_dict.keys())
    num_nodes = max_idx + 1
    base_x6 = torch.zeros((num_nodes, 6), dtype=torch.float32)
    for i, (z, x, y) in coords.items():
        base_x6[i, 0], base_x6[i, 1], base_x6[i, 2] = z / (d + 1), x / (d + 1), y / (d + 1)
    if bnd_z < num_nodes: base_x6[bnd_z, 3] = 1.0
    if bnd_x < num_nodes: base_x6[bnd_x, 4] = 1.0
        
    src, dst, attr, par, canon_edges = [], [], [], [], []
    for (u, v), data in raw_edge_dict.items():
        src.extend([u, v]); dst.extend([v, u])
        w = float(data.get('weight', 0.0))
        w_norm = min(w, 50.0) / 10.0  # Gradient-safe normalization
        e_feat = [w_norm, data.get('prob', 0.0), data.get('error_type', 0.0) / 2.0, 1.0]
        attr.extend([e_feat, e_feat])
        p = 1.0 if data.get('is_parity', False) else 0.0
        par.extend([p, p])
        canon_edges.extend([standardize_edge(u, v, bnd_z, bnd_x), standardize_edge(v, u, bnd_z, bnd_x)])
        
    base_e_idx = torch.tensor([src, dst], dtype=torch.long)
    base_e_attr = torch.tensor(attr, dtype=torch.float32)
    base_e_par = torch.tensor(par, dtype=torch.float32)
    # ---------------------------

    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    syn, flips = sampler.sample(shots=shots, separate_observables=True)
    flips = flips.flatten().astype(np.int64)
    preds_mwpm = matcher.decode_batch(syn).flatten().astype(np.int64)
    
    samples = []
    skipped_api_mismatch = 0
    active_indices = np.where(np.sum(syn, axis=1) >= 2)[0]
    
    for idx in active_indices:
        s = syn[idx].astype(np.uint8)
        batch_pred = preds_mwpm[idx]
        
        edges_a_raw = matcher.decode_to_edges_array(s)
        C_A = set(standardize_edge(int(e[0]), int(e[1]), bnd_z, bnd_x) for e in edges_a_raw)
        obs_A = compute_chain_observable(C_A, edge_dict)
        
        if obs_A != batch_pred:
            skipped_api_mismatch += 1
            continue
            
        C_B = C_A.symmetric_difference(R_L)
        C_0 = C_A if obs_A == 0 else C_B
        C_1 = C_B if obs_A == 0 else C_A
        w_0, w_1 = compute_chain_weight(C_0, edge_dict), compute_chain_weight(C_1, edge_dict)
        
        # INSTANT TENSOR CLONING
        x6 = base_x6.clone()
        active_nodes = np.where(s)[0]
        valid_nodes = active_nodes[active_nodes < num_nodes]
        x6[valid_nodes, 5] = 1.0
        
        mask_diff = torch.zeros((len(canon_edges), 1), dtype=torch.float32)
        for i, canon in enumerate(canon_edges):
            in_0 = canon in C_0
            in_1 = canon in C_1
            if in_1 and not in_0: mask_diff[i, 0] = 1.0
            elif in_0 and not in_1: mask_diff[i, 0] = -1.0
            
        samples.append({"x6": x6, "e_idx": base_e_idx, "e_attr": base_e_attr, "e_par": (base_e_par.view(-1) > 0).float(),
                        "mask_diff": mask_diff, "y_target": 1.0 if flips[idx] == 0 else -1.0, 
                        "delta_w": w_1 - w_0, "obs_A": obs_A, "idx": idx})
                        
    print(f"  [+] Prepared d={d} ({shots:,} shots) in {time.time()-t0:.2f}s | Dropped mismatch: {skipped_api_mismatch}")
    return {"samples": samples, "total_shots": shots, "preds_mwpm": preds_mwpm, "flips": flips}

def run_gate3_instant():
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cpu") # CPU is safe, avoids index panics
    p_val, eta, shots = 0.0035, 100.0, 10000

    print("=" * 90)
    print("PHASE G.25: ULTRA-FAST GATE 3 CLEAR")
    print("=" * 90 + "\n")

    train_samples = []
    for d in [3, 5, 7]:
        dat = collect_single_distance_fast(d, p_val, eta, shots)
        for s in dat["samples"]:
            if abs(s["delta_w"]) <= (4.0 * d):
                train_samples.append(s)

    pool_fail = [s for s in train_samples if (0 if s["y_target"] == 1.0 else 1) != s["obs_A"]]
    pool_corr = [s for s in train_samples if (0 if s["y_target"] == 1.0 else 1) == s["obs_A"]]

    model = TopoOracle(in_node_dim=6, in_edge_dim=4, hidden_dim=64, num_layers=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print(f"  Training Pool: {len(pool_fail):,d} Fails | {len(pool_corr):,d} Correct Shots")
    for epoch in range(1, 9):
        model.train()
        tot_loss, correct, items = 0.0, 0, 0
        for _ in range(100):
            batch = [pool_fail[i] for i in np.random.choice(len(pool_fail), 64, replace=True)] + \
                    [pool_corr[j] for j in np.random.choice(len(pool_corr), 64, replace=True)]
            bx6, be_idx, be_attr, be_par, bmask, _, bmap, targets, n_g = collate_cycle_batch(batch, device)
            optimizer.zero_grad()
            phi_out = model(bx6, be_idx, be_attr, be_par, bmask, batch_map=bmap, num_graphs=n_g)
            loss = criterion(phi_out, targets)
            loss.backward(); optimizer.step()
            tot_loss += loss.item() * n_g; items += n_g
            correct += (torch.where(phi_out > 0, 1.0, -1.0) == targets).sum().item()
        print(f"  Epoch {epoch:2d}/8 | Loss: {tot_loss/items:.4f} | Acc: {(correct/items)*100:6.2f}%")

    model.eval()
    print("\n[+] Evaluating Zero-Shot Transfer at Distance d = 11...\n")
    test_d11 = collect_single_distance_fast(11, p_val, eta, shots)
    active, preds_mwpm, flips = test_d11["samples"], test_d11["preds_mwpm"], test_d11["flips"]
    mwpm_errs = int(np.sum(preds_mwpm != flips))
    
    phi_list, obs_A_list, d_w_list, shot_idx = [], [], [], []
    for i in range(0, len(active), 128):
        batch_raw = active[i:i + 128]
        bx6, be_idx, be_attr, be_par, bmask, d_w, bmap, _, n_g = collate_cycle_batch(batch_raw, device)
        with torch.no_grad():
            phi_out = model(bx6, be_idx, be_attr, be_par, bmask, batch_map=bmap, num_graphs=n_g)
            phi_list.extend(phi_out.cpu().numpy().flatten())
        for s in batch_raw:
            obs_A_list.append(s["obs_A"]); d_w_list.append(s["delta_w"]); shot_idx.append(s["idx"])

    phi_arr, obs_A_arr, d_w_arr = np.array(phi_list), np.array(obs_A_list), np.array(d_w_list)

    print(f"\n>>> ZERO-SHOT DISTANCE d = 11 (MWPM P_L: {mwpm_errs/shots*100:.3f}%) <<<")
    print(f"{'Max dW (k*d)':<14} | {'Min Topo (tau)':<14} | {'Topo P_L':<10} | {'Rec':<6} | {'Reg':<6} | {'Net'}")
    print("-" * 75)

    for k in [2.5, 3.0, 3.5, 4.0]:
        for tau in [0.5, 0.65, 0.75]:
            topo_preds = preds_mwpm.copy()
            for phi, obs_a, dw, idx in zip(phi_arr, obs_A_arr, d_w_arr, shot_idx):
                if abs(dw) <= (k * 11.0):
                    if obs_a == 0 and phi < -tau: topo_preds[idx] = 1
                    elif obs_a == 1 and phi > tau: topo_preds[idx] = 0
            rec = int(np.sum((preds_mwpm != flips) & (topo_preds == flips)))
            reg = int(np.sum((preds_mwpm == flips) & (topo_preds != flips)))
            print(f"<= {k*11.0:<11.1f} | > {tau:<12.2f} | {(np.sum(topo_preds != flips) / shots) * 100.0:6.3f}%   | {rec:>4d} | {reg:>4d} | {rec-reg:>+4d}")

if __name__ == "__main__":
    run_gate3_instant()
