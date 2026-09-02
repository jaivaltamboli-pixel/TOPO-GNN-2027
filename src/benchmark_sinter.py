import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import time
import torch
import torch.nn as nn
import numpy as np
import stim
import pymatching
import sinter
import matplotlib.pyplot as plt

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph
from topo_oracle_model import (
    build_parity_expanded_graph,
    find_exact_logical_reference_chain,
    standardize_edge,
    compute_chain_observable,
    compute_chain_weight,
    MultiscaleTopoOracle
)

# --- 1. NEURAL ARCHITECTURE ---
def collate_cycle_batch(samples, device):
    x6_list, e_idx_list, e_attr_list, e_par_list, mask_diff_list, batch_map = [], [], [], [], [], []
    node_offset = 0
    for i, s in enumerate(samples):
        n = s["x6"].shape[0]
        x6_list.append(s["x6"]); e_idx_list.append(s["e_idx"].long() + node_offset)
        e_attr_list.append(s["e_attr"]); e_par_list.append(s["e_par"].view(-1, 1) if s["e_par"].dim() == 1 else s["e_par"])
        mask_diff_list.append(s["mask_diff"]); batch_map.append(torch.full((n,), i, dtype=torch.long))
        node_offset += n
    return (torch.cat(x6_list, dim=0).to(device), torch.cat(e_idx_list, dim=1).long().to(device),
            torch.cat(e_attr_list, dim=0).to(device), torch.cat(e_par_list, dim=0).to(device),
            torch.cat(mask_diff_list, dim=0).to(device), torch.cat(batch_map, dim=0).long().to(device), len(samples))

# --- 2. FAST INFERENCE ENGINE ---
def run_benchmark_point(d, p_val, eta, shots, model, device, k_thresh=3.5, tau_thresh=0.75):
    t0 = time.time()
    circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
    dem = circuit.detector_error_model(decompose_errors=True)
    coords = circuit.get_detector_coordinates()
    num_dets = circuit.num_detectors
    edge_dict, adj, bnd_z, bnd_x = build_parity_expanded_graph(dem, num_dets, coords, d)
    raw_edge_dict, _, _, _ = extract_complete_dem_graph(dem, num_dets, coords, d)
    R_L, _ = find_exact_logical_reference_chain(adj, num_dets)
    
    max_idx = max(max(u, v) for u, v in raw_edge_dict.keys())
    num_nodes = max(max_idx + 1, bnd_z + 1, bnd_x + 1)
    base_x6 = torch.zeros((num_nodes, 6), dtype=torch.float32)
    for i, (z, x, y) in coords.items():
        base_x6[i, 0], base_x6[i, 1], base_x6[i, 2] = z / (d + 1), x / (d + 1), y / (d + 1)
    if bnd_z < num_nodes: base_x6[bnd_z, 3] = 1.0
    if bnd_x < num_nodes: base_x6[bnd_x, 4] = 1.0
        
    src, dst, attr, par, canon_edges = [], [], [], [], []
    for (u, v), data in raw_edge_dict.items():
        src.extend([u, v]); dst.extend([v, u])
        w_norm = min(float(data.get('weight', 0.0)), 50.0) / 10.0 
        e_feat = [w_norm, data.get('prob', 0.0), data.get('error_type', 0.0) / 2.0, 1.0]
        attr.extend([e_feat, e_feat])
        p_flag = 1.0 if data.get('is_parity', False) else 0.0
        par.extend([p_flag, p_flag])
        canon_edges.extend([standardize_edge(u, v, bnd_z, bnd_x), standardize_edge(v, u, bnd_z, bnd_x)])
        
    base_e_idx = torch.tensor([src, dst], dtype=torch.long)
    base_e_attr = torch.tensor(attr, dtype=torch.float32)
    base_e_par = torch.tensor(par, dtype=torch.float32)

    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    syn, flips = sampler.sample(shots=shots, separate_observables=True)
    flips = flips.flatten().astype(np.int64)
    preds_mwpm = matcher.decode_batch(syn).flatten().astype(np.int64)
    
    topo_preds = preds_mwpm.copy()
    active_indices = np.where(np.sum(syn, axis=1) >= 2)[0]
    
    batch_samples = []
    shot_mappings = []
    VIRTUAL_BOUNDARY = -1
    
    for idx in active_indices:
        s = syn[idx].astype(np.uint8)
        batch_pred = preds_mwpm[idx]
        edges_a_raw = matcher.decode_to_edges_array(s)
        C_A = set(standardize_edge(int(e[0]), int(e[1]), bnd_z, bnd_x) for e in edges_a_raw)
        obs_A = compute_chain_observable(C_A, edge_dict)
        
        if obs_A != batch_pred: continue
            
        C_B = C_A.symmetric_difference(R_L)
        C_0 = C_A if obs_A == 0 else C_B
        C_1 = C_B if obs_A == 0 else C_A
        w_0, w_1 = compute_chain_weight(C_0, edge_dict), compute_chain_weight(C_1, edge_dict)
        delta_w = w_1 - w_0
        
        # Immediate k-filter rejection
        if abs(delta_w) > (k_thresh * d):
            continue
        
        chain_nodes = set()
        for u, v in C_0.union(C_1):
            if u != VIRTUAL_BOUNDARY: chain_nodes.add(u)
            if v != VIRTUAL_BOUNDARY: chain_nodes.add(v)
            
        valid_nodes = np.where(s)[0]
        valid_nodes = valid_nodes[valid_nodes < num_nodes]
        
        sparse_nodes = set(valid_nodes).union(chain_nodes)
        sparse_nodes.add(bnd_z)
        sparse_nodes.add(bnd_x)
        sparse_nodes_list = sorted(list(sparse_nodes))
        node_to_idx = {n: i for i, n in enumerate(sparse_nodes_list)}
        
        x6_sparse = base_x6[sparse_nodes_list].clone()
        for n in valid_nodes:
            if n in node_to_idx:
                x6_sparse[node_to_idx[n], 5] = 1.0

        src, dst, attr, par, canon_edges_sparse = [], [], [], [], []
        for (u, v), data in raw_edge_dict.items():
            if u in sparse_nodes and v in sparse_nodes:
                idx_u, idx_v = node_to_idx[u], node_to_idx[v]
                src.extend([idx_u, idx_v]); dst.extend([idx_v, idx_u])
                w_norm = min(float(data.get('weight', 0.0)), 50.0) / 10.0 
                e_feat = [w_norm, data.get('prob', 0.0), data.get('error_type', 0.0) / 2.0, 1.0]
                attr.extend([e_feat, e_feat])
                p_flag = 1.0 if data.get('is_parity', False) else 0.0
                par.extend([p_flag, p_flag])
                canon_edges_sparse.extend([standardize_edge(u, v, bnd_z, bnd_x), standardize_edge(v, u, bnd_z, bnd_x)])
                
        e_idx_sparse = torch.tensor([src, dst], dtype=torch.long)
        e_attr_sparse = torch.tensor(attr, dtype=torch.float32)
        e_par_sparse = torch.tensor(par, dtype=torch.float32)
        
        mask_diff = torch.zeros((len(canon_edges_sparse), 1), dtype=torch.float32)
        for i, canon in enumerate(canon_edges_sparse):
            in_0, in_1 = canon in C_0, canon in C_1
            if in_1 and not in_0: mask_diff[i, 0] = 1.0
            elif in_0 and not in_1: mask_diff[i, 0] = -1.0
            
        batch_samples.append({
            "x6": x6_sparse, "e_idx": e_idx_sparse, "e_attr": e_attr_sparse, 
            "e_par": (e_par_sparse.view(-1) > 0).float(), "mask_diff": mask_diff
        })
        shot_mappings.append({"idx": idx, "obs_A": obs_A})

    # Execute GNN on ambiguous subset
    if batch_samples:
        for i in range(0, len(batch_samples), 64):
            sub_batch = batch_samples[i:i+64]
            sub_maps = shot_mappings[i:i+64]
            bx6, be_idx, be_attr, be_par, bmask, bmap, n_g = collate_cycle_batch(sub_batch, device)
            with torch.no_grad():
                edge_logits, pred_delta_w, logical_logits = model(bx6, be_idx, be_attr, be_par, bmask, batch_map=bmap, num_graphs=n_g)
                logits = logical_logits.cpu().numpy().flatten()
            
            for logit, sm in zip(logits, sub_maps):
                if sm["obs_A"] == 0 and logit > tau_thresh: topo_preds[sm["idx"]] = 1
                elif sm["obs_A"] == 1 and logit < -tau_thresh: topo_preds[sm["idx"]] = 0

    mwpm_errors = int(np.sum(preds_mwpm != flips))
    topo_errors = int(np.sum(topo_preds != flips))
    return mwpm_errors, topo_errors, time.time() - t0

# --- 3. BENCHMARK EXECUTION ---
def run_gate4_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Gate 4 Sinter Benchmark on {device.type.upper()}")
    
    model = MultiscaleTopoOracle(in_node_dim=6, in_edge_dim=4, hidden_dim=64, num_layers=6, bins=3).to(device)
    model_path = "models/topo_gnn_gate4.pt"
    
    if not os.path.exists(model_path):
        print(f"[!] Critical Error: {model_path} not found. Run H.1 save_model script first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("[+] Model weights loaded successfully.")

    distances = [3, 5, 7, 9]
    error_rates = [0.001, 0.002, 0.0035, 0.005, 0.008]
    eta = 100.0
    shots = 50000  # High shot count for clean threshold lines

    csv_file = "gate4_benchmark_results.csv"
    with open(csv_file, "w") as f:
        f.write("shots,errors,discards,seconds,decoder,strong_id,json_metadata\n")

    print(f"\n[+] Commencing Data Collection (Shots per point: {shots:,})")
    print(f"{'d':<3} | {'p_val':<7} | {'MWPM Err':<10} | {'Topo Err':<10} | {'Net Gain':<9} | {'Time (s)'}")
    print("-" * 65)

    for d in distances:
        for p in error_rates:
            mwpm_errs, topo_errs, t_run = run_benchmark_point(d, p, eta, shots, model, device, k_thresh=3.5, tau_thresh=0.75)
            net = mwpm_errs - topo_errs
            print(f"{d:<3} | {p:<7.4f} | {mwpm_errs:<10} | {topo_errs:<10} | {net:<+9} | {t_run:.1f}s")
            
            # Write Sinter standard format rows
            meta = f'{{"d": {d}, "p": {p}, "eta": {eta}}}'
            with open(csv_file, "a") as f:
                f.write(f"{shots},{mwpm_errs},0,{t_run},mwpm,_,{meta}\n")
                f.write(f"{shots},{topo_errs},0,{t_run},topo_gnn,_,{meta}\n")

    print(f"\n[+] Benchmark complete. Saved to {csv_file}")
    
    # Generate Sinter Plot
    print("[+] Generating formal Sinter threshold plot...")
    stats = sinter.read_stats_from_csv_files(csv_file)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda stat: stat.json_metadata["p"],
        group_func=lambda stat: f"d={stat.json_metadata['d']} ({stat.decoder})",
        failure_units_per_quantity_func=lambda stat: stat.json_metadata["d"]
    )
    
    ax.loglog()
    ax.set_title(r"Gate 4: Topo-GNN vs MWPM Threshold Benchmark (Biased Noise $\eta=100$)")
    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Logical Error Rate per Round ($P_L$)")
    ax.grid(which="major", color="black", alpha=0.3)
    ax.grid(which="minor", color="gray", alpha=0.1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plot_path = "images/gate4_threshold_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"[+] Gate 4 Official Plot saved to: {plot_path}")

if __name__ == "__main__":
    run_gate4_benchmark()
