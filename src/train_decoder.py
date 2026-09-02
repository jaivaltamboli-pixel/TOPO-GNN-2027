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
from topo_oracle_model import (
    build_parity_expanded_graph,
    find_exact_logical_reference_chain,
    standardize_edge,
    compute_chain_observable,
    compute_chain_weight,
    MultiscaleTopoOracle,
    physics_informed_loss
)

def collate_cycle_batch(samples, device):
    x6_list, e_idx_list, e_attr_list, e_par_list, mask_diff_list, batch_map = [], [], [], [], [], []
    target_edges_list, target_delta_w_list, target_logical_list = [], [], []
    node_offset = 0
    for i, s in enumerate(samples):
        n = s["x6"].shape[0]
        x6_list.append(s["x6"]); e_idx_list.append(s["e_idx"].long() + node_offset)
        e_attr_list.append(s["e_attr"]); e_par_list.append(s["e_par"].view(-1, 1) if s["e_par"].dim() == 1 else s["e_par"])
        mask_diff_list.append(s["mask_diff"]); batch_map.append(torch.full((n,), i, dtype=torch.long))
        
        target_edges_list.append(s["target_edges"])
        target_delta_w_list.append(s["delta_w"])
        target_logical_list.append(s["target_logical"])
        node_offset += n
        
    return (torch.cat(x6_list, dim=0).to(device), torch.cat(e_idx_list, dim=1).long().to(device),
            torch.cat(e_attr_list, dim=0).to(device), torch.cat(e_par_list, dim=0).to(device),
            torch.cat(mask_diff_list, dim=0).to(device), torch.cat(target_edges_list, dim=0).to(device),
            torch.cat(batch_map, dim=0).long().to(device), 
            torch.tensor(target_delta_w_list, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.tensor(target_logical_list, dtype=torch.float32, device=device).unsqueeze(-1), len(samples))

def collect_distance_cache(d, p_val, eta, shots):
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
        p = 1.0 if data.get('is_parity', False) else 0.0
        par.extend([p, p])
        canon_edges.extend([standardize_edge(u, v, bnd_z, bnd_x), standardize_edge(v, u, bnd_z, bnd_x)])
        
    base_e_idx = torch.tensor([src, dst], dtype=torch.long)
    base_e_attr = torch.tensor(attr, dtype=torch.float32)
    base_e_par = torch.tensor(par, dtype=torch.float32)

    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    syn, flips = sampler.sample(shots=shots, separate_observables=True)
    flips = flips.flatten().astype(np.int64)
    preds_mwpm = matcher.decode_batch(syn).flatten().astype(np.int64)
    
    samples = []
    VIRTUAL_BOUNDARY = -1
    for idx in np.where(np.sum(syn, axis=1) >= 2)[0]:
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
            
        target_edges = torch.zeros((len(canon_edges_sparse), 1), dtype=torch.float32)
        target_logical = float(flips[idx])
        C_true = C_0 if flips[idx] == 0 else C_1
        for j, canon in enumerate(canon_edges_sparse):
            if canon in C_true:
                target_edges[j, 0] = 1.0
                
        samples.append({"x6": x6_sparse, "e_idx": e_idx_sparse, "e_attr": e_attr_sparse, "e_par": (e_par_sparse.view(-1) > 0).float(),
                        "mask_diff": mask_diff, "target_edges": target_edges, 
                        "target_logical": target_logical, "delta_w": w_1 - w_0, "obs_A": obs_A, "idx": idx})
                        
    print(f"  [+] d={d} ({shots:,} shots) mapped in {time.time()-t0:.2f}s")
    return {"samples": samples, "total_shots": shots, "preds_mwpm": preds_mwpm, "flips": flips}

def save_gate4_model():
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Backend executing on: {device.type.upper()}")
    
    p_val, eta, shots = 0.0035, 100.0, 10000

    print("=" * 80)
    print("PHASE H.1: TRAINING & SAVING PRODUCTION MODEL")
    print("=" * 80 + "\n")

    train_samples = []
    for d in [5, 7, 9]:
        dat = collect_distance_cache(d, p_val, eta, shots)
        for s in dat["samples"]:
            if abs(s["delta_w"]) <= (4.0 * d): train_samples.append(s)

    pool_fail = [s for s in train_samples if int(s["target_logical"]) != s["obs_A"]]
    pool_corr = [s for s in train_samples if int(s["target_logical"]) == s["obs_A"]]
    print(f"  Final Training Pool: {len(pool_fail):,d} Fails | {len(pool_corr):,d} Correct Shots")

    model = MultiscaleTopoOracle(in_node_dim=6, in_edge_dim=4, hidden_dim=64, num_layers=6, bins=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)


    for epoch in range(1, 16):
        model.train(); tot_loss, correct, items = 0.0, 0, 0
        for _ in range(150):
            batch = [pool_fail[i] for i in np.random.choice(len(pool_fail), 16, replace=True)] + \
                    [pool_corr[j] for j in np.random.choice(len(pool_corr), 16, replace=True)]
            bx6, be_idx, be_attr, be_par, bmask, btarget_edges, bmap, btarget_delta_w, btarget_logical, n_g = collate_cycle_batch(batch, device)
            optimizer.zero_grad()
            edge_logits, pred_delta_w, logical_logits = model(bx6, be_idx, be_attr, be_par, bmask, batch_map=bmap, num_graphs=n_g)
            loss = physics_informed_loss(edge_logits, pred_delta_w, logical_logits, btarget_edges, btarget_delta_w, btarget_logical, bmask)
            loss.backward(); optimizer.step()
            tot_loss += loss.item() * n_g; items += n_g
            correct += ((logical_logits > 0) == (btarget_logical > 0.5)).sum().item()
        print(f"  Epoch {epoch:2d}/12 | Loss: {tot_loss/items:.4f} | Acc: {(correct/items)*100:6.1f}%")

    os.makedirs("models", exist_ok=True)
    save_path = "models/topo_gnn_gate4.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\n[+] Production model weights successfully saved to: {save_path}")

if __name__ == "__main__":
    save_gate4_model()
