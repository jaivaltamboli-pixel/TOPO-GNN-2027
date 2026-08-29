import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import torch
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import TopoDephaseGNN

def audit_microscopic_shot(d=9, p_val=0.002, eta=100.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print(f"MICROSCOPIC INJECTION & PYMATCHING FAULT-ID AUDIT (d={d})")
    print("=" * 100 + "\n")

    model = TopoDephaseGNN().to(device)
    model.load_state_dict(torch.load("checkpoints/topo_dephase_gnn.pt", map_location=device))
    model.eval()

    circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
    dem = circuit.detector_error_model(decompose_errors=True)
    coords = circuit.get_detector_coordinates()
    num_dets = circuit.num_detectors
    edge_dict, bnd_z_idx, bnd_x_idx, dem_fault_to_edge = extract_complete_dem_graph(dem, num_dets, coords, d)

    sampler = circuit.compile_detector_sampler()
    dem_sampler = dem.compile_sampler()
    
    det_data, obs_data, err_data = dem_sampler.sample(shots=200, return_errors=True)
    
    # Find first multi-defect shot where GNN triggers p >= 0.95
    target_idx = None
    selected_edges = []
    
    for idx in range(len(det_data)):
        s = det_data[idx]
        if np.sum(s) < 2:
            continue
            
        active_fault_indices = np.where(err_data[idx])[0]
        active_fault_pairs = set()
        for f_idx in active_fault_indices:
            if f_idx < len(dem_fault_to_edge):
                pair = dem_fault_to_edge[f_idx]
                if pair is not None:
                    active_fault_pairs.add(pair)
                    
        x4, x6, e_idx, e_attr, e_par, s_t, e_targ, global_pairs = extract_active_subgraph_tensors(
            s, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device, active_fault_pairs=active_fault_pairs
        )
        if e_idx.numel() == 0:
            continue
            
        with torch.no_grad():
            _, edge_logits = model(x6, e_idx, e_attr, e_par)
            if edge_logits.numel() == 0:
                continue
            probs = torch.sigmoid(edge_logits).cpu().numpy().flatten()
            targs = e_targ.cpu().numpy().flatten()

        high_conf = np.where(probs >= 0.95)[0]
        if len(high_conf) >= 2:
            target_idx = idx
            for k in high_conf[:5]:
                u, v = global_pairs[k]
                selected_edges.append({
                    "u": int(u), "v": int(v),
                    "p": float(probs[k]),
                    "y_true": int(targs[k]),
                    "canon": tuple(sorted((int(u), int(v))))
                })
            break

    if target_idx is None:
        print("  [-] No multi-edge p >= 0.95 shot found in sample.")
        return

    s_vec = det_data[target_idx]
    obs_true = int(obs_data[target_idx, 0])
    base_matcher = pymatching.Matching.from_detector_error_model(dem)
    base_pred = int(base_matcher.decode(s_vec)[0])

    print(f"  [+] Inspecting Shot Index: {target_idx}")
    print(f"      Active Detector Count: {np.sum(s_vec)}")
    print(f"      Ground Truth Observable Flip: {obs_true}")
    print(f"      Base MWPM Prediction:        {base_pred} ({'CORRECT' if base_pred == obs_true else 'ERROR'})")
    print(f"      High Confidence (p>=0.95) Edges Selected: {len(selected_edges)}\n")

    print(f"  {'Edge (u, v)':<18} | {'p_GNN':<8} | {'y_true':<7} | {'base_w':<8} | {'new_w (0.85)':<12} | {'has_obs in DEM'}")
    print("  " + "-" * 75)

    for item in selected_edges:
        canon = item["canon"]
        props = edge_dict.get(canon, {"weight": 5.0, "has_obs": False})
        base_w = float(props["weight"])
        new_w = base_w * 0.85
        has_obs = props["has_obs"]
        print(f"  ({item['u']:>4d}, {item['v']:>4d})     | {item['p']:.4f} | {item['y_true']:<7d} | {base_w:6.3f}   | {new_w:6.3f}       | {has_obs}")

    print("\n  -------------------- PYMATCHING REPLACEMENT BEHAVIOR TEST --------------------")
    
    # Test 1: Full replace using default add_edge (drops fault_ids)
    matcher_untracked = pymatching.Matching.from_detector_error_model(dem)
    for item in selected_edges:
        u, v = item["u"], item["v"]
        props = edge_dict.get(item["canon"], {"weight": 5.0, "has_obs": False})
        new_w = float(props["weight"]) * 0.85
        if u == bnd_z_idx or u == bnd_x_idx:
            matcher_untracked.add_boundary_edge(v, weight=new_w, merge_strategy="replace")
        elif v == bnd_z_idx or v == bnd_x_idx:
            matcher_untracked.add_boundary_edge(u, weight=new_w, merge_strategy="replace")
        else:
            matcher_untracked.add_edge(u, v, weight=new_w, merge_strategy="replace")

    pred_untracked = int(matcher_untracked.decode(s_vec)[0])

    # Test 2: Replacement with explicit fault_ids preservation
    matcher_tracked = pymatching.Matching.from_detector_error_model(dem)
    for item in selected_edges:
        u, v = item["u"], item["v"]
        props = edge_dict.get(item["canon"], {"weight": 5.0, "has_obs": False})
        new_w = float(props["weight"]) * 0.85
        fault_ids = {0} if props["has_obs"] else set()
        
        if u == bnd_z_idx or u == bnd_x_idx:
            matcher_tracked.add_boundary_edge(v, weight=new_w, fault_ids=fault_ids, merge_strategy="replace")
        elif v == bnd_z_idx or v == bnd_x_idx:
            matcher_tracked.add_boundary_edge(u, weight=new_w, fault_ids=fault_ids, merge_strategy="replace")
        else:
            matcher_tracked.add_edge(u, v, weight=new_w, fault_ids=fault_ids, merge_strategy="replace")

    pred_tracked = int(matcher_tracked.decode(s_vec)[0])

    print(f"  Raw Base MWPM Prediction:                          {base_pred}")
    print(f"  Prediction after add_edge WITHOUT fault_ids:        {pred_untracked}")
    print(f"  Prediction after add_edge WITH fault_ids preserved: {pred_tracked}")
    print("=" * 100 + "\n")

if __name__ == "__main__":
    audit_microscopic_shot()
