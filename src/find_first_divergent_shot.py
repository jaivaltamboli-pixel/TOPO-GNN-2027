import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import torch
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import TopoDephaseGNN

def find_divergent_shot(d=9, p_val=0.002, eta=100.0, max_shots=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105)
    print(f"SEARCHING FOR FIRST DIVERGENT SHOT ON DISTANCE d={d} (Shots: {max_shots})")
    print("=" * 105 + "\n")

    model = TopoDephaseGNN().to(device)
    model.load_state_dict(torch.load("checkpoints/topo_dephase_gnn.pt", map_location=device))
    model.eval()

    circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
    dem = circuit.detector_error_model(decompose_errors=True)
    coords = circuit.get_detector_coordinates()
    num_dets = circuit.num_detectors
    edge_dict, bnd_z_idx, bnd_x_idx, dem_fault_to_edge = extract_complete_dem_graph(dem, num_dets, coords, d)

    dem_sampler = dem.compile_sampler()
    det_data, obs_data, err_data = dem_sampler.sample(shots=max_shots, return_errors=True)
    base_matcher = pymatching.Matching.from_detector_error_model(dem)

    found = False

    for shot_idx in range(max_shots):
        s_vec = det_data[shot_idx]
        if np.sum(s_vec) < 2:
            continue

        obs_true = int(obs_data[shot_idx, 0])
        base_pred = int(base_matcher.decode(s_vec)[0])

        active_fault_indices = np.where(err_data[shot_idx])[0]
        active_fault_pairs = set()
        for f_idx in active_fault_indices:
            if f_idx < len(dem_fault_to_edge):
                pair = dem_fault_to_edge[f_idx]
                if pair is not None:
                    active_fault_pairs.add(pair)

        x4, x6, e_idx, e_attr, e_par, s_t, e_targ, global_pairs = extract_active_subgraph_tensors(
            s_vec, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device, active_fault_pairs=active_fault_pairs
        )
        if e_idx.numel() == 0:
            continue

        with torch.no_grad():
            _, edge_logits = model(x6, e_idx, e_attr, e_par)
            if edge_logits.numel() == 0:
                continue
            probs = torch.sigmoid(edge_logits).cpu().numpy().flatten()
            targs = e_targ.cpu().numpy().flatten()

        processed = set()
        edges_to_mod = []

        for k_e, pair in enumerate(global_pairs):
            u, v = int(pair[0]), int(pair[1])
            canon = tuple(sorted((u, v)))
            if canon in processed:
                continue
            processed.add(canon)

            rev_idx = [j for j, gp in enumerate(global_pairs) if {int(gp[0]), int(gp[1])} == {u, v}]
            p_edge = float(np.mean(probs[rev_idx])) if len(rev_idx) > 0 else float(probs[k_e])
            y_edge = int(targs[k_e])

            if p_edge >= 0.95:
                props = edge_dict.get(canon)
                if props is not None:
                    edges_to_mod.append({
                        "u": u, "v": v, "canon": canon,
                        "p": p_edge, "y_true": y_edge,
                        "base_w": float(props["weight"]),
                        "has_obs": props.get("has_obs", False)
                    })

        if len(edges_to_mod) == 0:
            continue

        matcher_lower = pymatching.Matching.from_detector_error_model(dem)
        matcher_higher = pymatching.Matching.from_detector_error_model(dem)

        for item in edges_to_mod:
            u, v = item["u"], item["v"]
            base_w = item["base_w"]
            w_low = max(0.01, base_w * 0.85)
            w_high = base_w * 1.15
            fault_ids = {0} if item["has_obs"] else set()

            if u == bnd_z_idx or u == bnd_x_idx:
                if v < num_dets:
                    matcher_lower.add_boundary_edge(v, weight=w_low, fault_ids=fault_ids, merge_strategy="replace")
                    matcher_higher.add_boundary_edge(v, weight=w_high, fault_ids=fault_ids, merge_strategy="replace")
            elif v == bnd_z_idx or v == bnd_x_idx:
                if u < num_dets:
                    matcher_lower.add_boundary_edge(u, weight=w_low, fault_ids=fault_ids, merge_strategy="replace")
                    matcher_higher.add_boundary_edge(u, weight=w_high, fault_ids=fault_ids, merge_strategy="replace")
            else:
                matcher_lower.add_edge(u, v, weight=w_low, fault_ids=fault_ids, merge_strategy="replace")
                matcher_higher.add_edge(u, v, weight=w_high, fault_ids=fault_ids, merge_strategy="replace")

        pred_low = int(matcher_lower.decode(s_vec)[0])
        pred_high = int(matcher_higher.decode(s_vec)[0])

        # Target criterion: MWPM was correct, but hybrid diverged to wrong prediction
        if base_pred == obs_true and (pred_low != obs_true or pred_high != obs_true):
            found = True
            print("=" * 70)
            print(f"FIRST DIVERGENT SHOT FOUND: Shot #{shot_idx}")
            print("=" * 70)
            print(f"  Ground Truth Observable Flip:   {obs_true}")
            print(f"  Base MWPM Prediction:          {base_pred}  (CORRECT)")
            print(f"  Lower-Cost Hybrid Prediction:   {pred_low}  ({'CORRECT' if pred_low == obs_true else 'WRONG'})")
            print(f"  Higher-Cost Hybrid Prediction:  {pred_high}  ({'CORRECT' if pred_high == obs_true else 'WRONG'})")
            print(f"  Active Detector Defects ({np.sum(s_vec)} total): {np.where(s_vec)[0].tolist()}\n")

            print(f"  {'Physical Edge (u <-> v)':<26} | {'p_GNN':<7} | {'y_true':<7} | {'base_w':<7} | {'lower_w':<8} | {'higher_w':<9} | {'has_obs'}")
            print("  " + "-" * 90)
            for item in edges_to_mod:
                u, v = item["u"], item["v"]
                edge_label = f"({u}, {v})" if u < num_dets and v < num_dets else f"({min(u,v)}, BND)"
                print(f"  {edge_label:<26} | {item['p']:.4f}  | {item['y_true']:<7d} | {item['base_w']:6.3f}  | {item['base_w']*0.85:6.3f}   | {item['base_w']*1.15:6.3f}    | {item['has_obs']}")
            print("=" * 70 + "\n")
            break

    if not found:
        print("  [-] No divergent shot found in the sampled batch.")

if __name__ == "__main__":
    find_divergent_shot(d=9, max_shots=500)
