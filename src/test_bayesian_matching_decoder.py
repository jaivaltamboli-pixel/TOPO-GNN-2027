import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import torch
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import TopoDephaseGNN

def evaluate_bayesian_residual_decoder(d=9, p_val=0.002, eta=100.0, alpha_grid=[0.0, 0.02, 0.05, 0.10, 0.15]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 110)
    print(f"BAYESIAN RESIDUAL LOG-ODDS MATCHING DECODER (d={d}, p={p_val}, Bias eta={eta})")
    print("=" * 110 + "\n")

    model = TopoDephaseGNN().to(device)
    model.load_state_dict(torch.load("checkpoints/topo_dephase_gnn.pt", map_location=device))
    model.eval()

    circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
    dem = circuit.detector_error_model(decompose_errors=True)
    coords = circuit.get_detector_coordinates()
    num_dets = circuit.num_detectors
    edge_dict, bnd_z_idx, bnd_x_idx, _ = extract_complete_dem_graph(dem, num_dets, coords, d)

    syn = np.load("results/debug_syn.npy")
    flips = np.load("results/debug_flips.npy")
    shots = len(flips)

    # Base Classical MWPM
    base_matcher = pymatching.Matching.from_detector_error_model(dem)
    preds_base = base_matcher.decode_batch(syn).flatten().astype(np.int64)
    base_errors = int(np.sum(preds_base != flips))

    print(f"  [Baseline] Classical MWPM Errors: {base_errors}/{shots} ({base_errors/shots*100:5.2f}%)\n")
    print(f"{'Alpha (Residual Gain)':<22} | {'Total Errors':<14} | {'Changed Shots':<15} | {'Recoveries':<12} | {'Regressions':<12}")
    print("-" * 85)

    active_shots = np.where(np.sum(syn, axis=1) >= 2)[0]

    # Pre-extract GNN edge logits for all active shots
    gnn_edge_cache = {}
    for idx in active_shots:
        s = syn[idx].astype(np.uint8)
        x4, x6, e_idx, e_attr, e_par, _, _, global_pairs = extract_active_subgraph_tensors(
            s, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device
        )
        if e_idx.numel() == 0:
            continue

        with torch.no_grad():
            _, edge_logits = model(x6, e_idx, e_attr, e_par)
            if edge_logits.numel() == 0:
                continue
            raw_z = edge_logits.cpu().numpy().flatten()

        edge_updates = []
        processed = set()

        for k_e, pair in enumerate(global_pairs):
            u, v = int(pair[0]), int(pair[1])
            canon = tuple(sorted((u, v)))
            if canon in processed:
                continue
            processed.add(canon)

            rev_idx = [j for j, gp in enumerate(global_pairs) if {int(gp[0]), int(gp[1])} == {u, v}]
            z_gnn = float(np.mean(raw_z[rev_idx])) if len(rev_idx) > 0 else float(raw_z[k_e])

            props = edge_dict.get(canon)
            if props is not None:
                base_w = float(props["weight"])
                # Dem weight corresponds to log((1-p)/p) = -logit_dem
                # Information gain: delta_z = z_gnn - z_dem
                z_dem = -base_w
                delta_z = z_gnn - z_dem
                has_obs = props.get("has_obs", False)

                edge_updates.append((u, v, base_w, delta_z, has_obs))

        if len(edge_updates) > 0:
            gnn_edge_cache[idx] = edge_updates

    for alpha in alpha_grid:
        preds_hybrid = preds_base.copy()

        for idx, updates in gnn_edge_cache.items():
            s = syn[idx].astype(np.uint8)
            matcher = pymatching.Matching.from_detector_error_model(dem)

            for u, v, base_w, delta_z, has_obs in updates:
                # Bayesian update bounded by alpha
                new_w = max(0.01, float(base_w - alpha * delta_z))
                f_ids = {0} if has_obs else set()

                if u == bnd_z_idx or u == bnd_x_idx:
                    if v < num_dets:
                        matcher.add_boundary_edge(v, weight=new_w, fault_ids=f_ids, merge_strategy="replace")
                elif v == bnd_z_idx or v == bnd_x_idx:
                    if u < num_dets:
                        matcher.add_boundary_edge(u, weight=new_w, fault_ids=f_ids, merge_strategy="replace")
                else:
                    matcher.add_edge(u, v, weight=new_w, fault_ids=f_ids, merge_strategy="replace")

            preds_hybrid[idx] = int(matcher.decode(s)[0])

        total_err = int(np.sum(preds_hybrid != flips))
        diff_shots = np.where(preds_hybrid != preds_base)[0]
        recoveries = [i for i in diff_shots if preds_base[i] != flips[i] and preds_hybrid[i] == flips[i]]
        regressions = [i for i in diff_shots if preds_base[i] == flips[i] and preds_hybrid[i] != flips[i]]

        print(f"alpha = {alpha:<15.3f} | {total_err:>3d}/{shots} ({total_err/shots*100:5.2f}%) | {len(diff_shots):<15d} | {len(recoveries):<12d} | {len(regressions):<12d}")

    print("=" * 110 + "\n")

if __name__ == "__main__":
    evaluate_bayesian_residual_decoder()
