import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import torch
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import TopoDephaseGNN

def sweep_beta_values(d=9, p_val=0.002, eta=100.0, beta_list=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print(f"BETA LLR COUPLING SWEEP ON FIXED DETERMINISTIC BATCH (d={d}, Shots=500)")
    print("=" * 115 + "\n")

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

    base_matcher = pymatching.Matching.from_detector_error_model(dem)
    preds_base = base_matcher.decode_batch(syn).flatten().astype(np.int64)
    base_err_idx = np.where(preds_base != flips)[0]
    
    print(f"  [Baseline] MWPM Total Errors: {len(base_err_idx)}/500 ({len(base_err_idx)/shots*100:.3f}%)")
    print(f"  [Baseline] MWPM Error Shot Indices: {base_err_idx.tolist()}\n")

    active_shots = np.where(np.sum(syn, axis=1) >= 2)[0]

    # Pre-extract confident edge predictions once across all active shots
    shot_edge_cache = {}
    for idx in active_shots:
        s = syn[idx].astype(np.uint8)
        x4, x6, e_idx, e_attr, e_par, s_t, _, global_pairs = extract_active_subgraph_tensors(
            s, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device
        )
        if e_idx.numel() == 0:
            continue

        with torch.no_grad():
            _, edge_logits = model(x6, e_idx, e_attr, e_par)
            if edge_logits.numel() == 0:
                continue
            probs = torch.sigmoid(edge_logits).cpu().numpy().flatten()

        processed = set()
        confident_edges = []
        for k_e, pair in enumerate(global_pairs):
            u, v = int(pair[0]), int(pair[1])
            canon = tuple(sorted((u, v)))
            if canon in processed:
                continue
            processed.add(canon)

            rev_idx = [j for j, gp in enumerate(global_pairs) if {int(gp[0]), int(gp[1])} == {u, v}]
            p_edge = float(np.mean(probs[rev_idx])) if len(rev_idx) > 0 else float(probs[k_e])

            if p_edge >= 0.95:
                props = edge_dict.get(canon)
                if props is not None:
                    p_clamped = np.clip(p_edge, 1e-4, 1.0 - 1e-4)
                    llr = float(np.log(p_clamped / (1.0 - p_clamped)))
                    confident_edges.append({
                        "u": u, "v": v,
                        "llr": llr,
                        "base_w": float(props["weight"]),
                        "has_obs": props.get("has_obs", False)
                    })
        
        if len(confident_edges) > 0:
            shot_edge_cache[idx] = confident_edges

    print(f"{'Beta (LLR Scale)':<18} | {'Total Errors':<14} | {'Changed Shots':<15} | {'Recoveries':<12} | {'Regressions':<12} | {'Net Gain':<10}")
    print("-" * 95)

    for beta in beta_list:
        preds_hybrid = preds_base.copy()

        for idx, edges in shot_edge_cache.items():
            s = syn[idx].astype(np.uint8)
            matcher = pymatching.Matching.from_detector_error_model(dem)

            for item in edges:
                u, v, base_w, llr, has_obs = item["u"], item["v"], item["base_w"], item["llr"], item["has_obs"]
                new_w = max(0.001, float(base_w - beta * llr))
                fault_ids = {0} if has_obs else set()

                if u == bnd_z_idx or u == bnd_x_idx:
                    if v < num_dets:
                        matcher.add_boundary_edge(v, weight=new_w, fault_ids=fault_ids, merge_strategy="replace")
                elif v == bnd_z_idx or v == bnd_x_idx:
                    if u < num_dets:
                        matcher.add_boundary_edge(u, weight=new_w, fault_ids=fault_ids, merge_strategy="replace")
                else:
                    matcher.add_edge(u, v, weight=new_w, fault_ids=fault_ids, merge_strategy="replace")

            preds_hybrid[idx] = int(matcher.decode(s)[0])

        total_err = int(np.sum(preds_hybrid != flips))
        diff_shots = np.where(preds_hybrid != preds_base)[0]
        recoveries = [i for i in diff_shots if preds_base[i] != flips[i] and preds_hybrid[i] == flips[i]]
        regressions = [i for i in diff_shots if preds_base[i] == flips[i] and preds_hybrid[i] != flips[i]]
        net_gain = len(recoveries) - len(regressions)

        print(f"beta = {beta:<11.2f} | {total_err:>3d}/{shots} ({total_err/shots*100:5.2f}%) | {len(diff_shots):<15d} | {len(recoveries):<12d} | {len(regressions):<12d} | {net_gain:<+10d}")

    print("=" * 115 + "\n")

if __name__ == "__main__":
    sweep_beta_values()
