import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import torch
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import TopoDephaseGNN

def audit_bayesian_injection_strategies(d=9, p_val=0.002, eta=100.0, beta=0.10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 110)
    print(f"CONTROLLED BAYESIAN PRIOR & CONTRACTION AUDIT (d={d}, Fixed 500-Shot Deterministic Batch)")
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

    base_matcher = pymatching.Matching.from_detector_error_model(dem)
    preds_mwpm = base_matcher.decode_batch(syn).flatten().astype(np.int64)

    preds_bayes_res = preds_mwpm.copy()
    preds_bayes_rep = preds_mwpm.copy()
    preds_contract  = preds_mwpm.copy()

    active_shots = np.where(np.sum(syn, axis=1) >= 2)[0]
    annihilated_count = 0

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
                    confident_edges.append({
                        "u": u, "v": v, "canon": canon,
                        "p": p_edge,
                        "base_w": float(props["weight"]),
                        "has_obs": props.get("has_obs", False)
                    })

        if len(confident_edges) == 0:
            continue

        # Strategy 1A: Controlled LLR Residual (w_base - beta * LLR)
        matcher_res = pymatching.Matching.from_detector_error_model(dem)
        # Strategy 1B: Direct Shifted Replacement (log((1-p)/p) + 6.0)
        matcher_rep = pymatching.Matching.from_detector_error_model(dem)

        for item in confident_edges:
            u, v, p_e, base_w = item["u"], item["v"], item["p"], item["base_w"]
            p_clamped = np.clip(p_e, 1e-4, 1.0 - 1e-4)
            llr = float(np.log(p_clamped / (1.0 - p_clamped)))
            
            # 1A: Controlled bounded residual
            w_res = max(0.001, float(base_w - beta * llr))
            # 1B: Replacement
            w_rep = max(0.001, float(np.log((1.0 - p_clamped) / p_clamped) + 6.0))
            
            fault_ids = {0} if item["has_obs"] else set()

            if u == bnd_z_idx or u == bnd_x_idx:
                if v < num_dets:
                    matcher_res.add_boundary_edge(v, weight=w_res, fault_ids=fault_ids, merge_strategy="replace")
                    matcher_rep.add_boundary_edge(v, weight=w_rep, fault_ids=fault_ids, merge_strategy="replace")
            elif v == bnd_z_idx or v == bnd_x_idx:
                if u < num_dets:
                    matcher_res.add_boundary_edge(u, weight=w_res, fault_ids=fault_ids, merge_strategy="replace")
                    matcher_rep.add_boundary_edge(u, weight=w_rep, fault_ids=fault_ids, merge_strategy="replace")
            else:
                matcher_res.add_edge(u, v, weight=w_res, fault_ids=fault_ids, merge_strategy="replace")
                matcher_rep.add_edge(u, v, weight=w_rep, fault_ids=fault_ids, merge_strategy="replace")

        preds_bayes_res[idx] = int(matcher_res.decode(s)[0])
        preds_bayes_rep[idx] = int(matcher_rep.decode(s)[0])

        # Strategy 2: Pre-Matching Defect Annihilation
        s_contract = s.copy()
        obs_correction = 0
        for item in confident_edges:
            if item["p"] >= 0.99:
                u, v = item["u"], item["v"]
                if u < num_dets and v < num_dets and s_contract[u] == 1 and s_contract[v] == 1:
                    s_contract[u] = 0
                    s_contract[v] = 0
                    if item["has_obs"]:
                        obs_correction ^= 1
                    annihilated_count += 1

        preds_contract[idx] = int(base_matcher.decode(s_contract)[0]) ^ obs_correction

    # Metric summaries
    def analyze_strategy(name, p_vec):
        total_err = int(np.sum(p_vec != flips))
        diff_from_base = np.where(p_vec != preds_mwpm)[0]
        recoveries = [i for i in diff_from_base if preds_mwpm[i] != flips[i] and p_vec[i] == flips[i]]
        regressions = [i for i in diff_from_base if preds_mwpm[i] == flips[i] and p_vec[i] != flips[i]]
        return total_err, len(diff_from_base), recoveries, regressions

    err_base = int(np.sum(preds_mwpm != flips))
    err_res, diff_res, rec_res, reg_res = analyze_strategy("Controlled Residual (1A)", preds_bayes_res)
    err_rep, diff_rep, rec_rep, reg_rep = analyze_strategy("Direct Replacement (1B)", preds_bayes_rep)
    err_con, diff_con, rec_con, reg_con = analyze_strategy("Defect Annihilation (2)", preds_contract)

    print("==============================================================================================================")
    print(f"{'Decoding Paradigm':<35} | {'Errors':<10} | {'Logical Error':<14} | {'Mod Shots':<10} | {'Recoveries':<12} | {'Regressions':<12}")
    print("-" * 110)
    print(f"{'1. Pure Classical MWPM':<35} | {err_base:>3d}/{shots} | {err_base/shots*100:6.3f}%       | {'-':<10} | {'-':<12} | {'-':<12}")
    print(f"{'2. Strategy 1A (Controlled LLR)':<35} | {err_res:>3d}/{shots} | {err_res/shots*100:6.3f}%       | {diff_res:<10d} | {len(rec_res):<12d} | {len(reg_res):<12d}")
    print(f"{'3. Strategy 1B (Shifted Replacement)':<35} | {err_rep:>3d}/{shots} | {err_rep/shots*100:6.3f}%       | {diff_rep:<10d} | {len(rec_rep):<12d} | {len(reg_rep):<12d}")
    print(f"{'4. Strategy 2 (Defect Annihilation)':<35} | {err_con:>3d}/{shots} | {err_con/shots*100:6.3f}%       | {diff_con:<10d} | {len(rec_con):<12d} | {len(reg_con):<12d}")
    print("=" * 110 + "\n")

    print(f"  [Detailed Diagnostics]")
    print(f"    Shot #57 Ground Truth: {flips[57]}")
    print(f"    Shot #57 Predictions: MWPM={preds_mwpm[57]} | Strat 1A={preds_bayes_res[57]} | Strat 1B={preds_bayes_rep[57]} | Strat 2={preds_contract[57]}")
    print(f"    Strategy 1A Regressions (Shot indices): {reg_res}")
    print(f"    Strategy 1A Recoveries   (Shot indices): {rec_res}")
    print(f"    Strategy 2 Total Edges Annihilated:     {annihilated_count}")
    print("=" * 110 + "\n")

if __name__ == "__main__":
    audit_bayesian_injection_strategies()
