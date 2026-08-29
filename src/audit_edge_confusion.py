import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import torch
import numpy as np
import stim
import pymatching
import time

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import TopoDephaseGNN

def run_confusion_matrix_audit(distances=[3, 5, 7], p_val=0.002, eta=100.0, shots=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105)
    print(f"EDGE-LEVEL CONFUSION MATRIX & PRECISION AUDIT @ p >= 0.95 (Distances: {distances}, Shots: {shots})")
    print("=" * 105 + "\n")

    model = TopoDephaseGNN().to(device)
    model.load_state_dict(torch.load("checkpoints/topo_dephase_gnn.pt", map_location=device))
    model.eval()

    for d in distances:
        circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
        dem = circuit.detector_error_model(decompose_errors=True)
        coords = circuit.get_detector_coordinates()
        num_dets = circuit.num_detectors
        edge_dict, bnd_z_idx, bnd_x_idx, dem_fault_to_edge = extract_complete_dem_graph(dem, num_dets, coords, d)
        dem_sampler = dem.compile_sampler()

        det_data, _, err_data = dem_sampler.sample(shots=shots, return_errors=True)

        all_y_true = []
        all_y_pred = []
        all_probs = []

        for i in range(shots):
            s = det_data[i]
            if np.sum(s) < 1:
                continue

            active_fault_indices = np.where(err_data[i])[0]
            active_fault_pairs = set()
            for f_idx in active_fault_indices:
                if f_idx < len(dem_fault_to_edge):
                    pair = dem_fault_to_edge[f_idx]
                    if pair is not None:
                        active_fault_pairs.add(pair)

            x4, x6, e_idx, e_attr, e_par, s_t, e_targ, _ = extract_active_subgraph_tensors(
                s, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device, active_fault_pairs=active_fault_pairs
            )

            if e_idx.size(1) == 0:
                continue

            with torch.no_grad():
                _, edge_logits = model(x6, e_idx, e_attr, e_par)
                if edge_logits.numel() > 0:
                    probs = torch.sigmoid(edge_logits).cpu().numpy().flatten()
                    targs = e_targ.cpu().numpy().flatten()

                    all_y_true.extend(targs.tolist())
                    all_probs.extend(probs.tolist())
                    all_y_pred.extend((probs >= 0.95).astype(int).tolist())

        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)
        probs_arr = np.array(all_probs)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        precision_95 = tp / max(tp + fp, 1)
        recall_95 = tp / max(tp + fn, 1)
        f1_95 = (2 * precision_95 * recall_95) / max(precision_95 + recall_95, 1e-12)

        # Baseline rate for calibration reference
        base_pos_rate = np.mean(y_true) * 100

        print(f"---------------------- DISTANCE d = {d} ----------------------")
        print(f"  Total Subgraph Edges Evaluated: {len(y_true):,}")
        print(f"  Ground-Truth Positive Rate:     {base_pos_rate:.2f}% ({int(np.sum(y_true)):,}/{len(y_true):,})")
        print(f"  Predicted Positive @ p >= 0.95:  {(tp + fp)/len(y_true)*100:.2f}% ({tp + fp:,}/{len(y_true):,})")
        print(f"  [Matrix] TP: {tp:5d}  |  FP: {fp:5d}")
        print(f"  [Matrix] FN: {fn:5d}  |  TN: {tn:5d}")
        print(f"  Precision @ p >= 0.95:          {precision_95 * 100:6.2f}%")
        print(f"  Recall @ p >= 0.95:             {recall_95 * 100:6.2f}%")
        print(f"  F1 Score @ p >= 0.95:           {f1_95:.4f}")
        print()

def run_counterfactual_hybrid_test(d=9, shots=500, p_val=0.002, eta=100.0):
    print("=" * 105)
    print(f"3-WAY COUNTERFACTUAL DECODER TEST ON ZERO-SHOT DISTANCE d={d} (Shots={shots})")
    print("=" * 105 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TopoDephaseGNN().to(device)
    model.load_state_dict(torch.load("checkpoints/topo_dephase_gnn.pt", map_location=device))
    model.eval()

    circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
    dem = circuit.detector_error_model(decompose_errors=True)
    coords = circuit.get_detector_coordinates()
    num_dets = circuit.num_detectors
    edge_dict, bnd_z_idx, bnd_x_idx, _ = extract_complete_dem_graph(dem, num_dets, coords, d)

    sampler = circuit.compile_detector_sampler()
    syn, flips = sampler.sample(shots=shots, separate_observables=True)
    flips = flips.flatten().astype(np.int64)

    # 1. Classical MWPM (Mode C)
    base_matcher = pymatching.Matching.from_detector_error_model(dem)
    preds_c = base_matcher.decode_batch(syn).flatten().astype(np.int64)

    preds_a = preds_c.copy()  # Mode A: Reduce cost for high-confidence edges
    preds_b = preds_c.copy()  # Mode B: Increase cost for high-confidence edges

    active_shots = np.where(np.sum(syn, axis=1) >= 2)[0]

    for idx in active_shots:
        s = syn[idx].astype(np.uint8)
        _, x6, e_idx, e_attr, e_par, _, _, global_pairs = extract_active_subgraph_tensors(
            s, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device
        )
        if e_idx.numel() == 0:
            continue

        with torch.no_grad():
            _, edge_logits = model(x6, e_idx, e_attr, e_par)
            if edge_logits.numel() == 0:
                continue
            probs = torch.sigmoid(edge_logits).cpu().numpy().flatten()

        edges_to_mod = []
        processed = set()
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
                    edges_to_mod.append((u, v, float(props["weight"])))

        if len(edges_to_mod) > 0:
            # Mode A: Favor edge (reduce weight by 15%)
            matcher_a = pymatching.Matching.from_detector_error_model(dem)
            # Mode B: Penalize edge (increase weight by 15%)
            matcher_b = pymatching.Matching.from_detector_error_model(dem)

            for u, v, base_w in edges_to_mod:
                w_a = max(0.01, base_w * 0.85)
                w_b = base_w * 1.15

                if u == bnd_z_idx or u == bnd_x_idx:
                    if v < num_dets:
                        matcher_a.add_boundary_edge(v, weight=w_a, merge_strategy="replace")
                        matcher_b.add_boundary_edge(v, weight=w_b, merge_strategy="replace")
                elif v == bnd_z_idx or v == bnd_x_idx:
                    if u < num_dets:
                        matcher_a.add_boundary_edge(u, weight=w_a, merge_strategy="replace")
                        matcher_b.add_boundary_edge(u, weight=w_b, merge_strategy="replace")
                else:
                    matcher_a.add_edge(u, v, weight=w_a, merge_strategy="replace")
                    matcher_b.add_edge(u, v, weight=w_b, merge_strategy="replace")

            preds_a[idx] = int(matcher_a.decode(s)[0])
            preds_b[idx] = int(matcher_b.decode(s)[0])

    err_c = int(np.sum(preds_c != flips))
    err_a = int(np.sum(preds_a != flips))
    err_b = int(np.sum(preds_b != flips))

    print(f"  Mode C (Classical Pure MWPM):          {err_c:>3d}/{shots} errors ({err_c/shots*100:6.3f}%)")
    print(f"  Mode A (Prior: Lower Cost on p>=0.95):  {err_a:>3d}/{shots} errors ({err_a/shots*100:6.3f}%)")
    print(f"  Mode B (Prior: Raise Cost on p>=0.95):  {err_b:>3d}/{shots} errors ({err_b/shots*100:6.3f}%)")
    print("=" * 105 + "\n")

if __name__ == "__main__":
    run_confusion_matrix_audit(distances=[3, 5, 7], shots=500)
    run_counterfactual_hybrid_test(d=9, shots=500)
