import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import pymatching
import numpy as np
import torch
import time
import copy

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import (
    extract_complete_dem_graph,
    extract_active_subgraph_tensors,
)
from utils.metrics import wilson_score_interval
from models import (
    LangeIsotropicMPNN,
    NeuralBeliefPropagation,
    SpatioTemporalGNN,
    TopoDephaseGNN,
)

def run_evaluation(
    test_distances=[9, 11, 13],
    p_val=0.002,
    eta=100.0,
    shots=1000,
    positive_p_threshold=0.95,
    alpha_scale=0.05,
    max_delta=0.25,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 118)
    print(
        f"ZERO-SHOT EDGE PROBABILITY DISTRIBUTION & HIGH-CONFIDENCE PROFILE "
        f"(Distances: {test_distances}, Shots: {shots:,}, Gate: p > {positive_p_threshold:.2f})"
    )
    print("=" * 118 + "\n")

    models = {
        "Lange-inspired MPNN": LangeIsotropicMPNN().to(device),
        "Neural BP-inspired Net": NeuralBeliefPropagation().to(device),
        "ST-GNN-inspired Net": SpatioTemporalGNN().to(device),
        "Topo-DephaseGNN": TopoDephaseGNN().to(device),
    }

    models["Lange-inspired MPNN"].load_state_dict(
        torch.load("checkpoints/lange_mpnn.pt", map_location=device)
    )
    models["Neural BP-inspired Net"].load_state_dict(
        torch.load("checkpoints/neural_bp.pt", map_location=device)
    )
    models["ST-GNN-inspired Net"].load_state_dict(
        torch.load("checkpoints/st_gnn.pt", map_location=device)
    )
    models["Topo-DephaseGNN"].load_state_dict(
        torch.load("checkpoints/topo_dephase_gnn.pt", map_location=device)
    )

    for model in models.values():
        model.eval()

    for d in test_distances:
        t_dist_start = time.time()

        circuit = make_biased_surface_code(
            d=d,
            rounds=d,
            p_total=p_val,
            eta=eta,
        )

        dem = circuit.detector_error_model(decompose_errors=True)
        base_matcher = pymatching.Matching.from_detector_error_model(dem)
        coords = circuit.get_detector_coordinates()
        num_dets = circuit.num_detectors

        edge_dict, bnd_z_idx, bnd_x_idx, _ = extract_complete_dem_graph(
            dem,
            num_dets,
            coords,
            d,
        )

        sampler = circuit.compile_detector_sampler()
        syn, flips = sampler.sample(
            shots=shots,
            separate_observables=True,
        )
        flips = flips.flatten().astype(np.int64)

        # 1. Classical MWPM baseline
        t0 = time.time()
        mwpm_preds = base_matcher.decode_batch(syn).flatten().astype(np.int64)
        mwpm_lat = (time.time() - t0) * 1000.0 / (shots / 1000.0)

        # 2. Standalone neural classifiers
        standalone_preds = {
            "Lange-inspired MPNN (Classifier)": np.zeros(shots, dtype=np.int64),
            "Neural BP-inspired (Classifier)": np.zeros(shots, dtype=np.int64),
            "ST-GNN-inspired (Classifier)": np.zeros(shots, dtype=np.int64),
            "Topo-DephaseGNN (Classifier)": np.zeros(shots, dtype=np.int64),
        }

        # 3. Hybrid
        hybrid_preds = mwpm_preds.copy()
        syn_weights = np.sum(syn, axis=1)
        active_shots = np.where(syn_weights >= 2)[0]

        dynamic_prior_updates = 0
        total_edges_considered = 0
        total_edges_modified = 0

        all_sampled_probs = []
        neural_total_start = time.time()

        for shot_idx_enum, idx in enumerate(active_shots):
            s = syn[idx].astype(np.uint8)

            (
                x4,
                x6,
                e_idx,
                e_attr,
                e_par,
                s_t,
                _,
                global_pairs,
            ) = extract_active_subgraph_tensors(
                s,
                coords,
                edge_dict,
                bnd_z_idx,
                bnd_x_idx,
                d,
                device,
            )

            if e_idx.numel() == 0:
                continue

            with torch.no_grad():
                p1 = models["Lange-inspired MPNN"](x4, e_idx, e_attr).item()
                standalone_preds["Lange-inspired MPNN (Classifier)"][idx] = int(p1 > 0.5)

                p2 = models["Neural BP-inspired Net"](s_t, e_idx, e_attr).item()
                standalone_preds["Neural BP-inspired (Classifier)"][idx] = int(p2 > 0.5)

                p3 = models["ST-GNN-inspired Net"](x4, e_idx, e_attr).item()
                standalone_preds["ST-GNN-inspired (Classifier)"][idx] = int(p3 > 0.5)

                log_pred, edge_logits = models["Topo-DephaseGNN"](x6, e_idx, e_attr, e_par)
                standalone_preds["Topo-DephaseGNN (Classifier)"][idx] = int(log_pred.item() > 0.5)

                if edge_logits.numel() == 0:
                    continue

                raw_logits = edge_logits.detach().cpu().numpy().flatten()
                probs = 1.0 / (1.0 + np.exp(-raw_logits))

            if shot_idx_enum < 50:
                all_sampled_probs.extend(probs.tolist())

            # ============================================================
            # EDGE-PROBABILITY DISTRIBUTION DIAGNOSTICS (First Active Shot)
            # ============================================================
            if idx == active_shots[0]:
                pct = np.percentile(
                    probs,
                    [50, 75, 90, 95, 97.5, 99, 99.5, 99.9]
                )

                print(f"\n  [Distance d={d} Edge Diagnostic on First Active Shot (len={len(probs)})]")
                print(f"    Logit range:        [{raw_logits.min():.4f}, {raw_logits.max():.4f}]")
                print(f"    Probability range:  [{probs.min():.6f}, {probs.max():.6f}]")
                print(f"    Median p:            {pct[0]:.6f}")
                print(f"    75th percentile:     {pct[1]:.6f}")
                print(f"    90th percentile:     {pct[2]:.6f}")
                print(f"    95th percentile:     {pct[3]:.6f}")
                print(f"    97.5th percentile:   {pct[4]:.6f}")
                print(f"    99th percentile:     {pct[5]:.6f}")
                print(f"    99.5th percentile:   {pct[6]:.6f}")
                print(f"    99.9th percentile:   {pct[7]:.6f}")

                print("    Threshold counts:")
                print(f"      p > 0.50:  {np.sum(probs > 0.50):6d} / {len(probs)}")
                print(f"      p > 0.70:  {np.sum(probs > 0.70):6d} / {len(probs)}")
                print(f"      p > 0.80:  {np.sum(probs > 0.80):6d} / {len(probs)}")
                print(f"      p > 0.85:  {np.sum(probs > 0.85):6d} / {len(probs)}")
                print(f"      p > 0.90:  {np.sum(probs > 0.90):6d} / {len(probs)}")
                print(f"      p > 0.95:  {np.sum(probs > 0.95):6d} / {len(probs)}")
                print(f"      p > 0.99:  {np.sum(probs > 0.99):6d} / {len(probs)}")

                top_idx = np.argsort(probs)[-10:][::-1]
                print("    Top 10 directed edge predictions:")
                for j in top_idx:
                    print(
                        f"      edge[{j:5d}] "
                        f"logit={raw_logits[j]:8.4f} "
                        f"p={probs[j]:.6f}"
                    )
                print()

            processed_edges = set()
            edges_to_update = []

            for k_e, pair in enumerate(global_pairs):
                u, v = int(pair[0]), int(pair[1])
                canonical = tuple(sorted((u, v)))

                if canonical in processed_edges:
                    continue
                processed_edges.add(canonical)
                total_edges_considered += 1

                reverse_indices = [
                    j for j, gp in enumerate(global_pairs)
                    if {int(gp[0]), int(gp[1])} == {u, v}
                ]

                if len(reverse_indices) > 0:
                    p_edge = float(np.mean(probs[reverse_indices]))
                    logit_edge = float(np.mean(raw_logits[reverse_indices]))
                else:
                    p_edge = float(probs[k_e])
                    logit_edge = float(raw_logits[k_e])

                if p_edge <= positive_p_threshold:
                    continue

                props = edge_dict.get(canonical)
                if props is None:
                    continue

                base_w = float(props["weight"])
                delta_w = float(np.clip(logit_edge * alpha_scale, 0.0, max_delta))
                new_w = max(0.01, base_w - delta_w)

                edges_to_update.append((u, v, new_w))

            if len(edges_to_update) > 0:
                try:
                    matcher_shot = copy.deepcopy(base_matcher)
                except Exception:
                    matcher_shot = pymatching.Matching.from_detector_error_model(dem)

                for u, v, new_w in edges_to_update:
                    if u == bnd_z_idx or u == bnd_x_idx:
                        detector = v
                        if detector < num_dets:
                            matcher_shot.add_boundary_edge(detector, weight=new_w, merge_strategy="replace")
                            total_edges_modified += 1
                    elif v == bnd_z_idx or v == bnd_x_idx:
                        detector = u
                        if detector < num_dets:
                            matcher_shot.add_boundary_edge(detector, weight=new_w, merge_strategy="replace")
                            total_edges_modified += 1
                    else:
                        matcher_shot.add_edge(u, v, weight=new_w, merge_strategy="replace")
                        total_edges_modified += 1

                hybrid_preds[idx] = int(matcher_shot.decode(s)[0])
                dynamic_prior_updates += 1

        neural_total_time = time.time() - neural_total_start
        neural_lat = (neural_total_time * 1000.0 / (shots / 1000.0) / 4.0)

        all_results = {
            "MWPM (Classical Baseline)": (mwpm_preds, mwpm_lat),
            "Lange-inspired MPNN (Classifier)": (standalone_preds["Lange-inspired MPNN (Classifier)"], neural_lat),
            "Neural BP-inspired (Classifier)": (standalone_preds["Neural BP-inspired (Classifier)"], neural_lat),
            "ST-GNN-inspired (Classifier)": (standalone_preds["ST-GNN-inspired (Classifier)"], neural_lat),
            "Topo-DephaseGNN (Classifier)": (standalone_preds["Topo-DephaseGNN (Classifier)"], neural_lat),
            "Topo-DephaseGNN + MWPM (Learned Prior)": (hybrid_preds, mwpm_lat + neural_lat),
        }

        print(f"============================== ZERO-SHOT EVALUATION: DISTANCE d = {d} ==============================")
        print(f"{'Decoder / Classifier Architecture':<44} | {'Logical Error':<14} | {'95% Wilson CI':<18} | {'Errors':<10} | {'Latency (ms/1k)':<15}")
        print("-" * 118)

        for name, (p_vec, lat) in all_results.items():
            k_err = int(np.sum(p_vec != flips))
            p_hat, lower, upper = wilson_score_interval(k_err, shots)
            print(f"{name:<44} | {p_hat * 100:6.3f}%       | [{lower * 100:.3f}%, {upper * 100:.3f}%]   | {k_err:>5d}/{shots} | {lat:6.2f} ms")

        mod_rate = (dynamic_prior_updates / max(len(active_shots), 1) * 100.0)
        edge_mod_rate = (total_edges_modified / max(total_edges_considered, 1) * 100.0)

        probs_arr = np.array(all_sampled_probs)
        print("-" * 118)
        if len(probs_arr) > 0:
            print(f"  [Diagnostic] Aggregate Sampled Edge Probs (First 50 shots):")
            print(f"      Median: {np.median(probs_arr):.4f} | Mean: {np.mean(probs_arr):.4f} | 90th %ile: {np.percentile(probs_arr, 90):.4f} | 99th %ile: {np.percentile(probs_arr, 99):.4f} | Max: {np.max(probs_arr):.4f}")
        print(f"  [Diagnostic] Active Syndrome Events: {len(active_shots):,}/{shots:,} ({len(active_shots) / shots * 100:.2f}%)")
        print(f"  [Diagnostic] Shots With Learned Prior Updates: {dynamic_prior_updates:,} ({mod_rate:.2f}% of active events)")
        print(f"  [Diagnostic] Unique Candidate Edges Considered: {total_edges_considered:,}")
        print(f"  [Diagnostic] Unique Edges Modified: {total_edges_modified:,} ({edge_mod_rate:.2f}%)")
        print(f"  [Timing] Distance d={d} completed in {time.time() - t_dist_start:.2f}s\n")

if __name__ == "__main__":
    run_evaluation(
        test_distances=[9, 11, 13],
        shots=1000,
        positive_p_threshold=0.95,
        alpha_scale=0.05,
        max_delta=0.25,
    )
