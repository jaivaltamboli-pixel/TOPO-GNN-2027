import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import stim
import pymatching
import numpy as np
import torch
import json
import time

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from utils.metrics import wilson_score_interval
from models import LangeIsotropicMPNN, NeuralBeliefPropagation, SpatioTemporalGNN, TopoDephaseGNN

def run_quick_test(d=9, p_val=0.002, eta=100.0, shots=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        with open("checkpoints/meta.json", "r") as f:
            meta = json.load(f)
        tau = meta.get("optimal_tau", 0.15)
    except FileNotFoundError:
        tau = 0.15

    print("=" * 110)
    print(f"QUICK DIAGNOSTIC BENCHMARK (Distance: d={d}, Shots: {shots:,}, Validation-Locked Tau: {tau:.2f})")
    print("=" * 110 + "\n")

    models = {
        "Lange-inspired MPNN": LangeIsotropicMPNN().to(device),
        "Neural BP-inspired Net": NeuralBeliefPropagation().to(device),
        "ST-GNN-inspired Net": SpatioTemporalGNN().to(device),
        "Topo-DephaseGNN": TopoDephaseGNN().to(device)
    }

    models["Lange-inspired MPNN"].load_state_dict(torch.load("checkpoints/lange_mpnn.pt", map_location=device))
    models["Neural BP-inspired Net"].load_state_dict(torch.load("checkpoints/neural_bp.pt", map_location=device))
    models["ST-GNN-inspired Net"].load_state_dict(torch.load("checkpoints/st_gnn.pt", map_location=device))
    models["Topo-DephaseGNN"].load_state_dict(torch.load("checkpoints/topo_dephase_gnn.pt", map_location=device))

    for m in models.values():
        m.eval()

    t_dist_start = time.time()
    circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
    dem = circuit.detector_error_model(decompose_errors=True)
    base_matcher = pymatching.Matching.from_detector_error_model(dem)
    coords = circuit.get_detector_coordinates()
    num_dets = circuit.num_detectors
    edge_dict, bnd_idx = extract_complete_dem_graph(dem, num_dets, coords, d)
    sampler = circuit.compile_detector_sampler()

    syn, flips = sampler.sample(shots=shots, separate_observables=True)
    flips = flips.flatten()

    # 1. Classical MWPM
    t0 = time.time()
    mwpm_preds = base_matcher.decode_batch(syn).flatten()
    mwpm_lat = (time.time() - t0) * 1000 / (shots / 1000)

    # 2. Standalone Zero-Shot Predictions
    standalone_preds = {
        "Lange-inspired MPNN (Classifier)": np.zeros(shots, dtype=np.int64),
        "Neural BP-inspired (Classifier)": np.zeros(shots, dtype=np.int64),
        "ST-GNN-inspired (Classifier)": np.zeros(shots, dtype=np.int64),
        "Topo-DephaseGNN (Classifier)": np.zeros(shots, dtype=np.int64)
    }

    # 3. Confidence-Gated Hybrid
    hybrid_preds = mwpm_preds.copy()
    syn_weights = np.sum(syn, axis=1)
    active_shots = np.where(syn_weights > 0)[0]
    complex_cluster_shots = np.where(syn_weights >= 4)[0]
    dynamic_interventions = 0

    t_neural_start = time.time()
    for idx in active_shots:
        s = syn[idx]
        x4, x6, e_idx, e_attr, e_par, s_t, _, _ = extract_active_subgraph_tensors(
            s, coords, edge_dict, bnd_idx, d, device
        )

        with torch.no_grad():
            p1 = models["Lange-inspired MPNN"](x4, e_idx, e_attr).item()
            standalone_preds["Lange-inspired MPNN (Classifier)"][idx] = int(p1 > 0.5)

            p2 = models["Neural BP-inspired Net"](s_t, e_idx, e_attr).item()
            standalone_preds["Neural BP-inspired (Classifier)"][idx] = int(p2 > 0.5)

            p3 = models["ST-GNN-inspired Net"](x4, e_idx, e_attr).item()
            standalone_preds["ST-GNN-inspired (Classifier)"][idx] = int(p3 > 0.5)

            log_pred, _ = models["Topo-DephaseGNN"](x6, e_idx, e_attr, e_par)
            p4 = log_pred.item()
            topo_choice = int(p4 > 0.5)
            standalone_preds["Topo-DephaseGNN (Classifier)"][idx] = topo_choice

            # Conservative gating: only override when model has extreme certainty on multi-defect clusters
            if syn_weights[idx] >= 4 and abs(p4 - 0.5) >= 0.45:
                hybrid_preds[idx] = topo_choice
                if topo_choice != mwpm_preds[idx]:
                    dynamic_interventions += 1

    neural_lat = (time.time() - t_neural_start) * 1000 / (shots / 1000) / 4.0

    all_results = {
        "MWPM (Classical Baseline)": (mwpm_preds, mwpm_lat),
        "Lange-inspired MPNN (Classifier)": (standalone_preds["Lange-inspired MPNN (Classifier)"], neural_lat),
        "Neural BP-inspired (Classifier)": (standalone_preds["Neural BP-inspired (Classifier)"], neural_lat),
        "ST-GNN-inspired (Classifier)": (standalone_preds["ST-GNN-inspired (Classifier)"], neural_lat),
        "Topo-DephaseGNN (Classifier)": (standalone_preds["Topo-DephaseGNN (Classifier)"], neural_lat),
        "Topo-DephaseGNN Gated Hybrid": (hybrid_preds, mwpm_lat + neural_lat)
    }

    print(f"============================== QUICK TEST RESULT: DISTANCE d = {d} ==============================")
    print(f"{'Decoder / Classifier Architecture':<42} | {'Logical Error':<14} | {'95% Wilson CI':<18} | {'Errors':<10} | {'Latency (ms/1k)':<15}")
    print("-" * 115)
    for name, (p_vec, lat) in all_results.items():
        k_err = int(np.sum(p_vec != flips))
        p_hat, l, u = wilson_score_interval(k_err, shots)
        print(f"{name:<42} | {p_hat*100:6.3f}%       | [{l*100:.3f}%, {u*100:.3f}%]   | {k_err:>5d}/{shots} | {lat:6.2f} ms")
    
    print("-" * 115)
    print(f"  [Diagnostic] Active Events: {len(active_shots):,}/{shots:,} ({len(active_shots)/shots*100:.2f}%)")
    print(f"  [Diagnostic] Complex Clusters (k>=4): {len(complex_cluster_shots):,}/{shots:,} ({len(complex_cluster_shots)/shots*100:.2f}%)")
    print(f"  [Diagnostic] Hybrid Overrides: {dynamic_interventions:,}")
    print(f"  [Timing] Completed 1,000 shots in {time.time() - t_dist_start:.2f}s\n")

if __name__ == "__main__":
    run_quick_test()
