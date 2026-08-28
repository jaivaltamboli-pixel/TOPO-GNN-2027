import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import stim
import pymatching
import numpy as np
import torch
import time

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from utils.metrics import wilson_score_interval
from models import LangeIsotropicMPNN, NeuralBeliefPropagation, SpatioTemporalGNN, TopoDephaseGNN

def run_evaluation(test_distances=[9, 11, 13], p_val=0.002, eta=100.0, shots=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 118)
    print(f"ZERO-SHOT LEARNED EDGE-PRIOR BENCHMARK (Distances: {test_distances}, Shots: {shots:,})")
    print("=" * 118 + "\n")

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

    for d in test_distances:
        t_dist_start = time.time()
        circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
        dem = circuit.detector_error_model(decompose_errors=True)
        base_matcher = pymatching.Matching.from_detector_error_model(dem)
        coords = circuit.get_detector_coordinates()
        num_dets = circuit.num_detectors
        edge_dict, bnd_z_idx, bnd_x_idx, _ = extract_complete_dem_graph(dem, num_dets, coords, d)
        dem_sampler = dem.compile_sampler()

        det_data, obs_data, _ = dem_sampler.sample(shots=shots, return_errors=True)
        flips = obs_data.flatten()

        # 1. Classical MWPM (Baseline)
        t0 = time.time()
        mwpm_preds = base_matcher.decode_batch(det_data).flatten()
        mwpm_lat = (time.time() - t0) * 1000 / (shots / 1000)

        # 2. Strict Standalone Zero-Shot Predictions
        standalone_preds = {
            "Lange-inspired MPNN (Classifier)": np.zeros(shots, dtype=np.int64),
            "Neural BP-inspired (Classifier)": np.zeros(shots, dtype=np.int64),
            "ST-GNN-inspired (Classifier)": np.zeros(shots, dtype=np.int64),
            "Topo-DephaseGNN (Classifier)": np.zeros(shots, dtype=np.int64)
        }

        # 3. Topo-DephaseGNN + MWPM: True Prior-Injected Matching
        hybrid_preds = mwpm_preds.copy()
        syn_weights = np.sum(det_data, axis=1)
        active_shots = np.where(syn_weights >= 2)[0]
        dynamic_prior_updates = 0

        t_neural_start = time.time()
        for idx in active_shots:
            s = det_data[idx]
            x4, x6, e_idx, e_attr, e_par, s_t, _, global_pairs = extract_active_subgraph_tensors(
                s, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device
            )

            with torch.no_grad():
                # Baselines
                p1 = models["Lange-inspired MPNN"](x4, e_idx, e_attr).item()
                standalone_preds["Lange-inspired MPNN (Classifier)"][idx] = int(p1 > 0.5)

                p2 = models["Neural BP-inspired Net"](s_t, e_idx, e_attr).item()
                standalone_preds["Neural BP-inspired (Classifier)"][idx] = int(p2 > 0.5)

                p3 = models["ST-GNN-inspired Net"](x4, e_idx, e_attr).item()
                standalone_preds["ST-GNN-inspired (Classifier)"][idx] = int(p3 > 0.5)

                # Proposed Model
                log_pred, edge_logits = models["Topo-DephaseGNN"](x6, e_idx, e_attr, e_par)
                standalone_preds["Topo-DephaseGNN (Classifier)"][idx] = int(log_pred.item() > 0.5)

                # Inject Edge LLR Modifications into Matching Graph
                if edge_logits.numel() > 0:
                    probs = torch.sigmoid(edge_logits).cpu().numpy().flatten()
                    # Only modify when edge prediction deviates significantly
                    high_conf = np.where(probs > 0.70)[0]
                    if len(high_conf) > 0:
                        matcher_shot = pymatching.Matching.from_detector_error_model(dem)
                        for k_e in high_conf:
                            u, v = global_pairs[k_e]
                            base_w = edge_dict.get((min(u, v), max(u, v)), {}).get('weight', 5.0)
                            # Favor active dephasing edge by reducing its graph cost
                            new_w = max(0.01, base_w * (1.0 - probs[k_e] * 0.5))
                            
                            if v == bnd_z_idx or v == bnd_x_idx:
                                matcher_shot.add_boundary_edge(u, weight=new_w, merge_strategy="replace")
                            else:
                                matcher_shot.add_edge(u, v, weight=new_w, merge_strategy="replace")
                                
                        hybrid_preds[idx] = matcher_shot.decode(s)[0]
                        dynamic_prior_updates += 1

        neural_lat = (time.time() - t_neural_start) * 1000 / (shots / 1000) / 4.0

        all_results = {
            "MWPM (Classical Baseline)": (mwpm_preds, mwpm_lat),
            "Lange-inspired MPNN (Classifier)": (standalone_preds["Lange-inspired MPNN (Classifier)"], neural_lat),
            "Neural BP-inspired (Classifier)": (standalone_preds["Neural BP-inspired (Classifier)"], neural_lat),
            "ST-GNN-inspired (Classifier)": (standalone_preds["ST-GNN-inspired (Classifier)"], neural_lat),
            "Topo-DephaseGNN (Classifier)": (standalone_preds["Topo-DephaseGNN (Classifier)"], neural_lat),
            "Topo-DephaseGNN + MWPM (Learned Prior)": (hybrid_preds, mwpm_lat + neural_lat)
        }

        print(f"============================== ZERO-SHOT EVALUATION: DISTANCE d = {d} ==============================")
        print(f"{'Decoder / Classifier Architecture':<44} | {'Logical Error':<14} | {'95% Wilson CI':<18} | {'Errors':<10} | {'Latency (ms/1k)':<15}")
        print("-" * 118)
        for name, (p_vec, lat) in all_results.items():
            k_err = int(np.sum(p_vec != flips))
            p_hat, l, u = wilson_score_interval(k_err, shots)
            print(f"{name:<44} | {p_hat*100:6.3f}%       | [{l*100:.3f}%, {u*100:.3f}%]   | {k_err:>5d}/{shots} | {lat:6.2f} ms")
        
        mod_rate = (dynamic_prior_updates / max(len(active_shots), 1)) * 100
        print("-" * 118)
        print(f"  [Diagnostic] Active Syndrome Events: {len(active_shots):,}/{shots:,} ({len(active_shots)/shots*100:.2f}%)")
        print(f"  [Diagnostic] Edge-Prior PyMatching Modulations: {dynamic_prior_updates:,} ({mod_rate:.2f}% of active events)")
        print(f"  [Timing] Distance d={d} completed in {time.time() - t_dist_start:.2f}s\n")

if __name__ == "__main__":
    run_evaluation(test_distances=[9, 11, 13], shots=1000)
