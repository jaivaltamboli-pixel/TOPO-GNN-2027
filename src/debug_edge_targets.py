import torch
import torch.nn as nn
import numpy as np
import stim

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import TopoDephaseGNN

def diagnose_training_targets(distances=[3, 5, 7], p_val=0.002, eta=100.0, shots=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print(f"TRAINING TARGET & SUBGRAPH DENSITY AUDIT (Distances: {distances}, Shots: {shots})")
    print("=" * 100 + "\n")

    model = TopoDephaseGNN().to(device)
    try:
        model.load_state_dict(torch.load("checkpoints/topo_dephase_gnn.pt", map_location=device))
        model.eval()
        has_weights = True
    except Exception:
        has_weights = False

    for d in distances:
        circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
        dem = circuit.detector_error_model(decompose_errors=True)
        coords = circuit.get_detector_coordinates()
        num_dets = circuit.num_detectors
        edge_dict, bnd_z_idx, bnd_x_idx, dem_fault_to_edge = extract_complete_dem_graph(dem, num_dets, coords, d)
        dem_sampler = dem.compile_sampler()

        det_data, obs_data, err_data = dem_sampler.sample(shots=shots, return_errors=True)

        total_edges = 0
        positive_targets = 0
        predicted_over_50 = 0
        predicted_over_95 = 0
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

            n_e = e_targ.size(0)
            n_pos = int(e_targ.sum().item())
            total_edges += n_e
            positive_targets += n_pos

            if has_weights:
                with torch.no_grad():
                    _, edge_logits = model(x6, e_idx, e_attr, e_par)
                    if edge_logits.numel() > 0:
                        probs = torch.sigmoid(edge_logits).cpu().numpy().flatten()
                        all_probs.extend(probs.tolist())
                        predicted_over_50 += int(np.sum(probs > 0.50))
                        predicted_over_95 += int(np.sum(probs > 0.95))

        target_pos_rate = (positive_targets / max(total_edges, 1)) * 100
        print(f"--- Code Distance d={d} Audit ---")
        print(f"  Total Subgraph Edges Sampled: {total_edges:,}")
        print(f"  True Positive Edge Targets:   {positive_targets:,} ({target_pos_rate:.2f}%)")
        if has_weights and len(all_probs) > 0:
            pred_50_rate = (predicted_over_50 / total_edges) * 100
            pred_95_rate = (predicted_over_95 / total_edges) * 100
            print(f"  Predicted p > 0.50:           {predicted_over_50:,} ({pred_50_rate:.2f}%)")
            print(f"  Predicted p > 0.95:           {predicted_over_95:,} ({pred_95_rate:.2f}%)")
            print(f"  Mean Model Probability:       {np.mean(all_probs):.4f}")
        print()

if __name__ == "__main__":
    diagnose_training_targets()
