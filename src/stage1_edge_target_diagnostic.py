import numpy as np
import stim
import torch

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors

def run_edge_diagnostic(d=3, shots=200):
    print("=" * 70)
    print(f"STAGE 1: PHYSICAL FAULT-TO-EDGE GROUND-TRUTH DIAGNOSTIC (d={d}, Shots={shots})")
    print("=" * 70)

    circuit = make_biased_surface_code(d=d, rounds=d, p_total=0.002, eta=100.0)
    dem = circuit.detector_error_model(decompose_errors=True)
    coords = circuit.get_detector_coordinates()
    num_dets = circuit.num_detectors

    edge_dict, bnd_z_idx, bnd_x_idx, dem_fault_to_edge = extract_complete_dem_graph(dem, num_dets, coords, d)
    
    # Compile DEM error sampler to sample exact fault mechanisms
    dem_sampler = dem.compile_sampler()
    det_data, obs_data, err_data = dem_sampler.sample(shots=shots, return_errors=True)

    total_subgraph_edges = 0
    positive_edge_targets = 0
    active_shots_count = 0

    for i in range(shots):
        s_vec = det_data[i]
        if np.sum(s_vec) < 1:
            continue
        active_shots_count += 1

        # Extract active physical fault pairs triggered in this shot
        active_fault_indices = np.where(err_data[i])[0]
        active_fault_pairs = set()
        for f_idx in active_fault_indices:
            if f_idx < len(dem_fault_to_edge):
                pair = dem_fault_to_edge[f_idx]
                if pair is not None:
                    active_fault_pairs.add(pair)

        x4, x6, e_idx, e_attr, e_par, s_t, e_targ, global_pairs = extract_active_subgraph_tensors(
            s_vec, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, "cpu", active_fault_pairs=active_fault_pairs
        )

        num_edges = e_targ.size(0)
        pos_edges = int(e_targ.sum().item())

        total_subgraph_edges += num_edges
        positive_edge_targets += pos_edges

    pos_rate = (positive_edge_targets / max(total_subgraph_edges, 1)) * 100

    print(f"  [+] Physical DEM Total Edges in Dictionary: {len(edge_dict)}")
    print(f"  [+] DEM Fault Mechanisms Mapped: {len(dem_fault_to_edge)}")
    print(f"  [+] Active Syndrome Shots Sampled: {active_shots_count}/{shots}")
    print(f"  [+] Subgraph Edges Evaluated Across Shots: {total_subgraph_edges:,}")
    print(f"  [+] Positive Physical Fault Target Labels: {positive_edge_targets:,} ({pos_rate:.2f}% active)")
    print("=" * 70)

    assert positive_edge_targets > 0, "CRITICAL ERROR: Zero positive edge targets found! Do not train."
    print("  [SUCCESS] Physical edge ground-truth mapping verified. Stage 1 passed!\n")

if __name__ == "__main__":
    run_edge_diagnostic()
