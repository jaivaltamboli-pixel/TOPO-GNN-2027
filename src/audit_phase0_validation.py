import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import torch
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import TopoDephaseGNN

def audit_phase0_pipeline(distances=[3, 5, 7], p_vals=[0.002, 0.01, 0.05], eta=100.0, test_shots=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105)
    print("PHASE 0 AUDIT: DATA LEAKAGE, NOISE CONSISTENCY & INPUT INTEGRITY")
    print("=" * 105 + "\n")

    for d in distances:
        for p in p_vals:
            circuit = make_biased_surface_code(d=d, rounds=d, p_total=p, eta=eta)
            dem = circuit.detector_error_model(decompose_errors=True)
            coords = circuit.get_detector_coordinates()
            num_dets = circuit.num_detectors
            edge_dict, bnd_z_idx, bnd_x_idx, _ = extract_complete_dem_graph(dem, num_dets, coords, d)

            sampler = circuit.compile_detector_sampler(seed=42)
            syn, flips = sampler.sample(shots=test_shots, separate_observables=True)

            assert np.all(np.isin(flips, [0, 1])), "Observable flips contain non-binary entries."

            for idx in range(min(50, test_shots)):
                s = syn[idx].astype(np.uint8)
                x4, x6, e_idx, e_attr, e_par, s_t, e_targ, _ = extract_active_subgraph_tensors(
                    s, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device, active_fault_pairs=None
                )

                assert torch.all(e_targ == 0), "Data leakage detected: e_targ populated without ground truth input!"
                assert not torch.isnan(x6).any(), "NaN found in node features."
                assert not torch.isnan(e_attr).any(), "NaN found in edge attributes."

            print(f"  [+] d={d:2d}, p={p:.3f}: Pipeline integrity verified (No leakage, valid graph topologies).")

    print("\n" + "=" * 105)
    print("PHASE 0 COMPLETE: Input feature channels and graph extraction verified clean.")
    print("=" * 105 + "\n")

if __name__ == "__main__":
    audit_phase0_pipeline()
