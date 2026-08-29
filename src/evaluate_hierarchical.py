import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import torch
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import TopoDephaseGNN

def evaluate_hierarchical_decoder(d=9, p_val=0.002, eta=100.0, conf_threshold=0.80):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105)
    print(f"HIERARCHICAL COSET-ARBITRATED HYBRID EVALUATION (d={d}, Confidence Gate={conf_threshold})")
    print("=" * 105 + "\n")

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
    preds_hierarchical = preds_base.copy()

    active_shots = np.where(np.sum(syn, axis=1) >= 2)[0]
    overridden_shots = 0

    for idx in active_shots:
        s = syn[idx].astype(np.uint8)
        x4, x6, e_idx, e_attr, e_par, s_t, _, _ = extract_active_subgraph_tensors(
            s, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device
        )
        if e_idx.numel() == 0:
            continue

        with torch.no_grad():
            log_pred, _ = model(x6, e_idx, e_attr, e_par)
            p_coset = log_pred.item()

        # High-confidence coset arbitration
        if p_coset > conf_threshold:
            nn_decision = 1
            if preds_base[idx] != nn_decision:
                preds_hierarchical[idx] = nn_decision
                overridden_shots += 1
        elif p_coset < (1.0 - conf_threshold):
            nn_decision = 0
            if preds_base[idx] != nn_decision:
                preds_hierarchical[idx] = nn_decision
                overridden_shots += 1

    err_base = int(np.sum(preds_base != flips))
    err_hier = int(np.sum(preds_hierarchical != flips))

    diff_shots = np.where(preds_hierarchical != preds_base)[0]
    recoveries = [i for i in diff_shots if preds_base[i] != flips[i] and preds_hierarchical[i] == flips[i]]
    regressions = [i for i in diff_shots if preds_base[i] == flips[i] and preds_hierarchical[i] != flips[i]]

    print("============================================================")
    print("HIERARCHICAL DECODER PERFORMANCE SUMMARY")
    print("============================================================")
    print(f"  Total Shots Evaluated:              {shots}")
    print(f"  Pure MWPM Errors:                   {err_base:>3d}/{shots} ({err_base/shots*100:6.3f}%)")
    print(f"  Hierarchical Hybrid Errors:         {err_hier:>3d}/{shots} ({err_hier/shots*100:6.3f}%)")
    print("-" * 60)
    print(f"  Total MWPM Decisions Overridden:    {overridden_shots}")
    print(f"  Recoveries (MWPM wrong -> NN right): {len(recoveries):>3d} {recoveries}")
    print(f"  Regressions (MWPM right -> NN wrong):{len(regressions):>3d} {regressions}")
    print(f"  Net Improvement:                    {len(recoveries) - len(regressions):+3d}")
    print("============================================================\n")

if __name__ == "__main__":
    evaluate_hierarchical_decoder()
