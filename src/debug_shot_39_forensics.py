import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import torch
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import TopoDephaseGNN

def audit_shot_39_forensics(d=9, p_val=0.002, eta=100.0, target_shot=39):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105)
    print(f"SHOT #{target_shot} FORENSIC AUDIT: MWPM FAILURE VS. GNN PREDICTIONS (d={d})")
    print("=" * 105 + "\n")

    model = TopoDephaseGNN().to(device)
    model.load_state_dict(torch.load("checkpoints/topo_dephase_gnn.pt", map_location=device))
    model.eval()

    circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
    dem = circuit.detector_error_model(decompose_errors=True)
    coords = circuit.get_detector_coordinates()
    num_dets = circuit.num_detectors
    edge_dict, bnd_z_idx, bnd_x_idx, dem_fault_to_edge = extract_complete_dem_graph(dem, num_dets, coords, d)

    syn = np.load("results/debug_syn.npy")
    flips = np.load("results/debug_flips.npy")
    s_vec = syn[target_shot].astype(np.uint8)
    obs_true = int(flips[target_shot])

    base_matcher = pymatching.Matching.from_detector_error_model(dem)
    base_pred = int(base_matcher.decode(s_vec)[0])

    print(f"  Target Shot #{target_shot} Metadata:")
    print(f"    Active Syndrome Defects: {np.sum(s_vec)} defects at indices: {np.where(s_vec)[0].tolist()}")
    print(f"    Ground-Truth Observable: {obs_true}")
    print(f"    Base MWPM Decoded:       {base_pred} (ERROR)")
    print()

    x4, x6, e_idx, e_attr, e_par, s_t, _, global_pairs = extract_active_subgraph_tensors(
        s_vec, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device
    )

    if e_idx.numel() == 0:
        print("  [-] No subgraph edges constructed for this syndrome.")
        return

    with torch.no_grad():
        log_pred, edge_logits = model(x6, e_idx, e_attr, e_par)
        probs = torch.sigmoid(edge_logits).cpu().numpy().flatten()
        raw_logits = edge_logits.cpu().numpy().flatten()

    print(f"  Topo-DephaseGNN Global Coset Head Prediction: {log_pred.item():.4f} (Threshold=0.5 -> {int(log_pred.item() > 0.5)})")
    print(f"  Total Subgraph Edges: {len(probs)}")
    print(f"  Max Edge Prob: {probs.max():.4f} | Min Edge Prob: {probs.min():.4f} | Mean: {probs.mean():.4f}\n")

    print(f"  {'Candidate Edge (u <-> v)':<26} | {'Logit':<8} | {'p_GNN':<8} | {'Base Weight':<12} | {'has_obs'}")
    print("  " + "-" * 75)

    processed = set()
    for k_e, pair in enumerate(global_pairs):
        u, v = int(pair[0]), int(pair[1])
        canon = tuple(sorted((u, v)))
        if canon in processed:
            continue
        processed.add(canon)

        rev_idx = [j for j, gp in enumerate(global_pairs) if {int(gp[0]), int(gp[1])} == {u, v}]
        p_edge = float(np.mean(probs[rev_idx])) if len(rev_idx) > 0 else float(probs[k_e])
        l_edge = float(np.mean(raw_logits[rev_idx])) if len(rev_idx) > 0 else float(raw_logits[k_e])
        props = edge_dict.get(canon, {})
        base_w = float(props.get("weight", 0.0))
        has_obs = props.get("has_obs", False)

        edge_label = f"({u}, {v})" if u < num_dets and v < num_dets else f"({min(u,v)}, BND)"
        print(f"  {edge_label:<26} | {l_edge:8.4f} | {p_edge:8.4f} | {base_w:12.4f} | {has_obs}")

    print("=" * 105 + "\n")

if __name__ == "__main__":
    audit_shot_39_forensics()
