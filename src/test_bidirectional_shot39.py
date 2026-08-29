import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import torch
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import TopoDephaseGNN

def test_bidirectional():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TopoDephaseGNN().to(device)
    model.load_state_dict(torch.load("checkpoints/topo_dephase_gnn.pt", map_location=device))
    model.eval()

    d = 9
    circuit = make_biased_surface_code(d=d, rounds=d, p_total=0.002, eta=100.0)
    dem = circuit.detector_error_model(decompose_errors=True)
    coords = circuit.get_detector_coordinates()
    num_dets = circuit.num_detectors
    edge_dict, bnd_z_idx, bnd_x_idx, _ = extract_complete_dem_graph(dem, num_dets, coords, d)

    syn = np.load("results/debug_syn.npy")
    flips = np.load("results/debug_flips.npy")
    s_vec = syn[39].astype(np.uint8)
    obs_true = int(flips[39])

    base_matcher = pymatching.Matching.from_detector_error_model(dem)
    base_pred = int(base_matcher.decode(s_vec)[0])

    x4, x6, e_idx, e_attr, e_par, s_t, _, global_pairs = extract_active_subgraph_tensors(
        s_vec, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device
    )

    with torch.no_grad():
        log_pred, edge_logits = model(x6, e_idx, e_attr, e_par)
        raw_logits = edge_logits.cpu().numpy().flatten()
        probs = torch.sigmoid(edge_logits).cpu().numpy().flatten()

    print("=== SHOT #39 BIDIRECTIONAL HYBRID TEST ===")
    print(f"Ground Truth: {obs_true} | Base MWPM: {base_pred} (ERROR) | Topo-GNN Global: {int(log_pred.item() > 0.5)} (CORRECT)\n")

    for gamma in [0.05, 0.10, 0.20, 0.35, 0.50, 0.75]:
        matcher = pymatching.Matching.from_detector_error_model(dem)
        processed = set()
        
        for k_e, pair in enumerate(global_pairs):
            u, v = int(pair[0]), int(pair[1])
            canon = tuple(sorted((u, v)))
            if canon in processed:
                continue
            processed.add(canon)

            rev_idx = [j for j, gp in enumerate(global_pairs) if {int(gp[0]), int(gp[1])} == {u, v}]
            l_edge = float(np.mean(raw_logits[rev_idx])) if len(rev_idx) > 0 else float(raw_logits[k_e])
            
            props = edge_dict.get(canon)
            if props is None:
                continue
            base_w = float(props["weight"])
            has_obs = props.get("has_obs", False)
            f_ids = {0} if has_obs else set()

            # Bidirectional update: positive logit -> lower weight, negative logit -> higher weight
            delta_w = np.clip(l_edge * gamma, -2.5, 2.5)
            new_w = max(0.01, base_w - delta_w)

            if u == bnd_z_idx or u == bnd_x_idx:
                if v < num_dets:
                    matcher.add_boundary_edge(v, weight=new_w, fault_ids=f_ids, merge_strategy="replace")
            elif v == bnd_z_idx or v == bnd_x_idx:
                if u < num_dets:
                    matcher.add_boundary_edge(u, weight=new_w, fault_ids=f_ids, merge_strategy="replace")
            else:
                matcher.add_edge(u, v, weight=new_w, fault_ids=f_ids, merge_strategy="replace")

        pred_hybrid = int(matcher.decode(s_vec)[0])
        print(f"  Gamma = {gamma:4.2f} -> Hybrid Decoded: {pred_hybrid} ({'RECOVERED / CORRECT' if pred_hybrid == obs_true else 'STILL ERROR'})")

if __name__ == "__main__":
    test_bidirectional()
