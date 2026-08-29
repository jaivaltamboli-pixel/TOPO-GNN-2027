import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import torch
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import TopoDephaseGNN

def evaluate_deterministic_batch(d=9, p_val=0.002, eta=100.0, shots=500, seed=42):
    # Set deterministic seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("results", exist_ok=True)
    print("=" * 105)
    print(f"DETERMINISTIC 3-WAY COUNTERFACTUAL AUDIT (d={d}, shots={shots}, seed={seed})")
    print("=" * 105 + "\n")

    model = TopoDephaseGNN().to(device)
    model.load_state_dict(torch.load("checkpoints/topo_dephase_gnn.pt", map_location=device))
    model.eval()

    circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
    dem = circuit.detector_error_model(decompose_errors=True)
    coords = circuit.get_detector_coordinates()
    num_dets = circuit.num_detectors
    edge_dict, bnd_z_idx, bnd_x_idx, _ = extract_complete_dem_graph(dem, num_dets, coords, d)

    sampler = circuit.compile_detector_sampler(seed=seed)
    syn, flips = sampler.sample(shots=shots, separate_observables=True)
    flips = flips.flatten().astype(np.int64)

    np.save("results/debug_syn.npy", syn)
    np.save("results/debug_flips.npy", flips)

    base_matcher = pymatching.Matching.from_detector_error_model(dem)
    preds_base = base_matcher.decode_batch(syn).flatten().astype(np.int64)
    preds_lower = preds_base.copy()
    preds_higher = preds_base.copy()

    active_shots = np.where(np.sum(syn, axis=1) >= 2)[0]
    first_divergence_logged = False

    for idx in active_shots:
        s = syn[idx].astype(np.uint8)
        x4, x6, e_idx, e_attr, e_par, s_t, _, global_pairs = extract_active_subgraph_tensors(
            s, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device
        )
        if e_idx.numel() == 0:
            continue

        with torch.no_grad():
            _, edge_logits = model(x6, e_idx, e_attr, e_par)
            if edge_logits.numel() == 0:
                continue
            probs = torch.sigmoid(edge_logits).cpu().numpy().flatten()

        processed = set()
        edges_to_mod = []

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
                    edges_to_mod.append({
                        "u": u, "v": v, "canon": canon,
                        "p": p_edge,
                        "base_w": float(props["weight"]),
                        "has_obs": props.get("has_obs", False)
                    })

        if len(edges_to_mod) > 0:
            matcher_low = pymatching.Matching.from_detector_error_model(dem)
            matcher_high = pymatching.Matching.from_detector_error_model(dem)

            for item in edges_to_mod:
                u, v = item["u"], item["v"]
                base_w = item["base_w"]
                w_low = max(0.01, base_w * 0.85)
                w_high = base_w * 1.15
                fault_ids = {0} if item["has_obs"] else set()

                if u == bnd_z_idx or u == bnd_x_idx:
                    if v < num_dets:
                        matcher_low.add_boundary_edge(v, weight=w_low, fault_ids=fault_ids, merge_strategy="replace")
                        matcher_high.add_boundary_edge(v, weight=w_high, fault_ids=fault_ids, merge_strategy="replace")
                elif v == bnd_z_idx or v == bnd_x_idx:
                    if u < num_dets:
                        matcher_low.add_boundary_edge(u, weight=w_low, fault_ids=fault_ids, merge_strategy="replace")
                        matcher_high.add_boundary_edge(u, weight=w_high, fault_ids=fault_ids, merge_strategy="replace")
                else:
                    matcher_low.add_edge(u, v, weight=w_low, fault_ids=fault_ids, merge_strategy="replace")
                    matcher_high.add_edge(u, v, weight=w_high, fault_ids=fault_ids, merge_strategy="replace")

            p_l = int(matcher_low.decode(s)[0])
            p_h = int(matcher_high.decode(s)[0])
            preds_lower[idx] = p_l
            preds_higher[idx] = p_h

            if (p_l != preds_base[idx] or p_h != preds_base[idx]) and not first_divergence_logged:
                first_divergence_logged = True
                print("=" * 60)
                print(f"FIRST DIVERGENT SHOT FOUND: Shot #{idx}")
                print("=" * 60)
                print(f"  Ground Truth Observable Flip:   {flips[idx]}")
                print(f"  Base MWPM Prediction:          {preds_base[idx]}")
                print(f"  Lower-Cost Hybrid Prediction:   {p_l}")
                print(f"  Higher-Cost Hybrid Prediction:  {p_h}")
                print(f"  Modified Edges ({len(edges_to_mod)} total):")
                for e in edges_to_mod[:8]:
                    print(f"    ({e['u']:>3d}, {e['v']:>3d}) | p={e['p']:.4f} | base_w={e['base_w']:.3f} | has_obs={e['has_obs']}")
                print("=" * 60 + "\n")

    # Metrics
    all_three_agree = int(np.sum((preds_base == preds_lower) & (preds_base == preds_higher)))
    base_diff_lower = int(np.sum(preds_base != preds_lower))
    base_diff_higher = int(np.sum(preds_base != preds_higher))
    lower_diff_higher = int(np.sum(preds_lower != preds_higher))

    err_base = int(np.sum(preds_base != flips))
    err_lower = int(np.sum(preds_lower != flips))
    err_higher = int(np.sum(preds_higher != flips))

    print("============================================================")
    print("DETERMINISTIC PAIRWISE AGREEMENT SUMMARY")
    print("============================================================")
    print(f"  All Three Decoders Agree:           {all_three_agree:>3d}/{shots} ({all_three_agree/shots*100:6.2f}%)")
    print(f"  Base MWPM != Lower-Cost (Mode A):   {base_diff_lower:>3d}/{shots} ({base_diff_lower/shots*100:6.2f}%)")
    print(f"  Base MWPM != Higher-Cost (Mode B):  {base_diff_higher:>3d}/{shots} ({base_diff_higher/shots*100:6.2f}%)")
    print(f"  Lower-Cost != Higher-Cost (A != B): {lower_diff_higher:>3d}/{shots} ({lower_diff_higher/shots*100:6.2f}%)")
    print("-" * 60)
    print(f"  Base MWPM Logical Errors:           {err_base:>3d}/{shots} ({err_base/shots*100:6.3f}%)")
    print(f"  Lower-Cost Prior Logical Errors:    {err_lower:>3d}/{shots} ({err_lower/shots*100:6.3f}%)")
    print(f"  Higher-Cost Prior Logical Errors:   {err_higher:>3d}/{shots} ({err_higher/shots*100:6.3f}%)")
    print("============================================================\n")

if __name__ == "__main__":
    evaluate_deterministic_batch()
