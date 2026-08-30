import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import TopoDephaseGNN
from verify_exact_dual_coset import find_exact_logical_reference_chain, compute_chain_observable, compute_chain_weight

os.makedirs("checkpoints", exist_ok=True)

def train_ranker(distances=[3, 5, 7], epochs=10, shots_per_dist=3000, p_val=0.002, eta=100.0, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105)
    print(f"PAIRWISE HOMOLOGY CANDIDATE RANKER TRAINING (Distances: {distances}, Epochs: {epochs}, Bias eta={eta})")
    print("=" * 105 + "\n")

    model = TopoDephaseGNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_ranks = 0
        total_samples = 0
        t0 = time.time()

        for d in distances:
            circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
            dem = circuit.detector_error_model(decompose_errors=True)
            coords = circuit.get_detector_coordinates()
            num_dets = circuit.num_detectors
            edge_dict, bnd_z_idx, bnd_x_idx, _ = extract_complete_dem_graph(dem, num_dets, coords, d)

            matcher = pymatching.Matching.from_detector_error_model(dem)
            R_L, s_ref = find_exact_logical_reference_chain(dem, num_dets, edge_dict, bnd_z_idx, bnd_x_idx)

            sampler = circuit.compile_detector_sampler()
            syn, flips = sampler.sample(shots=shots_per_dist, separate_observables=True)
            flips = flips.flatten().astype(np.int64)

            active_shots = np.where(np.sum(syn, axis=1) >= 2)[0]

            for idx in active_shots:
                s = syn[idx].astype(np.uint8)
                s_sh = (s ^ s_ref).astype(np.uint8)
                y_true = flips[idx]

                # Extract C_A and C_B
                C_A = set(tuple(sorted((int(e[0]), int(e[1])))) for e in matcher.decode_to_edges_array(s))
                C_shift = set(tuple(sorted((int(e[0]), int(e[1])))) for e in matcher.decode_to_edges_array(s_sh))
                C_B = C_shift.symmetric_difference(R_L)

                obs_A = compute_chain_observable(C_A, edge_dict)
                obs_B = compute_chain_observable(C_B, edge_dict)

                # Designate C0 (obs=0) and C1 (obs=1)
                C_0 = C_A if obs_A == 0 else C_B
                C_1 = C_B if obs_A == 0 else C_A

                w_0 = compute_chain_weight(C_0, edge_dict)
                w_1 = compute_chain_weight(C_1, edge_dict)

                # True label: y = +1 if C_0 is correct, -1 if C_1 is correct
                y_target = 1.0 if y_true == 0 else -1.0

                # Neural feature extraction
                x4, x6, e_idx, e_attr, e_par, _, _, global_pairs = extract_active_subgraph_tensors(
                    s, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device
                )
                if e_idx.numel() == 0:
                    continue

                optimizer.zero_grad()
                log_pred, edge_logits = model(x6, e_idx, e_attr, e_par)

                # Build edge logit mapping
                canon_to_idx = {tuple(sorted((int(gp[0]), int(gp[1])))): i for i, gp in enumerate(global_pairs)}

                phi_0 = torch.tensor(0.0, device=device)
                phi_1 = torch.tensor(0.0, device=device)

                for e in C_0:
                    if e in canon_to_idx:
                        phi_0 = phi_0 + edge_logits[canon_to_idx[e]]
                for e in C_1:
                    if e in canon_to_idx:
                        phi_1 = phi_1 + edge_logits[canon_to_idx[e]]

                # Delta Energy: (E1 - E0)
                # E_k = W_k - phi_k
                delta_E = (w_1 - w_0) - (phi_1 - phi_0)

                # Pairwise Margin Loss: log(1 + exp(-y * delta_E))
                loss = torch.log(1.0 + torch.exp(-y_target * delta_E))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predicted_coset = 0 if delta_E.item() > 0 else 1
                if predicted_coset == y_true:
                    correct_ranks += 1
                total_samples += 1

        acc = (correct_ranks / total_samples) * 100.0 if total_samples > 0 else 0.0
        print(f"Epoch {epoch:2d}/{epochs:2d} | Loss: {total_loss/total_samples:6.4f} | Candidate Ranking Acc: {acc:6.2f}% | Time: {time.time()-t0:5.2f}s")

    torch.save(model.state_dict(), "checkpoints/topo_candidate_ranker.pt")
    print("\n[+] Saved candidate ranker checkpoint to checkpoints/topo_candidate_ranker.pt\n")

if __name__ == "__main__":
    train_ranker()
