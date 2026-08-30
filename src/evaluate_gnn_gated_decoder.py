import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import stim
import pymatching
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from utils.metrics import wilson_score_interval
from models import TopoDephaseGNN
from audit_phase_a_d_opportunity import (
    build_parity_expanded_graph,
    find_exact_logical_reference_chain,
    standardize_edge,
    compute_chain_observable
)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

def collect_shot_indexed_dataset(distances, p_val, eta, total_shots, device):
    """
    Collects dataset while preserving 100% exact original shot index mapping.
    """
    dataset = {}
    for d in distances:
        t0 = time.time()
        circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
        dem = circuit.detector_error_model(decompose_errors=True)
        coords = circuit.get_detector_coordinates()
        num_dets = circuit.num_detectors

        edge_dict, adj, bnd_z_idx, bnd_x_idx = build_parity_expanded_graph(dem, num_dets, coords, d)
        raw_edge_dict, _, _, _ = extract_complete_dem_graph(dem, num_dets, coords, d)
        R_L, _ = find_exact_logical_reference_chain(adj, num_dets)

        matcher = pymatching.Matching.from_detector_error_model(dem)
        sampler = circuit.compile_detector_sampler()
        syn, flips = sampler.sample(shots=total_shots, separate_observables=True)
        flips = flips.flatten().astype(np.int64)

        preds_mwpm = matcher.decode_batch(syn).flatten().astype(np.int64)
        mwpm_wrong = (preds_mwpm != flips).astype(np.int64)

        tensors_by_shot = {}
        active_indices = np.where(np.sum(syn, axis=1) >= 2)[0]

        for idx in active_indices:
            s = syn[idx].astype(np.uint8)
            x4, x6, e_idx, e_attr, e_par, _, _, _ = extract_active_subgraph_tensors(
                s, coords, raw_edge_dict, bnd_z_idx, bnd_x_idx, d, torch.device("cpu")
            )
            if e_idx.numel() > 0:
                tensors_by_shot[idx] = (x6, e_idx, e_attr, e_par)

        dataset[d] = {
            "circuit": circuit, "dem": dem, "coords": coords, "num_dets": num_dets,
            "edge_dict": edge_dict, "raw_edge_dict": raw_edge_dict, "R_L": R_L,
            "bnd_z_idx": bnd_z_idx, "bnd_x_idx": bnd_x_idx, "matcher": matcher,
            "syn": syn, "flips": flips, "preds_mwpm": preds_mwpm, "mwpm_wrong": mwpm_wrong,
            "tensors_by_shot": tensors_by_shot, "total_shots": total_shots,
            "total_fails": int(np.sum(mwpm_wrong))
        }
        print(f"  [+] Collected d={d:2d} ({total_shots:,} shots, {np.sum(mwpm_wrong):>4d} failures, {len(tensors_by_shot):,d} active subgraphs) in {time.time()-t0:.2f}s")
    return dataset

def train_or_load_failure_gate(train_data, train_distances, target_epochs=3, lr=3e-4, device="cuda"):
    """
    Loads epoch-3 checkpoint if present; otherwise trains for exactly 3 epochs, saving after every epoch.
    """
    ckpt_path = "checkpoints/topo_failure_gate_epoch3.pt"
    model = TopoDephaseGNN().to(device)

    if os.path.exists(ckpt_path):
        print(f"\n[Phase F - Step 1] Found existing checkpoint '{ckpt_path}'. Loading directly...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        return model

    print(f"\n[Phase F - Step 1] Checkpoint not found. Running lightweight training ({target_epochs} epochs with per-epoch saves)...")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    pos_samples = []
    neg_samples = []

    for d in train_distances:
        dat = train_data[d]
        for idx, (x6, e_idx, e_attr, e_par) in dat["tensors_by_shot"].items():
            y_fail = dat["mwpm_wrong"][idx]
            if y_fail == 1:
                pos_samples.append((x6, e_idx, e_attr, e_par))
            else:
                neg_samples.append((x6, e_idx, e_attr, e_par))

    print(f"  Training Pool: {len(pos_samples):,d} MWPM Failures (Positives) | {len(neg_samples):,d} Correct Shots (Negatives)")

    batch_pos = 64
    batch_neg = 64
    steps_per_epoch = 400

    for epoch in range(1, target_epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for _ in range(steps_per_epoch):
            p_idx = np.random.choice(len(pos_samples), batch_pos, replace=True)
            n_idx = np.random.choice(len(neg_samples), batch_neg, replace=False)

            batch = [(pos_samples[i], 1.0) for i in p_idx] + [(neg_samples[j], 0.0) for j in n_idx]
            np.random.shuffle(batch)

            optimizer.zero_grad()
            batch_loss = 0.0

            for (x6, e_idx, e_attr, e_par), y_val in batch:
                pred_prob, _ = model(x6.to(device), e_idx.to(device), e_attr.to(device), e_par.to(device))
                loss = criterion(pred_prob.squeeze(-1), torch.tensor([y_val], device=device))
                batch_loss += loss

            batch_loss = batch_loss / (batch_pos + batch_neg)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        epoch_save_path = f"checkpoints/topo_failure_gate_epoch{epoch}.pt"
        torch.save(model.state_dict(), epoch_save_path)
        print(f"  Epoch {epoch:2d}/{target_epochs:2d} | Balanced BCE Loss: {total_loss/steps_per_epoch:6.4f} | Saved: {epoch_save_path} | Time: {time.time()-t0:5.2f}s")

    return model

def evaluate_gated_decoder(
    train_distances=[3, 5, 7],
    test_distance=9,
    p_val=0.002,
    eta=100.0,
    train_shots_per_dist=100000,
    test_shots=100000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 130)
    print(f"PHASE F: EPOCH-3 GNN-GATED DUAL-CANDIDATE DECODER EVALUATION")
    print(f"Configuration: p={p_val}, Bias eta={eta}, Train Distances: {train_distances}, Zero-Shot Held-Out: d={test_distance}")
    print("=" * 130 + "\n")

    print("[Step 1/4] Preparing dataset for training distances...")
    train_data = collect_shot_indexed_dataset(train_distances, p_val, eta, train_shots_per_dist, device)

    # Train or load epoch-3 model
    model = train_or_load_failure_gate(train_data, train_distances, target_epochs=3, lr=3e-4, device=device)
    model.eval()

    print("[Step 2/4] Generating independent evaluation datasets (with held-out zero-shot d=9)...")
    eval_distances = train_distances + [test_distance]
    eval_data = collect_shot_indexed_dataset(eval_distances, p_val, eta, test_shots, device)

    print("\n" + "=" * 130)
    print("[Step 3/4] ORACLE GATE VERIFICATION (Testing theoretical 0-error ceiling)")
    print("=" * 130)
    print(f"{'Distance':<12} | {'MWPM P_L':<10} | {'Oracle Gate P_L':<16} | {'MWPM Errors':<12} | {'Oracle Errors':<14} | {'Recoverability':<14}")
    print("-" * 130)

    for d in eval_distances:
        dat = eval_data[d]
        shots = dat["total_shots"]
        flips = dat["flips"]
        preds_mwpm = dat["preds_mwpm"]
        mwpm_wrong = dat["mwpm_wrong"]
        R_L = dat["R_L"]
        edge_dict = dat["edge_dict"]
        bnd_z = dat["bnd_z_idx"]
        bnd_x = dat["bnd_x_idx"]
        matcher = dat["matcher"]
        syn = dat["syn"]

        oracle_preds = preds_mwpm.copy()
        for idx in range(shots):
            if mwpm_wrong[idx] == 1:
                s = syn[idx].astype(np.uint8)
                edges_a_raw = matcher.decode_to_edges_array(s)
                C_A = set(standardize_edge(int(e[0]), int(e[1]), bnd_z, bnd_x) for e in edges_a_raw)
                C_B = C_A.symmetric_difference(R_L)
                oracle_preds[idx] = compute_chain_observable(C_B, edge_dict)

        mwpm_errs = int(np.sum(preds_mwpm != flips))
        oracle_errs = int(np.sum(oracle_preds != flips))
        rec_pct = ((mwpm_errs - oracle_errs) / mwpm_errs * 100.0) if mwpm_errs > 0 else 0.0
        print(f"d = {d:<8d} | {mwpm_errs/shots*100:6.3f}%   | {oracle_errs/shots*100:6.3f}%         | {mwpm_errs:>5d}/{shots}   | {oracle_errs:>5d}/{shots}     | {rec_pct:6.2f}%")

    print("-" * 130)

    print("\n" + "=" * 130)
    print("[Step 4/4] LEARNED GNN-GATED DECODER THRESHOLD SWEEP (tau in [0.01, 0.99])")
    print("=" * 130 + "\n")

    tau_sweep = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
    optimal_results = []

    for d in eval_distances:
        t_dist = time.time()
        dat = eval_data[d]
        shots = dat["total_shots"]
        flips = dat["flips"]
        preds_mwpm = dat["preds_mwpm"]
        mwpm_wrong = dat["mwpm_wrong"]
        R_L = dat["R_L"]
        edge_dict = dat["edge_dict"]
        bnd_z = dat["bnd_z_idx"]
        bnd_x = dat["bnd_x_idx"]
        matcher = dat["matcher"]
        syn = dat["syn"]
        tensors_by_shot = dat["tensors_by_shot"]

        pred_probs = np.zeros(shots, dtype=np.float64)
        for idx, (x6, e_idx, e_attr, e_par) in tensors_by_shot.items():
            with torch.no_grad():
                prob, _ = model(x6.to(device), e_idx.to(device), e_attr.to(device), e_par.to(device))
                pred_probs[idx] = prob.item()

        auroc = roc_auc_score(mwpm_wrong, pred_probs) if np.sum(mwpm_wrong) > 0 else 0.0
        precisions, recalls, _ = precision_recall_curve(mwpm_wrong, pred_probs)
        auprc = auc(recalls, precisions) if np.sum(mwpm_wrong) > 0 else 0.0

        print(f">>> DISTANCE d = {d:2d} {'(HELD-OUT ZERO-SHOT)' if d==test_distance else ''} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} <<<")
        print(f"{'Tau':<6} | {'Gated P_L':<10} | {'MWPM P_L':<10} | {'Recoveries':<11} | {'Regressions':<12} | {'Net Gain':<10} | {'Recall':<8} | {'Precision':<10} | {'FPR':<8}")
        print("-" * 115)

        best_net_gain = -float("inf")
        best_row = None

        obs_B_cache = {}
        for idx in tensors_by_shot.keys():
            s = syn[idx].astype(np.uint8)
            edges_a_raw = matcher.decode_to_edges_array(s)
            C_A = set(standardize_edge(int(e[0]), int(e[1]), bnd_z, bnd_x) for e in edges_a_raw)
            C_B = C_A.symmetric_difference(R_L)
            obs_B_cache[idx] = compute_chain_observable(C_B, edge_dict)

        for tau in tau_sweep:
            gated_preds = preds_mwpm.copy()
            altered_shots = np.where(pred_probs >= tau)[0]

            for idx in altered_shots:
                if idx in obs_B_cache:
                    gated_preds[idx] = obs_B_cache[idx]

            gated_errs = int(np.sum(gated_preds != flips))
            mwpm_errs = int(np.sum(preds_mwpm != flips))

            rec = int(np.sum((preds_mwpm != flips) & (gated_preds == flips)))
            reg = int(np.sum((preds_mwpm == flips) & (gated_preds != flips)))
            net_gain = rec - reg

            pred_fail_binary = (pred_probs >= tau).astype(np.int64)
            tp = int(np.sum((pred_fail_binary == 1) & (mwpm_wrong == 1)))
            fp = int(np.sum((pred_fail_binary == 1) & (mwpm_wrong == 0)))
            tn = int(np.sum((pred_fail_binary == 0) & (mwpm_wrong == 0)))
            fn = int(np.sum((pred_fail_binary == 0) & (mwpm_wrong == 1)))

            rec_pct = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
            fpr_pct = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
            prec_pct = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0

            gated_pl = (gated_errs / shots) * 100.0
            mwpm_pl = (mwpm_errs / shots) * 100.0

            print(f"{tau:<6.2f} | {gated_pl:6.3f}%   | {mwpm_pl:6.3f}%   | {rec:>5d}/{mwpm_errs:<4d}  | {reg:>5d}/{shots-mwpm_errs:<6d} | {net_gain:>+5d}     | {rec_pct:6.2f}% | {prec_pct:6.2f}%    | {fpr_pct:5.2f}%")

            if net_gain > best_net_gain:
                best_net_gain = net_gain
                best_row = {
                    "d": d, "tau": tau, "gated_pl": gated_pl, "mwpm_pl": mwpm_pl,
                    "mwpm_errs": mwpm_errs, "gated_errs": gated_errs,
                    "rec": rec, "reg": reg, "net_gain": net_gain,
                    "recall": rec_pct, "precision": prec_pct, "fpr": fpr_pct,
                    "auroc": auroc, "auprc": auprc
                }

        optimal_results.append(best_row)
        print("-" * 115)
        print(f"  [Best Operating Point at d={d:2d}]: tau = {best_row['tau']:.2f} | Net Gain = {best_row['net_gain']:+d} | Gated P_L = {best_row['gated_pl']:.3f}% (MWPM: {best_row['mwpm_pl']:.3f}%)\n")

    print("=" * 135)
    print("FINAL PHASE F GNN-GATED DUAL-CANDIDATE SUMMARY TABLE (OPTIMAL THRESHOLDS)")
    print("=" * 135)
    print(f"{'Distance':<14} | {'MWPM P_L (95% CI)':<26} | {'Gated P_L (95% CI)':<26} | {'Optimal Tau':<12} | {'Rec / Reg':<12} | {'Net Gain':<10} | {'AUROC / AUPRC'}")
    print("-" * 135)

    for r in optimal_results:
        d = r["d"]
        shots = test_shots
        _, m_l, m_u = wilson_score_interval(r["mwpm_errs"], shots)
        _, g_l, g_u = wilson_score_interval(r["gated_errs"], shots)
        mwpm_str = f"{r['mwpm_pl']:5.3f}% [{m_l*100:.3f}%, {m_u*100:.3f}%]"
        gated_str = f"{r['gated_pl']:5.3f}% [{g_l*100:.3f}%, {g_u*100:.3f}%]"
        rec_reg_str = f"{r['rec']:>4d} / {r['reg']:<4d}"
        held_str = " (HELD-OUT)" if d == test_distance else ""
        print(f"d = {d:<2d}{held_str:<9} | {mwpm_str:<26} | {gated_str:<26} | tau = {r['tau']:<6.2f} | {rec_reg_str:<12} | {r['net_gain']:>+5d}      | {r['auroc']:.3f} / {r['auprc']:.3f}")

    print("=" * 135 + "\n")

if __name__ == "__main__":
    evaluate_gated_decoder()
