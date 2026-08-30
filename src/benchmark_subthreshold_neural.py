import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import json
import time
import torch
import numpy as np
import stim
import pymatching
import matplotlib.pyplot as plt

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from utils.metrics import wilson_score_interval
from models import TopoDephaseGNN, LangeIsotropicMPNN

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

def benchmark_subthreshold(
    distances=[3, 5, 7, 9],
    p_grid=[0.0008, 0.0012, 0.0016, 0.0020, 0.0025, 0.0035],
    eta=100.0,
    shots=5000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 110)
    print(f"SUB-THRESHOLD DIAGNOSTIC SWEEP (Distances: {distances}, Shots/pt: {shots:,}, Bias eta={eta})")
    print("=" * 110 + "\n")

    models = {
        "Topo-DephaseGNN": TopoDephaseGNN().to(device),
        "Lange-MPNN": LangeIsotropicMPNN().to(device),
    }

    models["Topo-DephaseGNN"].load_state_dict(torch.load("checkpoints/topo_dephase_gnn.pt", map_location=device))
    models["Lange-MPNN"].load_state_dict(torch.load("checkpoints/lange_mpnn.pt", map_location=device))

    for m in models.values():
        m.eval()

    results = {
        "metadata": {"distances": distances, "p_grid": p_grid, "eta": eta, "shots": shots},
        "data": {"MWPM": {}, "Topo-DephaseGNN": {}, "Lange-MPNN": {}}
    }

    for dec in results["data"]:
        results["data"][dec] = {str(d): [] for d in distances}

    for p in p_grid:
        print(f"\n--- Testing Physical Error Rate p = {p:.4f} ---")
        for d in distances:
            circuit = make_biased_surface_code(d=d, rounds=d, p_total=p, eta=eta)
            dem = circuit.detector_error_model(decompose_errors=True)
            coords = circuit.get_detector_coordinates()
            num_dets = circuit.num_detectors
            edge_dict, bnd_z_idx, bnd_x_idx, _ = extract_complete_dem_graph(dem, num_dets, coords, d)

            sampler = circuit.compile_detector_sampler()
            syn, flips = sampler.sample(shots=shots, separate_observables=True)
            flips = flips.flatten().astype(np.int64)

            # MWPM baseline
            base_matcher = pymatching.Matching.from_detector_error_model(dem)
            mwpm_preds = base_matcher.decode_batch(syn).flatten().astype(np.int64)

            preds_topo = np.zeros(shots, dtype=np.int64)
            preds_lange = np.zeros(shots, dtype=np.int64)

            active_shots = np.where(np.sum(syn, axis=1) >= 2)[0]

            for idx in active_shots:
                s = syn[idx].astype(np.uint8)
                x4, x6, e_idx, e_attr, e_par, _, _, _ = extract_active_subgraph_tensors(
                    s, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device
                )
                if e_idx.numel() == 0:
                    continue

                with torch.no_grad():
                    p_lange = models["Lange-MPNN"](x4, e_idx, e_attr).item()
                    preds_lange[idx] = int(p_lange > 0.5)

                    log_topo, _ = models["Topo-DephaseGNN"](x6, e_idx, e_attr, e_par)
                    preds_topo[idx] = int(log_topo.item() > 0.5)

            batch = {"MWPM": mwpm_preds, "Topo-DephaseGNN": preds_topo, "Lange-MPNN": preds_lange}

            for name, preds in batch.items():
                k_err = int(np.sum(preds != flips))
                p_hat, l_ci, u_ci = wilson_score_interval(k_err, shots)
                record = {
                    "p_phys": p,
                    "p_logical": float(p_hat),
                    "ci_lower": float(l_ci),
                    "ci_upper": float(u_ci),
                    "errors": k_err,
                    "shots": shots
                }
                results["data"][name][str(d)].append(record)

            print(f"  d={d:2d} | MWPM: {np.mean(mwpm_preds != flips)*100:6.3f}% | Topo-GNN: {np.mean(preds_topo != flips)*100:6.3f}% | Lange: {np.mean(preds_lange != flips)*100:6.3f}%")

    with open("results/subthreshold_scaling_data.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot Sub-Threshold Diagnostic Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax, dec in zip(axes, ["MWPM", "Topo-DephaseGNN", "Lange-MPNN"]):
        for d in distances:
            pts = results["data"][dec][str(d)]
            p_vals = np.array([pt["p_phys"] for pt in pts])
            p_logs = np.array([pt["p_logical"] for pt in pts])
            lows = np.array([pt["ci_lower"] for pt in pts])
            highs = np.array([pt["ci_upper"] for pt in pts])

            lower_err = np.clip(p_logs - lows, 0.0, None)
            upper_err = np.clip(highs - p_logs, 0.0, None)
            yerr = [lower_err, upper_err]

            ax.errorbar(
                p_vals, p_logs, yerr=yerr,
                label=f"d = {d}",
                marker="o", markersize=4,
                capsize=3, linewidth=1.8
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical Error Rate ($p$)", fontsize=11, fontweight="bold")
        ax.set_title(rf"{dec} Scaling Diagnostic ($\eta={eta}$)", fontsize=12, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)

    axes[0].set_ylabel("Logical Error Rate ($P_L$)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/subthreshold_scaling_curves.png", dpi=300)
    plt.close()
    print("\n[+] Saved sub-threshold diagnostic curves to figures/subthreshold_scaling_curves.png\n")

if __name__ == "__main__":
    benchmark_subthreshold()
